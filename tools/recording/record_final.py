"""Record a rendered replay of one final-evaluation campaign (winning attempts) and verify it is the same game.

Each level is played again by the same operator network with the journaled plan, save profile, seed and
protocol, on a recording copy of the host that keeps rendering (draw-time randomness isolated). The end state
is compared with the final run's journaled final_state_sha256.

usage: record_final.py <seed> [levels e.g. 1,2,3] ; env REC_TICKS_PER_FRAME (default 6), REC_FPS (default 60)
"""
import ctypes
import ctypes.wintypes
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "devrepo"))
os.environ["ALPHARUSH_RECORD"] = "1"
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.engine import DETERMINISTIC_MODE  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol, run_episode  # noqa: E402
from alpharush_rl.operator_net import OperatorPolicy, OptionScorer  # noqa: E402

FFMPEG = (r"C:\Users\<user>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
RUN = ROOT / "runtime/rl/campaign-v1/runs/native-final-1456be77e13c46598acab730468ec34c"
WEIGHTS = ROOT / "runtime/rl/campaign-v1/models/operator-d86c06c72ae1.json"
OUT = HERE / "video"
OUT.mkdir(exist_ok=True)
K = int(os.environ.get("REC_TICKS_PER_FRAME", "6"))
FPS = float(os.environ.get("REC_FPS", "60"))

seed = int(sys.argv[1])
levels = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 13))
rows = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
attempts = {p["level"]: p for r in rows if r["kind"] == "campaign_attempt"
            for p in [r["payload"]] if p["seed"] == seed and p["won"]}
net = OptionScorer.from_json(json.loads(WEIGHTS.read_text(encoding="utf-8")))

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # physical pixel coordinates, as gdigrab uses
except Exception:
    user32.SetProcessDPIAware()
EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)


def windows_of(pid):
    found = []

    def callback(hwnd, _):
        owner = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value:
                found.append((hwnd, buf.value))
        return True
    user32.EnumWindows(EnumProc(callback), 0)
    return found


WINDOW = 3  # pipelined quiet steps in flight; the host serves one request per rendered frame


def chunk_steps(worker):
    """Advance in K-tick quiet steps, pipelined so every rendered frame advances K ticks; then one state read."""
    original = worker.rpc

    def read_reply(expected_id):
        while b"\n" not in worker.buffer:
            data = worker.sock.recv(1024 * 1024)
            if not data:
                raise ConnectionError("Isolated game disconnected")
            worker.buffer += data
        line, worker.buffer = worker.buffer.split(b"\n", 1)
        response = json.loads(line)
        if response.get("id") != expected_id or not response.get("ok"):
            raise RuntimeError(f"pipelined step failed: {response}")
        return response["result"]

    def rpc(action, **kw):
        if action != "step" or not kw.get("ticks"):
            return original(action, **kw)
        ticks = kw["ticks"]
        sizes = [K] * (ticks // K) + ([ticks % K] if ticks % K else [])
        pending, results = [], []
        for n in sizes:
            worker.sequence += 1
            request = {"id": str(worker.sequence), "action": "step", "ticks": n, "quiet": True,
                       "token": worker.token}
            worker.sock.sendall((json.dumps(request) + "\n").encode())
            pending.append(request["id"])
            if len(pending) >= WINDOW:
                results.append(read_reply(pending.pop(0)))
        while pending:
            results.append(read_reply(pending.pop(0)))
        advanced = sum(r["tick_after"] - r["tick_before"] for r in results)
        terminated = any(r.get("terminated") for r in results)
        state = original("state")
        return {"tick_before": results[0]["tick_before"], "tick_after": results[0]["tick_before"] + advanced,
                "requested_ticks": ticks, "terminated": terminated, "state": state}
    worker.rpc = rpc


class Recorder:
    """Starts the window capture at the first decision (the level is loaded and under control)."""

    def __init__(self, inner, env, path):
        self.inner, self.name, self.env, self.path = inner, inner.name, env, path
        self.ffmpeg = None
        self.title = None

    def choose(self, state, menu, context):
        if self.ffmpeg is None:
            self.start()
        return self.inner.choose(state, menu, context)

    def start(self):
        pid = self.env.worker.process.pid
        for _ in range(100):
            found = windows_of(pid)
            if found:
                break
            time.sleep(0.05)
        hwnd, self.title = found[0]
        # Shown without activation and at the bottom of the Z-order: it never covers other windows.
        user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
        user32.SetWindowPos(hwnd, 1, 40, 40, 0, 0, 0x0001 | 0x0010)  # HWND_BOTTOM, keep size, no activate
        time.sleep(0.8)
        # Windows Graphics Capture of this window's own surface only (never other screen content).
        if os.environ.get("REC_ALPHA_PNG"):
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", f"gfxcapture=hwnd={hwnd}:max_framerate=30:capture_cursor=0",
                            "-vf", "hwdownload,format=bgra", "-frames:v", "3", str(OUT / "alpha_test_%d.png")])
        self.ffmpeg = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"gfxcapture=hwnd={hwnd}:max_framerate=30:capture_cursor=0",
             "-vf", "hwdownload,format=bgra,format=yuv420p", "-fps_mode", "cfr", "-r", "30",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(self.path)], stdin=subprocess.PIPE)
        time.sleep(0.8)

    def stop(self):
        if self.ffmpeg is not None:
            try:
                self.ffmpeg.communicate(b"q", timeout=60)
            except Exception:
                self.ffmpeg.kill()


report = []
for level in levels:
    p = attempts[level]
    expected = p["result"]
    protocol = EpisodeProtocol(**expected["protocol"])
    env = NativeEnv(seed=seed, level=level, port=9971, identity=f"devrec_{uuid.uuid4().hex[:8]}",
                    rng_mode=DETERMINISTIC_MODE, action_scope="v2", profile=p["profile"])
    chunk_steps(env.worker)
    path = OUT / f"seed{seed}_L{level:02d}_raw.mp4"
    recorder = Recorder(OperatorPolicy(net, p["genome"]), env, path)
    started = time.time()
    try:
        result = run_episode(env, recorder, protocol, seed=seed, level=level, difficulty=expected["difficulty"],
                             episode_id=f"recording-{seed}-L{level:02d}", observe_meta=True)
    finally:
        time.sleep(1.0)
        recorder.stop()
        env.close()
    out = result.get("outcome") or {}
    row = {"seed": seed, "level": level, "attempt": p["attempt"], "genome_id": p["genome_id"],
           "window_title": recorder.title, "video": path.name,
           "expected_final_state_sha256": expected["final_state_sha256"],
           "replayed_final_state_sha256": result.get("final_state_sha256"),
           "identical": result.get("final_state_sha256") == expected["final_state_sha256"],
           "expected_final_tick": expected["final_tick"], "replayed_final_tick": result.get("final_tick"),
           "expected_decisions": expected["n_decisions"], "replayed_decisions": result.get("n_decisions"),
           "won": out.get("level_won"), "lives": out.get("lives"), "wall_seconds": round(time.time() - started, 1),
           "ticks_per_frame": K}
    report.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (OUT / f"seed{seed}_verification.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
