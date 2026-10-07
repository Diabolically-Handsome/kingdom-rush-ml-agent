"""Re-execute every game of the elite-v1 one-shot final with the same operator networks and compare hashes.

usage: verify_final_reexec.py [workers] [port_base]
Each journaled game (campaign_attempt, star_replay, challenge_attempt) is played again from its journaled profile,
seed, level, mode and plan by the network that played it (campaign-v1 d86c06c72ae1 for main levels and star replays,
elite 1521c538876c for elite stages and challenges); trace and final-state hashes must be identical. Read-only.
"""
import json
import queue
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl.engine import DETERMINISTIC_MODE, level_scope  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol, run_episode  # noqa: E402
from alpharush_rl.operator_net import OperatorPolicy, OptionScorer  # noqa: E402

if (ROOT / "runtime/rl/elite-v1/job.lock").exists():
    raise SystemExit("refused: a job holds the lock")
RUN = ROOT / "runtime/rl/elite-v1/runs/native-final-0468386df8444b648525882472495279"
cfg = json.loads((ROOT / "configs/phases/elite-v1.json").read_text(encoding="utf-8"))["jobs"]["native-final"]
assert cfg["rng_mode"] == DETERMINISTIC_MODE == "isolate_sound+stable_pairs"  # muted, as in the final
nets = {}
for key in ("operator_weights", "elite_operator_weights"):
    w = cfg[key]
    nets[key] = (OptionScorer.from_json(json.loads((ROOT / w["path"]).read_text(encoding="utf-8"))),
                 "operator:" + w["sha256"][:12])
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
port_base = int(sys.argv[2]) if len(sys.argv) > 2 else 9960
games = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
games = [(r["kind"], r["payload"]) for r in games if r["kind"] in ("campaign_attempt", "star_replay", "challenge_attempt")]
PORTS = queue.Queue()
for offset in range(workers):
    PORTS.put(port_base + offset)


def play(index):
    port = PORTS.get()
    try:
        return play_on(index, port)
    finally:
        PORTS.put(port)


def play_on(index, port):
    kind, a = games[index]
    r = a["result"]
    mode = a.get("mode", 1)
    elite = a["level"] > 12 or kind == "challenge_attempt"
    net, name = nets["elite_operator_weights" if elite else "operator_weights"]
    env = NativeEnv(seed=a["seed"], level=a["level"], port=port, difficulty=r["difficulty"],
                    identity=f"verify_{uuid.uuid4().hex[:10]}", rng_mode=DETERMINISTIC_MODE,
                    action_scope=level_scope(cfg["action_scope"], a["level"]), profile=a["profile"], mode=mode)
    try:
        res = run_episode(env, OperatorPolicy(net, a["genome"], name=name), EpisodeProtocol(**r["protocol"]),
                          seed=a["seed"], level=a["level"], difficulty=r["difficulty"], episode_id=r["episode_id"],
                          observe_meta=True)
    finally:
        env.close()
    return {"kind": kind, "seed": a["seed"], "level": a["level"], "mode": mode,
            "attempt": a.get("attempt", a.get("replay")), "won": a["won"],
            "trace_equal": res["trace_sha256"] == r["trace_sha256"],
            "final_equal": res["final_state_sha256"] == r["final_state_sha256"]}


print(json.dumps({"games": len(games)}), flush=True)
with ThreadPoolExecutor(max_workers=workers) as pool:
    results = list(pool.map(play, range(len(games))))
for row in results:
    print(json.dumps(row), flush=True)
bad = [r for r in results if not (r["trace_equal"] and r["final_equal"])]
print(json.dumps({"checked": len(results), "identical": len(results) - len(bad), "different": bad}), flush=True)
