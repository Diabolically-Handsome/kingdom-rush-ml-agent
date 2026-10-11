"""Re-execute every game of an elite-v2 campaign/final run with the same operator networks and compare hashes.

usage: verify_final_reexec_v2.py <run id> <job kind native-final|native-campaign> [workers] [port_base]
Each journaled game (campaign_attempt, star_replay, challenge_attempt) is played again from its journaled profile,
seed, level, mode and plan by the network that played it (the job's main operator for main levels and star replays,
its elite operator for elite stages and challenges), with the job's host features; trace and final-state hashes must
be identical. Read-only; games muted (deterministic headless mode).
"""
import json
import os
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

if (ROOT / "runtime/rl/elite-v2/job.lock").exists():
    raise SystemExit("refused: a job holds the lock")
RUN = ROOT / "runtime/rl/elite-v2/runs" / sys.argv[1]
cfg = json.loads((ROOT / "configs/phases/elite-v2.json").read_text(encoding="utf-8"))["jobs"][sys.argv[2]]
os.environ["ALPHARUSH_HOST_FEATURES"] = ",".join(sorted(cfg.get("host_features", [])))
assert cfg["rng_mode"] == DETERMINISTIC_MODE == "isolate_sound+stable_pairs"  # muted, as in the final
nets = {}
for key in ("operator_weights", "elite_operator_weights"):
    w = cfg[key]
    nets[key] = (OptionScorer.from_json(json.loads((ROOT / w["path"]).read_text(encoding="utf-8"))),
                 "operator:" + w["sha256"][:12])
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
port_base = int(sys.argv[4]) if len(sys.argv) > 4 else 9960
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
