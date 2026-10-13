"""Cold replay in scope v3 on a development copy's host: 20 recorded elite-v2 final games are played again by the networks
that played them; traces and end states must equal the journal (proves the new host leaves scope v3 unchanged natively).

usage: probe_cold_v3.py <dev copy dir> [workers] [port_base]   (ports >= 9950; muted deterministic headless games)"""
import json
import os
import queue
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEV = Path(sys.argv[1]).resolve()
WS = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(DEV))
from alpharush_rl import engine  # noqa: E402
from alpharush_rl.engine import DETERMINISTIC_MODE, level_scope  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol, run_episode  # noqa: E402
from alpharush_rl.operator_net import OperatorPolicy, OptionScorer  # noqa: E402

assert Path(engine.__file__).resolve().is_relative_to(DEV), engine.__file__
RUN = WS / "runtime/rl/elite-v2/runs/native-final-af081b54e5f54478aa76b24bc02bdbf5"
cfg = json.loads((WS / "configs/phases/elite-v2.json").read_text(encoding="utf-8"))["jobs"]["native-final"]
os.environ["ALPHARUSH_HOST_FEATURES"] = ",".join(sorted(cfg.get("host_features", [])))
nets = {key: (OptionScorer.from_json(json.loads((WS / cfg[key]["path"]).read_text(encoding="utf-8"))),
              "operator:" + cfg[key]["sha256"][:12]) for key in ("operator_weights", "elite_operator_weights")}
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
port_base = int(sys.argv[3]) if len(sys.argv) > 3 else 9960
rows = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
pick = json.loads((SCRATCH / "cold_replay_rows.json").read_text(encoding="utf-8"))
games = [(rows[i]["kind"], rows[i]["payload"]) for i in pick]
PORTS = queue.Queue()
for offset in range(workers):
    PORTS.put(port_base + offset)


def play(index):
    port = PORTS.get()
    try:
        kind, a = games[index]
        r = a["result"]
        elite = a["level"] > 12 or kind == "challenge_attempt"
        net, name = nets["elite_operator_weights" if elite else "operator_weights"]
        env = NativeEnv(seed=a["seed"], level=a["level"], port=port, difficulty=r["difficulty"],
                        identity=f"cold_{uuid.uuid4().hex[:10]}", rng_mode=DETERMINISTIC_MODE,
                        action_scope=level_scope(cfg["action_scope"], a["level"]), profile=a["profile"],
                        mode=a.get("mode", 1))
        try:
            res = run_episode(env, OperatorPolicy(net, a["genome"], name=name), EpisodeProtocol(**r["protocol"]),
                              seed=a["seed"], level=a["level"], difficulty=r["difficulty"], episode_id=r["episode_id"],
                              observe_meta=True)
        finally:
            env.close()
        return {"seed": a["seed"], "level": a["level"], "attempt": a.get("attempt"), "won": a["won"],
                "trace_equal": res["trace_sha256"] == r["trace_sha256"],
                "final_equal": res["final_state_sha256"] == r["final_state_sha256"]}
    finally:
        PORTS.put(port)


with ThreadPoolExecutor(max_workers=workers) as pool:
    results = list(pool.map(play, range(len(games))))
for row in results:
    print(json.dumps(row), flush=True)
bad = [r for r in results if not (r["trace_equal"] and r["final_equal"])]
print(json.dumps({"checked": len(results), "identical": len(results) - len(bad), "different": bad}), flush=True)
