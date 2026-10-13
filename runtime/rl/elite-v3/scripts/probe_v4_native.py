"""Native v4 probes on a development copy (ports >= 9950, muted deterministic headless games, train seeds only).

usage: probe_v4_native.py <dev copy dir> <mode> [port_base]
  mode "exports": one L22 and one L19 reset in v4: hero exports and sample aim_power / move_hero catalog items.
  mode "passive": v3 plan executor vs v4 cmd_passive on the same tasks: executed actions, final tick and outcome must match.
  mode "smoke":   v4 cmd_rule on the weak elite stages: outcome, void, commands, consults per wave, pace check, wall time.
"""
import json
import os
import queue
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEV = Path(sys.argv[1]).resolve()
MODE = sys.argv[2]
PORT_BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 9950
SCRATCH = Path(__file__).parent
os.environ["ALPHARUSH_HOST_FEATURES"] = "timed_clicks"
sys.path.insert(0, str(DEV))
from alpharush_rl import commander, engine  # noqa: E402
from alpharush_rl.campaign import task_profile  # noqa: E402
from alpharush_rl.engine import DETERMINISTIC_MODE  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol, run_episode  # noqa: E402
from alpharush_rl.search import BuildOrderPolicy, check_genome  # noqa: E402

assert Path(engine.__file__).resolve().is_relative_to(DEV), engine.__file__
PROTOCOL = EpisodeProtocol(wait_ticks=120, max_interval_ticks=600, max_ticks=240000, max_decisions=3000, click_ticks=9)
PORTFOLIOS = json.loads((SCRATCH / "portfolios_v2_r3.json").read_text(encoding="utf-8"))["elite"]
PORTS = queue.Queue()
for offset in range(8):
    PORTS.put(PORT_BASE + offset)


def env_for(level, seed, genome, scope):
    profile = task_profile(level, [3, 1.2, 13], hero=genome["hero"], package=genome.get("pkg", "balanced"))
    return NativeEnv(seed=seed, level=level, port=PORTS.get(), difficulty=2, identity=f"v4probe_{uuid.uuid4().hex[:8]}",
                     rng_mode=DETERMINISTIC_MODE, action_scope=scope, profile=profile)


def play(level, seed, genome, scope, policy):
    env = env_for(level, seed, genome, scope)
    port = env.worker.port
    started = time.time()
    try:
        res = run_episode(env, policy, PROTOCOL, seed=seed, level=level, difficulty=2,
                          episode_id=f"v4probe-{scope}-{level}-{seed}", observe_meta=True)
    finally:
        env.close()
        PORTS.put(port)
    res["wall"] = round(time.time() - started, 1)
    return res


def actions(res):
    return [json.dumps({k: v for k, v in a.items() if k != "label"}, sort_keys=True) for a in (res.get("plan") or [])]


def outcome(res):
    out = res.get("outcome") or {}
    return {"status": res.get("status"), "void": res.get("void_reason"), "won": out.get("level_won"),
            "lives": out.get("lives"), "wave": (res.get("final_summary") or {}).get("wave"), "tick": res.get("final_tick")}


if MODE == "exports":
    for level in (22, 19):
        genome = check_genome(PORTFOLIOS[str(level)][0]["genome"])
        env = env_for(level, 1201, genome, "v4")
        try:
            state = env.reset()
            for _ in range(40):
                if any(i["action"] in ("aim_power", "move_hero") for i in state.get("action_catalog", [])) \
                        and state.get("enemies"):
                    break
                state = env.advance(240)
            heroes = state.get("heroes") or []
            print(level, "heroes", json.dumps(heroes)[:600])
            items = [i for i in state.get("action_catalog", []) if i["action"] in ("aim_power", "move_hero")]
            print(level, "tick", state.get("tick"), "wave", state.get("wave"), "v4 items", len(items))
            for item in items[:12]:
                print("   ", json.dumps(item))
        finally:
            PORTS.put(env.worker.port)
            env.close()
elif MODE == "passive":
    tasks = [(level, seed, check_genome(PORTFOLIOS[str(level)][0]["genome"])) for level in (15, 20, 22, 26)
             for seed in (1201, 1202)]

    def pair(task):
        level, seed, genome = task
        a = play(level, seed, genome, "v3", BuildOrderPolicy(genome))
        b = play(level, seed, genome, "v4", commander.make_player("cmd_passive", genome, {"seed": seed, "level": level}))
        same = actions(a) == actions(b) and outcome(a) == outcome(b)
        return {"level": level, "seed": seed, "same": same, "v3": outcome(a), "v4_passive": outcome(b),
                "actions": [len(actions(a)), len(actions(b))], "wall": [a["wall"], b["wall"]]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(pair, tasks))
    for row in rows:
        print(json.dumps(row), flush=True)
    print(json.dumps({"pairs": len(rows), "identical": sum(r["same"] for r in rows)}))
elif MODE.startswith("smoke"):
    PLAYER = MODE.partition(":")[2] or "cmd_rule"
    tasks = [(level, seed, check_genome(PORTFOLIOS[str(level)][0]["genome"])) for level in (15, 17, 19, 20, 22, 26)
             for seed in (1201, 1202)]

    def one(task):
        level, seed, genome = task
        policy = commander.make_player(PLAYER, genome, {"seed": seed, "level": level})
        res = play(level, seed, genome, "v4", policy)
        summary = policy.summary()
        waves = (res.get("final_summary") or {}).get("wave") or 1
        return {"player": PLAYER, "level": level, "seed": seed, **outcome(res), "commands": summary["commands"],
                "consults": len(summary["consults"]), "consults_per_wave": round(len(summary["consults"]) / waves, 2),
                "pace": commander.pace_violations(res.get("decisions") or []), "decisions": res.get("n_decisions"),
                "wall": res["wall"]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one, tasks))
    for row in rows:
        print(json.dumps(row), flush=True)
else:
    raise SystemExit("mode must be exports, passive or smoke")
