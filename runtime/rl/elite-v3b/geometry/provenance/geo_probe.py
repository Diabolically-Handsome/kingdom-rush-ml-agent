"""elite-v3b: one-off geometry dump per elite stage from the scratch dev copy (v4_merged + DEV-ONLY "paths" RPC).
Muted deterministic headless games, ports >= 9950, train seed 1201, one reset per level, no actions taken.
Writes runtime/rl/elite-v3b/geometry/level_<L>.json in the real repository: the RPC's paths/lanes/connections/exits/
ice flags plus the reset state's holders (build spots) and heroes (start point).

usage: geo_probe.py <dev copy dir> [port]"""
import json
import os
import sys
import uuid
from pathlib import Path

DEV = Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9950
SCRATCH = Path(__file__).parent
OUT = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v3b\geometry")
os.environ["ALPHARUSH_HOST_FEATURES"] = "timed_clicks"
sys.dont_write_bytecode = True
sys.path.insert(0, str(DEV))
from alpharush_rl import engine  # noqa: E402
from alpharush_rl.campaign import task_profile  # noqa: E402
from alpharush_rl.engine import DETERMINISTIC_MODE  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.search import check_genome  # noqa: E402

assert Path(engine.__file__).resolve().is_relative_to(DEV), engine.__file__
PORTFOLIOS = json.loads((SCRATCH / "portfolios_v2_r3.json").read_text(encoding="utf-8"))["elite"]
OUT.mkdir(parents=True, exist_ok=True)
for level in (15, 17, 19, 20, 21, 22, 26):
    genome = check_genome(PORTFOLIOS[str(level)][0]["genome"])
    profile = task_profile(level, [3, 1.2, 13], hero=genome["hero"], package=genome.get("pkg", "balanced"))
    env = NativeEnv(seed=1201, level=level, port=PORT, difficulty=2, identity=f"geo_{uuid.uuid4().hex[:8]}",
                    rng_mode=DETERMINISTIC_MODE, action_scope="v4", profile=profile)
    try:
        state = env.reset()
        geo = env.worker.rpc("paths")
        keep = ("id", "x", "y", "template", "home_x", "home_y", "level")
        geo["holders"] = [{k: h.get(k) for k in ("id", "mesh_id", "x", "y") if k in h} for h in state.get("holders") or []]
        geo["heroes"] = [{k: h.get(k) for k in keep if k in h} for h in state.get("heroes") or []]
        geo["level"], geo["hero"] = level, genome["hero"]
        (OUT / f"level_{level}.json").write_text(json.dumps(geo, indent=1), encoding="utf-8")
        lanes = [len(p) for p in geo.get("paths", [])]
        print(level, "paths", len(lanes), "lanes", lanes, "nodes(l1)", [len(p[0]) if p else 0 for p in geo.get("paths", [])],
              "conn", geo.get("connections"), "exits", geo.get("exits"), "ice", [sum(1 for x in i if x) for i in geo.get("ice", [])],
              "holders", len(geo["holders"]), "heroes", geo["heroes"], "loc", geo.get("location_keys"), flush=True)
    finally:
        env.close()
