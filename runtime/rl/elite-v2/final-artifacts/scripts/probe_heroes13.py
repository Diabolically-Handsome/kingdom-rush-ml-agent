"""Probe (muted, train seed 1001, ports >= 9950): from a save that has won the main campaign, does each of the 7
heroes outside FREE_HEROES load as the selected hero and appear on the field of an elite stage?

usage: probe_heroes13.py [level] [port_base]
"""
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\<user>\Documents\AlphaRush")
from alpharush_rl.campaign import level_profile  # noqa: E402
from alpharush_rl.engine import DETERMINISTIC_MODE  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402

level = int(sys.argv[1]) if len(sys.argv) > 1 else 14
port_base = int(sys.argv[2]) if len(sys.argv) > 2 else 9960
assert DETERMINISTIC_MODE == "isolate_sound+stable_pairs"  # muted
HEROES = ["hero_denas", "hero_elora", "hero_ingvar", "hero_hacksaw", "hero_oni", "hero_thor", "hero_10yr",
          "hero_gerald"]  # gerald: a free hero as the control


def probe(args):
    hero, port = args
    profile = level_profile(level, [3, 1.0])  # no hero: the availability check is what we are testing
    profile = {**profile, "hero": hero, "hero_xp": 0}
    env = NativeEnv(seed=1001, level=level, port=port, identity=f"devhero13_{uuid.uuid4().hex[:8]}",
                    rng_mode=DETERMINISTIC_MODE, action_scope="v3", profile=profile)
    try:
        state = env.reset()
        meta = env.meta()
        seen = []
        for _ in range(4):
            state = env.advance(300)
            seen.append([(h.get("template"), h.get("level"), h.get("hp"), h.get("dead")) for h in state.get("heroes") or []])
        return {"hero": hero, "selected_hero": meta.get("selected_hero"), "hero_count": meta.get("hero_count"),
                "on_field": seen[-1], "ok": meta.get("selected_hero") == hero and any(t == hero for t, *_ in seen[-1])}
    except Exception as exc:  # noqa: BLE001
        return {"hero": hero, "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        env.close()


with ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(probe, [(h, port_base + i) for i, h in enumerate(HEROES)]):
        print(json.dumps(row), flush=True)
