"""Native smoke probe of the human-timing patches in the route_dev copy (its own engine runtime; ports >= 9950).
usage: probe_human_timing.py <case> <port>   case: l12_off | l12_on | l11_e1 | l11_e2
Plays train seeds 1001-1002 with the plan executor and prints wins, lives, trap clicks / wave calls."""
import collections
import json
import os
import sys
import uuid
from pathlib import Path

DEV = Path(__file__).parent / "route_dev"
SCRATCH = Path(__file__).parent
case, port = sys.argv[1], int(sys.argv[2])
os.environ["ALPHARUSH_HOST_FEATURES"] = "" if case in ("l12_off", "l11_e1_off") else "timed_clicks,early_waves"
sys.path.insert(0, str(DEV))
from alpharush_rl import engine  # noqa: E402
from alpharush_rl.campaign import task_profile  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol, run_episode  # noqa: E402
from alpharush_rl.search import BuildOrderPolicy, check_genome  # noqa: E402

assert Path(engine.__file__).resolve().is_relative_to(DEV.resolve()), engine.__file__
v1 = json.loads((Path(r"C:\Users\<user>\Documents\AlphaRush") / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
if case.startswith("l12"):
    level, e = 12, v1["jobs"]["native-final"]["campaign"]["plans"]["12"][0]
    genome = e["genome"] if isinstance(e, dict) and "genome" in e else e
else:
    level = 11
    genome = json.loads((SCRATCH / "v2_l11_portfolio.json").read_text(encoding="utf-8"))[0]["genome"]
    if case == "l11_e2":
        genome = {**genome, "early": 2}
genome = check_genome(genome)
protocol = EpisodeProtocol(name="search", wait_ticks=120, max_interval_ticks=600, max_ticks=240000,
                           max_decisions=3000, **({"click_ticks": 9} if case == "l12_on" else {}))


class Spy:
    def __init__(self, inner):
        self.inner, self.name, self.log = inner, inner.name, []

    def choose(self, state, menu, context):
        c = self.inner.choose(state, menu, context)
        act = next(m["action"] for m in menu if m["label"] == c["label"])
        if act.get("action") in ("click_entity", "send_wave"):
            enemies = sum(1 for e in state.get("enemies", []))
            self.log.append((state["tick"], state["wave"], act["action"], enemies, state["lives"]))
        return c


for seed in (1001, 1002):
    profile = task_profile(level, [2.75, 1.0], hero=genome["hero"], package=genome.get("pkg", "balanced"))
    env = NativeEnv(seed=seed, level=level, port=port, difficulty=2, identity=f"probe_{case}_{seed}_{uuid.uuid4().hex[:6]}",
                    rng_mode=engine.DETERMINISTIC_MODE, action_scope="v2", profile=profile)
    spy = Spy(BuildOrderPolicy(genome))
    try:
        res = run_episode(env, spy, protocol, seed=seed, level=level)
    finally:
        env.close()
    kinds = collections.Counter(a for _, _, a, _, _ in spy.log)
    with_enemies = sum(1 for _, w, a, n, _ in spy.log if a == "send_wave" and n and w)
    print(case, "seed", seed, "won", (res.get("outcome") or {}).get("level_won"), "lives",
          (res.get("outcome") or {}).get("lives"), "wave", (res.get("final_summary") or {}).get("wave"),
          "ticks", res.get("final_tick"), dict(kinds), "calls_with_enemies", with_enemies, flush=True)
    if case.startswith("l12"):
        clicks = [(t, w) for t, w, a, _, _ in spy.log if a == "click_entity"]
        print("  clicks (tick, wave):", clicks[:40], flush=True)
