"""Raw check: does the dev host offer send_wave with enemies on the field when early_waves is on?"""
import json, os, sys, uuid
from pathlib import Path
DEV = Path(__file__).parent / "route_dev"
os.environ["ALPHARUSH_HOST_FEATURES"] = "timed_clicks,early_waves"
sys.path.insert(0, str(DEV))
from alpharush_rl import engine
from alpharush_rl.campaign import task_profile
from alpharush_rl.env import NativeEnv
from alpharush_rl.episode import EpisodeProtocol, run_episode
from alpharush_rl.search import BuildOrderPolicy, check_genome
genome = json.loads((Path(__file__).parent / "v2_l11_portfolio.json").read_text(encoding="utf-8"))[0]["genome"]
genome = check_genome({**genome, "early": 2})
class Spy:
    def __init__(self, inner): self.inner, self.name = inner, inner.name
    def choose(self, state, menu, context):
        c = self.inner.choose(state, menu, context)
        send = any(m["action"].get("action") == "send_wave" for m in menu)
        if state["wave"] <= 6:
            print(state["tick"], "w", state["wave"], "nw", state.get("next_wave") is not None, "spawning", state.get("wave_spawning"),
                  "ec", state.get("enemy_count"), "aec", state.get("active_enemy_count"), "n", len(state.get("enemies", [])),
                  "send", send, "chose", c["label"], c["meta"].get("rule"), flush=True)
        return c
seed = 1001
profile = task_profile(11, [2.75, 1.0], hero=genome["hero"], package=genome.get("pkg", "balanced"))
env = NativeEnv(seed=seed, level=11, port=9956, difficulty=2, identity=f"probe_raw_{uuid.uuid4().hex[:6]}",
                rng_mode=engine.DETERMINISTIC_MODE, action_scope="v2", profile=profile)
try:
    print("host features env", os.environ.get("ALPHARUSH_HOST_FEATURES"))
    res = run_episode(env, Spy(BuildOrderPolicy(genome)), EpisodeProtocol(name="search", wait_ticks=60, max_interval_ticks=120,
                      max_ticks=30000, max_decisions=3000), seed=seed, level=11)
finally:
    env.close()
