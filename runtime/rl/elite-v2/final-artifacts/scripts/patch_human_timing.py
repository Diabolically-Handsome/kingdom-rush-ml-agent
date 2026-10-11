"""Human-capability items 3 and 4 (owner-approved 2026-10-10), opt-in per job via "host_features":

- "timed_clicks": a timed click trap (Vez'nan's tower hold: mod_veznan_tower, click_time) is offered only while its
  click prompt (render sprite 2) is visible, i.e. inside the game's own click window (main/181 L5283-5337); expired
  and released holds are no longer offered. With the protocol's "click_ticks", a menu with a pending click is decided
  at once and each click advances click_ticks (a human taps a few times a second) instead of 30.
- "early_waves": the next-wave flag is offered whenever the game shows it (store.next_wave and not spawning), not
  only on a clear field, as a human can tap it with enemies on the field. Plan gene early=2 calls it as soon as it
  shows unless a live enemy has covered EARLY2_MAX_PROGRESS of its path; early 0/1 keep their meaning (the executor
  checks the clear field itself). Searches draw/mutate early 2 only while this feature is on.

Without host_features (and click_ticks None) every path is unchanged. Apply only while no runtime/rl/*/job.lock.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", r"C:\Users\<user>\Documents\AlphaRush"))
REAL = Path(r"C:\Users\<user>\Documents\AlphaRush")
if ROOT.resolve() == REAL.resolve() and any((REAL / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")


def patch(rel, pairs):
    path = ROOT / rel
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:90])
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


patch("alpharush_rl/assets/host.lua", [
    ('''local ACTION_SCOPE = os.getenv("ALPHARUSH_ACTION_SCOPE") or "v1"''',
     '''local ACTION_SCOPE = os.getenv("ALPHARUSH_ACTION_SCOPE") or "v1"
-- Opt-in human-capability features of a job (comma list): timed_clicks, early_waves.
local HOST_FEATURES = {}
for name in string.gmatch(os.getenv("ALPHARUSH_HOST_FEATURES") or "", "[%w_]+") do HOST_FEATURES[name] = true end'''),
    ('''    if type(e.modifier) == "table" and finite(e.required_clicks) and e.required_clicks > 0 then
        return "tower_trap"
    end''',
     '''    if type(e.modifier) == "table" and finite(e.required_clicks) and e.required_clicks > 0 then
        if HOST_FEATURES.timed_clicks and finite(e.click_time) then
            -- A timed hold counts clicks only while its click prompt (sprite 2) shows: before it (start
            -- animation, clicks not yet routed to it) and after the window (held, then released) a tap is lost.
            local prompt = type(e.render) == "table" and type(e.render.sprites) == "table" and e.render.sprites[2]
            if type(prompt) ~= "table" or prompt.hidden then return nil end
        end
        return "tower_trap"
    end'''),
    ('''    if s.next_wave and not s.wave_spawning and (s.wave==0 or wave_count(s)==0) then
        add({action="send_wave", cost=0})
    end''',
     '''    -- A human can tap the next-wave flag whenever it shows (early_waves); the default offers it on a clear field.
    if s.next_wave and not s.wave_spawning and (s.wave==0 or wave_count(s)==0 or HOST_FEATURES.early_waves) then
        add({action="send_wave", cost=0})
    end'''),
    # The Python menu offers send_wave only when the state's wave_ready is true: same rule as the catalog.
    ('''        s.wave_ready = s.next_wave ~= nil and not s.wave_spawning and (s.wave==0 or wave_count(s)==0)''',
     '''        s.wave_ready = s.next_wave ~= nil and not s.wave_spawning
            and (s.wave==0 or wave_count(s)==0 or HOST_FEATURES.early_waves == true)'''),
])

patch("alpharush_rl/env.py", [
    ('''        self.mode = mode
        self.profile = profile''',
     '''        self.mode = mode
        self.profile = profile
        self.action_ticks = dict(ACTION_TICKS)  # an episode protocol may shorten the click step (click_ticks)'''),
    ('''            after = self.advance(ACTION_TICKS.get(name, 2), record_plan=False)''',
     '''            after = self.advance(self.action_ticks.get(name, 2), record_plan=False)'''),
])

patch("alpharush_rl/episode.py", [
    ('''    reset_on_trivial_menu: bool = True
''',
     '''    reset_on_trivial_menu: bool = True
    # A pending click (boss finishing tap, tower trap) is decided at once and each click advances this many
    # ticks (a human taps a few times a second); None keeps the old 30-tick click step and decision rule.
    click_ticks: int | None = None
'''),
    ('''        if not isinstance(self.reset_on_trivial_menu, bool):''',
     '''        if self.click_ticks is not None and not _positive_int(self.click_ticks):
            raise ValueError("click_ticks must be None or a positive integer")
        if not isinstance(self.reset_on_trivial_menu, bool):'''),
    ('''        due = bool(option_keys) and (option_keys != decided_keys
                                     or tick - decided_tick >= protocol.max_interval_ticks)''',
     '''        due = bool(option_keys) and (option_keys != decided_keys
                                     or tick - decided_tick >= protocol.max_interval_ticks)
        if protocol.click_ticks and any(item["action"].get("action") == "click_entity" for item in menu[1:]):
            due = True  # a tap the game is waiting for: a human does not wait for the menu to change'''),
])
# run_episode: give the env the protocol's click step.
episode = (ROOT / "alpharush_rl/episode.py").read_bytes().decode("utf-8")
marker = '''    decided_keys, decided_tick = frozenset(), None
'''
assert episode.count(marker) == 1
episode = episode.replace(marker, marker + '''    if protocol.click_ticks and isinstance(getattr(env, "action_ticks", None), dict):
        env.action_ticks["click_entity"] = protocol.click_ticks
''')
(ROOT / "alpharush_rl/episode.py").write_bytes(episode.encode("utf-8"))

patch("alpharush_rl/search.py", [
    ('''    if genome["early"] not in (0, 1) or isinstance(genome["early"], bool):
        raise ValueError("early must be 0 or 1")''',
     '''    if genome["early"] not in (0, 1, 2) or isinstance(genome["early"], bool):
        raise ValueError("early must be 0, 1 or 2")'''),
    ('''            if sends and (state.get("wave") == 0 or (self.genome["early"] and not enemies)):''',
     '''            if sends and (state.get("wave") == 0 or (self.genome["early"] == 1 and not enemies)
                          or (self.genome["early"] == 2 and _calm(enemies))):'''),
    ('''        choice = self.fallback.choose(state, menu, context)
        rule = choice["meta"].get("rule")''',
     '''        if self.genome["early"] == 2 and state.get("wave") != 0 and _calm(enemies):
            sends = [item for item in menu if item["action"].get("action") == "send_wave"]
            if sends:
                return _choice(self, sends[0]["label"], rule="early_wave", plan_cursor=self.cursor)
        choice = self.fallback.choose(state, menu, context)
        rule = choice["meta"].get("rule")'''),
    ('''                         "cast": rng.choice((30, 40, 50, 60, 70)), "early": rng.choice((0, 1)),''',
     '''                         "cast": rng.choice((30, 40, 50, 60, 70)), "early": rng.choice(early_values()),'''),
    ('''            elif which == "early":
                g["early"] = 1 - g["early"]''',
     '''            elif which == "early":
                values = early_values()
                g["early"] = (1 - g["early"]) if values == (0, 1) and g["early"] in (0, 1) else rng.choice(
                    [v for v in values if v != g["early"]])'''),
    ('''class BuildOrderPolicy:''',
     '''# early=2: call the next wave as soon as its flag shows (host feature early_waves) unless a live enemy has
# already covered this much of its path.
EARLY2_MAX_PROGRESS = 0.5


def early_values():
    """Plan values of the early gene a search may draw: 2 only while the job offers the flag with enemies on the
    field (host feature early_waves, exported to ALPHARUSH_HOST_FEATURES by the job worker)."""
    features = os.environ.get("ALPHARUSH_HOST_FEATURES", "").split(",")
    return (0, 1, 2) if "early_waves" in features else (0, 1)


def _calm(enemies):
    return all(_number(e.get("path_progress")) < EARLY2_MAX_PROGRESS for e in enemies)


class BuildOrderPolicy:'''),
])
search = (ROOT / "alpharush_rl/search.py").read_bytes().decode("utf-8")
if "\nimport os\n" not in search:
    search = search.replace("\nimport json\n", "\nimport json\nimport os\n", 1)
    assert "\nimport os\n" in search
(ROOT / "alpharush_rl/search.py").write_bytes(search.encode("utf-8"))

patch("alpharush_rl/strategy_brain.py", [
    ('''            + ("calls the next wave early when the field is clear" if genome["early"] else "lets waves come on their own")''',
     '''            + ("calls each next wave as soon as its flag shows (unless an enemy is near the exit)" if genome["early"] == 2
               else "calls the next wave early when the field is clear" if genome["early"]
               else "lets waves come on their own")'''),
])

patch("tools/campaign-survey.py", [
    ('''    job, pools = verify_frozen(config, permit, job_kind)
    if job_kind in SEARCH_JOB_KINDS:''',
     '''    job, pools = verify_frozen(config, permit, job_kind)
    features = job.get("host_features", [])
    if not isinstance(features, list) or not set(features) <= HOST_FEATURES:
        raise GateRefused(f"host_features must be a subset of {sorted(HOST_FEATURES)}")
    # Every game this worker starts inherits the job's opt-in human-capability features.
    os.environ["ALPHARUSH_HOST_FEATURES"] = ",".join(sorted(features))
    if features:
        ctx.record("host_features", features=sorted(features))
    if job_kind in SEARCH_JOB_KINDS:'''),
    ('''BRAIN_ENDPOINT = "http://127.0.0.1:12081"''',
     '''BRAIN_ENDPOINT = "http://127.0.0.1:12081"
# Opt-in host features a job may switch on (host.lua HOST_FEATURES): human-timed trap clicks, early wave calls.
HOST_FEATURES = {"timed_clicks", "early_waves"}'''),
])

tp = ROOT / "tests/test_elite.py"
t = tp.read_bytes().decode("utf-8")
anchor = '''class JournalTests(unittest.TestCase):'''
assert t.count(anchor) == 1
t = t.replace(anchor, '''class HumanTimingTests(unittest.TestCase):
    def test_click_ticks_protocol_field(self):
        from alpharush_rl.episode import EpisodeProtocol
        self.assertIsNone(EpisodeProtocol().click_ticks)
        self.assertEqual(9, EpisodeProtocol(click_ticks=9).click_ticks)
        for bad in (0, -3, 1.5, True):
            with self.assertRaises(ValueError):
                EpisodeProtocol(click_ticks=bad)

    def test_early_two_calls_the_flag_unless_an_enemy_is_near_the_exit(self):
        import os
        from alpharush_rl import search as search_module
        g = genome([], early=2)
        self.assertEqual(2, g["early"])
        self.assertIn("as soon as its flag shows", strategy_brain.describe_plan(g))
        menu = [WAIT, option("B", {"action": "send_wave"}, 0.0)]
        far = {"gold": 0, "wave": 3, "towers": [], "holders": [],
               "enemies": [{"id": 1, "path_progress": 0.2, "hp": 10, "hp_max": 10}]}
        near = {**far, "enemies": [{"id": 1, "path_progress": 0.8, "hp": 10, "hp_max": 10}]}
        self.assertEqual("B", BuildOrderPolicy(g).choose(far, menu, {"decision_index": 0})["label"])
        self.assertNotEqual("B", BuildOrderPolicy(g).choose(near, menu, {"decision_index": 0})["label"])
        self.assertNotEqual("B", BuildOrderPolicy(genome([], early=1)).choose(far, menu, {"decision_index": 0})["label"])
        old = os.environ.get("ALPHARUSH_HOST_FEATURES")
        try:
            os.environ["ALPHARUSH_HOST_FEATURES"] = ""
            self.assertEqual((0, 1), search_module.early_values())
            os.environ["ALPHARUSH_HOST_FEATURES"] = "early_waves,timed_clicks"
            self.assertEqual((0, 1, 2), search_module.early_values())
            rng = random.Random(4)
            seen = {random_genome(HOLDERS, [], rng, rally=True)["early"] for _ in range(60)}
            self.assertEqual({0, 1, 2}, seen)
        finally:
            if old is None:
                os.environ.pop("ALPHARUSH_HOST_FEATURES", None)
            else:
                os.environ["ALPHARUSH_HOST_FEATURES"] = old
        with self.assertRaises(ValueError):
            genome([], early=3)


class JournalTests(unittest.TestCase):''')
tp.write_bytes(t.encode("utf-8"))
print("patched", ROOT)
