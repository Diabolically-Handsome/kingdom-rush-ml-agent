"""Commander design change (merged dev copy, after the native smoke): the steady play P0 leaves spells and hero to the
executor exactly (the plans were searched together with the executor's spell rule and a hero at its start point); a
play takes over only the parts it changes: a power whose spec is not steady, and the hero when the play sends it away
from home (reflexes only then). New brain "steady" (always P0, not passive) = the fork baseline; cmd_steady player."""
from pathlib import Path

ROOT = Path(__file__).parent / "v4_merged"
p = ROOT / "alpharush_rl/commander.py"
s = p.read_text(encoding="utf-8")


def rep(old, new, path=None):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


# Translator: which powers the current play owns; the hero reflexes only while the play sends the hero out.
rep('''    def adopt(self, play, state):
        self.play = play''',
    '''    def owned_powers(self):
        """Powers the current play casts itself: those whose spec is not steady (and only if this commander controls
        the spells); the executor keeps casting the others exactly as without a commander."""
        if "spells" not in self.parts:
            return set()
        return {power for power, part in POWER_PARTS.items() if (self.play.get(part) or STEADY)["mode"] != "steady"}

    def hero_out(self):
        """True while the play sends the hero away from its start point (hero reflexes apply only then)."""
        spec = self.play.get("hero")
        return "hero" in self.parts and bool(spec) and spec.get("to") != "home"

    def adopt(self, play, state):
        self.play = play''')
rep('''        out = ((hero and self._reflex(state, menu, tick)) or (spells and self._spells(state, menu))''',
    '''        out = ((hero and self.hero_out() and self._reflex(state, menu, tick)) or (spells and self._spells(state, menu))''')
rep('''    def _spells(self, state, menu):
        for power in (1, 2):
            if self.cast_issued[power]:
                continue''',
    '''    def _spells(self, state, menu):
        owned = self.owned_powers()
        for power in (1, 2):
            if self.cast_issued[power] or power not in owned:
                continue''')
# CommanderPolicy: hide use_power only for the powers the current play owns.
rep('''        hidden = {"aim_power", "move_hero"} | (set() if self.passive or "spells" not in self.parts else {"use_power"})
        choice = self.inner.choose(state, [item for item in menu if item["action"].get("action") not in hidden],
                                   context)''',
    '''        owned = set() if self.passive else self.translator.owned_powers()
        shown = [item for item in menu if item["action"].get("action") not in ("aim_power", "move_hero")
                 and not (item["action"].get("action") == "use_power" and item["action"].get("power") in owned)]
        choice = self.inner.choose(state, shown, context)''')
# The steady brain.
rep('''class RuleSpellsCommander(RuleCommander):''',
    '''class SteadyBrain:
    """Always the steady play P0 without being passive: plays exactly as the executor alone (the fork baseline)."""
    name = "steady"

    def choose(self, briefing, plays, key):
        return 0, {"brain": self.name}


class RuleSpellsCommander(RuleCommander):''')
rep('''RULE_BRAINS = {"passive": PassiveBrain, "rule": RuleCommander, "rule_spells": RuleSpellsCommander,''',
    '''RULE_BRAINS = {"passive": PassiveBrain, "steady": SteadyBrain, "rule": RuleCommander, "rule_spells": RuleSpellsCommander,''')
rep('''PLAYER_BRAINS = {"cmd_passive": "passive", "cmd_rule": "rule", "cmd_rule_spells": "rule_spells",''',
    '''PLAYER_BRAINS = {"cmd_passive": "passive", "cmd_steady": "steady", "cmd_rule": "rule", "cmd_rule_spells": "rule_spells",''')
p.write_text(s, encoding="utf-8")

e = ROOT / "alpharush_rl/eval_job.py"
es = e.read_text(encoding="utf-8")
old = 'FORK_BASES = ("rule", "random", "placebo")'
assert es.count(old) == 1
e.write_text(es.replace(old, 'FORK_BASES = ("steady", "rule", "random", "placebo")'), encoding="utf-8")

t = ROOT / "tests/test_commander.py"
ts = t.read_text(encoding="utf-8")
ts += '''

class SteadyPlayTests(unittest.TestCase):
    def test_steady_brain_plays_exactly_as_the_executor(self):
        from alpharush_rl.commander import SteadyBrain
        alone = play(BuildOrderPolicy(PLAN))
        steady = play(CommanderPolicy(BuildOrderPolicy(PLAN), SteadyBrain(), seed=1001))
        for key in ("plan", "trace_sha256", "outcome", "n_forced_waits"):
            self.assertEqual(alone[key], steady[key], key)
        self.assertEqual([(d["tick"], d["action"]) for d in alone["decisions"]],
                         [(d["tick"], d["action"]) for d in steady["decisions"]])

    def test_a_play_owns_only_the_powers_it_changes(self):
        from alpharush_rl.commander import STEADY, Translator
        tr = Translator()
        tr.play = {"id": "P3", "hero": {"to": "exit", "exit": "2"}, "fire": STEADY, "reinf": STEADY}
        self.assertEqual(set(), tr.owned_powers())
        self.assertTrue(tr.hero_out())
        tr.play = {"id": "P4", "hero": {"to": "home"}, "fire": {"mode": "reserve", "template": "x", "path": 1},
                   "reinf": STEADY}
        self.assertEqual({1}, tr.owned_powers())
        self.assertFalse(tr.hero_out())
        self.assertEqual(set(), Translator(parts=("hero",)).owned_powers())
'''
t.write_text(ts, encoding="utf-8")
print("deviation design patched")
