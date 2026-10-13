"""Commander component ablation (merged dev copy): a brain may control only some parts of the commander - "spells"
(Rain of Fire and reinforcements) and/or "hero" (moves and the hero reflexes); parts it does not control stay with the
executor exactly as without a commander. New arms: cmd_rule_spells and cmd_rule_hero."""
from pathlib import Path

ROOT = Path(__file__).parent / "v4_merged"
p = ROOT / "alpharush_rl/commander.py"
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


rep('''    def __init__(self, cast=50, boss_focus=False):
        self.cast, self.boss_focus = cast, boss_focus''',
    '''    def __init__(self, cast=50, boss_focus=False, parts=("spells", "hero")):
        self.cast, self.boss_focus = cast, boss_focus
        self.parts = frozenset(parts)  # what this commander controls: "spells" and/or "hero"''')
rep('''        out = self._reflex(state, menu, tick) or self._spells(state, menu) or self._hero(state, menu, tick)''',
    '''        hero, spells = "hero" in self.parts, "spells" in self.parts
        out = ((hero and self._reflex(state, menu, tick)) or (spells and self._spells(state, menu))
               or (hero and self._hero(state, menu, tick)) or None)''')
rep('''class PlaceboCommander(RuleCommander):''',
    '''class RuleSpellsCommander(RuleCommander):
    """The rule brain controlling only the spells; the hero stays where the executor leaves it (component ablation)."""
    name = "rule_spells"
    parts = ("spells",)


class RuleHeroCommander(RuleCommander):
    """The rule brain controlling only the hero; the executor casts the spells as without a commander (component
    ablation)."""
    name = "rule_hero"
    parts = ("hero",)


class PlaceboCommander(RuleCommander):''')
rep('''RULE_BRAINS = {"passive": PassiveBrain, "rule": RuleCommander, "random": RandomCommander,''',
    '''RULE_BRAINS = {"passive": PassiveBrain, "rule": RuleCommander, "rule_spells": RuleSpellsCommander,
               "rule_hero": RuleHeroCommander, "random": RandomCommander,''')
rep('''        self.translator = Translator(cast=cast, boss_focus=bool((self.genome or {}).get("boss")))''',
    '''        self.parts = tuple(getattr(brain, "parts", ("spells", "hero")))
        self.translator = Translator(cast=cast, boss_focus=bool((self.genome or {}).get("boss")), parts=self.parts)''')
rep('''        hidden = {"aim_power", "move_hero"} | (set() if self.passive else {"use_power"})''',
    '''        hidden = {"aim_power", "move_hero"} | (set() if self.passive or "spells" not in self.parts else {"use_power"})''')
rep('''PLAYER_BRAINS = {"cmd_passive": "passive", "cmd_rule": "rule", "cmd_random": "random", "cmd_placebo": "placebo",''',
    '''PLAYER_BRAINS = {"cmd_passive": "passive", "cmd_rule": "rule", "cmd_rule_spells": "rule_spells",
                 "cmd_rule_hero": "rule_hero", "cmd_random": "random", "cmd_placebo": "placebo",''')
p.write_text(s, encoding="utf-8")

t = ROOT / "tests/test_commander.py"
ts = t.read_text(encoding="utf-8")
ts += '''

class CommanderPartsTests(unittest.TestCase):
    def test_component_arms_control_only_their_parts(self):
        from alpharush_rl import commander as c
        spells, hero = c.make_brain("rule_spells"), c.make_brain("rule_hero")
        self.assertEqual(("spells",), spells.parts)
        self.assertEqual(("hero",), hero.parts)
        self.assertNotIn("parts", vars(c.RuleCommander))
        tr = c.Translator(parts=("hero",))
        self.assertFalse("spells" in tr.parts)
        self.assertEqual({"cmd_rule_spells", "cmd_rule_hero"} <= set(c.PLAYER_BRAINS), True)
'''
t.write_text(ts, encoding="utf-8")
print("parts patched")
