"""Campaign progression rules and the save-slot profile; no game is started."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from alpharush_rl import campaign
from alpharush_rl.campaign import (STAR_PRICES, STAR_TREES, buy_upgrades, check_profile, heroes_available,
                                   level_profile, profile_matches, slot_lua, stars_for_lives, upgrades_cost,
                                   write_profile)


class StarsTests(unittest.TestCase):
    def test_victory_star_thresholds(self):
        self.assertEqual([1, 1, 2, 2, 2, 3, 3], [stars_for_lives(n) for n in (1, 5, 6, 12, 17, 18, 20)])
        for bad in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                stars_for_lives(bad)


class UpgradeRuleTests(unittest.TestCase):
    def test_rule_spends_evenly_in_priority_order(self):
        self.assertEqual({tree: 0 for tree in STAR_TREES}, buy_upgrades(0))
        self.assertEqual({**{tree: 0 for tree in STAR_TREES}, "rain": 1, "archers": 1}, buy_upgrades(3))
        self.assertEqual({tree: 5 for tree in STAR_TREES}, buy_upgrades(65))

    def test_rule_never_overspends_and_leaves_nothing_affordable(self):
        for stars in range(0, 70):
            levels = buy_upgrades(stars)
            spent = upgrades_cost(levels)
            self.assertLessEqual(spent, stars)
            for tree in STAR_TREES:
                if levels[tree] < 5:
                    self.assertGreater(STAR_PRICES[tree][levels[tree]], stars - spent)

    def test_rule_builds_on_owned_levels_and_refuses_overdrafts(self):
        owned = {"reinforcements": 2}
        levels = buy_upgrades(10, owned)
        self.assertEqual(2, levels["reinforcements"])
        with self.assertRaises(ValueError):
            buy_upgrades(4, owned)  # 2 + 3 stars already spent
        with self.assertRaises(ValueError):
            buy_upgrades(5, {"bogus": 1})


class ProfileTests(unittest.TestCase):
    def test_level_profile_buys_after_every_victory(self):
        profile = level_profile(5)
        self.assertEqual({1: 3, 2: 3, 3: 3, 4: 3}, profile["levels"])
        owned = {}
        for stars in (3, 6, 9, 12):
            owned = buy_upgrades(stars, owned)
        self.assertEqual(owned, profile["upgrades"])
        self.assertEqual({tree: 0 for tree in STAR_TREES}, level_profile(1)["upgrades"])
        previous = level_profile(1, 2)["upgrades"]
        for level in range(2, 14):  # bought upgrades are kept: later levels never have fewer
            current = level_profile(level, 2)["upgrades"]
            self.assertTrue(all(current[t] >= previous[t] for t in STAR_TREES))
            self.assertLessEqual(upgrades_cost(current), 2 * (level - 1))
            previous = current

    def test_average_star_rate_profiles(self):
        profile = level_profile(9, 2.75)
        self.assertEqual(22, sum(profile["levels"].values()))  # round(2.75 * 8)
        self.assertEqual({1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 2, 8: 2}, profile["levels"])
        self.assertEqual(level_profile(9, 2), level_profile(9, 2.0))
        self.assertEqual(level_profile(5, 3)["upgrades"], level_profile(5, 3.0)["upgrades"])
        for bad in (0.5, 3.5, True, "2"):
            with self.assertRaises(ValueError):
                level_profile(4, bad)

    def test_allocation_packages_spend_all_stars_on_their_trees_first(self):
        from alpharush_rl.campaign import PACKAGES, package_upgrades
        balanced = level_profile(9, 2.75)
        self.assertEqual(balanced, level_profile(9, 2.75, package="balanced"))
        rain = level_profile(9, 2.75, package="rain")
        self.assertEqual({"archers": 2, "barracks": 2, "mages": 2, "engineers": 1, "rain": 5, "reinforcements": 1},
                         rain["upgrades"])
        self.assertEqual(balanced["levels"], rain["levels"])
        for name in PACKAGES:
            upgrades = level_profile(12, 3, package=name)["upgrades"]
            self.assertLessEqual(upgrades_cost(upgrades), 33)
        self.assertEqual(5, package_upgrades("mages", {1: 3, 2: 3, 3: 3})["mages"])
        with self.assertRaises(ValueError):
            package_upgrades("wizards", {1: 3})

    def test_hero_must_be_unlocked_for_the_level(self):
        self.assertEqual([], heroes_available(3))
        self.assertEqual(["hero_gerald"], heroes_available(4))
        self.assertIn("hero_ignus", heroes_available(11))
        self.assertNotIn("hero_ignus", heroes_available(10))
        with self.assertRaises(ValueError):
            level_profile(3, hero="hero_gerald")
        self.assertEqual("hero_gerald", level_profile(4, hero="hero_gerald")["hero"])

    def test_profile_validation(self):
        with self.assertRaises(ValueError):
            check_profile({"upgrades": {"archers": 6}})
        with self.assertRaises(ValueError):
            check_profile({"hero": "hero_nobody"})
        with self.assertRaises(ValueError):
            check_profile({"hero_xp": 5})  # xp without a hero
        with self.assertRaises(ValueError):
            check_profile({"upgrades": {"rain": 5}, "levels": {1: 3}})  # 13 stars from one level
        with self.assertRaises(ValueError):
            check_profile({"levels": {1: 4}})

    def test_slot_text_holds_profile_in_game_format(self):
        text = slot_lua({"upgrades": {"archers": 1, "rain": 1}, "hero": "hero_gerald", "hero_xp": 40,
                         "levels": {1: 3}})
        self.assertTrue(text.startswith("local obj1 = {\n") and text.endswith("}\nreturn obj1\n"))
        self.assertIn('\t["heroes"] = {\n\t\t["selected"] = "hero_gerald";', text)
        self.assertIn('["archers"] = 1;', text)
        self.assertIn('["rain"] = 1;', text)
        self.assertIn('\t\t[1] = {\n\t\t\t[1] = 2;\n\t\t\t["stars"] = 3;\n\t\t};\n\t\t[2] = {\n\t\t};', text)
        self.assertIn('["xp"] = 40;', text)
        self.assertEqual(1, text.count("40"))
        self.assertNotIn("selected", slot_lua({}))

    def test_write_profile_only_into_a_fresh_identity(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"APPDATA": tmp}):
            path = write_profile("ident_a", level_profile(4, hero="hero_gerald"))
            self.assertEqual(Path(tmp) / "ident_a" / "slot_1.lua", path)
            self.assertIn("hero_gerald", path.read_text(encoding="utf-8"))
            with self.assertRaises(RuntimeError):
                write_profile("ident_a", level_profile(4))

    def test_profile_matches_loaded_level(self):
        profile = level_profile(4, hero="hero_gerald")
        meta = {"star_upgrades": dict(profile["upgrades"]), "selected_hero": "hero_gerald"}
        self.assertEqual([], profile_matches(profile, meta))
        self.assertEqual(1, len(profile_matches(profile, {**meta, "selected_hero": None})))
        self.assertEqual(1, len(profile_matches(profile, {**meta, "star_upgrades": {}})))
        self.assertEqual([], profile_matches(level_profile(1), {"star_upgrades": buy_upgrades(0)}))


if __name__ == "__main__":
    unittest.main()
