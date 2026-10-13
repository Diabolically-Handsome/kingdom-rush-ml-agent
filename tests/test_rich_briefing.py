"""elite-v3b: the commander's screen capture and memory ledger, and the fair re-test prompts (rich_briefing). No game,
process or model is started."""
from __future__ import annotations

import unittest

from alpharush_rl import commander, eval_job, rich_briefing as rb
from alpharush_rl.commander import Briefing, CommanderPolicy, SteadyBrain, consult_user, request_user
from alpharush_rl.search import BuildOrderPolicy

from test_commander import PLAN, enemy, play, preview, snapshot

CARDS = {"enemy_gargoyle": {"name": "Gargoyle", "card": ["flying", "average speed"], "special": "Flying",
                            "flying_template": True},
         "enemy_zombie": {"name": "Husk", "card": ["slow speed", "medium armor"], "special": "",
                          "flying_template": False},
         "enemy_spider_rotten": {"name": "Noxious Creeper", "card": ["medium magic resistance", "spawns spiderlings",
                                                                     "costs 3 lives"], "special": "",
                                 "flying_template": False}}
GEO = {"paths": [[[[x, 100] for x in range(0, 301, 10)]] * 3,
                 [[[x, 300] for x in range(0, 201, 10)] + [[200 + 10 * i, 300 - 20 * i] for i in range(1, 11)]] * 3],
       "connections": [], "exits": [{"k": 1, "x": 310, "y": 100}], "ice": [[False] * 31, [False] * 31]}


def capture_policy(capture=True):
    return CommanderPolicy(BuildOrderPolicy(PLAN), SteadyBrain(), seed=1001, capture=capture)


class CaptureTests(unittest.TestCase):
    def test_capturing_changes_no_decision_and_records_the_screen(self):
        plain = play(capture_policy(capture=False))
        captured = play(capture_policy(capture=True))
        for key in ("plan", "trace_sha256", "outcome", "n_forced_waits"):
            self.assertEqual(plain[key], captured[key], key)
        self.assertEqual([(d["tick"], d["action"]) for d in plain["decisions"]],
                         [(d["tick"], d["action"]) for d in captured["decisions"]])
        consults = [c for d in captured["decisions"] for c in (d.get("meta") or {}).get("commander_consults") or []]
        self.assertTrue(consults)
        for c in consults:
            self.assertEqual({"tick", "wave", "wave_total", "lives", "gold", "level_idx", "towers", "enemies",
                              "heroes", "next_wave", "powers_ui"}, set(c["capture"]))
            self.assertEqual({"lives_lost", "hero_deaths"}, set(c["ledger"]))
            self.assertEqual({"cast", "boss_focus"}, set(c["translator"]))
        bare = [c for d in plain["decisions"] for c in (d.get("meta") or {}).get("commander_consults") or []]
        self.assertTrue(all("capture" not in c and "ledger" not in c for c in bare))
        # The old prompt is unchanged by capturing (the replay check compares these hashes).
        self.assertEqual([c["prompt_sha256"] for c in bare], [c["prompt_sha256"] for c in consults])

    def test_ledger_puts_each_life_on_the_path_that_got_out(self):
        policy = capture_policy()
        s1 = snapshot(tick=100, lives=20, enemies=[enemy(1, path=2, progress=0.95, lives_cost=1),
                                                    enemy(2, path=1, progress=0.5, lives_cost=1),
                                                    enemy(3, path=1, progress=0.97, lives_cost=3)])
        policy._observe(s1, [])
        s2 = snapshot(tick=160, lives=16, enemies=[enemy(2, path=1, progress=0.6, lives_cost=1)])
        policy._observe(s2, [])
        self.assertEqual({"1": 3, "2": 1}, policy.ledger["lives_lost"])
        s3 = snapshot(tick=220, lives=14, enemies=[enemy(2, path=1, progress=0.7, lives_cost=1)])
        policy._observe(s3, [])  # nobody near an exit vanished: the loss is not attributed to a path
        self.assertEqual({"1": 3, "2": 1, "unknown": 2}, policy.ledger["lives_lost"])
        s4 = snapshot(tick=280, lives=14, hero={"dead": True})
        policy._observe(s4, [])
        s5 = snapshot(tick=340, lives=14, hero={"dead": True})
        policy._observe(s5, [])
        self.assertEqual(1, policy.ledger["hero_deaths"])

    def test_capture_player_is_a_commander_player_with_capture_on(self):
        self.assertIn("cmd_steady_capture", eval_job.COMMANDER_PLAYERS)
        task = {"seed": 1201, "level": 15}
        self.assertTrue(commander.make_player("cmd_steady_capture", PLAN, task).capture)
        self.assertFalse(commander.make_player("cmd_steady", PLAN, task).capture)


def _plays_and_data():
    state = snapshot(enemies=[enemy(1, template="enemy_gargoyle", path=1, progress=0.3),
                              enemy(2, template="enemy_zombie", path=2, progress=0.6)],
                     preview=preview(("enemy_spider_rotten", 1, 4), ("enemy_zombie", 2, 6)))
    data = Briefing.from_state(state).data
    geometry = {"exits": {"1": (330, 100), "2": (330, 300)}, "home_exit": "1", "path_exit": {"1": "1", "2": "2"}}
    plays = commander.PlayGenerator(cast=40).plays(data, geometry)
    return state, data, plays


class PlayTextTests(unittest.TestCase):
    def test_every_play_text_states_what_the_translator_does(self):
        _, data, plays = _plays_and_data()
        texts = rb.options_v2(plays, data, CARDS, boss=True)
        self.assertEqual([p["id"] for p in plays], [t[1:t.index("]")] for t in texts])
        steady = texts[0]
        self.assertIn("covered 40% of its path", steady)
        self.assertIn("only at the boss", steady)
        by_name = {p["name"]: t for p, t in zip(plays, texts)}
        self.assertIn("35% along path", by_name["counter"])
        self.assertIn("released if 2 lives are lost within 10 seconds", by_name["counter"])
        self.assertIn("90% along the path that leaves by exit 2", by_name["guard_exit"])
        self.assertIn("held until", by_name["focus"])
        self.assertIn("Noxious Creeper", by_name["focus"])
        for t in texts:
            self.assertNotIn("block them", t)
            self.assertEqual([], rb.leaks(t))

    def test_on_field_targets_are_named(self):
        _, data, plays = _plays_and_data()
        counter = next(p for p in plays if p["name"] == "counter" and p["threat"]["template"] == "enemy_zombie")
        text = rb.play_text_v2(counter, data, 40, CARDS)
        self.assertIn("(some are on the field now)", text)


class RequestTests(unittest.TestCase):
    def test_label_rotations_match_the_live_commander_prompt(self):
        _, data, plays = _plays_and_data()
        briefing = Briefing(data)
        options = [p["text"] for p in plays]
        labels, rotations = rb.label_requests(consult_user(briefing), options)
        self.assertEqual(request_user(consult_user(briefing), options), rotations[0][1])
        self.assertEqual(list("ABCDEF"[:len(plays)]), labels)
        self.assertEqual(min(len(plays), 6), len(rotations))
        for order, text in rotations:
            self.assertEqual(sorted(order), list(range(len(plays))))
            self.assertTrue(text.endswith("Answer with one label."))

    def test_final_answers_parse_and_strip(self):
        ids = ["P0", "P1", "P4"]
        self.assertEqual(2, rb.parse_final("The spiders matter most.\nFINAL: [P4]", ids))
        self.assertEqual(1, rb.parse_final("final: p1", ids))
        self.assertIsNone(rb.parse_final("no decision", ids))
        self.assertIsNone(rb.parse_final("FINAL: [P3]", ids))
        self.assertEqual("Reasoning here.", rb.strip_final("Reasoning here.\nFINAL: [P1]"))


class LevelMapTests(unittest.TestCase):
    def test_geometry_queries(self):
        m = rb.LevelMap(GEO)
        self.assertEqual(1, m.exit_of(1))
        self.assertEqual(1, m.exit_of(2))
        path, progress, distance = m.nearest(150, 104)
        self.assertEqual((1, 0.5), (path, round(progress, 2)))
        self.assertLess(distance, 5)
        self.assertEqual([(2, 0.9)], [(p, round(a, 2)) for p, a in m.joins(1)])  # they meet at the exit
        self.assertIsNone(m.ice_span(1))
        quarter = m.band_nodes(1, 0)
        self.assertTrue(all(0 <= x < 80 for x, _ in quarter))

    def test_coverage_uses_the_game_ellipse_and_barracks_rally_points(self):
        m = rb.LevelMap(GEO)
        towers = [{"template": "tower_mage_3", "x": 160, "y": 130, "range": 60},
                  {"template": "tower_engineer_2", "x": 20, "y": 170, "range": 100},
                  {"template": "tower_barrack_3", "x": 250, "y": 60, "soldier_count": 3, "max_soldiers": 3,
                   "rally_x": 260, "rally_y": 100}]
        quarters = rb.coverage(m, towers, 1)
        self.assertIn("Wizard Tower", quarters[1])
        self.assertIn("Wizard Tower", quarters[2])
        # 70 px below the path is exactly the ellipse's vertical edge (0.7 x 100): the node straight above is in.
        self.assertIn("Dwarven Artillery (no air)", quarters[0])
        self.assertTrue(any(q.startswith("Knights Barracks (3/3 soldiers holding the road at 87%)") for q in quarters[3]))

    def test_rich_text_shows_map_defences_cards_and_memory_without_leaks(self):
        state, data, plays = _plays_and_data()
        capture = commander.capture_state(state)
        capture["towers"] = [{"template": "tower_archer_2", "x": 100, "y": 130, "range": 160}]
        text = rb.rich_text(data, capture, {"lives_lost": {"2": 3, "unknown": 1}, "hero_deaths": 1},
                            rb.LevelMap(GEO), CARDS)
        for part in ("Map (paths are numbered", "Your defences reaching", "Enemy cards: ", "Noxious Creeper",
                     "Gargoyle: flying", "lives lost path 2: 3, unclear where: 1", "the hero has fallen 1 time"):
            self.assertIn(part, text)
        self.assertEqual([], rb.leaks(text))
        thin = rb.thin_text_v2(data, CARDS)
        self.assertIn("Husk", thin)
        self.assertNotIn("Your defences", thin)


if __name__ == "__main__":
    unittest.main()
