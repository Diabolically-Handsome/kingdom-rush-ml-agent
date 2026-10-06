"""Episode loop on a deterministic toy env; never starts the game or a process."""
from __future__ import annotations

import copy
import json
import random
import unittest

from alpharush_rl.episode import EpisodeProtocol, replay_unsupported, replay_verify, run_episode, validate_choice
from alpharush_rl.journal import canonical_json, sha256_data
from alpharush_rl.menus import build_menu

COSTS = {"archer": 70, "barrack": 70, "mage": 100, "engineer": 125}
POWER = {"archer": 1, "barrack": 1, "mage": 2, "engineer": 2}


class FakeEnv:
    """Toy level with NativeEnv's plan/trace/act/replay/terminal semantics.

    Rules: +1 gold every ``income_every`` ticks; a build costs its catalog price
    and takes 180 ticks, send_wave takes 2; enemies walk one step per tick and
    cost one life on reaching ``path_ticks``; each tower hits the leading enemy
    every tick; a kill pays 5 gold. Waves after the first auto-start after
    ``wave_gap`` calm ticks. Spawn spacing is a deterministic function of seed.
    """

    def __init__(self, seed=1001, level=1, *, holders=3, gold=150, lives=20, waves=3,
                 income_every=10, path_ticks=400, wave_gap=300, fail_action=None, frozen=False):
        self.seed, self.level = seed, level
        self.config = {"holders": holders, "gold": gold, "lives": lives, "waves": waves,
                       "income_every": income_every, "path_ticks": path_ticks, "wave_gap": wave_gap}
        self.fail_action, self.frozen = fail_action, frozen
        self.state, self.trace, self.plan = None, [], []
        self.resets = self.closes = 0

    def reset(self):
        self.resets += 1
        c = self.config
        state = {"type": "game_state", "level_idx": self.level, "tick": 1, "gold": c["gold"],
                 "lives": c["lives"], "wave": 0, "wave_total": c["waves"], "next_wave_ticks": None,
                 "level_won": False, "level_lost": False, "game_over": False, "spawned": 0,
                 "holders": [{"id": index + 1, "mesh_id": str(index + 1), "blocked": False,
                              "path_score": 10 - index, "x": 100 * (index + 1), "y": 200}
                             for index in range(c["holders"])],
                 "towers": [], "enemies": [], "heroes": [],
                 "level_path_wave_counts": [[4 + 2 * wave] for wave in range(1, c["waves"] + 1)]}
        self.state = self._refresh(state)
        self.trace = [{"kind": "reset", "state_sha256": sha256_data(self.state),
                       "tick": self.state["tick"], "seed": self.seed, "level": self.level}]
        self.plan = []
        return self.state

    def _refresh(self, s):
        if s["lives"] <= 0:
            s["lives"], s["level_lost"] = 0, True
        elif s["wave"] >= s["wave_total"] and not s["enemies"]:
            s["level_won"] = True
        s["game_over"] = s["level_won"] or s["level_lost"]
        s["wave_ready"] = not s["game_over"] and s["wave"] < s["wave_total"] and not s["enemies"]
        s["enemy_count"] = len(s["enemies"])
        catalog = [{"action": "build_tower", "holder_id": holder["id"], "tower_type": kind,
                    "cost": cost, "available": s["gold"] >= cost}
                   for holder in s["holders"] for kind, cost in COSTS.items()]
        if s["wave_ready"]:
            catalog.append({"action": "send_wave", "available": True})
        s["action_catalog"] = catalog
        return s

    def _spawn(self, s):
        s["wave"] += 1
        s["next_wave_ticks"] = None
        rng = random.Random(f"{self.seed}:{s['wave']}")
        offset = 0
        for _ in range(4 + 2 * s["wave"]):
            s["spawned"] += 1
            s["enemies"].append({"id": 1000 + s["spawned"], "hp": 30 + 10 * s["wave"], "progress": -offset})
            offset += rng.randint(15, 25)

    def _tick(self, s):
        c = self.config
        s["tick"] += 1
        if c["income_every"] and s["tick"] % c["income_every"] == 0:
            s["gold"] += 1
        for tower in s["towers"]:
            alive = [e for e in s["enemies"] if e["progress"] >= 0 and e["hp"] > 0]
            if not alive:
                break
            max(alive, key=lambda e: (e["progress"], -e["id"]))["hp"] -= POWER[tower["type"]]
        survivors = []
        for enemy in s["enemies"]:
            if enemy["hp"] <= 0:
                s["gold"] += 5
                continue
            enemy["progress"] += 1
            if enemy["progress"] >= c["path_ticks"]:
                s["lives"] -= 1
            else:
                survivors.append(enemy)
        s["enemies"] = survivors
        if 1 <= s["wave"] < s["wave_total"] and not s["enemies"]:
            s["next_wave_ticks"] = c["wave_gap"] if s["next_wave_ticks"] is None else s["next_wave_ticks"] - 1
            if s["next_wave_ticks"] <= 0:
                self._spawn(s)
        self._refresh(s)

    def advance(self, ticks, *, record_plan=True):
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        state = copy.deepcopy(self.state)
        before_tick = state["tick"]
        for _ in range(0 if self.frozen else ticks):
            if state["game_over"]:
                break
            self._tick(state)
        self.state = state
        self.trace.append({"kind": "step", "ticks": ticks, "advanced_ticks": state["tick"] - before_tick,
                           "tick": state["tick"], "state_sha256": sha256_data(state)})
        if record_plan:
            self.plan.append({"ticks": ticks})
        return state

    def act(self, action):
        action = dict(action)
        before = self.state
        name = action.get("action")
        if name == self.fail_action:
            raise RuntimeError(f"Native action rejected: injected {name} failure")
        if name == "wait":
            after = self.advance(action.get("ticks", 30), record_plan=False)
        else:
            if not any(item["action"] == action for item in build_menu(before)):
                raise ValueError("Action is not in the native legal menu")
            state = copy.deepcopy(before)
            if name == "build_tower":
                holder = next(h for h in state["holders"] if h["id"] == action["holder_id"])
                state["holders"].remove(holder)
                state["gold"] -= COSTS[action["tower_type"]]
                kind = action["tower_type"]
                state["towers"].append({"id": 100 + len(state["towers"]), "holder_id": holder["mesh_id"],
                                        "template": f"tower_{kind}_1", "type": kind,
                                        "is_special": False, "level": 1})
                ticks = 180
            elif name == "send_wave":
                self._spawn(state)
                ticks = 2
            else:
                raise RuntimeError("Unverified action scope")
            self.state = self._refresh(state)
            after = self.advance(ticks, record_plan=False)
        receipt = {"accepted": True, "executed": True, "action": action,
                   "tick_before": before["tick"], "tick_after": after["tick"],
                   "before_sha256": sha256_data(before), "after_sha256": sha256_data(self.state)}
        self.trace.append({"kind": "action", "receipt": receipt})
        self.plan.append({"action": action})
        return receipt

    def terminal(self):
        s = self.state
        if not (s.get("level_won") or s.get("level_lost")):
            return None
        return {"source": "native", "terminal": True, "level_won": bool(s["level_won"]),
                "level_lost": bool(s["level_lost"]), "lives": s["lives"],
                "wave": s["wave"], "tick": s["tick"], "state_sha256": sha256_data(s)}

    def replay(self, plan):
        for command in plan:
            if "action" in command:
                self.act(command["action"])
            else:
                self.advance(command["ticks"])
        return self.state

    def close(self):
        self.closes += 1


def label_of(menu, predicate):
    return next((item["label"] for item in menu if predicate(item["action"])), None)


def pick_wait(state, menu, context):
    return "A"


def pick_send_wave(state, menu, context):
    return label_of(menu, lambda a: a["action"] == "send_wave") or "A"


def pick_greedy(state, menu, context):
    builds = [item for item in menu if item["action"]["action"] == "build_tower"
              and item["action"]["tower_type"] == "archer"]
    if builds:
        return min(builds, key=lambda item: item["action"]["holder_id"])["label"]
    return pick_send_wave(state, menu, context)


class RecordingPolicy:
    def __init__(self, pick, name="test-policy", distribution=None, meta=None):
        self.pick, self.name = pick, name
        self.distribution, self.meta = distribution, meta
        self.calls = []

    def choose(self, state, menu, context):
        self.calls.append({"state": copy.deepcopy(state), "menu": copy.deepcopy(menu), "context": dict(context)})
        reply = {"label": self.pick(state, menu, context), "provenance": f"scripted:{self.name}",
                 "distribution": self.distribution(menu) if self.distribution else None}
        if self.meta is not None:
            reply["meta"] = self.meta
        return reply


class FailingPolicy:
    name = "failing"

    def __init__(self, fail_at=0):
        self.fail_at, self.calls = fail_at, 0

    def choose(self, state, menu, context):
        self.calls += 1
        if context["decision_index"] >= self.fail_at:
            raise RuntimeError("model broker unavailable")
        return {"label": "A", "provenance": "model:test", "distribution": None}


class StepClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


def uniform(menu):
    return {"labels": [item["label"] for item in menu], "p": [1.0 / len(menu)] * len(menu)}


def run(env, policy, **protocol):
    return run_episode(env, policy, EpisodeProtocol(**protocol), seed=env.seed, level=env.level,
                       episode_id="ep-test", clock=StepClock())


class FakeEnvTests(unittest.TestCase):
    def test_fake_env_is_deterministic_and_seed_dependent(self):
        results = []
        for seed in (1001, 1001, 1002):
            env = FakeEnv(seed=seed)
            results.append(run(env, RecordingPolicy(pick_greedy)))
        self.assertEqual(results[0]["trace_sha256"], results[1]["trace_sha256"])
        self.assertEqual(results[0]["plan"], results[1]["plan"])
        self.assertNotEqual(results[0]["trace_sha256"], results[2]["trace_sha256"])

    def test_towers_matter_in_the_toy_rules(self):
        won = run(FakeEnv(), RecordingPolicy(pick_greedy))
        lost = run(FakeEnv(), RecordingPolicy(pick_send_wave))
        self.assertEqual(won["status"], "terminal")
        self.assertTrue(won["outcome"]["level_won"])
        # Every wave is called by the policy, none auto-started behind its back.
        self.assertEqual(sum(d["action"] == {"action": "send_wave"} for d in won["decisions"]), 3)
        self.assertEqual(lost["status"], "terminal")
        self.assertTrue(lost["outcome"]["level_lost"])
        self.assertEqual(lost["outcome"]["lives"], 0)


class DecisionEventTests(unittest.TestCase):
    def test_trivial_menu_never_calls_policy(self):
        # No holders: after the wave is sent only "wait" is legal until the end.
        env = FakeEnv(holders=0, waves=1, lives=3)
        policy = RecordingPolicy(pick_send_wave)
        result = run(env, policy)
        self.assertEqual(result["status"], "terminal")
        self.assertEqual(len(policy.calls), 1)
        self.assertEqual(result["n_decisions"], 1)
        self.assertGreater(result["n_forced_waits"], 0)
        self.assertEqual(result["plan"][0], {"action": {"action": "send_wave"}})
        self.assertTrue(all(step == {"action": {"action": "wait", "ticks": 60}} for step in result["plan"][1:]))
        self.assertEqual(len(result["plan"]), 1 + result["n_forced_waits"])

    def test_only_option_set_changes_trigger_decisions(self):
        # Gold 60 -> 70 (archer, barrack) -> 100 (mage) -> 125 (engineer), then nothing changes.
        env = FakeEnv(holders=1, gold=60)
        policy = RecordingPolicy(pick_wait)
        result = run(env, policy, max_interval_ticks=10 ** 9, max_ticks=2000)
        self.assertEqual(result["status"], "stalled")
        self.assertEqual(result["stall_reason"], "max_ticks")
        self.assertEqual(len(policy.calls), 4)
        self.assertEqual([len(d["labels"]) for d in result["decisions"]], [2, 4, 5, 6])
        keysets = [frozenset(canonical_json(m["action"]) for m in call["menu"][1:]) for call in policy.calls]
        self.assertTrue(all(a != b for a, b in zip(keysets, keysets[1:])))
        self.assertEqual([d["gold"] >= cost for d, cost in zip(result["decisions"][1:], (70, 100, 125))],
                         [True, True, True])
        self.assertGreater(result["n_forced_waits"], 20)

    def test_options_reappearing_after_trivial_gap_are_a_new_event(self):
        # {send_wave} -> wait-only while the wave runs -> {send_wave}: same set as the
        # last decision, but nothing was offered in between, so the policy is asked again.
        env = FakeEnv(holders=0, waves=3, income_every=0, wave_gap=10 ** 6)
        policy = RecordingPolicy(pick_send_wave)
        result = run(env, policy, max_interval_ticks=10 ** 9)
        self.assertEqual(result["status"], "terminal")
        self.assertEqual([d["action"] for d in result["decisions"]], [{"action": "send_wave"}] * 3)
        self.assertEqual(result["outcome"]["wave"], 3)
        self.assertTrue(all(d["labels"] == ["A", "B"] for d in result["decisions"]))

    def test_max_interval_reasks_unchanged_menu(self):
        env = FakeEnv(holders=1, gold=0, income_every=0)
        policy = RecordingPolicy(pick_wait)
        result = run(env, policy, max_interval_ticks=300, max_ticks=1500)
        start = 1
        self.assertEqual([d["tick"] - start for d in result["decisions"]], [0, 300, 600, 900, 1200])
        self.assertEqual(result["n_forced_waits"], 20)
        self.assertEqual(result["stall_reason"], "max_ticks")
        self.assertEqual(result["final_tick"] - start, 1500)
        self.assertEqual(len(result["plan"]), 25)

    def test_max_decisions_stalls(self):
        env = FakeEnv(holders=1, gold=0, income_every=0)
        policy = RecordingPolicy(pick_wait)
        result = run(env, policy, max_interval_ticks=60, max_decisions=5)
        self.assertEqual((result["status"], result["stall_reason"]), ("stalled", "max_decisions"))
        self.assertEqual(result["n_decisions"], 5)
        self.assertEqual(len(policy.calls), 5)
        self.assertEqual(result["n_forced_waits"], 0)
        self.assertIsNone(result["outcome"])

    def test_first_wave_deadline_stalls_only_while_wave_zero(self):
        env = FakeEnv(holders=1, gold=0, income_every=0)
        result = run(env, RecordingPolicy(pick_wait), max_interval_ticks=10 ** 6, first_wave_deadline_ticks=600)
        self.assertEqual((result["status"], result["stall_reason"]), ("stalled", "first_wave_not_started"))
        self.assertEqual(result["final_tick"] - 1, 600)
        self.assertEqual(result["n_decisions"], 1)
        sent = run(FakeEnv(), RecordingPolicy(pick_greedy), first_wave_deadline_ticks=600)
        self.assertEqual(sent["status"], "terminal")
        self.assertIsNone(sent["stall_reason"])

    def test_decision_record_context_and_isolation(self):
        env = FakeEnv()
        seen = []

        def vandal(state, menu, context):
            seen.append(sha256_data(state))
            state["gold"] = -999  # must not reach the env
            menu.clear()
            return pick_greedy(state, build_menu(env.state, wait_ticks=60), context)

        policy = RecordingPolicy(vandal, name="vandal", meta={"note": "unit"})
        result = run_episode(env, policy, EpisodeProtocol(), seed=1001, level=1, difficulty=2,
                             episode_id="ep-ctx", clock=StepClock())
        self.assertEqual(result["status"], "terminal")
        self.assertGreaterEqual(env.state["gold"], 0)
        self.assertEqual(env.closes, 0)  # caller owns close()
        self.assertEqual(env.resets, 1)
        expected_keys = {"index", "tick", "wave", "gold", "lives", "state_sha256", "menu_sha256", "labels",
                         "label", "action", "provenance", "distribution", "latency_seconds", "meta"}
        for index, (decision, call) in enumerate(zip(result["decisions"], policy.calls)):
            self.assertEqual(set(decision), expected_keys)
            self.assertEqual(decision["index"], index)
            self.assertEqual(decision["state_sha256"], seen[index])
            self.assertEqual(decision["latency_seconds"], 1.0)
            self.assertEqual(decision["meta"], {"note": "unit"})
            self.assertEqual(decision["provenance"], "scripted:vandal")
            self.assertEqual(call["context"], {"episode_id": "ep-ctx", "decision_index": index, "level": 1,
                                               "seed": 1001, "difficulty": 2, "tick": decision["tick"],
                                               "protocol": "kr1-episode-v1"})
        self.assertEqual(set(result), {"schema", "protocol", "policy", "episode_id", "seed", "level",
                                       "difficulty", "status", "stall_reason", "void_reason", "outcome",
                                       "final_tick", "final_state_sha256", "final_summary", "decisions",
                                       "n_decisions", "n_forced_waits", "plan", "trace_sha256", "wall_seconds"})
        self.assertEqual(result["schema"], "alpharush-episode-v1")
        self.assertEqual(result["protocol"]["wait_ticks"], 60)
        self.assertEqual(result["final_state_sha256"], sha256_data(env.state))
        self.assertEqual(result["trace_sha256"], sha256_data(env.trace))
        json.dumps(result, allow_nan=False)  # whole result is plain JSON


class VoidTests(unittest.TestCase):
    def test_policy_exception_voids_without_fallback(self):
        env = FakeEnv()
        policy = FailingPolicy(fail_at=1)
        result = run(env, policy, max_interval_ticks=60)
        self.assertEqual(result["status"], "void")
        self.assertTrue(result["void_reason"].startswith("policy_error: RuntimeError"))
        self.assertEqual(result["n_decisions"], 1)
        self.assertEqual(policy.calls, 2)
        # Only the first (successful) decision was executed; nothing was substituted.
        self.assertEqual(result["plan"], [{"action": {"action": "wait", "ticks": 60}}])
        self.assertIsNone(result["outcome"])

    def test_illegal_label_or_reply_voids_before_any_action(self):
        for reply in ({"label": "ZZ", "provenance": "model:test", "distribution": None},
                      {"label": "a", "provenance": "model:test"},
                      {"label": "A", "provenance": ""},
                      {"label": "A", "provenance": "model:test", "meta": "not-a-dict"},
                      {"label": "A", "provenance": "model:test", "meta": {"x": float("nan")}},
                      "A", None):
            with self.subTest(reply=reply):
                policy = RecordingPolicy(pick_wait)
                policy.choose = lambda state, menu, context, reply=reply: reply
                env = FakeEnv()
                result = run(env, policy)
                self.assertEqual(result["status"], "void")
                self.assertTrue(result["void_reason"].startswith("invalid_choice: "))
                self.assertEqual(result["n_decisions"], 0)
                self.assertEqual(result["plan"], [])

    def test_native_action_error_voids(self):
        env = FakeEnv(fail_action="build_tower")
        result = run(env, RecordingPolicy(pick_greedy))
        self.assertEqual(result["status"], "void")
        self.assertTrue(result["void_reason"].startswith("native_action_error: RuntimeError"))
        self.assertEqual(result["n_decisions"], 1)
        self.assertEqual(result["plan"], [])
        forced = run(FakeEnv(holders=0, waves=1, fail_action="wait"), RecordingPolicy(pick_send_wave))
        self.assertEqual(forced["status"], "void")
        self.assertEqual(forced["n_forced_waits"], 0)

    def test_frozen_native_clock_voids_instead_of_looping(self):
        result = run(FakeEnv(holders=0, waves=1, frozen=True), RecordingPolicy(pick_wait))
        self.assertEqual(result["status"], "void")
        self.assertEqual(result["void_reason"], "native_action_error: native tick did not advance")

    def test_check_exception_propagates_and_env_stays_open(self):
        env, calls = FakeEnv(), []

        def check():
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("STOP")

        with self.assertRaisesRegex(RuntimeError, "STOP"):
            run_episode(env, RecordingPolicy(pick_greedy), EpisodeProtocol(), seed=1001, level=1, check=check)
        self.assertEqual(env.closes, 0)
        stopped = FakeEnv()
        with self.assertRaises(RuntimeError):
            run_episode(stopped, RecordingPolicy(pick_greedy), EpisodeProtocol(), seed=1001, level=1,
                        check=lambda: (_ for _ in ()).throw(RuntimeError("STOP")))
        self.assertEqual(stopped.resets, 0)  # STOP before reset never starts the level


class DistributionTests(unittest.TestCase):
    MENU = [{"label": "A", "action": {"action": "wait", "ticks": 60}},
            {"label": "B", "action": {"action": "send_wave"}}]

    def test_valid_distribution_is_normalized_and_kept(self):
        choice = validate_choice({"label": "B", "provenance": "model:8b",
                                  "distribution": {"labels": ["A", "B"], "p": [0.25, 0.75],
                                                   "logp": [-1.386, -0.288], "model": "m"}}, self.MENU)
        self.assertEqual(choice["distribution"]["p"], [0.25, 0.75])
        self.assertEqual(choice["distribution"]["model"], "m")
        self.assertEqual(choice["meta"], {})
        ok = validate_choice({"label": "A", "provenance": "x",
                              "distribution": {"labels": ["A", "B"], "p": [1, 0]}}, self.MENU)
        self.assertEqual(ok["distribution"]["p"], [1.0, 0.0])
        self.assertIsNone(validate_choice({"label": "A", "provenance": "x"}, self.MENU)["distribution"])

    def test_invalid_distributions_are_rejected(self):
        bad = [{"labels": ["B", "A"], "p": [0.5, 0.5]},
               {"labels": ["A"], "p": [1.0]},
               {"labels": ["A", "B"], "p": [1.0]},
               {"labels": ["A", "B"], "p": [0.6, 0.6]},
               {"labels": ["A", "B"], "p": [1.5, -0.5]},
               {"labels": ["A", "B"], "p": [float("nan"), 1.0]},
               {"labels": ["A", "B"], "p": [float("inf"), 0.0]},
               {"labels": ["A", "B"], "p": [True, False]},
               {"labels": ["A", "B"], "p": "0.5,0.5"},
               {"labels": ["A", "B"], "p": [0.5, 0.5], "extra": {1, 2}},
               ["A", "B"]]
        for distribution in bad:
            with self.subTest(distribution=distribution):
                with self.assertRaises(ValueError):
                    validate_choice({"label": "A", "provenance": "x", "distribution": distribution}, self.MENU)
        within = validate_choice({"label": "A", "provenance": "x",
                                  "distribution": {"labels": ["A", "B"], "p": [0.5, 0.5 + 5e-7]}}, self.MENU)
        self.assertEqual(within["label"], "A")

    def test_episode_records_distribution_and_voids_on_bad_one(self):
        result = run(FakeEnv(), RecordingPolicy(pick_greedy, distribution=uniform))
        self.assertEqual(result["status"], "terminal")
        for decision in result["decisions"]:
            self.assertEqual(decision["distribution"]["labels"], decision["labels"])
            self.assertAlmostEqual(sum(decision["distribution"]["p"]), 1.0)

        def reversed_labels(menu):
            dist = uniform(menu)
            dist["labels"].reverse()
            return dist

        result = run(FakeEnv(), RecordingPolicy(pick_greedy, distribution=reversed_labels))
        self.assertEqual(result["status"], "void")
        self.assertIn("distribution labels", result["void_reason"])
        self.assertEqual(result["plan"], [])


class SpellEnv:
    """Mid-wave v2-style toy state whose only options are spells anchored on the leading enemy.

    The leader walks one unit per tick, so every spell's x changes on every step; the
    leader itself changes from enemy 7 to enemy 8 at ``swap_tick``; reinforcements
    (power 2) are offered from ``power_2_tick``. A cast is recorded and changes nothing.
    """

    def __init__(self, swap_tick=200, power_2_tick=401):
        self.seed, self.level = 1001, 1
        self.swap_tick, self.power_2_tick = swap_tick, power_2_tick
        self.casts, self.plan, self.trace = [], [], []

    def _state(self, tick):
        leader = {"id": 7 if tick < self.swap_tick else 8, "x": 100.4 + tick, "y": 400.6, "path_progress": 0.5}
        catalog = [{"action": "use_power", "power": power, "x": round(leader["x"]), "y": round(leader["y"]),
                    "anchor_id": leader["id"], "cost": 0, "available": True, "legal": True}
                   for power in (1, 2) if power == 1 or tick >= self.power_2_tick]
        return {"type": "game_state", "tick": tick, "wave": 1, "wave_total": 1, "gold": 0, "lives": 20,
                "level_won": False, "level_lost": False, "wave_ready": False, "enemies": [leader],
                "towers": [], "holders": [], "action_catalog": catalog}

    def reset(self):
        self.state, self.plan, self.trace = self._state(1), [], [{"kind": "reset"}]
        return self.state

    def advance(self, ticks, *, record_plan=True):
        self.state = self._state(self.state["tick"] + ticks)
        self.trace.append({"kind": "step", "tick": self.state["tick"]})
        if record_plan:
            self.plan.append({"ticks": ticks})
        return self.state

    def act(self, action):
        if action["action"] == "wait":
            self.advance(action["ticks"], record_plan=False)
        else:
            if not any(item["action"] == action for item in build_menu(self.state)):
                raise ValueError("Action is not in the native legal menu")
            self.casts.append(dict(action))
            self.advance(2, record_plan=False)
        self.plan.append({"action": dict(action)})

    def terminal(self):
        return None

    def close(self):
        pass


class SpellAnchorDecisionTests(unittest.TestCase):
    def test_moving_spell_anchors_alone_never_reask_the_policy(self):
        env = SpellEnv()
        policy = RecordingPolicy(pick_wait)
        result = run(env, policy, max_interval_ticks=10 ** 9, max_ticks=600)
        self.assertEqual((result["status"], result["stall_reason"]), ("stalled", "max_ticks"))
        # Asked at the start and when reinforcements first appear (tick 421 >= 401), never for a
        # spell whose coordinates moved every block or whose anchor enemy changed at tick 200.
        self.assertEqual([d["tick"] for d in result["decisions"]], [1, 421])
        self.assertEqual(result["n_forced_waits"], 8)
        menus = [[item["action"] for item in call["menu"][1:]] for call in policy.calls]
        self.assertEqual(menus, [
            [{"action": "use_power", "power": 1, "x": 101, "y": 401, "anchor_id": 7}],
            [{"action": "use_power", "power": 1, "x": 521, "y": 401, "anchor_id": 8},
             {"action": "use_power", "power": 2, "x": 521, "y": 401, "anchor_id": 8}],
        ])
        # The full menu really changed at every forced wait (anchor x, then the enemy id).
        full = [frozenset(canonical_json(item["action"]) for item in build_menu(env._state(t))[1:])
                for t in range(1, 421, 60)]
        self.assertEqual(len(set(full)), len(full))
        # The max-interval rule still re-asks an unchanged spell menu.
        result = run(SpellEnv(), RecordingPolicy(pick_wait), max_interval_ticks=300, max_ticks=600)
        self.assertEqual([d["tick"] for d in result["decisions"]], [1, 301, 421])

    def test_executed_spell_keeps_its_anchor_coordinates(self):
        def pick_fire(state, menu, context):
            return label_of(menu, lambda a: a["action"] == "use_power" and a["power"] == 1)

        env = SpellEnv()
        result = run(env, RecordingPolicy(pick_fire), max_interval_ticks=10 ** 9, max_ticks=600)
        self.assertEqual([d["tick"] for d in result["decisions"]], [1, 423])
        expected = [{"action": "use_power", "power": 1, "x": 101, "y": 401, "anchor_id": 7},
                    {"action": "use_power", "power": 1, "x": 523, "y": 401, "anchor_id": 8}]
        self.assertEqual([d["action"] for d in result["decisions"]], expected)
        self.assertEqual(env.casts, expected)
        self.assertEqual([p["action"] for p in result["plan"] if p["action"]["action"] == "use_power"], expected)

    def test_option_key_reduces_only_spells(self):
        from alpharush_rl.episode import option_key
        for action in ({"action": "build_tower", "holder_id": 1, "tower_type": "archer"}, {"action": "send_wave"},
                       {"action": "wait", "ticks": 60},
                       {"action": "upgrade_tower", "tower_id": 3, "target": "tower_archer_2"},
                       {"action": "upgrade_power", "tower_id": 3, "power": "poison"},
                       {"action": "sell_tower", "tower_id": 3}):
            self.assertEqual(option_key(action), canonical_json(action))
        spell = {"action": "use_power", "power": 2, "x": 5, "y": 6, "anchor_id": 9}
        self.assertEqual(option_key(spell), canonical_json({"action": "use_power", "power": 2}))
        self.assertEqual(spell, {"action": "use_power", "power": 2, "x": 5, "y": 6, "anchor_id": 9})  # not mutated
        self.assertNotEqual(option_key(spell), option_key(dict(spell, power=1)))


class ProtocolRuleTests(unittest.TestCase):
    def test_reset_on_trivial_menu_is_recorded_and_can_be_disabled(self):
        self.assertTrue(EpisodeProtocol().reset_on_trivial_menu)
        with self.assertRaises(ValueError):
            EpisodeProtocol(reset_on_trivial_menu=1)
        result = run(FakeEnv(seed=7), RecordingPolicy(pick_send_wave))
        self.assertIs(result["protocol"]["reset_on_trivial_menu"], True)


class MetaAndSummaryTests(unittest.TestCase):
    def test_meta_is_observed_once_after_reset_and_failures_are_recorded(self):
        env = FakeEnv(seed=7)
        calls = []
        env.meta = lambda: calls.append(env.state["tick"]) or {"level_idx": 1, "locked_towers": ["tower_mage_2"]}
        result = run_episode(env, RecordingPolicy(pick_greedy), EpisodeProtocol(), seed=7, level=1,
                             observe_meta=True, clock=StepClock())
        self.assertEqual(result["level_meta"], {"level_idx": 1, "locked_towers": ["tower_mage_2"]})
        self.assertEqual(len(calls), 1)
        # meta() is not part of the trace: replay without meta reproduces the episode.
        self.assertTrue(replay_verify(lambda: FakeEnv(seed=7), json.loads(json.dumps(result)))["replay_verified"])

        broken = FakeEnv(seed=7)
        broken.meta = lambda: (_ for _ in ()).throw(RuntimeError("no store"))
        result = run_episode(broken, RecordingPolicy(pick_greedy), EpisodeProtocol(), seed=7, level=1,
                             observe_meta=True, clock=StepClock())
        self.assertEqual(result["level_meta"], {"error": "RuntimeError: no store"})
        self.assertEqual(result["status"], "terminal")

        result = run_episode(FakeEnv(seed=7), RecordingPolicy(pick_greedy), EpisodeProtocol(), seed=7, level=1,
                             observe_meta=True, clock=StepClock())
        self.assertEqual(result["level_meta"], {"error": "env has no meta()"})
        self.assertNotIn("level_meta", run(FakeEnv(seed=7), RecordingPolicy(pick_greedy)))

    def test_final_summary_digests_the_last_state(self):
        result = run(FakeEnv(seed=7), RecordingPolicy(pick_greedy))
        summary = result["final_summary"]
        self.assertEqual(set(summary), {"wave", "wave_total", "lives", "gold", "level_won", "level_lost", "towers"})
        self.assertEqual(summary["lives"], result["outcome"]["lives"])
        self.assertTrue(summary["towers"])
        self.assertEqual(summary["towers"], sorted(summary["towers"]))


class ReplaySkipTests(unittest.TestCase):
    def test_void_after_native_side_effect_is_skipped_not_failed(self):
        made = []
        result = {"status": "void", "void_reason": "native_action_error: RuntimeError: receipt failed",
                  "plan": [], "trace_sha256": "x", "outcome": None}
        verdict = replay_verify(lambda: made.append(1), result)
        self.assertIsNone(verdict["replay_verified"])
        self.assertEqual(verdict["replay_skipped"], "void_after_native_side_effect")
        self.assertEqual(made, [])
        self.assertTrue(replay_unsupported(result))
        self.assertFalse(replay_unsupported({**result, "void_reason": "policy_error: ValueError: x"}))


class ReplayTests(unittest.TestCase):
    def test_full_episode_replays_from_plan(self):
        for pick in (pick_greedy, pick_send_wave):
            with self.subTest(pick=pick.__name__):
                result = run(FakeEnv(seed=1003), RecordingPolicy(pick))
                self.assertEqual(result["status"], "terminal")
                envs = []

                def make_env():
                    envs.append(FakeEnv(seed=1003))
                    return envs[-1]

                verdict = replay_verify(make_env, json.loads(json.dumps(result)))
                self.assertEqual(verdict, {"replay_verified": True, "outcome_match": True,
                                           "replay_trace_sha256": result["trace_sha256"], "replay_error": None})
                self.assertEqual(envs[0].closes, 1)

    def test_stalled_episode_replays_with_no_outcome(self):
        result = run(FakeEnv(holders=1, gold=0, income_every=0), RecordingPolicy(pick_wait),
                     max_interval_ticks=300, max_ticks=1500)
        verdict = replay_verify(lambda: FakeEnv(holders=1, gold=0, income_every=0), result)
        self.assertTrue(verdict["replay_verified"])
        self.assertTrue(verdict["outcome_match"])

    def test_tampered_plan_or_env_fails_verification(self):
        result = run(FakeEnv(), RecordingPolicy(pick_greedy))
        waits = [i for i, step in enumerate(result["plan"]) if step["action"]["action"] == "wait"]
        builds = [i for i, step in enumerate(result["plan"]) if step["action"]["action"] == "build_tower"]
        longer = copy.deepcopy(result)
        longer["plan"][waits[0]]["action"]["ticks"] = 61
        moved = copy.deepcopy(result)
        moved["plan"][builds[0]]["action"]["holder_id"] = 3
        truncated = copy.deepcopy(result)
        truncated["plan"] = truncated["plan"][:-1]
        illegal = copy.deepcopy(result)
        illegal["plan"].insert(0, {"action": {"action": "build_tower", "holder_id": 99, "tower_type": "archer"}})
        for name, tampered in (("longer", longer), ("moved", moved), ("truncated", truncated), ("illegal", illegal)):
            with self.subTest(name=name):
                env = FakeEnv()
                verdict = replay_verify(lambda: env, tampered)
                self.assertFalse(verdict["replay_verified"])
                self.assertEqual(env.closes, 1)
        self.assertIn("legal menu", replay_verify(FakeEnv, illegal)["replay_error"])
        self.assertFalse(replay_verify(lambda: FakeEnv(seed=1004), result)["replay_verified"])


class ScriptedPolicyIntegrationTests(unittest.TestCase):
    """The shipped scripted policies satisfy the Policy protocol end to end."""

    SPECS = ("send_wave_only", "pressure_greedy", "random:0", "single:archer", "single:barrack",
             "single:mage", "single:engineer")

    def test_every_scripted_policy_plays_a_replayable_episode(self):
        from alpharush_rl.scripted_policies import make_policy
        for spec in self.SPECS:
            with self.subTest(spec=spec):
                results = []
                for _ in range(2):
                    env = FakeEnv(seed=1001)
                    results.append(run_episode(env, make_policy(spec), EpisodeProtocol(), seed=1001, level=1,
                                               episode_id="ep-scripted", clock=StepClock()))
                    self.assertEqual(env.closes, 0)
                result = json.loads(json.dumps(results[0], allow_nan=False))
                self.assertEqual(("terminal", None), (result["status"], result["void_reason"]))
                self.assertEqual(spec, result["policy"])
                self.assertGreater(result["n_decisions"], 0)
                for decision in result["decisions"]:
                    self.assertEqual(f"scripted:{spec}", decision["provenance"])
                    self.assertIsNone(decision["distribution"])
                    self.assertIn(decision["label"], decision["labels"])
                # Deterministic: the same seed gives the same plan and native trace.
                self.assertEqual(results[0]["plan"], results[1]["plan"])
                self.assertEqual(results[0]["trace_sha256"], results[1]["trace_sha256"])
                self.assertTrue(replay_verify(lambda: FakeEnv(seed=1001), result)["replay_verified"])


class FakeWorker:
    """host.lua RPC semantics over the toy level: actions apply at once, ticks only via step.

    Replaces engine.Worker inside the real NativeEnv; no process or socket.
    """

    def __init__(self, seed=1001, port=9879, identity=None, level=1, difficulty=2, rng_mode=""):
        # No passive income, so NativeEnv's exact wave-0 build price receipt holds.
        self.sim = FakeEnv(seed=seed, level=level, income_every=0)
        self.identity, self.difficulty = identity, difficulty
        self.starts = self.closes = 0

    def start(self):
        self.starts += 1
        self.sim.reset()

    def close(self):
        self.closes += 1

    def _wire(self):
        return {"type": "game_state", "timestamp": 0.5, "save_directory": "fake", **copy.deepcopy(self.sim.state)}

    def rpc(self, action, **arguments):
        sim = self.sim
        if action == "state":
            return self._wire()
        if action == "step":
            before = sim.state["tick"]
            sim.advance(arguments["ticks"], record_plan=False)
            return {"tick_before": before, "tick_after": sim.state["tick"],
                    "terminated": sim.state["game_over"], "state": self._wire()}
        state = copy.deepcopy(sim.state)
        if action == "build_tower":
            holder = next(h for h in state["holders"] if h["id"] == arguments["holder_id"])
            state["holders"].remove(holder)
            state["gold"] -= COSTS[arguments["tower_type"]]
            state["towers"].append({"id": 100 + len(state["towers"]), "holder_id": holder["mesh_id"],
                                    "template": f"tower_{arguments['tower_type']}_1",
                                    "type": arguments["tower_type"], "is_special": False})
        elif action == "send_wave":
            sim._spawn(state)
        else:
            raise RuntimeError("Action not enabled in verified experiment scope")
        sim.state = sim._refresh(state)
        return {"type": "ok"}


class NativeEnvIntegrationTests(unittest.TestCase):
    """run_episode/replay_verify over the real NativeEnv class with a fake RPC worker."""

    def test_native_env_episode_and_cold_replay(self):
        from unittest import mock
        from alpharush_rl import env as env_module
        from alpharush_rl.scripted_policies import make_policy
        with mock.patch.object(env_module, "Worker", FakeWorker):
            for spec in ("send_wave_only", "pressure_greedy", "random:3"):
                with self.subTest(spec=spec):
                    native = env_module.NativeEnv(seed=1001, level=1, difficulty=2, identity="unit_episode")
                    try:
                        result = run_episode(native, make_policy(spec), EpisodeProtocol(), seed=1001, level=1,
                                             episode_id="ep-native", clock=StepClock())
                    finally:
                        native.close()
                    self.assertEqual((1, 1), (native.worker.starts, native.worker.closes - 1))
                    self.assertEqual(("terminal", None), (result["status"], result["void_reason"]))
                    self.assertEqual({"kind", "state_sha256", "tick", "seed", "level"}, set(native.trace[0]))
                    waits = [step["action"] for step in result["plan"] if step["action"]["action"] == "wait"]
                    self.assertTrue(waits)
                    self.assertTrue(all(wait == {"action": "wait", "ticks": 60} for wait in waits))
                    result = json.loads(json.dumps(result, allow_nan=False))
                    replays = []

                    def make_env():
                        replays.append(env_module.NativeEnv(seed=1001, level=1, identity="unit_replay"))
                        return replays[-1]

                    verdict = replay_verify(make_env, result)
                    self.assertTrue(verdict["replay_verified"], verdict)
                    self.assertEqual(1, replays[0].worker.closes - 1)
                    tampered = copy.deepcopy(result)
                    tampered["plan"][0]["action"] = {"action": "wait", "ticks": 61}
                    self.assertFalse(replay_verify(make_env, tampered)["replay_verified"])


class ProtocolTests(unittest.TestCase):
    def test_protocol_defaults_and_validation(self):
        protocol = EpisodeProtocol()
        self.assertEqual((protocol.name, protocol.wait_ticks, protocol.max_interval_ticks, protocol.max_ticks,
                          protocol.max_decisions, protocol.first_wave_deadline_ticks),
                         ("kr1-episode-v1", 60, 600, 60000, 400, None))
        for kwargs in ({"wait_ticks": 0}, {"max_ticks": -1}, {"max_decisions": True},
                       {"max_interval_ticks": 1.5}, {"first_wave_deadline_ticks": 0}, {"name": ""}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    EpisodeProtocol(**kwargs)
        with self.assertRaises(TypeError):
            run_episode(FakeEnv(), RecordingPolicy(pick_wait), {"wait_ticks": 60}, seed=1, level=1)


if __name__ == "__main__":
    unittest.main()
