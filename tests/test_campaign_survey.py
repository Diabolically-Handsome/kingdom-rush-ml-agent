"""campaign-survey tool tests: temporary workspaces and a toy env only.

No game, process, socket or GPU is started. ``--run`` is exercised only on its
refusal path while the native-survey job is disabled; ``worker_loop`` runs on
an injected toy env factory.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
import unittest
from unittest import mock

from alpharush_rl import phase
from alpharush_rl.episode import EpisodeProtocol
from alpharush_rl.journal import Journal, sha256_data
from alpharush_rl.menus import build_menu
from alpharush_rl.ops import GateRefused, sha256_file

WORKSPACE = Path(__file__).resolve().parents[1]
TOOL_PATH = WORKSPACE / "tools/campaign-survey.py"
SPEC = importlib.util.spec_from_file_location("campaign_survey_under_test", TOOL_PATH)
SURVEY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SURVEY)
REAL_ENGINE_READY = SURVEY.engine_ready
REAL_PHASE = WORKSPACE / "configs/phases/campaign-v1.json"
REAL_POOLS = WORKSPACE / "configs/pools-campaign-v1.json"
CONFIG = "configs/phases/campaign-v1.json"
STATE = "runtime/rl/campaign-v1"
DISABLED = "native-survey is not enabled: run list must be confirmed by the user before enabling"
NO_RUN_LIST = "Survey plan: native-survey declares no run_list; the user must confirm one before enabling"
CAP_KEYS = ("max_wall_seconds", "total_wall_seconds", "max_jobs", "max_games", "gpu", "optimizer_steps", "serial")


def disabled_baseline(cfg):
    """The shipped phase config as it was before the stage-1 approval: caps only, disabled."""
    cfg = copy.deepcopy(cfg)
    survey = cfg["jobs"]["native-survey"]
    cfg["jobs"]["native-survey"] = {"enabled": False, "pending": "run list must be confirmed by the user before enabling",
                                    **{k: survey[k] for k in CAP_KEYS}}
    return cfg


FAKE_FILES = {
    "alpharush_rl/fake_core.py": "VALUE = 1\n",
    "alpharush_rl/assets/host.lua": "return {}\n",
    "tools/fake-tool.py": "print('tool')\n",
    "docs/NOTES.md": "notes\n",
    "REPORT.md": "report\n",
    "tests/test_fake.py": "pass\n",
}

COSTS = {"archer": 70, "barrack": 70, "mage": 100, "engineer": 125}
POWER = {"archer": 1, "barrack": 1, "mage": 2, "engineer": 2}


class FakeEnv:
    """Minimal copy of tests/test_episode.py's toy level (NativeEnv semantics).

    +1 gold every 10 ticks; a build costs its catalog price and takes 180
    ticks, send_wave 2; enemies walk one step per tick and cost a life at 400;
    towers hit the leading enemy every tick; waves after the first auto-start.
    """

    def __init__(self, seed=1001, level=1, *, on_reset=None):
        self.seed, self.level, self.on_reset = seed, level, on_reset
        self.state, self.trace, self.plan = None, [], []
        self.resets = self.closes = 0

    def reset(self):
        self.resets += 1
        if self.on_reset:
            self.on_reset()
        state = {"type": "game_state", "level_idx": self.level, "tick": 1, "gold": 150, "lives": 20,
                 "wave": 0, "wave_total": 3, "next_wave_ticks": None, "level_won": False, "level_lost": False,
                 "game_over": False, "spawned": 0,
                 "holders": [{"id": i + 1, "mesh_id": str(i + 1), "blocked": False, "path_score": 10 - i}
                             for i in range(3)],
                 "towers": [], "enemies": [], "heroes": [], "level_path_wave_counts": [[6], [8], [10]]}
        self.state = self._refresh(state)
        self.trace = [{"kind": "reset", "state_sha256": sha256_data(self.state), "tick": 1,
                       "seed": self.seed, "level": self.level}]
        self.plan = []
        return self.state

    def _refresh(self, s):
        if s["lives"] <= 0:
            s["lives"], s["level_lost"] = 0, True
        elif s["wave"] >= s["wave_total"] and not s["enemies"]:
            s["level_won"] = True
        s["game_over"] = s["level_won"] or s["level_lost"]
        s["wave_ready"] = not s["game_over"] and s["wave"] < s["wave_total"] and not s["enemies"]
        s["action_catalog"] = [{"action": "build_tower", "holder_id": h["id"], "tower_type": kind, "cost": cost,
                                "available": s["gold"] >= cost} for h in s["holders"] for kind, cost in COSTS.items()]
        if s["wave_ready"]:
            s["action_catalog"].append({"action": "send_wave", "available": True})
        return s

    def _spawn(self, s):
        s["wave"] += 1
        s["next_wave_ticks"] = None
        rng, offset = random.Random(f"{self.seed}:{s['wave']}"), 0
        for _ in range(4 + 2 * s["wave"]):
            s["spawned"] += 1
            s["enemies"].append({"id": 1000 + s["spawned"], "hp": 30 + 10 * s["wave"], "progress": -offset})
            offset += rng.randint(15, 25)

    def _tick(self, s):
        s["tick"] += 1
        if s["tick"] % 10 == 0:
            s["gold"] += 1
        for tower in s["towers"]:
            alive = [e for e in s["enemies"] if e["progress"] >= 0 and e["hp"] > 0]
            if alive:
                max(alive, key=lambda e: (e["progress"], -e["id"]))["hp"] -= POWER[tower["type"]]
        survivors = []
        for enemy in s["enemies"]:
            if enemy["hp"] <= 0:
                s["gold"] += 5
                continue
            enemy["progress"] += 1
            if enemy["progress"] >= 400:
                s["lives"] -= 1
            else:
                survivors.append(enemy)
        s["enemies"] = survivors
        if 1 <= s["wave"] < s["wave_total"] and not s["enemies"]:
            s["next_wave_ticks"] = 300 if s["next_wave_ticks"] is None else s["next_wave_ticks"] - 1
            if s["next_wave_ticks"] <= 0:
                self._spawn(s)
        self._refresh(s)

    def advance(self, ticks, *, record_plan=True):
        state, before = copy.deepcopy(self.state), self.state["tick"]
        for _ in range(ticks):
            if state["game_over"]:
                break
            self._tick(state)
        self.state = state
        self.trace.append({"kind": "step", "ticks": ticks, "advanced_ticks": state["tick"] - before,
                           "tick": state["tick"], "state_sha256": sha256_data(state)})
        if record_plan:
            self.plan.append({"ticks": ticks})
        return state

    def act(self, action):
        action, before = dict(action), self.state
        if action["action"] == "wait":
            self.advance(action.get("ticks", 30), record_plan=False)
        else:
            if not any(item["action"] == action for item in build_menu(before)):
                raise ValueError("Action is not in the native legal menu")
            state = copy.deepcopy(before)
            if action["action"] == "build_tower":
                holder = next(h for h in state["holders"] if h["id"] == action["holder_id"])
                state["holders"].remove(holder)
                state["gold"] -= COSTS[action["tower_type"]]
                state["towers"].append({"id": 100 + len(state["towers"]), "holder_id": holder["mesh_id"],
                                        "template": f"tower_{action['tower_type']}_1",
                                        "type": action["tower_type"], "is_special": False, "level": 1})
                ticks = 180
            else:
                self._spawn(state)
                ticks = 2
            self.state = self._refresh(state)
            self.advance(ticks, record_plan=False)
        receipt = {"accepted": True, "executed": True, "action": action, "tick_before": before["tick"],
                   "tick_after": self.state["tick"], "before_sha256": sha256_data(before),
                   "after_sha256": sha256_data(self.state)}
        self.trace.append({"kind": "action", "receipt": receipt})
        self.plan.append({"action": action})
        return receipt

    def terminal(self):
        s = self.state
        if not (s.get("level_won") or s.get("level_lost")):
            return None
        return {"source": "native", "terminal": True, "level_won": bool(s["level_won"]),
                "level_lost": bool(s["level_lost"]), "lives": s["lives"], "wave": s["wave"],
                "tick": s["tick"], "state_sha256": sha256_data(s)}

    def replay(self, plan):
        for command in plan:
            if "action" in command:
                self.act(command["action"])
            else:
                self.advance(command["ticks"])
        return self.state

    def close(self):
        self.closes += 1


class FakeFactory:
    """Injected env factory; records every env it hands out."""

    def __init__(self, replay_seed_offset=0, on_create=None):
        self.created, self.replay_seed_offset, self.on_create = [], replay_seed_offset, on_create

    def __call__(self, item, replay=False):
        seed = item["seed"] + (self.replay_seed_offset if replay else 0)
        env = FakeEnv(seed=seed, level=item["level"])
        self.created.append((item["index"], replay, env))
        if self.on_create:
            self.on_create(len(self.created), env)
        return env


class ScopeFactory(FakeFactory):
    """FakeFactory that also records the action_scope each env would be started with."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scopes = []

    def __call__(self, item, replay=False):
        self.scopes.append((item["index"], replay, item.get("action_scope")))
        return super().__call__(item, replay)


def item(seed, policy="send_wave_only", level=1, difficulty=2):
    return {"level": level, "seed": seed, "policy": policy, "difficulty": difficulty}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def call(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = SURVEY.main(list(argv))
    return code, json.loads(out.getvalue())


class TempWorkspace(unittest.TestCase):
    """Copies of the real campaign configs in a temporary workspace."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-survey-")
        self.root = Path(self.tmp.name).resolve()
        for relative, text in FAKE_FILES.items():
            _write(self.root / relative, text)
        self.config = self.root / CONFIG
        self.config.parent.mkdir(parents=True)
        # Tests start from the disabled baseline and enable what each one needs.
        self.cfg = disabled_baseline(json.loads(REAL_PHASE.read_text(encoding="utf-8")))
        _write(self.config, json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n")
        shutil.copyfile(REAL_POOLS, self.root / "configs/pools-campaign-v1.json")
        self.pools = json.loads(REAL_POOLS.read_text(encoding="utf-8"))
        self.state = self.root / STATE
        # The real engine manifest is outside these temporary workspaces.
        ready = mock.patch.object(SURVEY, "engine_ready", return_value=None)
        ready.start()
        self.addCleanup(ready.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def configure(self, **job):
        self.cfg["jobs"]["native-survey"].update(job)
        _write(self.config, json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n")

    def check(self):
        return call("--check-only", "--config", str(self.config))

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): (p.is_dir(), None if p.is_dir() else sha256_file(p))
                for p in self.root.rglob("*")}

    def context(self, max_games=10, run_id="native-survey-0123456789abcdef"):
        run_dir = self.state / "runs" / run_id
        return phase.PhaseRunContext(run_id, run_dir, self.state, time.monotonic() + 300, max_games,
                                     job_kind="native-survey", stop_dirs=(self.root / "runtime/rl",))


class CheckFreezeRunTests(TempWorkspace):
    def test_check_only_refuses_disabled_job_and_is_read_only(self):
        before = self.snapshot()
        code, result = self.check()
        self.assertEqual(2, code)
        self.assertFalse(result["ok"])
        self.assertIn(DISABLED, result["issues"])
        self.assertIn(NO_RUN_LIST, result["issues"])
        self.assertTrue(any("No verified pin manifest" in issue for issue in result["issues"]))
        self.assertEqual(("campaign-v1", "native-survey", True), (result["phase_id"], result["job_kind"],
                                                                 result["check_only"]))
        self.assertIsNone(result["survey"]["items"])
        self.assertEqual((code, result), call("--config", str(self.config)))  # default mode is check-only
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.state.exists())

    def test_freeze_then_pins_verify_and_only_pinned_edits_matter(self):
        code, manifest = call("--freeze", "--config", str(self.config))
        self.assertEqual(0, code)
        self.assertEqual(sha256_file(self.config), manifest["files"][CONFIG])
        self.assertIn("configs/pools-campaign-v1.json", manifest["files"])
        self.assertIn("alpharush_rl/fake_core.py", manifest["files"])
        self.assertFalse(any(p.startswith(("docs/", "tests/")) or p.endswith(".md") for p in manifest["files"]))
        expected = [DISABLED, NO_RUN_LIST,
                    "Survey plan: native-survey declares no replay_fraction (cold replays are native games too)"]
        code, result = self.check()
        self.assertEqual((2, expected), (code, result["issues"]))  # pins verified; only the gate refuses
        self.assertEqual(sha256_file(self.state / "pins.json"), result["pins_sha256"])

        for relative in ("docs/NOTES.md", "REPORT.md", "tests/test_fake.py", "docs/new.md"):
            _write(self.root / relative, "edited\n")
        self.assertEqual(expected, self.check()[1]["issues"])
        core = self.root / "alpharush_rl/fake_core.py"
        _write(core, "VALUE = 2\n")
        self.assertCountEqual(expected + ["Code/config SHA mismatch: alpharush_rl/fake_core.py"],
                              self.check()[1]["issues"])
        _write(core, FAKE_FILES["alpharush_rl/fake_core.py"])
        self.assertEqual(expected, self.check()[1]["issues"])

        (self.state / "job.lock").write_text("{}", encoding="utf-8")
        code, refused = call("--freeze", "--config", str(self.config))
        self.assertEqual((2, False), (code, refused["ok"]))
        self.assertIn("lock", refused["error"])

    def test_run_refuses_disabled_job_before_any_process_or_ledger(self):
        call("--freeze", "--config", str(self.config))
        before = self.snapshot()
        forbidden = mock.Mock(side_effect=AssertionError("must not be reached"))
        with mock.patch.object(SURVEY.subprocess, "Popen", forbidden), \
                mock.patch.object(SURVEY, "_probe_port", forbidden), \
                mock.patch("alpharush_rl.windows_job.OwnedProcessJob", forbidden), \
                mock.patch.object(SURVEY.phase, "job_context", forbidden):
            code, result = call("--run", "--config", str(self.config))
            with self.assertRaises(SURVEY.SurveyRefused) as refused:
                SURVEY.supervise(self.config)
        self.assertEqual(2, code)
        self.assertFalse(result["ok"])
        self.assertIn(DISABLED, result["issues"])
        self.assertEqual(result["issues"], refused.exception.result["issues"])
        forbidden.assert_not_called()
        self.assertEqual(before, self.snapshot())
        for name in ("ledger.jsonl", "runs", "job.lock"):
            self.assertFalse((self.state / name).exists(), name)

    def test_enabled_plan_is_fully_checked_before_any_job(self):
        self.configure(enabled=True, replay_fraction=1.0, run_list=[item(1001), item(5001, "pressure_greedy")])
        call("--freeze", "--config", str(self.config))
        code, result = self.check()
        self.assertEqual((0, True, []), (code, result["ok"], result["issues"]))
        self.assertEqual((2, 4, 1.0, 450), (result["survey"]["items"], result["survey"]["planned_games"],
                                            result["survey"]["replay_fraction"], result["survey"]["max_games"]))
        self.assertEqual({"train": 1, "evaluation": 1}, result["survey"]["pools"])
        self.assertEqual(EpisodeProtocol().max_ticks, result["survey"]["episode_protocol"]["max_ticks"])

        cases = [
            ({"run_list": [item(1001), item(2001)]}, "seed 2001 is retired"),
            ({"replay_fraction": 1.5}, "replay_fraction must be a number in [0, 1]"),
            ({"max_games": 3}, "run_list needs 4 native games"),
            ({"episode_protocol": {"wait_ticks": 0}}, "episode_protocol refused"),
            ({"episode_protocol": {"unknown": 1}}, "episode_protocol refused"),
        ]
        for change, message in cases:
            with self.subTest(change=change):
                original = copy.deepcopy(self.cfg)
                self.configure(**change)
                call("--freeze", "--config", str(self.config))
                code, result = self.check()
                self.assertEqual(2, code)
                self.assertTrue(any(i.startswith("Survey plan: ") and message in i for i in result["issues"]),
                                result["issues"])
                self.cfg = original
                self.configure()
        del self.cfg["jobs"]["native-survey"]["replay_fraction"]
        self.configure()
        call("--freeze", "--config", str(self.config))
        self.assertTrue(any("declares no replay_fraction" in i for i in self.check()[1]["issues"]))
        self.assertFalse((self.state / "ledger.jsonl").exists())

    def test_real_campaign_run_list_is_the_approved_plan(self):
        plan = SURVEY.plan_survey(REAL_PHASE)  # reads only the shipped config, pools and historic evidence
        self.assertEqual([], plan["issues"])
        self.assertEqual(140, plan["items"])
        self.assertLessEqual(plan["planned_games"], plan["max_games"])
        self.assertEqual(450, plan["max_games"])
        self.assertEqual({"train": 140, "evaluation": 0}, plan["pools"])
        self.assertEqual(0.1, plan["replay_fraction"])
        self.assertEqual(150.0, plan["time_reserve_seconds"])
        run_list = json.loads(REAL_PHASE.read_text(encoding="utf-8"))["jobs"]["native-survey"]["run_list"]
        levels = [r.get("level") for r in run_list]
        # Historic replays first; non-main levels 13-26 last, after every main-campaign item.
        self.assertTrue(all("historic_replay" in r for r in run_list[:28]))
        self.assertEqual(list(range(13, 27)), levels[-14:])
        self.assertTrue(all(level is None or level <= 12 for level in levels[:-14]))


class EngineReadinessTests(TempWorkspace):
    def test_unprepared_engine_refuses_before_the_job_is_spent(self):
        self.configure(enabled=True, run_list=[item(1001)], replay_fraction=0.0)
        call("--freeze", "--config", str(self.config))
        before = self.snapshot()
        with mock.patch.object(SURVEY, "engine_ready", return_value="Engine not prepared: legacy manifest"):
            code, output = self.check()
            self.assertEqual(code, 2)
            self.assertIn("Engine not prepared: legacy manifest", output["issues"])
            with mock.patch.object(SURVEY.phase, "job_context", side_effect=AssertionError("job opened")), \
                    mock.patch.object(SURVEY, "_probe_port", side_effect=AssertionError("port probed")):
                with self.assertRaises(SURVEY.SurveyRefused):
                    SURVEY.supervise(self.config)
        self.assertEqual(self.snapshot(), before)

    def test_engine_ready_is_stat_only(self):
        from alpharush_rl import engine
        with mock.patch.object(engine, "prepare", side_effect=RuntimeError("legacy")) as prepare:
            self.assertEqual(REAL_ENGINE_READY(), "Engine not prepared: legacy")
            prepare.assert_called_once_with()
        with mock.patch.object(engine, "prepare", return_value=Path("exe")):
            self.assertIsNone(REAL_ENGINE_READY())


class HistoricReplayTests(TempWorkspace):
    """Earlier verified branches are replayed cold first; any difference stops the survey."""

    def write_branches(self, source, seed, labels, tamper=None):
        branches = []
        for label in labels:
            env = FakeEnv(seed=seed)
            env.reset()
            env.act({"action": "wait", "ticks": 30})
            for _ in range(200):  # same frozen continuation as validation.finish
                if env.terminal():
                    break
                if env.state.get("wave_ready"):
                    env.act({"action": "send_wave"})
                env.advance(600)
            self.assertTrue(env.terminal())
            branches.append({"label": label, "plan": copy.deepcopy(env.plan), "trace": copy.deepcopy(env.trace),
                             "outcome": {**env.terminal(), "level": 1, "native_raw": env.terminal()}})
        if tamper:
            tamper(branches)
        _write(self.root / source, json.dumps({"scope": {}, "branches": branches}))
        return branches

    def historic(self, source, label):
        return {"historic_replay": source, "branch": label}

    def run_loop(self, run_list, ctx, factory, **kwargs):
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        summary = SURVEY.worker_loop(run_list, factory, ctx, out, pools=self.pools, root=self.root, **kwargs)
        return summary, out

    def test_check_accepts_allowlisted_branches_and_refuses_others(self):
        source = SURVEY.HISTORIC_SOURCES[1]
        self.write_branches(source, 1002, ["A", "B"])
        items, issues = SURVEY.check_run_list([self.historic(source, "A"), self.historic(source, "B"),
                                               item(1001)], self.pools, self.root)
        self.assertEqual(issues, [])
        self.assertEqual([i.get("kind") for i in items], ["historic_replay", "historic_replay", None])
        self.assertEqual(items[0]["seed"], 1002)
        self.assertEqual(items[0]["pool"], "train")
        self.assertEqual(SURVEY.planned_games(items, 1.0), 2 + 2)  # historic items are never resampled
        bad = [self.historic("runtime/rl/other.json", "A"), self.historic(source, "Z"), self.historic(source, "A"),
               self.historic(source, "A")]
        _, issues = SURVEY.check_run_list(bad, self.pools, self.root)
        text = " | ".join(issues)
        self.assertIn("is not one of", text)
        self.assertIn("0 branches labelled 'Z'", text)
        self.assertIn("repeats an earlier item", text)
        _, issues = SURVEY.check_run_list([self.historic(source, "A")], self.pools, None)
        self.assertIn("need the workspace root", issues[0])
        retired = SURVEY.HISTORIC_SOURCES[0]
        self.write_branches(retired, 2001, ["A"])
        _, issues = SURVEY.check_run_list([self.historic(retired, "A")], self.pools, self.root)
        self.assertIn("retired", issues[0])

    def test_matching_replays_continue_and_episodes_record_meta(self):
        source = SURVEY.HISTORIC_SOURCES[1]
        self.write_branches(source, 1002, ["A", "B"])
        ctx = self.context(max_games=10)
        factory = FakeFactory()
        summary, out = self.run_loop([self.historic(source, "A"), self.historic(source, "B"), item(1001)],
                                     ctx, factory)
        self.assertEqual(summary["historic_replays"], {"replayed": 2, "verified": 2, "mismatches": [],
                                                       "engine_errors": []})
        self.assertIsNone(summary["stopped_reason"])
        rows = out.entries()
        self.assertEqual([r["kind"] for r in rows], ["historic_replay", "historic_replay", "episode"])
        self.assertTrue(all(r["payload"]["verified"] for r in rows[:2]))
        self.assertEqual(rows[0]["payload"]["source_sha256"], sha256_file(self.root / source))
        self.assertIn("level_meta", rows[2]["payload"]["result"])  # FakeEnv has no meta(): recorded as error
        self.assertEqual(ctx.games_played, 3)
        self.assertTrue(all(env.closes == 1 for _, _, env in factory.created))

    def test_first_difference_stops_the_whole_survey(self):
        source = SURVEY.HISTORIC_SOURCES[1]

        def tamper(branches):
            branches[0]["trace"][2]["state_sha256"] = "0" * 64
        self.write_branches(source, 1002, ["A", "B"], tamper=tamper)
        ctx = self.context(max_games=10)
        factory = FakeFactory()
        summary, out = self.run_loop([self.historic(source, "A"), self.historic(source, "B"), item(1001)],
                                     ctx, factory)
        self.assertEqual(summary["stopped_reason"], "determinism_mismatch")
        self.assertEqual(summary["not_run"], [1, 2])
        self.assertEqual(summary["historic_replays"]["mismatches"], [0])
        row = out.entries()[0]["payload"]
        self.assertFalse(row["verified"])
        self.assertEqual(row["first_trace_mismatch"], 2)
        self.assertEqual(row["expected_mismatch_entry"]["state_sha256"], "0" * 64)
        self.assertEqual(len(factory.created), 1)
        self.assertIn("determinism_mismatch", [r["kind"] for r in Journal(ctx.output_dir / SURVEY.EVENTS).entries()])

    def test_outcome_difference_alone_also_stops(self):
        source = SURVEY.HISTORIC_SOURCES[1]

        def tamper(branches):
            branches[0]["outcome"]["native_raw"]["lives"] += 1
        self.write_branches(source, 1002, ["A"], tamper=tamper)
        ctx = self.context(max_games=10)
        summary, out = self.run_loop([self.historic(source, "A"), item(1001)], ctx, FakeFactory())
        self.assertEqual(summary["stopped_reason"], "determinism_mismatch")
        row = out.entries()[0]["payload"]
        self.assertTrue(row["trace_match"])
        self.assertFalse(row["outcome_match"])

    def test_engine_start_failure_is_retried_once_and_never_a_mismatch(self):
        source = SURVEY.HISTORIC_SOURCES[1]
        self.write_branches(source, 1002, ["A", "B"])

        def fail_first(n, env):
            if n == 1:
                def boom():
                    raise RuntimeError("Isolated RPC start timed out")
                env.on_reset = boom
        ctx = self.context(max_games=10)
        summary, out = self.run_loop([self.historic(source, "A"), self.historic(source, "B")], ctx,
                                     FakeFactory(on_create=fail_first))
        self.assertIsNone(summary["stopped_reason"])
        self.assertEqual(summary["historic_replays"]["verified"], 2)
        kinds = [r["kind"] for r in out.entries()]
        self.assertEqual(kinds, ["historic_engine_error", "historic_replay", "historic_replay"])
        self.assertEqual(out.entries()[1]["payload"]["attempt"], 2)
        self.assertEqual(ctx.games_played, 3)  # the failed start is a claimed native game too

        def fail_always(n, env):
            def boom():
                raise RuntimeError("Isolated game exited")
            env.on_reset = boom
        ctx = self.context(max_games=10, run_id="native-survey-fedcba9876543210")
        summary, out = self.run_loop([self.historic(source, "A"), self.historic(source, "B")], ctx,
                                     FakeFactory(on_create=fail_always))
        self.assertEqual(summary["stopped_reason"], "historic_engine_error")
        self.assertEqual(summary["historic_replays"]["engine_errors"], [0])
        self.assertEqual(summary["historic_replays"]["mismatches"], [])
        self.assertEqual(summary["not_run"], [1])
        self.assertEqual([r["kind"] for r in out.entries()], ["historic_engine_error", "historic_engine_error"])

    def test_failure_after_the_level_loaded_is_a_mismatch(self):
        source = SURVEY.HISTORIC_SOURCES[1]

        def tamper(branches):
            branches[0]["plan"].insert(1, {"action": {"action": "build_tower", "holder_id": 99, "tower_type": "archer"}})
        self.write_branches(source, 1002, ["A"], tamper=tamper)
        summary, out = self.run_loop([self.historic(source, "A")], self.context(max_games=10), FakeFactory())
        self.assertEqual(summary["stopped_reason"], "determinism_mismatch")
        row = out.entries()[0]["payload"]
        self.assertEqual(row["stopped_at"], "replay")
        self.assertIsNotNone(row["error"])

    def test_historic_replays_keep_v1_in_a_v2_job(self):
        source = SURVEY.HISTORIC_SOURCES[1]
        self.write_branches(source, 1002, ["A"])
        items, issues = SURVEY.check_run_list([self.historic(source, "A"), item(1001)], self.pools, self.root,
                                              action_scope="v2")
        self.assertEqual(issues, [])
        self.assertEqual([i["action_scope"] for i in items], ["v1", "v2"])
        # The scope never enters an item's core, so dedupe keys and replay draws are unchanged.
        self.assertEqual([SURVEY._core(i) for i in items], [{"historic_replay": source, "branch": "A"}, item(1001)])
        factory = ScopeFactory()
        summary, out = self.run_loop([self.historic(source, "A"), item(1001)], self.context(max_games=10), factory,
                                     action_scope="v2")
        # The v1 trace is replayed under v1; only the job's own episodes use the job scope.
        self.assertEqual(factory.scopes, [(0, True, "v1"), (1, False, "v2")])
        self.assertEqual(summary["historic_replays"]["verified"], 1)
        self.assertEqual(summary["action_scope"], "v2")
        self.assertEqual([r["kind"] for r in out.entries()], ["historic_replay", "episode"])

    def test_sampled_replay_respects_the_time_reserve(self):
        ctx = self.context(max_games=10)
        now = [ctx.deadline - 1000]
        factory = FakeFactory(on_create=lambda n, env: now.__setitem__(0, now[0] + 600))
        summary, out = self.run_loop([item(1001)], ctx, factory, replay_fraction=1.0,
                                     time_reserve_seconds=500, clock=lambda: now[0])
        self.assertEqual([r["kind"] for r in out.entries()], ["episode", "replay_skipped"])
        self.assertEqual(out.entries()[1]["payload"]["reason"], "time_reserve")
        self.assertEqual(summary["replays_sampled"], 0)
        self.assertEqual(ctx.games_played, 1)

    def test_engine_identity_reads_the_prepared_manifest(self):
        manifest = {"schema_version": 2, "recipe_sha256": "r" * 64, "exe": {"sha256": "e" * 64},
                    "runtime": {"runtime/rl-engine/alpha_rl_host.lua": {"sha256": "h" * 64}}, "inputs": {}}
        _write(self.root / "runtime/rl-engine/manifest.json", json.dumps(manifest))
        from alpharush_rl import engine
        with mock.patch.object(SURVEY, "ROOT", self.root), mock.patch.object(engine, "prepare") as prepare:
            identity = SURVEY.engine_identity()
            prepare.assert_called_once_with()
        self.assertEqual(identity, {"manifest_sha256": sha256_file(self.root / "runtime/rl-engine/manifest.json"),
                                    "recipe_sha256": "r" * 64, "exe_sha256": "e" * 64, "host_copy_sha256": "h" * 64})

    def test_time_reserve_stops_before_starting_a_game(self):
        ctx = self.context(max_games=10)
        now = [ctx.deadline - 1000]
        factory = FakeFactory(on_create=lambda n, env: now.__setitem__(0, now[0] + 600))
        summary, _ = self.run_loop([item(1001), item(1002), item(1003)], ctx, factory,
                                   time_reserve_seconds=500, clock=lambda: now[0])
        self.assertEqual(summary["stopped_reason"], "time_reserve")
        self.assertEqual(summary["not_run"], [1, 2])
        self.assertEqual(len(factory.created), 1)
        self.assertEqual(summary["time_reserve_seconds"], 500)
        with self.assertRaises(GateRefused):
            self.run_loop([item(1001)], self.context(max_games=10, run_id="native-survey-fedcba9876543210"),
                          FakeFactory(), time_reserve_seconds=-1)


class EngineeringErrorTests(TempWorkspace):
    """A game that fails to start voids only itself; repeated failures stop the survey."""

    def run_loop(self, run_list, ctx, factory, **kwargs):
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        summary = SURVEY.worker_loop(run_list, factory, ctx, out, pools=self.pools, **kwargs)
        return summary, out

    @staticmethod
    def failing_reset(levels):
        def on_create(n, env):
            if env.level in levels:
                def boom():
                    raise RuntimeError(f"Isolated game exited (level {env.level})")
                env.on_reset = boom
        return on_create

    def test_single_engineering_error_is_recorded_and_the_survey_continues(self):
        ctx = self.context(max_games=10)
        factory = FakeFactory(on_create=self.failing_reset({13}))
        summary, out = self.run_loop([item(1001, level=13), item(1001), item(1002)], ctx, factory)
        self.assertIsNone(summary["stopped_reason"])
        self.assertEqual(summary["episode_errors"], [0])
        kinds = [r["kind"] for r in out.entries()]
        self.assertEqual(kinds, ["episode_error", "episode", "episode"])
        self.assertIn("level 13", out.entries()[0]["payload"]["error"])
        self.assertEqual(ctx.games_played, 3)  # the failed start still counts as a native game
        self.assertTrue(all(env.closes == 1 for _, _, env in factory.created))

    def test_consecutive_errors_stop_the_survey(self):
        ctx = self.context(max_games=20)
        levels = set(range(13, 19))
        factory = FakeFactory(on_create=self.failing_reset(levels))
        run_list = [item(1001, level=level) for level in sorted(levels)] + [item(1001)]
        summary, out = self.run_loop(run_list, ctx, factory)
        self.assertEqual(summary["stopped_reason"], "consecutive_errors")
        self.assertEqual(summary["episode_errors"], [0, 1, 2, 3, 4])
        self.assertEqual(summary["not_run"], [5, 6])
        self.assertEqual(len(factory.created), SURVEY.MAX_CONSECUTIVE_ERRORS)

    def test_a_success_resets_the_consecutive_error_count(self):
        ctx = self.context(max_games=20)
        factory = FakeFactory(on_create=self.failing_reset({13, 14, 15, 16, 17, 18}))
        run_list = ([item(1001, level=level) for level in (13, 14, 15, 16)] + [item(1001)]
                    + [item(1001, level=level) for level in (17, 18)])
        summary, _ = self.run_loop(run_list, ctx, factory)
        self.assertIsNone(summary["stopped_reason"])
        self.assertEqual(summary["episode_errors"], [0, 1, 2, 3, 5, 6])

    def test_stop_still_aborts_everything(self):
        ctx = self.context(max_games=10)
        stop = self.root / "runtime/rl/STOP"
        factory = FakeFactory(on_create=lambda n, env: setattr(env, "on_reset", stop.touch))
        with self.assertRaises(GateRefused):
            self.run_loop([item(1001), item(1002)], ctx, factory)

    def test_replay_engine_error_is_not_a_determinism_verdict(self):
        ctx = self.context(max_games=10)

        def on_create(n, env):
            if n == 2:  # the sampled replay's process fails to start
                def boom():
                    raise RuntimeError("Isolated RPC start timed out")
                env.on_reset = boom
        summary, out = self.run_loop([item(1001)], ctx, FakeFactory(on_create=on_create), replay_fraction=1.0)
        replay = [r["payload"] for r in out.entries() if r["kind"] == "replay"][0]
        self.assertIsNone(replay["replay_verified"])
        self.assertEqual(replay["replay_skipped"], "replay_engine_error")
        self.assertEqual(summary["replay_failures"], [])
        self.assertEqual(summary["replay_engine_errors"], [0])


class AuditEnv(FakeEnv):
    """FakeEnv with native-style RNG audit counts; ``perturb`` mimics a sound draw that shifts gameplay."""

    def __init__(self, *args, perturb=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb, self.draws, self.sound = perturb, 0, 0

    def _tick(self, s):
        super()._tick(s)
        self.draws += 1
        if self.perturb and s["tick"] == 400:
            self.sound += 1
            s["gold"] += 1

    def rng_audit(self):
        counts = {"math@@all/systems.lua:1": self.draws}
        if self.sound:
            counts["math@@all/sound_db.lua:1"] = self.sound
        return {"mode": "audit", "counts": counts}


class DiagFactory:
    def __init__(self, perturb_attempts=()):
        self.created, self.perturb_attempts = [], set(perturb_attempts)

    def __call__(self, item, replay=False, attempt=None):
        env = AuditEnv(seed=item["seed"], level=item["level"], perturb=attempt in self.perturb_attempts)
        self.created.append((item["index"], replay, attempt, env))
        return env


class PlanReplayDiagnosisTests(TempWorkspace):
    RUN_ID = "native-survey-" + "ab" * 16

    def write_survey(self, seed=1001, index=5, **row):
        """``row`` adds payload fields (e.g. action_scope); without them it is a pre-scope row."""
        from alpharush_rl.episode import run_episode
        from alpharush_rl.scripted_policies import make_policy
        result = run_episode(FakeEnv(seed=seed), make_policy("pressure_greedy"), EpisodeProtocol(), seed=seed, level=1)
        self.assertEqual(result["status"], "terminal")
        journal = Journal(self.root / f"runtime/rl/campaign-v1/runs/{self.RUN_ID}/episodes.jsonl")
        journal.append("episode", {"index": index, "item": item(seed, "pressure_greedy"), "result": result, **row})
        return result

    def diag(self, mode="audit", repeats=2, episode=5, run=None):
        return {"plan_replay": run or self.RUN_ID, "episode": episode, "rng_mode": mode, "repeats": repeats}

    def run_loop(self, run_list, ctx, factory, **kwargs):
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        summary = SURVEY.worker_loop(run_list, factory, ctx, out, pools=self.pools, root=self.root, **kwargs)
        return summary, out

    def test_plan_replay_items_are_checked_and_counted(self):
        self.write_survey()
        items, issues = SURVEY.check_run_list([self.diag(), self.diag("audit+isolate_sound", 3)], self.pools, self.root)
        self.assertEqual(issues, [])
        self.assertEqual([i["kind"] for i in items], ["plan_replay", "plan_replay"])
        self.assertEqual((items[0]["seed"], items[0]["pool"], items[0]["level"]), (1001, "train", 1))
        self.assertEqual(SURVEY.planned_games(items, 1.0), 5)  # repeats only, never resampled
        bad = [self.diag("isolate"), self.diag(repeats=1), self.diag(repeats=5), self.diag(episode=9),
               self.diag(run="native-survey-xyz"), self.diag(), self.diag()]
        _, issues = SURVEY.check_run_list(bad, self.pools, self.root)
        text = " | ".join(issues)
        for fragment in ("rng_mode 'isolate'", "repeats 1", "repeats 5", "0 episodes with index 9",
                         "not a native-survey run id", "repeats an earlier item"):
            self.assertIn(fragment, text)

    def test_identical_repeats_report_no_divergence(self):
        original = self.write_survey()
        ctx = self.context(max_games=10)
        factory = DiagFactory()
        summary, out = self.run_loop([self.diag()], ctx, factory)
        row = out.entries()[0]["payload"]
        self.assertTrue(row["all_repeats_equal"])
        self.assertEqual(row["action_scope"], "v1")  # a pre-scope source row was played under v1
        self.assertNotIn("action_scope", row["item"])  # the item core is unchanged
        self.assertEqual([r["matches_original"] for r in row["runs"]], [True, True])
        self.assertEqual(row["divergences"], [{"attempt": 2, "first_divergent_step": None}])
        self.assertEqual(row["runs"][0]["final_rng"]["math@@all/systems.lua:1"], original["final_tick"] - 1)
        self.assertEqual(summary["plan_replay_diagnoses"], {"items": 1, "repeats_equal": [0], "repeats_differ": []})
        self.assertEqual(ctx.games_played, 2)
        self.assertEqual([(c[2], c[3].closes) for c in factory.created], [(1, 1), (2, 1)])

    def test_divergence_is_located_with_state_and_rng_differences(self):
        self.write_survey()
        ctx = self.context(max_games=10)
        summary, out = self.run_loop([self.diag()], ctx, DiagFactory(perturb_attempts={2}))
        row = out.entries()[0]["payload"]
        self.assertFalse(row["all_repeats_equal"])
        divergence = row["divergences"][0]
        self.assertIsNotNone(divergence["first_divergent_step"])
        self.assertGreaterEqual(divergence["tick"][0], 400)
        self.assertIn("/gold", [path for path, _, _ in divergence["state_diff"]])
        self.assertEqual(divergence["rng_diff_at_step"], {"math@@all/sound_db.lua:1": [0, 1]})
        self.assertEqual(divergence["rng_diff_before"], {})
        self.assertEqual([r["matches_original"] for r in row["runs"]], [True, False])
        self.assertEqual(summary["plan_replay_diagnoses"]["repeats_differ"], [0])

    def test_plan_replay_uses_the_source_episode_action_scope(self):
        self.write_survey(index=5)  # written before action scopes: played under v1
        self.write_survey(seed=1002, index=6, action_scope="v2")
        self.write_survey(seed=1003, index=7, action_scope="v3")
        for job_scope in ("v1", "v2"):
            with self.subTest(job_scope=job_scope):
                items, issues = SURVEY.check_run_list([self.diag(episode=5), self.diag(episode=6)], self.pools,
                                                      self.root, action_scope=job_scope)
                self.assertEqual(issues, [])
                self.assertEqual([i["action_scope"] for i in items], ["v1", "v2"])
        _, issues = SURVEY.check_run_list([self.diag(episode=7)], self.pools, self.root)
        self.assertIn("episode action_scope 'v3'", " | ".join(issues))

        class Recording(DiagFactory):
            def __init__(self):
                super().__init__()
                self.scopes = []

            def __call__(self, item, replay=False, attempt=None):
                self.scopes.append((attempt, item.get("action_scope")))
                return super().__call__(item, replay, attempt)
        factory = Recording()
        summary, out = self.run_loop([self.diag(episode=6)], self.context(max_games=10), factory)
        self.assertEqual(factory.scopes, [(1, "v2"), (2, "v2")])
        self.assertEqual(summary["action_scope"], "v1")  # the job scope; the replay kept its source's
        [row] = out.entries()
        self.assertEqual((row["kind"], row["payload"]["action_scope"]), ("plan_replay_diagnosis", "v2"))
        self.assertEqual(row["payload"]["item"], self.diag(episode=6))

    def test_changed_source_aborts_with_its_action_scope(self):
        self.write_survey(index=6, action_scope="v2")
        real, loads = SURVEY.load_survey_episode, []

        def load(root, run_id, index):
            # The source changes between the run-list check and the diagnosis.
            result, digest, scope = real(root, run_id, index)
            loads.append(digest)
            return result, digest if len(loads) == 1 else "0" * 64, scope
        ctx = self.context(max_games=10)
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        factory = DiagFactory()
        with mock.patch.object(SURVEY, "load_survey_episode", side_effect=load):
            with self.assertRaisesRegex(GateRefused, "plan_replay source changed"):
                SURVEY.worker_loop([self.diag(episode=6)], factory, ctx, out, pools=self.pools, root=self.root)
        self.assertEqual(factory.created, [])
        [row] = out.entries()
        self.assertEqual((row["kind"], row["payload"]["stage"], row["payload"]["action_scope"]),
                         ("aborted", "plan_replay", "v2"))

    def test_state_diff_paths(self):
        a = {"gold": 5, "towers": [{"id": 1, "hp": 3}], "x": [1, 2]}
        b = {"gold": 6, "towers": [{"id": 1, "hp": 4}], "x": [1, 2, 3], "y": 1}
        self.assertEqual(SURVEY.state_diff(a, b), [("/gold", "5", "6"), ("/towers[0]/hp", "3", "4"),
                                                   ("/x#len", 2, 3), ("/y", "null", "1")])
        self.assertEqual(len(SURVEY.state_diff({"k": list(range(100))}, {"k": list(range(1, 101))}, limit=5)), 5)

    def test_diagnose_job_kind_uses_its_own_caps(self):
        self.write_survey()
        self.cfg["jobs"]["native-diagnose"] = {"enabled": True, "max_wall_seconds": 600, "total_wall_seconds": 600,
                                              "max_jobs": 1, "max_games": 4, "gpu": False, "optimizer_steps": 0,
                                              "serial": True, "replay_fraction": 0.0, "run_list": [self.diag()]}
        self.configure()
        call("--freeze", "--config", str(self.config))
        code, result = call("--check-only", "--job", "native-diagnose", "--config", str(self.config))
        self.assertEqual((code, result["issues"]), (0, []))
        self.assertEqual(result["survey"]["planned_games"], 2)
        code, result = call("--check-only", "--config", str(self.config))  # native-survey is still disabled
        self.assertEqual(code, 2)
        self.cfg["jobs"]["native-diagnose"]["max_games"] = 1
        self.configure()
        call("--freeze", "--config", str(self.config))
        code, result = call("--check-only", "--job", "native-diagnose", "--config", str(self.config))
        self.assertIn("max_games is 1", " | ".join(result["issues"]))


class JobRngModeTests(TempWorkspace):
    def test_job_rng_mode_reaches_every_episode_and_replay(self):
        seen = []

        class Recording(FakeFactory):
            def __call__(self, item, replay=False):
                seen.append((item["index"], replay, item.get("rng_mode")))
                return super().__call__(item, replay)
        ctx = self.context(max_games=10)
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        summary = SURVEY.worker_loop([item(1001), item(1002)], Recording(), ctx, out, pools=self.pools,
                                     replay_fraction=1.0, rng_mode="isolate_sound+stable_pairs")
        self.assertEqual(seen, [(0, False, "isolate_sound+stable_pairs"), (0, True, "isolate_sound+stable_pairs"),
                                (1, False, "isolate_sound+stable_pairs"), (1, True, "isolate_sound+stable_pairs")])
        self.assertEqual(summary["rng_mode"], "isolate_sound+stable_pairs")
        self.assertEqual({r["payload"]["rng_mode"] for r in out.entries() if r["kind"] == "episode"},
                         {"isolate_sound+stable_pairs"})
        _, issues = SURVEY.check_run_list([item(1001)], self.pools, None, "stable")
        self.assertIn("rng_mode 'stable'", issues[0])

    def test_plan_reports_the_job_rng_mode(self):
        self.configure(enabled=True, run_list=[item(1001)], replay_fraction=0.0, rng_mode="isolate_sound+stable_pairs")
        self.assertEqual(SURVEY.plan_survey(self.config)["rng_mode"], "isolate_sound+stable_pairs")
        self.configure(rng_mode="bogus")
        self.assertIn("rng_mode 'bogus'", " | ".join(SURVEY.plan_survey(self.config)["issues"]))


class JobActionScopeTests(TempWorkspace):
    def run_loop(self, run_list, ctx, factory, **kwargs):
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        summary = SURVEY.worker_loop(run_list, factory, ctx, out, pools=self.pools, **kwargs)
        return summary, out

    def test_job_action_scope_reaches_every_episode_and_replay(self):
        factory = ScopeFactory()
        summary, out = self.run_loop([item(1001, "teacher_v2"), item(1002)], self.context(max_games=10), factory,
                                     replay_fraction=1.0, action_scope="v2")
        self.assertEqual(factory.scopes, [(0, False, "v2"), (0, True, "v2"), (1, False, "v2"), (1, True, "v2")])
        self.assertEqual(summary["action_scope"], "v2")
        episodes = [r["payload"] for r in out.entries() if r["kind"] == "episode"]
        self.assertEqual([e["action_scope"] for e in episodes], ["v2", "v2"])
        # The scope is recorded beside the item, never inside its core.
        self.assertEqual([e["item"] for e in episodes], [item(1001, "teacher_v2"), item(1002)])
        self.assertEqual(("teacher_v2", "terminal"), (episodes[0]["result"]["policy"], episodes[0]["result"]["status"]))
        self.assertTrue(all(d["provenance"] == "scripted:teacher_v2" for d in episodes[0]["result"]["decisions"]))
        self.assertEqual((2, 2), (summary["replays_sampled"], summary["replays_verified"]))

    def test_default_action_scope_is_v1(self):
        factory = ScopeFactory()
        summary, out = self.run_loop([item(1001)], self.context(max_games=10), factory)
        self.assertEqual(factory.scopes, [(0, False, "v1")])
        self.assertEqual(summary["action_scope"], "v1")
        self.assertEqual(out.entries()[0]["payload"]["action_scope"], "v1")
        items, issues = SURVEY.check_run_list([item(1001)], self.pools)
        self.assertEqual(([], "v1"), (issues, items[0]["action_scope"]))

    def test_unknown_action_scope_refuses_before_any_game(self):
        for scope in ("v3", "", None, "V2", 2):
            with self.subTest(scope=scope):
                _, issues = SURVEY.check_run_list([item(1001)], self.pools, None, "", scope)
                self.assertEqual([f"action_scope {scope!r} is not one of ['v1', 'v2']"], issues)
                ctx, factory = self.context(), ScopeFactory()
                with self.assertRaisesRegex(GateRefused, "action_scope"):
                    self.run_loop([item(1001)], ctx, factory, action_scope=scope)
                self.assertEqual(([], 0), (factory.created, ctx.games_played))
                self.assertFalse((ctx.output_dir / SURVEY.EPISODES).exists())

    def test_plan_reports_the_job_action_scope(self):
        self.configure(enabled=True, run_list=[item(1001, "teacher_v2")], replay_fraction=0.0)
        plan = SURVEY.plan_survey(self.config)
        self.assertEqual(("v1", []), (plan["action_scope"], plan["issues"]))
        self.configure(action_scope="v2")
        plan = SURVEY.plan_survey(self.config)
        self.assertEqual(("v2", [], 1), (plan["action_scope"], plan["issues"], plan["items"]))
        self.configure(action_scope="bogus")
        self.assertIn("action_scope 'bogus'", " | ".join(SURVEY.plan_survey(self.config)["issues"]))


class WorkerLoopTests(TempWorkspace):
    def run_loop(self, run_list, ctx, factory, **kwargs):
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        summary = SURVEY.worker_loop(run_list, factory, ctx, out, pools=self.pools, **kwargs)
        return summary, out

    def events(self, ctx):
        return [row["kind"] for row in Journal(ctx.output_dir / SURVEY.EVENTS).entries()]

    def test_two_games_are_journaled_and_replayed(self):
        ctx, factory = self.context(max_games=4), FakeFactory()
        summary, out = self.run_loop([item(1001, "pressure_greedy"), item(5001)], ctx, factory, replay_fraction=1.0)
        rows = out.entries()
        self.assertEqual(4, out.verify()["entries"])
        self.assertEqual(["episode", "replay", "episode", "replay"], [row["kind"] for row in rows])
        episodes = [row["payload"] for row in rows if row["kind"] == "episode"]
        self.assertEqual([(0, 1001, "train"), (1, 5001, "evaluation")],
                         [(e["index"], e["item"]["seed"], e["pool"]) for e in episodes])
        for e in episodes:
            self.assertEqual("terminal", e["result"]["status"])
            self.assertEqual(f"{ctx.run_id}-{e['index']:04d}", e["result"]["episode_id"])
            self.assertTrue(e["replay_sampled"])
        self.assertEqual("pressure_greedy", episodes[0]["result"]["policy"])
        self.assertTrue(all(d["provenance"] == "scripted:pressure_greedy" for d in episodes[0]["result"]["decisions"]))
        for episode_row, replay_row in ((rows[0], rows[1]), (rows[2], rows[3])):
            self.assertEqual(episode_row["sha256"], replay_row["payload"]["episode_sha256"])
            self.assertIs(True, replay_row["payload"]["replay_verified"])
        self.assertEqual((4, 4, 0), (ctx.games_played, summary["games_played"], ctx.games_remaining))
        self.assertEqual({"terminal": 2}, summary["status_counts"])
        self.assertEqual((2, 2, [], None, []), (summary["replays_sampled"], summary["replays_verified"],
                                                summary["replay_failures"], summary["stopped_reason"],
                                                summary["not_run"]))
        self.assertEqual({"train": 1, "evaluation": 1}, summary["pools"])
        self.assertEqual(out.verify()["tip_sha256"], summary["episodes_journal"]["tip_sha256"])
        self.assertEqual(["survey_start"] + ["game_claimed"] * 4 + ["survey_end"], self.events(ctx))
        self.assertEqual([(0, False), (0, True), (1, False), (1, True)], [(i, r) for i, r, _ in factory.created])
        self.assertTrue(all(env.closes == 1 and env.resets == 1 for _, _, env in factory.created))
        json.dumps(summary, allow_nan=False)

        # A cold replay that diverges is reported, never silently accepted.
        ctx, factory = self.context(max_games=2, run_id="native-survey-fedcba9876543210"), FakeFactory(1)
        summary, _ = self.run_loop([item(1002, "single:archer")], ctx, factory, replay_fraction=1.0)
        self.assertEqual((1, 0, [0]), (summary["replays_sampled"], summary["replays_verified"],
                                       summary["replay_failures"]))

    def test_ineligible_seeds_and_bad_items_are_refused_before_any_game(self):
        good = item(1001)
        bad_lists = {
            "empty": [], "not a list": "1001", "retired": [good, item(2001)],
            "final campaign run": [item(6001)], "unlisted": [item(9999)], "bool seed": [item(True)],
            "string seed": [item("1001")], "float seed": [item(1001.0)], "difficulty": [item(1001, difficulty=3)],
            "string difficulty": [item(1001, difficulty="2")], "level zero": [item(1001, level=0)],
            "bool level": [item(1001, level=True)], "padded random seed": [item(1001, "random:007")],
            "unknown policy": [item(1001, "nope")], "unknown tower": [item(1001, "single:hero")],
            "policy type": [item(1001, 5)], "extra key": [dict(good, note="x")],
            "missing key": [{"level": 1, "seed": 1001, "policy": "send_wave_only"}],
            "duplicate": [good, dict(good)], "not an object": [["level", 1]],
        }
        for name, run_list in bad_lists.items():
            with self.subTest(name=name):
                ctx, factory = self.context(), FakeFactory()
                with self.assertRaises(GateRefused) as caught:
                    self.run_loop(run_list, ctx, factory)
                self.assertEqual([], factory.created)
                self.assertEqual(0, ctx.games_played)
                self.assertFalse((ctx.output_dir / SURVEY.EPISODES).exists())
                self.assertFalse((ctx.output_dir / SURVEY.EVENTS).exists())
        with self.assertRaisesRegex(GateRefused, "seed 2001 is retired"):
            self.run_loop([good, item(2001)], self.context(), FakeFactory())
        for fraction in (1.5, -0.1, True, float("nan"), None):
            with self.assertRaises(GateRefused):
                self.run_loop([good], self.context(), FakeFactory(), replay_fraction=fraction)
        with self.assertRaises(TypeError):
            self.run_loop([good], self.context(), FakeFactory(), protocol={"wait_ticks": 60})
        leaked = copy.deepcopy(self.pools)
        leaked["pools"]["train"]["seeds"].append(5001)
        with self.assertRaisesRegex(GateRefused, "overlap"):
            SURVEY.worker_loop([good], FakeFactory(), self.context(), Journal(self.root / "x.jsonl"), pools=leaked)

    def test_max_games_is_a_hard_stop(self):
        run_list = [item(seed) for seed in range(1001, 1006)]
        ctx, factory = self.context(max_games=3), FakeFactory()
        summary, out = self.run_loop(run_list, ctx, factory, replay_fraction=0.0)
        self.assertEqual((3, 3, "max_games", [3, 4]), (summary["episodes"], ctx.games_played,
                                                       summary["stopped_reason"], summary["not_run"]))
        self.assertEqual(["episode"] * 3, [row["kind"] for row in out.entries()])
        self.assertEqual(3, len(factory.created))
        self.assertIn("max_games_reached", self.events(ctx))
        with self.assertRaises(GateRefused):
            ctx.claim_game()
        # A sampled replay must fit too: the second item needs two games but only one is left.
        ctx, factory = self.context(max_games=3, run_id="native-survey-aaaaaaaaaaaaaaaa"), FakeFactory()
        summary, out = self.run_loop(run_list, ctx, factory, replay_fraction=1.0)
        self.assertEqual((1, 2, [1, 2, 3, 4]), (summary["episodes"], ctx.games_played, summary["not_run"]))
        self.assertEqual(["episode", "replay"], [row["kind"] for row in out.entries()])

    def test_stop_aborts_and_keeps_journaled_results(self):
        stop = self.root / "runtime/rl/STOP"

        def touch_stop_on_second_game(count, env):
            if count == 2:
                env.on_reset = stop.touch

        ctx, factory = self.context(), FakeFactory(on_create=touch_stop_on_second_game)
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        with self.assertRaisesRegex(GateRefused, "stop file"):
            SURVEY.worker_loop([item(1001), item(1002), item(1003)], factory, ctx, out, pools=self.pools)
        rows = out.entries()
        self.assertEqual(["episode", "aborted"], [row["kind"] for row in rows])
        self.assertEqual((1, "episode"), (rows[1]["payload"]["index"], rows[1]["payload"]["stage"]))
        self.assertTrue(rows[1]["payload"]["error"].startswith("GateRefused"))
        self.assertEqual(2, out.verify()["entries"])
        self.assertEqual([1, 1], [env.closes for _, _, env in factory.created])
        self.assertEqual(2, ctx.games_played)

    def test_survey_inside_a_real_phase_job_context(self):
        run_list = [item(1001, "pressure_greedy"), item(5002)]
        self.configure(enabled=True, replay_fraction=1.0, run_list=run_list)
        call("--freeze", "--config", str(self.config))
        check = SURVEY.check_only(self.config)
        self.assertTrue(check["ok"], check["issues"])
        job, pools = SURVEY.verify_frozen(self.config, check)
        self.assertEqual(run_list, job["run_list"])
        factory = FakeFactory()
        with phase.job_context(self.config, "native-survey") as ctx:
            summary = SURVEY.worker_loop(job["run_list"], factory, ctx, Journal(ctx.output_dir / SURVEY.EPISODES),
                                         pools=pools, replay_fraction=job["replay_fraction"],
                                         protocol=SURVEY.episode_protocol(job))
        receipt = json.loads((ctx.output_dir / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(("ok", 4, 450), (receipt["status"], receipt["games_played"], receipt["max_games"]))
        self.assertEqual((2, 2), (summary["episodes"], summary["replays_verified"]))
        self.assertEqual((4, "worker_events_journal"), SURVEY._games_claimed(ctx))
        ledger = Journal(self.state / "ledger.jsonl")
        self.assertEqual(["open", "close"], [row["kind"] for row in ledger.entries()])
        after = SURVEY.check_only(self.config)
        self.assertIn("No remaining native-survey jobs: used 1 of 1", after["issues"])
        # The worker's own recheck refuses edited pinned sources.
        _write(self.root / "tools/fake-tool.py", "print('edited')\n")
        with self.assertRaisesRegex(GateRefused, "Code/config SHA mismatch: tools/fake-tool.py"):
            SURVEY.verify_frozen(self.config, check)


class ToolBoundaryTests(TempWorkspace):
    def test_replay_sampling_is_deterministic(self):
        items = [{"index": i, **item(1001 + i % 10, level=1 + i // 10)} for i in range(400)]
        self.assertFalse(any(SURVEY.replay_sampled(x, 0.0) for x in items))
        self.assertTrue(all(SURVEY.replay_sampled(x, 1.0) for x in items))
        half = [SURVEY.replay_sampled(x, 0.5) for x in items]
        self.assertEqual(half, [SURVEY.replay_sampled(copy.deepcopy(x), 0.5) for x in items])
        self.assertTrue(150 < sum(half) < 250, sum(half))
        self.assertEqual(400 + sum(half), SURVEY.planned_games(items, 0.5))

    def test_native_env_factory_builds_without_starting(self):
        with mock.patch("alpharush_rl.env.NativeEnv") as native:
            make = SURVEY.native_env_factory("native-survey-0123456789abcdef")
            make({"index": 7, **item(5003, level=4)})
            make({"index": 7, **item(5003, level=4)}, replay=True)
            make({"index": 7, **item(5003, level=4), "rng_mode": "audit"}, replay=True, attempt=2)
            make({"index": 8, **item(5003, level=4), "action_scope": "v2"})
        self.assertEqual([mock.call(seed=5003, level=4, port=9879, difficulty=2, identity="survey_0123456789ab_0007",
                                    rng_mode="", action_scope="v1"),
                          mock.call(seed=5003, level=4, port=9879, difficulty=2,
                                    identity="survey_0123456789ab_0007_replay", rng_mode="", action_scope="v1"),
                          mock.call(seed=5003, level=4, port=9879, difficulty=2,
                                    identity="survey_0123456789ab_0007_replay_r2", rng_mode="audit",
                                    action_scope="v1"),
                          mock.call(seed=5003, level=4, port=9879, difficulty=2, identity="survey_0123456789ab_0008",
                                    rng_mode="", action_scope="v2")],
                         native.call_args_list)
        from alpharush_rl import engine
        self.assertEqual(SURVEY.RNG_MODES, engine.RNG_MODES)
        self.assertEqual(SURVEY.ACTION_SCOPES, engine.ACTION_SCOPES)

    def test_worker_refuses_missing_or_foreign_permit(self):
        run_dir = self.state / "runs" / "native-survey-0123456789abcdef"
        run_dir.mkdir(parents=True)
        with mock.patch("alpharush_rl.windows_job.confirm_current_job",
                        side_effect=AssertionError("must not be reached")) as confirm:
            with self.assertRaisesRegex(RuntimeError, "No owned Windows job permit"):
                SURVEY.worker_main(self.config, run_dir, permit_wait=0.1)
            _write(run_dir / SURVEY.PERMIT, json.dumps({"parent_pid": 0, "launcher_pid": 0,
                                                       "deadline": time.monotonic() + 60}))
            with self.assertRaisesRegex(RuntimeError, "Invalid live survey permit"):
                SURVEY.worker_main(self.config, run_dir, permit_wait=0.1)
            confirm.assert_not_called()
        self.assertEqual([SURVEY.PERMIT], [p.name for p in run_dir.iterdir()])

    def test_module_level_imports_never_reach_the_native_stack(self):
        tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
        modules = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        for native in ("alpharush_rl.env", "alpharush_rl.engine", "alpharush_rl.windows_job",
                       "alpharush_rl.validation"):
            self.assertNotIn(native, modules)
        self.assertEqual("native-survey", SURVEY.JOB_KIND)
        self.assertEqual(10, SURVEY.CLEANUP_RESERVE_SECONDS)


if __name__ == "__main__":
    unittest.main()
