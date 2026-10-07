"""Phase governance tests; every write happens inside a temporary workspace."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from alpharush_rl import phase
from alpharush_rl.journal import Journal, JournalError
from alpharush_rl.ops import GateRefused, sha256_file

WORKSPACE = Path(__file__).resolve().parents[1]
REAL_PHASE = WORKSPACE / "configs/phases/campaign-v1.json"
REAL_POOLS = WORKSPACE / "configs/pools-campaign-v1.json"
AUTHORIZATION = [
    "[用户原话已省略 / user's message omitted]",
    "[用户原话已省略 / user's message omitted]",
    "[用户原话已省略 / user's message omitted]",
    "[用户原话已省略 / user's message omitted]",
    "[用户原话已省略 / user's message omitted]",
]
CAP_KEYS = ("max_wall_seconds", "total_wall_seconds", "max_jobs", "max_games", "gpu", "optimizer_steps", "serial")
PENDING = "run list must be confirmed by the user before enabling"


def disabled_baseline(cfg):
    """The shipped phase config as it was before the stage-1 approval: caps only, disabled."""
    cfg = copy.deepcopy(cfg)
    survey = cfg["jobs"]["native-survey"]
    cfg["jobs"]["native-survey"] = {"enabled": False, "pending": PENDING, **{k: survey[k] for k in CAP_KEYS}}
    return cfg


FAKE_FILES = {
    "alpharush_rl/fake_core.py": "VALUE = 1\n",
    "alpharush_rl/sub/deep.py": "DEEP = 2\n",
    "alpharush_rl/__pycache__/stale.py": "# cache, never pinned\n",
    "alpharush_rl/assets/host.lua": "return {}\n",
    "tools/fake-tool.py": "print('tool')\n",
    "tools/fake-launch.ps1": "Write-Output 1\n",
    "docs/NOTES.md": "notes\n",
    "REPORT.md": "report\n",
    "tests/test_fake.py": "pass\n",
    "run.cmd": "echo run\n",
}
PINNED = ["alpharush_rl/assets/host.lua", "alpharush_rl/fake_core.py", "alpharush_rl/sub/deep.py",
          "configs/phases/test-phase.json", "configs/pools-test.json", "tools/fake-launch.ps1", "tools/fake-tool.py"]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _job(**overrides):
    job = {"enabled": True, "max_wall_seconds": 30, "total_wall_seconds": 60, "max_jobs": 2, "max_games": 2,
           "gpu": False, "optimizer_steps": 0, "serial": True}
    job.update(overrides)
    return job


class RealCampaignConfigTests(unittest.TestCase):
    """Read-only checks of the shipped configs; nothing is written in the real workspace."""

    def test_campaign_pools_are_disjoint_and_roles_resolve(self):
        pools = json.loads(REAL_POOLS.read_text(encoding="utf-8"))
        self.assertEqual([], phase.pool_problems(pools))
        self.assertEqual(("KR1-campaign-v1", "KR1-deployment-v1", 2, "campaign"),
                         (pools["pool_id"], pools["supersedes"], pools["difficulty"], pools["mode"]))
        self.assertEqual(list(range(1, 13)), pools["levels"]["main_campaign"])
        self.assertIn("main_campaign_levels=12", pools["levels"]["source"])
        self.assertEqual(list(range(1001, 1011)), pools["pools"]["train"]["seeds"])
        self.assertEqual(list(range(5001, 5021)), pools["pools"]["evaluation"]["seeds"])
        self.assertIs(True, pools["pools"]["evaluation"]["never_in_gradients"])
        self.assertEqual([6001, 6002, 6003, 6004, 6005], pools["pools"]["final_campaign_run"]["seeds"])
        self.assertEqual("one_frozen_evaluation", pools["pools"]["final_campaign_run"]["access"])
        self.assertEqual("never_use", pools["unlisted_seeds"])
        # A list, not a dict: 1001.0 == 1001 would collapse into one dict key.
        expected = [(1001, "train"), (1010, "train"), (5001, "evaluation"), (5020, "evaluation"),
                    (6001, "final_campaign_run"), (6005, "final_campaign_run"), (2001, "retired"),
                    (3004, "retired"), (4004, "retired"), (1011, "never_use"), (5021, "never_use"),
                    (9999, "never_use"), (True, "never_use"), (1001.0, "never_use"), ("1001", "never_use")]
        for seed, role in expected:
            self.assertEqual(role, phase.seed_role(pools, seed), repr(seed))
        # Every KR1-deployment-v1 non-training seed is retired, never reused.
        old = json.loads((WORKSPACE / "configs/pools.json").read_text(encoding="utf-8"))
        for role in ("validation", "heldout", "never_train"):
            for seed in old["pools"][role]["seeds"]:
                self.assertEqual("retired", phase.seed_role(pools, seed))

    def test_campaign_phase_config_is_approved_and_selects_pins_by_glob(self):
        cp, cfg, root = phase.load_phase(REAL_PHASE)
        self.assertEqual(WORKSPACE, root)
        self.assertEqual(("campaign-v1", "runtime/rl/campaign-v1", "runtime/rl/campaign-v1/pins.json",
                          "configs/pools-campaign-v1.json", 1),
                         (cfg["phase_id"], cfg["state_dir"], cfg["pins_manifest"], cfg["pools_path"],
                          cfg["max_concurrent_jobs"]))
        self.assertEqual(AUTHORIZATION, cfg["authorization"])
        self.assertEqual(["alpharush_rl/**/*.py", "alpharush_rl/assets/*.lua", "tools/*.py", "tools/*.ps1",
                          "tools/*.sh", "configs/**/*.json", "Lumi_Nox/games/kingdom_rush/bridge.lua",
                          "requirements-local.txt"], cfg["freeze"]["include"])
        self.assertEqual(["**/__pycache__/**"], cfg["freeze"]["exclude"])
        survey = cfg["jobs"]["native-survey"]
        # Enabled only after the user's stage-1 approval; the caps are unchanged.
        self.assertIs(True, survey["enabled"])
        self.assertEqual("[用户原话已省略 / user's message omitted]", survey["approval"])
        self.assertNotIn("pending", survey)
        self.assertEqual({"max_wall_seconds": 2700, "total_wall_seconds": 2700, "max_jobs": 1, "max_games": 450,
                          "gpu": False, "optimizer_steps": 0, "serial": True},
                         {k: survey[k] for k in CAP_KEYS})
        historic = [r for r in survey["run_list"] if "historic_replay" in r]
        episodes = [r for r in survey["run_list"] if "historic_replay" not in r]
        self.assertEqual((28, 112), (len(historic), len(episodes)))
        self.assertTrue(all("historic_replay" in r for r in survey["run_list"][:28]))  # determinism first
        self.assertEqual({1001, 1002}, {r["seed"] for r in episodes})
        self.assertEqual({2}, {r["difficulty"] for r in episodes})
        pins = phase.collect_pin_paths(cfg, root, config_path=cp)  # glob only; no writes
        self.assertEqual(sorted(set(pins)), pins)
        for required in ("configs/phases/campaign-v1.json", "configs/pools-campaign-v1.json",
                         "alpharush_rl/phase.py", "alpharush_rl/journal.py", "alpharush_rl/assets/host.lua"):
            self.assertIn(required, pins)
        for path in pins:
            self.assertFalse(path.startswith(("tests/", "docs/", "runtime/", ".venv/")), path)
            self.assertFalse(path.endswith((".md", ".cmd")), path)
            self.assertNotIn("__pycache__", path)

    def test_copied_disabled_baseline_refuses_only_because_survey_is_disabled(self):
        with tempfile.TemporaryDirectory(prefix="alpharush-phase-real-") as tmp:
            root = Path(tmp).resolve()
            for relative, text in FAKE_FILES.items():
                _write(root / relative, text)
            (root / "configs/phases").mkdir(parents=True)
            baseline = disabled_baseline(json.loads(REAL_PHASE.read_text(encoding="utf-8")))
            _write(root / "configs/phases/campaign-v1.json", json.dumps(baseline, ensure_ascii=False, indent=2))
            shutil.copyfile(REAL_POOLS, root / "configs/pools-campaign-v1.json")
            config = root / "configs/phases/campaign-v1.json"
            manifest = phase.freeze_phase(config)
            self.assertIn("configs/phases/campaign-v1.json", manifest["files"])
            self.assertIn("configs/pools-campaign-v1.json", manifest["files"])
            result = phase.preflight_phase(config, "native-survey")
            self.assertFalse(result["ok"])
            self.assertEqual(["native-survey is not enabled: run list must be confirmed by the user before enabling"],
                             result["issues"])
            with self.assertRaises(GateRefused):
                with phase.job_context(config, "native-survey"):
                    self.fail("disabled job must never run")
            state = root / "runtime/rl/campaign-v1"
            self.assertFalse((state / "ledger.jsonl").exists())
            self.assertFalse((state / "runs").exists())
            self.assertFalse((state / "job.lock").exists())


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-phase-")
        self.root = Path(self.tmp.name).resolve()
        for relative, text in FAKE_FILES.items():
            _write(self.root / relative, text)
        self.pools_path = self.root / "configs/pools-test.json"
        self.pools = json.loads(REAL_POOLS.read_text(encoding="utf-8"))
        self.write_pools()
        self.config = self.root / "configs/phases/test-phase.json"
        self.cfg = {
            "schema_version": 1, "phase_id": "test-phase", "workspace": "../..",
            "state_dir": "runtime/rl/test-phase", "pins_manifest": "runtime/rl/test-phase/pins.json",
            "pools_path": "configs/pools-test.json",
            "freeze": {"include": ["alpharush_rl/**/*.py", "alpharush_rl/assets/*.lua", "tools/*.py",
                                   "tools/*.ps1", "configs/**/*.json"],
                       "exclude": ["**/__pycache__/**"]},
            "max_concurrent_jobs": 1,
            "jobs": {"survey": _job(),
                     "dormant": _job(enabled=False, pending="run list must be confirmed by the user before enabling"),
                     "gpu-job": _job(gpu=True)},
        }
        self.write_cfg()
        self.state = self.root / "runtime/rl/test-phase"
        self.global_state = self.root / "runtime/rl"

    def tearDown(self):
        self.tmp.cleanup()

    def write_cfg(self):
        _write(self.config, json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n")

    def write_pools(self):
        _write(self.pools_path, json.dumps(self.pools, indent=2) + "\n")

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): (p.is_dir(), None if p.is_dir() else sha256_file(p))
                for p in self.root.rglob("*")}

    def issues(self, kind="survey"):
        return phase.preflight_phase(self.config, kind)["issues"]

    def run_job(self, kind="survey", games=0):
        with phase.job_context(self.config, kind) as ctx:
            for _ in range(games):
                ctx.claim_game()
        return ctx

    def receipt(self, ctx):
        return json.loads((ctx.output_dir / "receipt.json").read_text(encoding="utf-8"))

    def ledger(self):
        return Journal(self.state / "ledger.jsonl")

    # --- configuration and pin selection -------------------------------------------------

    def test_load_phase_requires_workspace_containing_config(self):
        cp, cfg, root = phase.load_phase(self.config)
        self.assertEqual((self.config, self.root), (cp, root))
        for key, value in (("workspace", "../../elsewhere"), ("schema_version", 2), ("state_dir", None),
                           ("state_dir", "../outside"), ("pools_path", "../../outside.json")):
            broken = copy.deepcopy(self.cfg)
            broken[key] = value
            _write(self.config, json.dumps(broken))
            with self.assertRaises(GateRefused, msg=f"{key}={value!r}"):
                phase.load_phase(self.config)
            result = phase.preflight_phase(self.config, "survey")
            self.assertFalse(result["ok"])
            self.assertTrue(result["issues"][0].startswith("Phase configuration unusable"))
        _write(self.config, "{not json")
        self.assertFalse(phase.preflight_phase(self.config, "survey")["ok"])

    def test_collect_pin_paths_globs_excludes_and_always_pins_config_and_pools(self):
        _, cfg, root = phase.load_phase(self.config)
        self.assertEqual(PINNED, phase.collect_pin_paths(cfg, root, config_path=self.config))
        narrow = copy.deepcopy(cfg)
        narrow["freeze"] = {"include": ["tools/*.py"], "exclude": []}
        self.assertEqual(["configs/phases/test-phase.json", "configs/pools-test.json", "tools/fake-tool.py"],
                         phase.collect_pin_paths(narrow, root, config_path=self.config))
        # Job-written state and the pins manifest are never pinned, even if a pattern matches them.
        _write(self.state / "pins.json", "{}")
        _write(self.state / "runs/x/receipt.json", "{}")
        _write(self.root / "runtime/other.json", "{}")
        wide = copy.deepcopy(cfg)
        wide["freeze"] = {"include": ["runtime/**/*.json"], "exclude": []}
        self.assertEqual(["configs/pools-test.json", "runtime/other.json"], phase.collect_pin_paths(wide, root))
        for bad in ("../outside/*.py", "/abs/*.py", "C:/x/*.py", "tools\\*.py", "", 7):
            broken = copy.deepcopy(cfg)
            broken["freeze"] = {"include": [bad], "exclude": []}
            with self.assertRaises(GateRefused, msg=repr(bad)):
                phase.collect_pin_paths(broken, root)
        broken = copy.deepcopy(cfg)
        del broken["freeze"]
        with self.assertRaises(GateRefused):
            phase.collect_pin_paths(broken, root)

    def test_freeze_writes_manifest_and_hash_chained_history(self):
        manifest = phase.freeze_phase(self.config)
        pins = self.state / "pins.json"
        self.assertEqual(manifest, json.loads(pins.read_text(encoding="utf-8")))
        self.assertEqual((1, "test-phase"), (manifest["schema_version"], manifest["phase_id"]))
        self.assertEqual(PINNED, sorted(manifest["files"]))
        for relative, digest in manifest["files"].items():
            self.assertEqual(sha256_file(self.root / relative), digest)
        history = Journal(self.state / "pins-history.jsonl")
        self.assertEqual(1, history.verify()["entries"])
        entry = history.entries()[0]
        self.assertEqual("freeze", entry["kind"])
        self.assertEqual(sha256_file(pins), entry["payload"]["manifest_sha256"])
        self.assertEqual(manifest["files"], entry["payload"]["files"])
        self.assertEqual([], [p.name for p in self.state.iterdir() if p.name.endswith(".tmp")])
        phase.freeze_phase(self.config)
        self.assertEqual(2, history.verify()["entries"])

    def test_freeze_refused_while_job_lock_held_or_history_tampered(self):
        _write(self.state / "job.lock", "{}")
        with self.assertRaises(GateRefused):
            phase.freeze_phase(self.config)
        self.assertFalse((self.state / "pins.json").exists())
        (self.state / "job.lock").unlink()
        phase.freeze_phase(self.config)
        history = self.state / "pins-history.jsonl"
        history.write_text(history.read_text(encoding="utf-8").replace("test-phase", "other-phase"), encoding="utf-8")
        before = sha256_file(self.state / "pins.json")
        with self.assertRaises(JournalError):
            phase.freeze_phase(self.config)
        self.assertEqual(before, sha256_file(self.state / "pins.json"))

    # --- preflight ----------------------------------------------------------------------

    def test_preflight_is_strictly_read_only(self):
        before = self.snapshot()
        result = phase.preflight_phase(self.config, "survey")
        self.assertFalse(result["ok"])
        self.assertTrue(any("No verified pin manifest" in issue for issue in result["issues"]))
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.state.exists())
        phase.freeze_phase(self.config)
        self.run_job()
        before = self.snapshot()
        result = phase.preflight_phase(self.config, "survey")
        self.assertEqual(before, self.snapshot())
        self.assertTrue(result["ok"], result["issues"])
        self.assertEqual([], result["issues"])
        self.assertEqual(("test-phase", "survey", 1), (result["phase_id"], result["job_kind"], result["used_jobs"]))
        self.assertEqual(sha256_file(self.state / "pins.json"), result["pins_sha256"])
        self.assertEqual(sha256_file(self.config), result["config_sha256"])
        self.assertEqual(sha256_file(self.pools_path), result["pools_sha256"])
        self.assertGreaterEqual(result["used_wall_seconds"], 0.0)

    def test_pin_drift_is_reported_and_unpinned_files_are_ignored(self):
        phase.freeze_phase(self.config)
        self.assertEqual([], self.issues())
        for relative in ("docs/NOTES.md", "REPORT.md", "tests/test_fake.py", "run.cmd",
                         "alpharush_rl/__pycache__/stale.py", "alpharush_rl/__pycache__/x.cpython-312.pyc",
                         "docs/new.md", "runtime/rl/test-phase/scratch.json"):
            _write(self.root / relative, "changed\n")
        self.assertEqual([], self.issues())

        core = self.root / "alpharush_rl/fake_core.py"
        _write(core, "VALUE = 2\n")
        self.assertEqual(["Code/config SHA mismatch: alpharush_rl/fake_core.py"], self.issues())
        _write(core, FAKE_FILES["alpharush_rl/fake_core.py"])
        self.assertEqual([], self.issues())

        added = self.root / "alpharush_rl/sub/new_module.py"
        _write(added, "NEW = 1\n")
        self.assertEqual(["File not covered by pin manifest: alpharush_rl/sub/new_module.py"], self.issues())
        added.unlink()

        tool = self.root / "tools/fake-tool.py"
        tool.unlink()
        issues = self.issues()
        self.assertIn("Pinned file no longer in pin set: tools/fake-tool.py", issues)
        self.assertTrue(any(i.startswith("Pinned file unreadable: tools/fake-tool.py") for i in issues))
        _write(tool, FAKE_FILES["tools/fake-tool.py"])
        self.assertEqual([], self.issues())

        self.pools["note"] = "edited"
        self.write_pools()
        self.assertEqual(["Code/config SHA mismatch: configs/pools-test.json"], self.issues())
        del self.pools["note"]
        self.write_pools()
        self.cfg["jobs"]["survey"]["max_games"] = 3
        self.write_cfg()
        self.assertEqual(["Code/config SHA mismatch: configs/phases/test-phase.json"], self.issues())
        phase.freeze_phase(self.config)
        self.assertEqual([], self.issues())

        pins = self.state / "pins.json"
        manifest = json.loads(pins.read_text(encoding="utf-8"))
        pins.write_text(json.dumps(dict(manifest, phase_id="another-phase")), encoding="utf-8")
        self.assertTrue(any("another schema or phase" in i for i in self.issues()))
        pins.unlink()
        self.assertTrue(any("No verified pin manifest" in i for i in self.issues()))

    def test_job_gates_enabled_gpu_unknown_and_caps(self):
        phase.freeze_phase(self.config)
        self.assertEqual(["dormant is not enabled: run list must be confirmed by the user before enabling"],
                         self.issues("dormant"))
        self.assertEqual(["Phase-0 tools never dispatch GPU work; the job must declare gpu=false "
                          "(or gpu=\"external-inference\" to query the separately started scoring server)"],
                         self.issues("gpu-job"))
        self.assertEqual(["Unknown job kind missing"], self.issues("missing"))
        for kind in ("dormant", "gpu-job", "missing"):
            with self.assertRaises(GateRefused):
                with phase.job_context(self.config, kind):
                    self.fail("refused job must never run")
        self.assertFalse((self.state / "ledger.jsonl").exists())
        self.assertFalse((self.state / "runs").exists())

        self.cfg["jobs"]["survey"].update(enabled="true", max_jobs=0, max_games=-1, max_wall_seconds=float("inf"))
        del self.cfg["jobs"]["survey"]["gpu"]
        self.cfg["max_concurrent_jobs"] = 2
        _write(self.config, json.dumps(self.cfg))  # json.dumps writes Infinity; parsing accepts it
        phase.freeze_phase(self.config)
        issues = self.issues()
        for expected in ("survey is not enabled", "gpu=false", "max_wall_seconds", "max_jobs", "max_games",
                         "max_concurrent_jobs"):
            self.assertTrue(any(expected in i for i in issues), (expected, issues))

    def test_stop_files_and_locks_refuse(self):
        phase.freeze_phase(self.config)
        for directory in (self.state, self.global_state):
            for name in ("STOP", "ENGINEERING-STOP"):
                stop = directory / name
                stop.touch()
                self.assertEqual([f"Stop file exists: {stop}"], self.issues())
                with self.assertRaises(GateRefused):
                    with phase.job_context(self.config, "survey"):
                        self.fail("stopped phase must never run")
                stop.unlink()
        lock = self.state / "job.lock"
        lock.write_text("{}", encoding="utf-8")
        self.assertEqual([phase.LOCK_ISSUE], self.issues())
        with self.assertRaises(GateRefused):
            with phase.job_context(self.config, "survey"):
                self.fail("locked phase must never run")
        self.assertTrue(lock.exists())  # another holder's lock is never removed
        lock.unlink()
        global_lock = self.global_state / "job.lock"
        global_lock.write_text("{}", encoding="utf-8")
        self.assertEqual([f"Global AlphaRush job lock exists: {global_lock}"], self.issues())
        global_lock.unlink()
        self.assertEqual([], self.issues())
        self.assertFalse((self.state / "ledger.jsonl").exists())

    # --- job context, ledger and budgets ------------------------------------------------

    def test_job_context_writes_hash_chained_ledger_events_and_receipt(self):
        phase.freeze_phase(self.config)
        with phase.job_context(self.config, "survey") as ctx:
            self.assertTrue((self.state / "job.lock").exists())
            outside = phase.preflight_phase(self.config, "survey")
            self.assertFalse(outside["ok"])
            self.assertIn(phase.LOCK_ISSUE, outside["issues"])
            self.assertTrue(any("Unclosed job" in i for i in outside["issues"]))
            with self.assertRaises(GateRefused):
                phase.freeze_phase(self.config)
            with self.assertRaises(GateRefused):
                with phase.job_context(self.config, "survey"):
                    self.fail("a second concurrent job must never run")
            ctx.check()
            ctx.record("note", value=1)
            self.assertEqual(0, ctx.claim_game())
            self.assertEqual(self.state / "runs", ctx.output_dir.parent)
            self.assertTrue(ctx.run_id.startswith("survey-"))
            self.assertEqual(ctx.run_id, ctx.output_dir.name)
            self.assertEqual(2, ctx.max_games)
        self.assertFalse((self.state / "job.lock").exists())
        receipt = self.receipt(ctx)
        self.assertEqual(("ok", None, 1, 2, False), (receipt["status"], receipt["error"], receipt["games_played"],
                                                     receipt["max_games"], receipt["verified"]))
        self.assertEqual(sha256_file(self.state / "pins.json"), receipt["pins_sha256"])
        ledger = self.ledger()
        self.assertEqual(2, ledger.verify()["entries"])
        opened, closed = ledger.entries()
        self.assertEqual(("open", "close"), (opened["kind"], closed["kind"]))
        self.assertEqual((ctx.run_id, ctx.run_id), (opened["payload"]["run_id"], closed["payload"]["run_id"]))
        self.assertEqual(2, opened["payload"]["caps"]["max_games"])
        self.assertIs(False, opened["payload"]["caps"]["gpu"])
        self.assertTrue(opened["payload"]["preflight"]["ok"])
        self.assertEqual([], opened["payload"]["preflight"]["issues"])
        self.assertEqual(("ok", 1), (closed["payload"]["status"], closed["payload"]["games_played"]))
        self.assertEqual((opened["sha256"], closed["sha256"]),
                         (receipt["ledger_open_sha256"], receipt["ledger_close_sha256"]))
        self.assertEqual(closed["payload"]["wall_seconds"], receipt["wall_seconds"])
        events = Journal(ctx.output_dir / "events.jsonl")
        self.assertEqual(1, events.verify()["entries"])
        self.assertEqual(("note", 1), (events.entries()[0]["kind"], events.entries()[0]["payload"]["value"]))
        after = phase.preflight_phase(self.config, "survey")
        self.assertTrue(after["ok"], after["issues"])
        self.assertEqual((1, closed["payload"]["wall_seconds"]), (after["used_jobs"], after["used_wall_seconds"]))

    def test_job_count_and_wall_budgets_are_exhausted(self):
        phase.freeze_phase(self.config)
        self.run_job()
        self.run_job()
        self.assertEqual(["No remaining survey jobs: used 2 of 2"], self.issues())
        with self.assertRaises(GateRefused):
            self.run_job()
        self.assertEqual(4, self.ledger().verify()["entries"])

        self.cfg["jobs"]["survey"]["max_jobs"] = 10
        self.write_cfg()
        phase.freeze_phase(self.config)
        self.assertEqual([], self.issues())
        ledger = self.ledger()
        ledger.append("open", {"run_id": "manual-1", "job_kind": "survey"})
        ledger.append("close", {"run_id": "manual-1", "job_kind": "survey", "wall_seconds": 31})
        self.assertEqual(["Insufficient remaining cumulative wall budget for the full job cap"], self.issues())
        # Other job kinds draw on their own budgets.
        result = phase.preflight_phase(self.config, "dormant")
        self.assertEqual(0, result["used_jobs"])
        self.assertTrue(all("budget" not in i for i in result["issues"]))

    def test_ledger_tampering_and_malformed_records_are_refused(self):
        phase.freeze_phase(self.config)
        self.run_job()
        path = self.state / "ledger.jsonl"
        good = path.read_text(encoding="utf-8")
        path.write_text(good.replace('"status":"ok"', '"status":"OK"'), encoding="utf-8")
        self.assertTrue(any(i.startswith("Ledger refused: hash mismatch") for i in self.issues()))
        path.write_text(good.splitlines(keepends=True)[0], encoding="utf-8")  # truncated: open only
        self.assertTrue(any("Unclosed job" in i for i in self.issues()))
        path.write_text(good + "\n", encoding="utf-8")
        self.assertTrue(any("blank journal line" in i for i in self.issues()))
        for rows in ([("note", {"run_id": "x"})],
                     [("close", {"run_id": "x", "job_kind": "survey", "wall_seconds": 1})],
                     [("open", {"run_id": "x", "job_kind": "survey"}), ("open", {"run_id": "x", "job_kind": "survey"})],
                     [("open", {"run_id": "x", "job_kind": "survey"}),
                      ("close", {"run_id": "x", "job_kind": "other", "wall_seconds": 1})],
                     [("open", {"run_id": "x", "job_kind": "survey"}),
                      ("close", {"run_id": "x", "job_kind": "survey", "wall_seconds": -1})]):
            path.unlink()
            ledger = Journal(path)
            for kind, payload in rows:
                ledger.append(kind, payload)
            issues = self.issues()
            self.assertTrue(any(i.startswith("Ledger refused") for i in issues), (rows, issues))
            with self.assertRaises(GateRefused):
                self.run_job()

    def test_job_context_stop_timeout_and_failure_statuses(self):
        self.cfg["jobs"]["survey"].update(max_jobs=10, total_wall_seconds=1000)
        self.write_cfg()
        phase.freeze_phase(self.config)
        cases = []
        for stop in (self.state / "STOP", self.global_state / "ENGINEERING-STOP"):
            with self.assertRaises(GateRefused):
                with phase.job_context(self.config, "survey") as ctx:
                    stop.touch()
                    ctx.check()
            cases.append((ctx, "stopped"))
            stop.unlink()
        with self.assertRaises(GateRefused):  # STOP after the body is caught by the closing check
            with phase.job_context(self.config, "survey") as ctx:
                (self.state / "STOP").touch()
        cases.append((ctx, "stopped"))
        (self.state / "STOP").unlink()
        with self.assertRaises(GateRefused):
            with phase.job_context(self.config, "survey") as ctx:
                ctx.deadline = time.monotonic() - 1
                ctx.check()
        cases.append((ctx, "stopped"))
        with self.assertRaises(ValueError):
            with phase.job_context(self.config, "survey") as ctx:
                raise ValueError("boom")
        cases.append((ctx, "failed"))
        closes = [row["payload"] for row in self.ledger().entries() if row["kind"] == "close"]
        self.assertEqual([status for _, status in cases], [row["status"] for row in closes])
        for ctx, status in cases:
            receipt = self.receipt(ctx)
            self.assertEqual(status, receipt["status"])
            self.assertIsNotNone(receipt["error"])
        self.assertEqual("ValueError: boom", self.receipt(cases[-1][0])["error"])
        self.assertFalse((self.state / "job.lock").exists())
        self.assertEqual([], self.issues())

    def test_max_games_is_a_hard_cap(self):
        phase.freeze_phase(self.config)
        with phase.job_context(self.config, "survey") as ctx:
            self.assertEqual([0, 1], [ctx.claim_game(), ctx.claim_game()])
            self.assertEqual(0, ctx.games_remaining)
            with self.assertRaises(GateRefused):
                ctx.claim_game()
        receipt = self.receipt(ctx)
        self.assertEqual(("ok", 2), (receipt["status"], receipt["games_played"]))
        with self.assertRaises(GateRefused):
            with phase.job_context(self.config, "survey") as ctx:
                ctx.claim_game()
                ctx.claim_game()
                ctx.claim_game()
        self.assertEqual(("stopped", 2), (self.receipt(ctx)["status"], self.receipt(ctx)["games_played"]))

    # --- seed pools ---------------------------------------------------------------------

    def test_seed_pools_must_be_disjoint_and_well_formed(self):
        def problems(mutate):
            pools = copy.deepcopy(self.pools)
            mutate(pools)
            return phase.pool_problems(pools)

        self.assertEqual([], problems(lambda p: None))
        overlap = problems(lambda p: p["pools"]["train"]["seeds"].append(5001))
        self.assertEqual(["Seed pools train and evaluation overlap: [5001]"], overlap)
        self.assertTrue(problems(lambda p: p["pools"]["final_campaign_run"]["seeds"].append(2001)))
        self.assertTrue(problems(lambda p: p["pools"]["train"]["seeds"].append(1001)))
        self.assertTrue(problems(lambda p: p["pools"]["train"]["seeds"].append(True)))
        self.assertTrue(problems(lambda p: p["pools"]["train"]["seeds"].append(-1)))
        self.assertTrue(problems(lambda p: p["pools"]["train"]["seeds"].append(1011.0)))
        self.assertTrue(problems(lambda p: p["pools"].pop("evaluation")))
        self.assertTrue(problems(lambda p: p["pools"].update(extra={"seeds": [7001]})))
        self.assertTrue(problems(lambda p: p["pools"]["evaluation"].update(never_in_gradients=False)))
        self.assertTrue(problems(lambda p: p.update(unlisted_seeds="never_train")))
        self.assertTrue(problems(lambda p: p.pop("pools")))
        leaked = copy.deepcopy(self.pools)
        leaked["pools"]["train"]["seeds"].append(6001)
        with self.assertRaises(GateRefused):
            phase.seed_role(leaked, 1001)

        phase.freeze_phase(self.config)
        self.pools = leaked
        self.write_pools()
        phase.freeze_phase(self.config)  # pins match, yet the pool gate still refuses
        self.assertEqual(["Seed pools train and final_campaign_run overlap: [6001]"], self.issues())
        self.pools_path.write_text("{broken", encoding="utf-8")
        phase.freeze_phase(self.config)
        self.assertTrue(any(i.startswith("Pool manifest unreadable") for i in self.issues()))


if __name__ == "__main__":
    unittest.main()
