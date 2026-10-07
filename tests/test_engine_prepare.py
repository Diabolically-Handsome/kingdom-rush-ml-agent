"""engine.prepare fast path / explicit rebuild, Worker hygiene, and evidence publishing.

Everything runs against tiny fake game files in a temporary directory: GAME,
ROOT and ASSETS are patched, so no real Steam file, engine exe or process is touched.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock
import zipfile

from alpharush_rl import engine, env as env_module, validation

STUB = b"MZ-fake-love-stub\x00" * 8
ORIGINAL_MAIN = b"-- original game main\nreturn true\n"
BRIDGE = "local M = {}\nlocal json = {}\nlocal function collect_game_state() return {} end\nreturn M\n"
EXPECTED_KEYS = {"GAME/Kingdom Rush.exe.bak", "GAME/SDL2.dll", "GAME/lua51.dll",
                 "Lumi_Nox/games/kingdom_rush/bridge.lua",
                 "alpharush_rl/assets/wrapper.lua", "alpharush_rl/assets/host.lua", "alpharush_rl/assets/conf.lua"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_game_zip(corrupt=False):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        z.writestr("main.lua", ORIGINAL_MAIN)
        z.writestr("conf.lua", b"function love.conf(t) end\n")
        z.writestr("kr1/game_settings.lua", b"return {main_campaign_levels = 12}\n")
    data = archive.getvalue()
    if corrupt:
        data = data.replace(b"original game main", b"0riginal game main", 1)
    return STUB + data


class TempGameMixin:
    def make_workspace(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-engine-")
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root = base / "workspace"
        self.game = base / "steam/Kingdom Rush"
        self.assets = self.root / "alpharush_rl/assets"
        for directory in (self.game, self.assets, self.root / "Lumi_Nox/games/kingdom_rush"):
            directory.mkdir(parents=True)
        (self.game / "Kingdom Rush.exe.bak").write_bytes(fake_game_zip())
        (self.game / "SDL2.dll").write_bytes(b"fake sdl")
        (self.game / "lua51.dll").write_bytes(b"fake lua")
        (self.game / "notes.txt").write_text("not an engine input", encoding="utf-8")
        (self.root / "Lumi_Nox/games/kingdom_rush/bridge.lua").write_text(BRIDGE, encoding="utf-8")
        (self.assets / "wrapper.lua").write_bytes(b"-- wrapper\n")
        (self.assets / "host.lua").write_bytes(b"-- host v1\n")
        (self.assets / "conf.lua").write_bytes(b"-- conf wrapper\n")
        for patcher in (mock.patch.object(engine, "ROOT", self.root), mock.patch.object(engine, "GAME", self.game),
                        mock.patch.object(engine, "ASSETS", self.assets),
                        mock.patch.dict(engine._VERIFIED, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        # Guard: every engine path must now point inside the temporary directory.
        for path in (engine.ROOT, engine.GAME, engine.ASSETS):
            self.assertTrue(Path(path).is_relative_to(base))
        self.destination = self.root / "runtime/rl-engine"
        self.exe = self.destination / "Kingdom Rush.exe"
        self.manifest = self.destination / "manifest.json"

    def records(self):
        return json.loads(self.manifest.read_text(encoding="utf-8"))


class EnginePrepareTests(TempGameMixin, unittest.TestCase):
    def setUp(self):
        self.make_workspace()

    def test_missing_manifest_refuses_without_explicit_rebuild(self):
        with self.assertRaisesRegex(RuntimeError, r"python -m alpharush_rl\.engine --prepare"):
            engine.prepare()
        self.assertFalse(self.destination.exists())

    def test_rebuild_writes_relative_v2_manifest_and_wrapped_exe(self):
        self.assertEqual(engine.prepare(rebuild=True), self.exe)
        manifest = self.records()
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(set(manifest), {"schema_version", "recipe_sha256", "inputs", "runtime", "exe"})
        self.assertEqual(manifest["recipe_sha256"], engine._recipe_sha256())
        self.assertEqual(set(manifest["inputs"]), EXPECTED_KEYS)
        self.assertEqual(set(manifest["runtime"]), {"runtime/rl-engine/SDL2.dll", "runtime/rl-engine/lua51.dll",
                                                    "runtime/rl-engine/alpha_rl_host.lua"})
        for key, record in manifest["runtime"].items():
            path = self.root / key
            self.assertEqual(record, {"sha256": sha(path), "size": path.stat().st_size,
                                      "mtime_ns": path.stat().st_mtime_ns})
        self.assertEqual((self.destination / "alpha_rl_host.lua").read_bytes(), b"-- host v1\n")
        text = self.manifest.read_text(encoding="utf-8")
        for absolute in (self.tmp.name, Path(self.tmp.name).as_posix(), json.dumps(self.tmp.name)[1:-1]):
            self.assertNotIn(absolute, text)
        self.assertIsNone(re.search(r"[A-Za-z]:(\\\\|/)", text))
        sources = {"GAME/Kingdom Rush.exe.bak": self.game / "Kingdom Rush.exe.bak",
                   "GAME/SDL2.dll": self.game / "SDL2.dll", "GAME/lua51.dll": self.game / "lua51.dll",
                   "Lumi_Nox/games/kingdom_rush/bridge.lua": self.root / "Lumi_Nox/games/kingdom_rush/bridge.lua",
                   "alpharush_rl/assets/wrapper.lua": self.assets / "wrapper.lua",
                   "alpharush_rl/assets/host.lua": self.assets / "host.lua",
                   "alpharush_rl/assets/conf.lua": self.assets / "conf.lua"}
        for key, path in sources.items():
            st = path.stat()
            self.assertEqual(manifest["inputs"][key], {"sha256": sha(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
        st = self.exe.stat()
        self.assertEqual(manifest["exe"], {"sha256": sha(self.exe), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
        self.assertTrue(self.exe.read_bytes().startswith(STUB))
        with zipfile.ZipFile(self.exe) as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
            self.assertTrue({"main.lua", "_alpha_original_main.lua", "alpha_rl_host.lua", "alpha_native_bridge.lua",
                             "conf.lua", "_alpha_original_conf.lua", "kr1/game_settings.lua"} <= names)
            self.assertEqual(archive.read("conf.lua"), b"-- conf wrapper\n")
            self.assertEqual(archive.read("_alpha_original_conf.lua"), b"function love.conf(t) end\n")
            self.assertEqual(archive.read("main.lua"), b"-- wrapper\n")
            self.assertEqual(archive.read("_alpha_original_main.lua"), ORIGINAL_MAIN)
            self.assertEqual(archive.read("alpha_rl_host.lua"), b"-- host v1\n")
            native = archive.read("alpha_native_bridge.lua")
            self.assertIn(b"M.command_json", native)
            self.assertTrue(native.rstrip().endswith(b"return M"))
        self.assertEqual((self.destination / "SDL2.dll").read_bytes(), b"fake sdl")
        self.assertEqual(sorted(p.name for p in self.destination.glob("*.tmp")), [])

    def test_fast_path_stats_only_and_caches_in_process(self):
        engine.prepare(rebuild=True)
        engine._VERIFIED.clear()
        with mock.patch.object(engine, "_sha256", side_effect=AssertionError("hashed on fast path")) as hashed, \
                mock.patch.object(engine, "_build", side_effect=AssertionError("rebuilt")):
            self.assertEqual(engine.prepare(), self.exe)
            with mock.patch.object(engine, "_load_manifest", side_effect=AssertionError("manifest re-read")) as load:
                self.assertEqual(engine.prepare(), self.exe)
            hashed.assert_not_called()
            load.assert_not_called()
        # An explicit rebuild always re-hashes, but does not rebuild identical inputs.
        with mock.patch.object(engine, "_build", side_effect=AssertionError("rebuilt")):
            self.assertEqual(engine.prepare(rebuild=True), self.exe)

    def test_changed_host_requires_explicit_rebuild_then_recovers(self):
        engine.prepare(rebuild=True)
        old = self.records()
        host = self.assets / "host.lua"
        host.write_bytes(b"-- host v2 with a different size\n")
        with self.assertRaisesRegex(RuntimeError, "Engine inputs changed or manifest is legacy"):
            engine.prepare()
        self.assertEqual(self.records(), old)
        engine.prepare(rebuild=True)
        new = self.records()
        self.assertEqual(new["inputs"]["alpharush_rl/assets/host.lua"]["sha256"], sha(host))
        self.assertNotEqual(new["exe"]["sha256"], old["exe"]["sha256"])
        with zipfile.ZipFile(self.exe) as archive:
            self.assertEqual(archive.read("alpha_rl_host.lua"), b"-- host v2 with a different size\n")
        engine._VERIFIED.clear()
        with mock.patch.object(engine, "_sha256", side_effect=AssertionError("hashed")):
            self.assertEqual(engine.prepare(), self.exe)

    def test_mtime_only_change_refreshes_stats_without_rebuilding(self):
        engine.prepare(rebuild=True)
        old = self.records()
        exe_bytes_sha, exe_mtime = sha(self.exe), self.exe.stat().st_mtime_ns
        wrapper = self.assets / "wrapper.lua"
        st = wrapper.stat()
        os.utime(wrapper, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        with self.assertRaises(RuntimeError):
            engine.prepare()
        with mock.patch.object(engine, "_build", side_effect=AssertionError("rebuilt")) as build:
            self.assertEqual(engine.prepare(rebuild=True), self.exe)
            build.assert_not_called()
        new = self.records()
        self.assertEqual(sha(self.exe), exe_bytes_sha)
        self.assertEqual(self.exe.stat().st_mtime_ns, exe_mtime)
        self.assertEqual(new["exe"], old["exe"])
        self.assertEqual(new["inputs"]["alpharush_rl/assets/wrapper.lua"]["sha256"],
                         old["inputs"]["alpharush_rl/assets/wrapper.lua"]["sha256"])
        self.assertEqual(new["inputs"]["alpharush_rl/assets/wrapper.lua"]["mtime_ns"], st.st_mtime_ns + 5_000_000_000)
        engine._VERIFIED.clear()
        self.assertEqual(engine.prepare(), self.exe)

    def test_runtime_copies_recipe_and_force_are_bound_to_the_manifest(self):
        engine.prepare(rebuild=True)
        exe_sha = sha(self.exe)
        # A replaced runtime DLL is refused on the fast path and repaired by an explicit rebuild.
        copied = self.destination / "lua51.dll"
        copied.write_bytes(b"debug luajit")
        engine._VERIFIED.clear()
        with self.assertRaises(RuntimeError):
            engine.prepare()
        engine.prepare(rebuild=True)
        self.assertEqual(copied.read_bytes(), b"fake lua")
        # Same size and restored mtime still fail on an explicit rebuild, which always hashes.
        st = copied.stat()
        copied.write_bytes(b"fake LUA")
        os.utime(copied, ns=(st.st_atime_ns, st.st_mtime_ns))
        engine.prepare(rebuild=True)
        self.assertEqual(copied.read_bytes(), b"fake lua")
        # The executed host copy must match assets/host.lua.
        (self.destination / "alpha_rl_host.lua").write_bytes(b"-- edited copy\n")
        engine._VERIFIED.clear()
        with self.assertRaises(RuntimeError):
            engine.prepare()
        engine.prepare(rebuild=True)
        self.assertEqual((self.destination / "alpha_rl_host.lua").read_bytes(), b"-- host v1\n")
        # Stale DLLs absent from the game directory are removed by a rebuild.
        (self.destination / "stale.dll").write_bytes(b"old")
        with self.assertRaises(RuntimeError):
            engine.prepare()
        engine.prepare(rebuild=True)
        self.assertFalse((self.destination / "stale.dll").exists())
        # A changed build recipe invalidates the prepared engine.
        engine._VERIFIED.clear()
        with mock.patch.object(engine, "RECIPE_VERSION", "alpharush-engine-recipe-test"):
            with self.assertRaises(RuntimeError):
                engine.prepare()
            with mock.patch.object(engine, "_build", wraps=engine._build) as build:
                engine.prepare(rebuild=True)
                build.assert_called_once()
        engine._VERIFIED.clear()
        with mock.patch.object(engine, "_build", wraps=engine._build) as build:
            engine.prepare(rebuild=True)
            build.assert_called_once()
            engine.prepare(rebuild=True)
            build.assert_called_once()
            engine.prepare(rebuild=True, force=True)
            self.assertEqual(build.call_count, 2)
        self.assertEqual(self.records()["exe"]["sha256"], sha(self.exe))
        self.assertEqual(sha(self.exe), exe_sha)

    def test_force_rebuilds_are_byte_identical(self):
        # An original archive with its own timestamps, a directory entry and a stored entry.
        original_times = {"main.lua": (2011, 3, 4, 5, 6, 8), "conf.lua": (2012, 7, 8, 9, 10, 12),
                          "kr1/": (2013, 1, 2, 3, 4, 6), "kr1/game_settings.lua": (2014, 5, 6, 7, 8, 10)}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
            for name, stamp in original_times.items():
                z.writestr(zipfile.ZipInfo(name, date_time=stamp),
                           b"" if name.endswith("/") else ORIGINAL_MAIN if name == "main.lua" else b"-- data\n")
        (self.game / "Kingdom Rush.exe.bak").write_bytes(STUB + archive.getvalue())
        engine.prepare(rebuild=True)
        first = self.exe.read_bytes()
        # zipfile stamps entries written by name with time.localtime(time.time()); force a
        # rebuild three days later, then one now, and both must equal the first byte for byte.
        later = time.time() + 3 * 86400 + 7
        with mock.patch.object(zipfile, "time", types.SimpleNamespace(time=lambda: later, localtime=time.localtime)):
            with mock.patch.object(engine, "_build", wraps=engine._build) as build:
                engine.prepare(rebuild=True, force=True)
                build.assert_called_once()
        self.assertEqual(self.exe.read_bytes(), first)
        engine.prepare(rebuild=True, force=True)
        self.assertEqual(self.exe.read_bytes(), first)
        self.assertEqual(self.records()["exe"]["sha256"], hashlib.sha256(first).hexdigest())
        with zipfile.ZipFile(self.exe) as built:
            self.assertIsNone(built.testzip())
            infos = {info.filename: info for info in built.infolist()}
        # Originals keep their date_time (the renamed main.lua and conf.lua too); added entries are fixed.
        self.assertEqual({name: infos[name].date_time for name in ("_alpha_original_main.lua",
                                                                    "_alpha_original_conf.lua", "kr1/",
                                                                    "kr1/game_settings.lua")},
                         {"_alpha_original_main.lua": original_times["main.lua"],
                          "_alpha_original_conf.lua": original_times["conf.lua"],
                          "kr1/": original_times["kr1/"],
                          "kr1/game_settings.lua": original_times["kr1/game_settings.lua"]})
        for name in ("main.lua", "conf.lua", "alpha_rl_host.lua", "alpha_native_bridge.lua"):
            self.assertEqual(infos[name].date_time, (1980, 1, 1, 0, 0, 0))
        # Every entry is deflated, with the attributes ZipFile.writestr(name, ...) gives.
        self.assertEqual({info.compress_type for info in infos.values()}, {zipfile.ZIP_DEFLATED})
        self.assertEqual({name: info.external_attr for name, info in infos.items() if name.endswith("/")},
                         {"kr1/": (0o40775 << 16) | 0x10})
        self.assertEqual({info.external_attr for name, info in infos.items() if not name.endswith("/")}, {0o600 << 16})
        self.assertEqual(engine.RECIPE_VERSION, "alpharush-engine-recipe-4")

    def test_legacy_manifest_new_dll_and_tampered_exe_are_refused(self):
        engine.prepare(rebuild=True)
        current = self.records()
        legacy = {"inputs": {str(self.assets / "host.lua"): current["inputs"]["alpharush_rl/assets/host.lua"]["sha256"]},
                  "exe_sha256": current["exe"]["sha256"]}
        self.manifest.write_text(json.dumps(legacy), encoding="utf-8")
        engine._VERIFIED.clear()
        with self.assertRaisesRegex(RuntimeError, "legacy"):
            engine.prepare()
        engine.prepare(rebuild=True)
        self.assertEqual(self.records()["schema_version"], 2)
        (self.game / "extra.dll").write_bytes(b"new dll")
        with self.assertRaises(RuntimeError):
            engine.prepare()
        engine.prepare(rebuild=True)
        self.assertIn("GAME/extra.dll", self.records()["inputs"])
        with self.exe.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(RuntimeError):
            engine.prepare()
        engine.prepare(rebuild=True)
        with zipfile.ZipFile(self.exe) as archive:
            self.assertIsNone(archive.testzip())
        self.assertEqual(self.records()["exe"]["sha256"], sha(self.exe))

    def test_corrupt_original_fails_with_runtime_error_and_leaves_nothing(self):
        (self.game / "Kingdom Rush.exe.bak").write_bytes(fake_game_zip(corrupt=True))
        with self.assertRaisesRegex(RuntimeError, "CRC"):
            engine.prepare(rebuild=True)
        self.assertFalse(self.exe.exists())
        self.assertFalse(self.manifest.exists())
        self.assertEqual(list(self.destination.glob("*.tmp")), [])
        source = inspect.getsource(engine)
        self.assertIsNone(re.search(r"^\s*assert\b", source, re.M))


class WorkerHygieneTests(TempGameMixin, unittest.TestCase):
    def setUp(self):
        self.make_workspace()

    def test_signature_identity_and_log_path(self):
        parameters = inspect.signature(engine.Worker.__init__).parameters
        self.assertEqual([(n, p.default) for n, p in parameters.items() if n != "self"],
                         [("seed", 1001), ("port", 9879), ("identity", None), ("level", 1), ("difficulty", 2),
                          ("rng_mode", ""), ("action_scope", "v1"), ("profile", None), ("mode", 1)])
        self.assertEqual(len(engine.RNG_MODES), 8)
        self.assertIn(engine.DETERMINISTIC_MODE, engine.RNG_MODES)
        for mode in engine.RNG_MODES:
            self.assertEqual(engine.Worker(rng_mode=mode).rng_mode, mode)
        with self.assertRaises(ValueError):
            engine.Worker(rng_mode="isolate")
        self.assertEqual(engine.ACTION_SCOPES, ("v1", "v2", "v3"))
        self.assertEqual(engine.Worker().action_scope, "v1")
        for scope in engine.ACTION_SCOPES:
            self.assertEqual(engine.Worker(action_scope=scope).action_scope, scope)
        for bad in ("v9", "V2", "", None, 2):
            with self.assertRaises(ValueError):
                engine.Worker(action_scope=bad)
        worker = engine.Worker(seed=5, identity="unit_worker-1")
        self.assertEqual(worker.log_path, self.root / "runtime/rl-engine/logs/unit_worker-1.log")
        self.assertTrue(engine.Worker(seed=7).identity.startswith("alpharush_rl_7_"))
        for bad in ("../escape", "a/b", "a\\b", ".hidden", "with space", "nul\0", "x" * 200):
            with self.assertRaises(ValueError):
                engine.Worker(identity=bad)

    def test_start_refuses_stale_engine_before_any_launch(self):
        with mock.patch.object(engine.subprocess, "Popen", side_effect=AssertionError("launched")) as popen:
            with self.assertRaisesRegex(RuntimeError, "--prepare"):
                engine.Worker(identity="unit_start").start()
            popen.assert_not_called()
        self.assertFalse((self.root / "runtime/rl-engine/logs").exists())

    def test_close_kills_after_terminate_timeout_and_always_closes_log(self):
        worker = engine.Worker(identity="unit_close")
        process, log, sock = mock.Mock(), mock.Mock(), mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("game", 5), 0]
        worker.process, worker.log, worker.sock = process, log, sock
        worker.close()
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=5), mock.call(timeout=5)])
        sock.close.assert_called_once_with()
        log.close.assert_called_once_with()
        self.assertIsNone(worker.sock)
        self.assertIsNone(worker.log)

        stuck, log = mock.Mock(), mock.Mock()
        stuck.poll.return_value = None
        stuck.wait.side_effect = subprocess.TimeoutExpired("game", 5)
        worker.process, worker.log = stuck, log
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.close()
        stuck.kill.assert_called_once_with()
        log.close.assert_called_once_with()

    def test_rpc_surfaces_malformed_reply_and_meta_uses_rpc(self):
        class FakeSocket:
            def __init__(self, reply):
                self.reply, self.sent = reply, []

            def sendall(self, data):
                self.sent.append(data)

            def recv(self, size):
                reply, self.reply = self.reply, b""
                return reply

        worker = engine.Worker(identity="unit_rpc")
        worker.sock = FakeSocket(b'{"ok":false,"error":"Malformed request: expected colon at byte 9"}\n')
        with self.assertRaisesRegex(RuntimeError, "Malformed request: expected colon"):
            worker.rpc("state")
        worker.sock = FakeSocket(b'{"id":"99","ok":true,"result":{}}\n')
        with self.assertRaisesRegex(RuntimeError, "ID mismatch"):
            worker.rpc("state")
        with mock.patch.object(worker, "rpc", return_value={"level_idx": 1, "errors": {}, "locked_towers": {},
                                                             "locked_powers": {"1": 2}}) as rpc:
            self.assertEqual(worker.meta(), {"level_idx": 1, "errors": [], "locked_towers": [],
                                             "locked_powers": {"1": 2}})
            rpc.assert_called_once_with("meta")
        worker.sock = FakeSocket(b'{"id":"3","ok":true,"result":{"type":"pong"}}\n')
        self.assertEqual(worker.rpc("state"), {"type": "pong"})
        sent = json.loads(worker.sock.sent[0])
        self.assertEqual(sent["token"], worker.token)
        self.assertEqual(len(worker.token), 32)

    def test_native_env_passes_difficulty_identity_and_keeps_reset_trace(self):
        with mock.patch.object(env_module, "Worker") as worker_class:
            default = env_module.NativeEnv()
            worker_class.assert_called_once_with(seed=1001, level=1, port=9879, difficulty=2, identity=None,
                                                 rng_mode="")
            self.assertEqual(default.difficulty, 2)
            worker_class.reset_mock()
            native = env_module.NativeEnv(seed=7, level=3, port=1234, difficulty=3, identity="unit_env",
                                          rng_mode="audit")
            worker_class.assert_called_once_with(seed=7, level=3, port=1234, difficulty=3, identity="unit_env",
                                                 rng_mode="audit")
            self.assertEqual((native.seed, native.level, native.difficulty), (7, 3, 3))
            self.assertEqual((default.action_scope, native.action_scope), ("v1", "v1"))
            # v1 keeps the exact Worker call; other scopes are passed through.
            worker_class.reset_mock()
            env_module.NativeEnv(identity="unit_env_v1", action_scope="v1")
            worker_class.assert_called_once_with(seed=1001, level=1, port=9879, difficulty=2, identity="unit_env_v1",
                                                 rng_mode="")
            worker_class.reset_mock()
            scoped = env_module.NativeEnv(identity="unit_env_v2", action_scope="v2")
            worker_class.assert_called_once_with(seed=1001, level=1, port=9879, difficulty=2, identity="unit_env_v2",
                                                 rng_mode="", action_scope="v2")
            self.assertEqual(scoped.action_scope, "v2")
            worker_class.reset_mock()
            with self.assertRaises(ValueError):
                env_module.NativeEnv(action_scope="v9")
            worker_class.assert_not_called()
            native.worker.rpc.return_value = {"type": "game_state", "tick": 5, "gold": 100, "towers": [],
                                              "holders": [], "enemies": [], "heroes": []}
            native.reset()
        self.assertEqual(set(native.trace[0]), {"kind", "state_sha256", "tick", "seed", "level"})


class EvidencePublishTests(TempGameMixin, unittest.TestCase):
    def setUp(self):
        self.make_workspace()
        patcher = mock.patch.object(validation, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root / "runtime/rl").mkdir(parents=True)

    def test_publish_refuses_existing_global_evidence_before_native_run(self):
        published = self.root / "runtime/rl/native-evidence.json"
        published.write_text('{"old": true}', encoding="utf-8")
        output = self.root / "runtime/rl/fresh-validation"
        with mock.patch.object(validation, "NativeEnv", side_effect=AssertionError("native run")) as native:
            with self.assertRaises(FileExistsError):
                validation.collect_verified(output, publish=True)
            native.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(published.read_text(encoding="utf-8"), '{"old": true}')

    def test_stop_is_checked_before_anything_runs(self):
        (self.root / "runtime/rl/STOP").write_text("", encoding="utf-8")
        output = self.root / "runtime/rl/stopped-validation"
        with mock.patch.object(validation, "NativeEnv", side_effect=AssertionError("native run")) as native:
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                validation.collect_verified(output)
            native.assert_not_called()
        self.assertFalse(output.exists())

    def test_write_new_json_never_overwrites(self):
        target = self.root / "runtime/rl/model-comparison.json"
        validation.write_new_json(target, {"first": 1})
        with self.assertRaises(FileExistsError):
            validation.write_new_json(target, {"second": 2})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"first": 1})


if __name__ == "__main__":
    unittest.main()
