"""Prepare and manage an isolated Windows LÖVE worker, preserving Steam files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GAME = Path(r"D:\SteamLibrary\steamapps\common\Kingdom Rush")
ASSETS = Path(__file__).parent / "assets"
SCHEMA_VERSION = 2
REBUILD_HINT = ("Engine inputs changed or manifest is legacy; "
                "run `python -m alpharush_rl.engine --prepare` explicitly")
HOST_COPY = "alpha_rl_host.lua"
# Bump when the build recipe below changes; its digest is part of the manifest.
# Recipe 3: deterministic archive (original entries keep their date_time, added entries use
# ADDED_DATE_TIME), so equal inputs always give a byte-identical exe.
RECIPE_VERSION = "alpharush-engine-recipe-4"
# Fixed DOS timestamp of the entries the recipe adds (the earliest a zip can store).
ADDED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
NATIVE_EXPORTS = '''
M.codec = json
M.collect_state = collect_game_state
M.command_json = function(cmd)
    local wire=nil
    handle_command(cmd,{send=function(self,s) wire=s; return #s end})
    return wire
end
return M
'''
# Manifest path -> records this process already matched by stat (fast path cache).
_VERIFIED: dict[str, dict] = {}


def _recipe_sha256():
    return hashlib.sha256(f"{RECIPE_VERSION}\n{NATIVE_EXPORTS}".encode("utf-8")).hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _key(path):
    """Workspace-relative posix key; game files are GAME/<name>. Never absolute."""
    path = Path(path)
    if path.parent == GAME:
        return f"GAME/{path.name}"
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        raise RuntimeError(f"Engine input is outside the workspace and game directory: {path.name}") from None


def _inputs():
    source = GAME / "Kingdom Rush.exe.bak"
    bridge = ROOT / "Lumi_Nox/games/kingdom_rush/bridge.lua"
    paths = [source, bridge, ASSETS / "wrapper.lua", ASSETS / "host.lua", ASSETS / "conf.lua",
             *sorted(GAME.glob("*.dll"))]
    return {_key(p): p for p in paths}


def _runtime_files(destination):
    """Files the game loads besides the exe: copied DLLs and the verified host copy."""
    names = sorted({p.name for p in GAME.glob("*.dll")} | {p.name for p in destination.glob("*.dll")} | {HOST_COPY})
    return {_key(destination / name): destination / name for name in names}


def _load_manifest(path):
    try:
        prior = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(prior, dict) or prior.get("schema_version") != SCHEMA_VERSION:
        return None
    if not all(isinstance(prior.get(k), dict) for k in ("inputs", "exe", "runtime")):
        return None
    return prior


def _stats_match(records, inputs, exe, runtime):
    """True only if recipe, file sets and every (size, mtime_ns), exe included, match."""
    if records.get("recipe_sha256") != _recipe_sha256():
        return False
    if set(records["inputs"]) != set(inputs) or set(records["runtime"]) != set(runtime):
        return False
    try:
        pairs = ([(records["inputs"][key], inputs[key]) for key in inputs]
                 + [(records["runtime"][key], runtime[key]) for key in runtime] + [(records["exe"], exe)])
        return all(isinstance(r, dict) and _stat(p) == {"size": r.get("size"), "mtime_ns": r.get("mtime_ns")}
                   for r, p in pairs)
    except OSError:
        return False


def _write_manifest(path, value):
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_inputs(inputs):
    records = {}
    for key, path in inputs.items():
        before = _stat(path)
        digest = _sha256(path)
        if _stat(path) != before:
            raise RuntimeError(f"Engine input changed while hashing: {key}")
        records[key] = {"sha256": digest, **before}
    return records


def _replace_copy(source, target):
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _zip_entry(name, date_time):
    """The entry ``ZipFile.writestr(name, ...)`` would make, with a given date_time instead of now."""
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o40775 << 16) | 0x10 if info.filename.endswith("/") else 0o600 << 16
    return info


def _build(source, bridge, destination, exe):
    native = bridge.read_text(encoding="utf-8").rsplit("return M", 1)[0] + NATIVE_EXPORTS
    temporary = destination / f"Kingdom Rush.exe.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(source) as original:
            if original.testzip() is not None:
                raise RuntimeError("Original game resources fail CRC validation")
            # Runtime files change only after the source archive is known good.
            game_dlls = {dll.name for dll in GAME.glob("*.dll")}
            for stale in destination.glob("*.dll"):
                if stale.name not in game_dlls:
                    stale.unlink()
            for dll in sorted(GAME.glob("*.dll")):
                _replace_copy(dll, destination / dll.name)
            _replace_copy(ASSETS / "host.lua", destination / HOST_COPY)
            offset = original.infolist()[0].header_offset
            with source.open("rb") as handle, temporary.open("wb") as out:
                out.write(handle.read(offset))
            with zipfile.ZipFile(temporary, "a", compression=zipfile.ZIP_DEFLATED) as out:
                # No wall-clock time enters the archive: originals (the renamed main.lua too)
                # keep their own date_time, added entries get ADDED_DATE_TIME.
                renamed = {"main.lua": "_alpha_original_main.lua", "conf.lua": "_alpha_original_conf.lua"}
                for info in original.infolist():
                    name = renamed.get(info.filename, info.filename)
                    out.writestr(_zip_entry(name, info.date_time), original.read(info))
                out.writestr(_zip_entry("main.lua", ADDED_DATE_TIME), (ASSETS / "wrapper.lua").read_bytes())
                out.writestr(_zip_entry("conf.lua", ADDED_DATE_TIME), (ASSETS / "conf.lua").read_bytes())
                out.writestr(_zip_entry("alpha_rl_host.lua", ADDED_DATE_TIME), (ASSETS / "host.lua").read_bytes())
                out.writestr(_zip_entry("alpha_native_bridge.lua", ADDED_DATE_TIME), native.encode("utf-8"))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Prepared engine archive fails CRC validation")
        os.replace(temporary, exe)
    finally:
        temporary.unlink(missing_ok=True)


def _shas(records):
    return {k: r.get("sha256") for k, r in records.items() if isinstance(r, dict)}


def prepare(rebuild: bool = False, force: bool = False):
    """Return the prepared exe; rebuilding happens only when explicitly requested.

    Without ``rebuild`` this is a stat-only check against the manifest. An explicit
    rebuild always re-hashes every input and runtime file, and rebuilds whenever any
    digest, the runtime file set or the recipe differs (or when ``force`` is set).
    """
    destination = ROOT / "runtime/rl-engine"
    manifest = destination / "manifest.json"
    exe = destination / "Kingdom Rush.exe"
    inputs = _inputs()
    cache_key = str(manifest)
    if not rebuild:
        runtime = _runtime_files(destination)
        cached = _VERIFIED.get(cache_key)
        if cached is not None and _stats_match(cached, inputs, exe, runtime):
            return exe  # same process, already matched: stat only, no manifest read
        records = _load_manifest(manifest)
        if records is not None and _stats_match(records, inputs, exe, runtime):
            _VERIFIED[cache_key] = records
            return exe
        _VERIFIED.pop(cache_key, None)
        raise RuntimeError(REBUILD_HINT)
    _VERIFIED.pop(cache_key, None)
    destination.mkdir(parents=True, exist_ok=True)
    hashed = _hash_inputs(inputs)
    host_sha = hashed[_key(ASSETS / "host.lua")]["sha256"]
    host_key = _key(destination / HOST_COPY)
    prior = _load_manifest(manifest)
    expected = _runtime_files(destination)
    unchanged = (not force and prior is not None and exe.exists()
                 and prior.get("recipe_sha256") == _recipe_sha256()
                 and _shas(prior["inputs"]) == _shas(hashed)
                 and set(prior["runtime"]) == set(expected)
                 and all(p.is_file() for p in expected.values())
                 and _shas(prior["runtime"]) == _shas(_hash_inputs(expected))
                 and _shas(prior["runtime"]).get(host_key) == host_sha
                 and prior["exe"].get("sha256") == _sha256(exe))
    if not unchanged:
        _build(inputs["GAME/Kingdom Rush.exe.bak"], inputs["Lumi_Nox/games/kingdom_rush/bridge.lua"],
               destination, exe)
    runtime_records = _hash_inputs(_runtime_files(destination))
    if runtime_records[host_key]["sha256"] != host_sha:
        raise RuntimeError("Verified host copy differs from assets/host.lua")
    exe_record = _stat(exe)
    exe_record["sha256"] = _sha256(exe)
    if _stat(exe) != {"size": exe_record["size"], "mtime_ns": exe_record["mtime_ns"]}:
        raise RuntimeError("Prepared engine changed while hashing")
    records = {"schema_version": SCHEMA_VERSION, "recipe_sha256": _recipe_sha256(), "inputs": hashed,
               "runtime": runtime_records, "exe": exe_record}
    _write_manifest(manifest, records)
    _VERIFIED[cache_key] = records
    return exe


RNG_TOKENS = ("audit", "isolate_sound", "stable_pairs")
# Every "+"-joined subset of the tokens, in this canonical order ("" = the unmodified game).
RNG_MODES = tuple("+".join(t for i, t in enumerate(RNG_TOKENS) if mask >> i & 1) for mask in range(8))
# Sound-RNG isolation plus creation-order iteration of coroutine/object-keyed tables.
DETERMINISTIC_MODE = "isolate_sound+stable_pairs"
# Native action catalogs the host can serve; "v1" (build/send only) is the replay-stable default,
# "v3" (elite stages) is v2 plus barracks rally points and dormant/boss-aware enemy data.
ACTION_SCOPES = ("v1", "v2", "v3")
MAIN_CAMPAIGN_LEVELS = 12


def level_scope(action_scope, level):
    """The scope a level is played in: v3 is the elite-stage scope, so a v3 job plays the main-campaign
    levels (1-12) in v2 and their validated plans and operator decisions stay exactly as before."""
    if action_scope == "v3" and isinstance(level, int) and not isinstance(level, bool) and level <= MAIN_CAMPAIGN_LEVELS:
        return "v2"
    return action_scope


# Game modes the director accepts with -mode: campaign, and a won level's Heroic and Iron challenges.
GAME_MODES = {1: "campaign", 2: "heroic", 3: "iron"}


class Worker:
    def __init__(self, seed=1001, port=9879, identity=None, level=1, difficulty=2, rng_mode="",
                 action_scope="v1", profile=None, mode=1):
        identity = identity or f"alpharush_rl_{seed}_{uuid.uuid4().hex[:12]}"
        # The identity names the save directory and log file, so keep it a plain file name.
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}", identity):
            raise ValueError("Worker identity must be a plain file name")
        if rng_mode not in RNG_MODES:
            raise ValueError(f"rng_mode must be one of {RNG_MODES}")
        if action_scope not in ACTION_SCOPES:
            raise ValueError(f"action_scope must be one of {ACTION_SCOPES}")
        if isinstance(mode, bool) or mode not in GAME_MODES:
            raise ValueError(f"mode must be one of {sorted(GAME_MODES)}")
        self.mode = mode
        self.rng_mode = rng_mode
        self.action_scope = action_scope
        # Headless (no joystick subsystem) in the isolated-sound modes: after hundreds of
        # short-lived parallel games, input-device enumeration stalled every new process at
        # startup. Other modes keep the original startup their recorded plans were made with.
        self.headless = "isolate_sound" in rng_mode.split("+")
        # Campaign progress written as save slot 1 before the first start (None = the game's new save).
        if profile is not None:
            from .campaign import check_profile
            profile = check_profile(profile)
        self.profile = profile
        self.seed, self.port, self.identity = seed, port, identity
        self.level, self.difficulty = level, difficulty
        self.process = self.sock = None
        self.token = uuid.uuid4().hex
        self.buffer = b""
        self.sequence = 0
        self.log = None

    @property
    def log_path(self):
        return ROOT / "runtime/rl-engine/logs" / f"{self.identity}.log"

    def start(self):
        exe = prepare()
        if self.profile is not None:
            from .campaign import write_profile
            write_profile(self.identity, self.profile)  # refuses an identity that already has a slot
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        environment = os.environ.copy()
        environment.update(ALPHARUSH_SEED=str(self.seed), ALPHARUSH_PORT=str(self.port),
                           ALPHARUSH_IDENTITY=self.identity, ALPHARUSH_TOKEN=self.token,
                           ALPHARUSH_RNG_MODE=self.rng_mode, ALPHARUSH_HEADLESS="1" if self.headless else "",
                           # Always explicit, so an inherited value never changes the catalog.
                           ALPHARUSH_ACTION_SCOPE=self.action_scope,
                           # The manifest-verified copy, not the editable asset, is executed.
                           ALPHARUSH_HOST_FILE=str(exe.parent / HOST_COPY))
        if self.headless:
            # OpenAL Soft's null output: sounds still play on an internal clock, but no system audio
            # session is opened (each game otherwise left a few handles in the audio service).
            environment["ALSOFT_DRIVERS"] = "null"
        else:
            environment.pop("ALSOFT_DRIVERS", None)
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        self.process = subprocess.Popen([str(exe), "-level", str(self.level), "-mode", str(self.mode),
                                        "-diff", str(self.difficulty), "-windowed"],
                                       cwd=exe.parent, env=environment,
                                       stdout=self.log, stderr=self.log, startupinfo=startup)
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Isolated game exited; inspect runtime/rl-engine/logs/{self.identity}.log")
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                self.sock.settimeout(30)
                result = self.rpc("hello")
                expected = "/" + self.identity
                if not result["save_directory"].replace("\\", "/").endswith(expected):
                    raise RuntimeError("Save isolation failed")
                if result.get("rng_mode", "") != self.rng_mode:
                    raise RuntimeError(f"Native RNG mode {result.get('rng_mode')!r} differs from {self.rng_mode!r}")
                if bool(result.get("headless", False)) != self.headless:
                    raise RuntimeError(f"Native headless mode {result.get('headless')!r} differs from {self.headless!r}")
                # A host that predates action scopes only serves the v1 catalog.
                if result.get("action_scope", "v1") != self.action_scope:
                    raise RuntimeError(f"Native action scope {result.get('action_scope')!r} "
                                       f"differs from {self.action_scope!r}")
                return result
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(0.2)
        raise RuntimeError("Isolated RPC start timed out")

    def rpc(self, action, **arguments):
        self.sequence += 1
        request = {"id": str(self.sequence), "action": action, **arguments, "token": self.token}
        self.sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode())
        while b"\n" not in self.buffer:
            data = self.sock.recv(1024 * 1024)
            if not data:
                raise ConnectionError("Isolated game disconnected")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if response.get("id") != request["id"]:
            # The host answers undecodable requests without an id.
            if response.get("id") is None and response.get("ok") is False:
                raise RuntimeError(response.get("error", "Native RPC rejected the request"))
            raise RuntimeError("RPC request ID mismatch")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Native RPC failed"))
        result = response["result"]
        if "native_wire" in result:
            return json.loads(result["native_wire"])
        return result

    def rng_audit(self):
        result = self.rpc("rng_audit")
        if result.get("counts") == []:
            result["counts"] = {}
        return result

    def meta(self):
        result = self.rpc("meta")
        # The Lua encoder writes empty tables as {}; these fields are lists.
        for key in ("errors", "locked_towers", "locked_powers", "main_campaign_levels", "heroes"):
            if result.get(key) == {}:
                result[key] = []
        return result

    def close(self):
        try:
            if self.sock:
                try:
                    if self.process and self.process.poll() is None:
                        # Ask for a normal shutdown first (it frees the audio device); kill below.
                        self.sock.settimeout(2)
                        self.rpc("quit")
                        self.process.wait(timeout=3)
                except Exception:
                    pass
                finally:
                    try:
                        self.sock.close()
                    finally:
                        self.sock = None
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.log:
                self.log.close()
                self.log = None

    def __enter__(self):
        try:
            self.start()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild even when every digest matches")
    args = parser.parse_args()
    if args.force and not args.prepare:
        parser.error("--force requires --prepare")
    if args.prepare:
        print(prepare(rebuild=True, force=args.force))
    else:
        with Worker() as worker:
            print(json.dumps(worker.rpc("diagnostics"), ensure_ascii=False, indent=2))
