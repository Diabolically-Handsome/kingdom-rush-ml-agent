"""Phase governance for campaign-v1: pins, seed pools, budgets and job ledger.

Stdlib only; never starts a game, model or GPU. ``preflight_phase`` is read-only.
``job_context`` is cooperative: call ``context.check()`` inside every loop.
A receipt certifies process/budget accounting only, never a scientific result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import time
import uuid

from .journal import Journal, JournalError
from .ops import GateRefused, _exclusive_lock, _inside, _utc, sha256_file

GLOBAL_STATE = "runtime/rl"
STOP_NAMES = ("STOP", "ENGINEERING-STOP")
POOL_ROLES = ("train", "evaluation", "final_campaign_run", "retired")
UNLISTED_ROLE = "never_use"
LOCK_ISSUE = "Another AlphaRush phase job holds the lock (or an interrupted job needs inspection)"
_REQUIRED_KEYS = ("phase_id", "workspace", "state_dir", "pins_manifest", "pools_path")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    """Temporary file plus os.replace, so readers never see a partial document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_phase(path):
    cp = Path(path).resolve()
    raw = cp.read_bytes()
    cfg = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise GateRefused("Phase configuration must be a schema_version 1 object")
    for key in _REQUIRED_KEYS:
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise GateRefused(f"Phase configuration requires a nonempty string {key}")
    root = (cp.parent / cfg["workspace"]).resolve()
    if not cp.is_relative_to(root):
        raise GateRefused("Phase configuration is outside its declared workspace")
    for key in ("state_dir", "pins_manifest", "pools_path"):
        _inside(root, cfg[key])
    # The digest is taken from the exact bytes that were parsed.
    return cp, cfg, root, hashlib.sha256(raw).hexdigest()


def load_phase(path):
    """Return (config_path, cfg, root); root is config dir / workspace and contains it."""
    cp, cfg, root, _ = _load_phase(path)
    return cp, cfg, root


def _pattern(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise GateRefused(f"Freeze patterns must be nonempty posix strings: {value!r}")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive or ".." in value.split("/"):
        raise GateRefused(f"Freeze pattern leaves the workspace: {value!r}")
    return value


def _match(parts, pattern):
    """Segment glob where ``**`` spans zero or more whole segments."""
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_match(parts[i:], rest) for i in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], head) and _match(parts[1:], rest)


def collect_pin_paths(cfg, root, config_path=None):
    """Sorted workspace-relative posix paths selected by freeze.include minus exclude.

    The pools file is always pinned; the phase config is pinned when its path is
    given (freeze/preflight always pass it). Files under state_dir and the pins
    manifest itself are never pinned because jobs write there.
    """
    root = Path(root).resolve()
    freeze = cfg.get("freeze")
    if not isinstance(freeze, dict) or not isinstance(freeze.get("include"), list) \
            or not isinstance(freeze.get("exclude", []), list):
        raise GateRefused("Phase freeze requires include/exclude pattern lists")
    include = [_pattern(value) for value in freeze["include"]]
    exclude = [_pattern(value).split("/") for value in freeze.get("exclude", [])]
    state = _inside(root, cfg["state_dir"])
    manifest = _inside(root, cfg["pins_manifest"])
    found = set()
    for pattern in include:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise GateRefused(f"Pinned path leaves AlphaRush workspace: {candidate}")
            if resolved == manifest or resolved.is_relative_to(state):
                continue
            relative = resolved.relative_to(root).as_posix()
            if not any(_match(relative.split("/"), parts) for parts in exclude):
                found.add(relative)
    found.add(_inside(root, cfg["pools_path"]).relative_to(root).as_posix())
    if config_path is not None:
        found.add(_inside(root, config_path).relative_to(root).as_posix())
    return sorted(found)


def freeze_phase(path):
    """Explicit pinning step; never re-freeze automatically to bypass a refusal."""
    cp, cfg, root, _ = _load_phase(path)
    state = _inside(root, cfg["state_dir"])
    if (state / "job.lock").exists():
        raise GateRefused("Cannot change pins while a phase job holds the lock")
    files = {}
    for relative in collect_pin_paths(cfg, root, config_path=cp):
        target = _inside(root, relative)
        if not target.is_file():
            raise GateRefused(f"Cannot pin missing file: {relative}")
        files[relative] = sha256_file(target)
    history = Journal(state / "pins-history.jsonl")  # verifies the chain before any write
    manifest = dict(schema_version=1, phase_id=cfg["phase_id"], created_at=_utc(), files=files)
    destination = _inside(root, cfg["pins_manifest"])
    _write_json(destination, manifest)
    history.append("freeze", dict(phase_id=cfg["phase_id"], time=manifest["created_at"],
                                  manifest_sha256=sha256_file(destination), files=files))
    return manifest


def pool_problems(pools):
    """Seeds are nonnegative integers, unique, and disjoint across every pool."""
    if not isinstance(pools, dict) or not isinstance(pools.get("pools"), dict):
        return ["Pool manifest requires a pools mapping"]
    issues, seeds_by_role = [], {}
    for role in POOL_ROLES:
        if role not in pools["pools"]:
            issues.append(f"Pool {role} must be declared")
    for role, pool in pools["pools"].items():
        if role not in POOL_ROLES:
            issues.append(f"Unknown pool {role}; campaign roles are {', '.join(POOL_ROLES)}")
        seeds = pool.get("seeds") if isinstance(pool, dict) else None
        if not isinstance(seeds, list):
            issues.append(f"{role} seeds: list required")
            continue
        valid = [x for x in seeds if not isinstance(x, bool) and isinstance(x, int) and x >= 0]
        if len(valid) != len(seeds):
            issues.append(f"{role} seeds: nonnegative integers required")
        if len(set(valid)) != len(valid):
            issues.append(f"{role} seeds: duplicate seed")
        seeds_by_role[role] = set(valid)
    names = list(seeds_by_role)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = seeds_by_role[left] & seeds_by_role[right]
            if overlap:
                issues.append(f"Seed pools {left} and {right} overlap: {sorted(overlap)}")
    evaluation = pools["pools"].get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("never_in_gradients") is not True:
        issues.append("Evaluation seeds must be declared never_in_gradients")
    if pools.get("unlisted_seeds") != UNLISTED_ROLE:
        issues.append(f"Unlisted seeds must be {UNLISTED_ROLE}")
    return issues


def seed_role(pools, seed):
    """train/evaluation/final_campaign_run/retired, else never_use."""
    issues = pool_problems(pools)
    if issues:
        raise GateRefused("; ".join(issues))
    if isinstance(seed, bool) or not isinstance(seed, int):
        return UNLISTED_ROLE
    for role, pool in pools["pools"].items():
        if seed in pool["seeds"]:
            return role
    return UNLISTED_ROLE


def ledger_usage(entries, job_kind):
    """(used_jobs, used_wall_seconds) for job_kind; every open must be closed."""
    opens, closes = {}, {}
    for n, row in enumerate(entries):
        kind, payload = row.get("kind"), row.get("payload")
        if kind not in ("open", "close") or not isinstance(payload, dict) or not isinstance(payload.get("run_id"), str):
            raise GateRefused(f"Ledger entry {n} is not a phase open/close record")
        rid = payload["run_id"]
        bucket = opens if kind == "open" else closes
        if rid in bucket:
            raise GateRefused(f"Duplicate ledger {kind} for {rid}")
        if kind == "close" and rid not in opens:
            raise GateRefused(f"Ledger close has no preceding open: {rid}")
        bucket[rid] = payload
    used_jobs, used_wall = 0, 0.0
    for rid, row in opens.items():
        if rid not in closes:
            raise GateRefused(f"Unclosed job {rid}; inspect it before another launch")
        close = closes[rid]
        if close.get("job_kind") != row.get("job_kind"):
            raise GateRefused(f"Ledger open/close job kinds differ for {rid}")
        seconds = close.get("wall_seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise GateRefused(f"Invalid ledger duration for {rid}")
        if row.get("job_kind") == job_kind:
            used_jobs += 1
            used_wall += float(seconds)
    return used_jobs, used_wall


def _number(value, *, integer, allow_zero=False):
    if isinstance(value, bool):
        return False
    if integer and not isinstance(value, int):
        return False
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value >= 0 if allow_zero else value > 0


def _caps(job, issues):
    caps = {key: job.get(key) for key in ("max_wall_seconds", "total_wall_seconds", "max_jobs", "max_games",
                                          "gpu", "optimizer_steps", "serial")}
    valid = True
    for key, integer, allow_zero in (("max_wall_seconds", False, False), ("total_wall_seconds", False, False),
                                     ("max_jobs", True, False), ("max_games", True, True)):
        if not _number(caps[key], integer=integer, allow_zero=allow_zero):
            issues.append(f"Job cap {key} must be a finite {'integer' if integer else 'number'}"
                          f" {'>= 0' if allow_zero else '> 0'}")
            valid = False
    return caps, valid


def _stop_dirs(root, state):
    global_state = _inside(root, GLOBAL_STATE)
    return (state,) if global_state == state else (state, global_state)


def preflight_phase(path, job_kind):
    """Read-only: no directory creation, lock, ledger write, GPU query, model or game."""
    result = dict(ok=False, check_only=True, issues=[], phase_id=None, job_kind=job_kind, pins_sha256=None,
                  config_sha256=None, pools_sha256=None, used_jobs=None, used_wall_seconds=None, caps=None)
    issues = result["issues"]
    try:
        cp, cfg, root, result["config_sha256"] = _load_phase(path)
    except (OSError, ValueError, GateRefused) as exc:
        issues.append(f"Phase configuration unusable: {exc}")
        return result
    result["phase_id"] = cfg["phase_id"]
    state = _inside(root, cfg["state_dir"])
    jobs = cfg.get("jobs")
    job = jobs.get(job_kind) if isinstance(jobs, dict) else None
    caps_valid = False
    if not isinstance(job, dict):
        issues.append(f"Unknown job kind {job_kind}")
    else:
        if job.get("enabled") is not True:
            pending = job.get("pending")
            issues.append(f"{job_kind} is not enabled" + (f": {pending}" if pending else ""))
        gpu = job.get("gpu")
        # A job never starts GPU work itself. "external-inference" declares that it queries the
        # separately started local scoring server (inference only, no weight updates).
        if gpu is not False and gpu != "external-inference":
            issues.append("Phase-0 tools never dispatch GPU work; the job must declare gpu=false "
                          "(or gpu=\"external-inference\" to query the separately started scoring server)")
        result["caps"], caps_valid = _caps(job, issues)
    if cfg.get("max_concurrent_jobs") != 1:
        issues.append("A phase supports exactly one job at a time (max_concurrent_jobs=1)")

    dirs = _stop_dirs(root, state)
    for directory in dirs:
        for name in STOP_NAMES:
            if (directory / name).exists():
                issues.append(f"Stop file exists: {directory / name}")
    if (state / "job.lock").exists():
        issues.append(LOCK_ISSUE)
    for directory in dirs[1:]:
        if (directory / "job.lock").exists():
            issues.append(f"Global AlphaRush job lock exists: {directory / 'job.lock'}")

    ledger = state / "ledger.jsonl"
    try:
        # Journal() only creates the parent directory, which exists when the ledger does.
        entries = Journal(ledger).entries() if ledger.exists() else []
        used_jobs, used_wall = ledger_usage(entries, job_kind)
        result.update(used_jobs=used_jobs, used_wall_seconds=used_wall)
        if caps_valid:
            caps = result["caps"]
            if used_jobs >= caps["max_jobs"]:
                issues.append(f"No remaining {job_kind} jobs: used {used_jobs} of {caps['max_jobs']}")
            if used_wall + caps["max_wall_seconds"] > caps["total_wall_seconds"]:
                issues.append("Insufficient remaining cumulative wall budget for the full job cap")
    except (GateRefused, JournalError, OSError, ValueError, TypeError) as exc:
        issues.append(f"Ledger refused: {exc}")

    try:
        expected = collect_pin_paths(cfg, root, config_path=cp)
    except (GateRefused, OSError, ValueError) as exc:
        issues.append(f"Pin set unavailable: {exc}")
        expected = None
    manifest_path = _inside(root, cfg["pins_manifest"])
    try:
        manifest = _read_json(manifest_path)
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict) or manifest.get("schema_version") != 1 \
                or manifest.get("phase_id") != cfg["phase_id"]:
            issues.append("Pin manifest has another schema or phase; run explicit freeze first")
        else:
            result["pins_sha256"] = sha256_file(manifest_path)
            if expected is not None:
                for relative in sorted(set(expected) - set(files)):
                    issues.append(f"File not covered by pin manifest: {relative}")
                for relative in sorted(set(files) - set(expected)):
                    issues.append(f"Pinned file no longer in pin set: {relative}")
            for relative, wanted in sorted(files.items()):
                try:
                    actual = sha256_file(_inside(root, relative))
                except (OSError, GateRefused) as exc:
                    issues.append(f"Pinned file unreadable: {relative}: {exc}")
                    continue
                if actual != wanted:
                    issues.append(f"Code/config SHA mismatch: {relative}")
    except (OSError, ValueError) as exc:
        issues.append(f"No verified pin manifest; run explicit freeze first: {exc}")

    pools_path = _inside(root, cfg["pools_path"])
    try:
        pools = _read_json(pools_path)
        result["pools_sha256"] = sha256_file(pools_path)
        issues.extend(pool_problems(pools))
    except (OSError, ValueError) as exc:
        issues.append(f"Pool manifest unreadable: {exc}")
    result["ok"] = not issues
    return result


@dataclass
class PhaseRunContext:
    run_id: str
    output_dir: Path
    state_dir: Path
    deadline: float
    max_games: int
    job_kind: str = ""
    stop_dirs: tuple = ()
    games_played: int = 0
    _events: object = None  # the run's events journal, opened (and verified) once

    def check(self):
        if time.monotonic() >= self.deadline:
            raise GateRefused("Phase job wall-clock budget exhausted")
        for directory in (self.state_dir, *self.stop_dirs):
            for name in STOP_NAMES:
                if (directory / name).exists():
                    raise GateRefused(f"AlphaRush stop file was created: {directory / name}")

    @property
    def games_remaining(self):
        return max(0, self.max_games - self.games_played)

    def claim_game(self):
        """Count one game before it starts; refuses beyond the max_games hard cap."""
        self.check()
        if self.games_played >= self.max_games:
            raise GateRefused(f"Phase job max_games cap reached: {self.max_games}")
        self.games_played += 1
        return self.games_played - 1

    def record(self, event, **details):
        self.check()
        if self._events is None:
            self._events = Journal(self.output_dir / "events.jsonl")
        return self._events.append(str(event), dict(run_id=self.run_id, time=_utc(), **details))


@contextmanager
def job_context(path, job_kind):
    pf = preflight_phase(path, job_kind)
    if not pf["ok"]:
        raise GateRefused("; ".join(pf["issues"]))
    _, cfg, root, _ = _load_phase(path)
    state = _inside(root, cfg["state_dir"])
    with _exclusive_lock(state / "job.lock"):
        # Recheck all gates after acquiring the lock; this job's own lock is expected.
        current = preflight_phase(path, job_kind)
        issues = [issue for issue in current["issues"] if issue != LOCK_ISSUE]
        if issues:
            raise GateRefused("; ".join(issues))
        current.update(issues=[], ok=True, own_lock_held=True)
        _, cfg, root, config_sha256 = _load_phase(path)
        if config_sha256 != current["config_sha256"] or current["pins_sha256"] != pf["pins_sha256"]:
            raise GateRefused("Phase configuration or pins changed while acquiring the lock")
        caps = current["caps"]
        ledger = Journal(state / "ledger.jsonl")
        rid = f"{job_kind}-{uuid.uuid4().hex}"
        output = state / "runs" / rid
        output.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        ctx = PhaseRunContext(rid, output, state, start + float(caps["max_wall_seconds"]), int(caps["max_games"]),
                              job_kind=job_kind, stop_dirs=_stop_dirs(root, state)[1:])
        opened = ledger.append("open", dict(run_id=rid, job_kind=job_kind, phase_id=cfg["phase_id"],
                                            pid=os.getpid(), time=_utc(), caps=caps, preflight=current))
        status, error = "ok", None
        try:
            yield ctx
            ctx.check()
        except BaseException as exc:
            status = "stopped" if isinstance(exc, GateRefused) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.monotonic() - start
            receipt = dict(schema_version=1, phase_id=cfg["phase_id"], run_id=rid, job_kind=job_kind,
                           status=status, error=error, wall_seconds=elapsed, games_played=ctx.games_played,
                           max_games=ctx.max_games, config_sha256=current["config_sha256"],
                           pins_sha256=current["pins_sha256"], pools_sha256=current["pools_sha256"],
                           ledger_open_sha256=opened["sha256"], ledger_close_sha256=None, verified=False,
                           meaning="Process/budget receipt only; the job must verify its own result")
            try:
                closed = ledger.append("close", dict(run_id=rid, job_kind=job_kind, time=_utc(),
                                                     wall_seconds=elapsed, status=status, error=error,
                                                     games_played=ctx.games_played))
                receipt["ledger_close_sha256"] = closed["sha256"]
            finally:
                _write_json(output / "receipt.json", receipt)
