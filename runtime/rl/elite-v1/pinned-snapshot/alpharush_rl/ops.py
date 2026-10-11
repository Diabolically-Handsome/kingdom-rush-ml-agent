"""Independent AlphaRush experiment gates; stdlib only, no model/GPU imports.

`launch` executes a picklable callback in a spawned child and enforces wall time.
`job_context` is the cooperative alternative: call context.check() each iteration.
Neither API certifies game fidelity or improves a model by itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import time
import uuid


class GateRefused(RuntimeError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise GateRefused(f"Path leaves AlphaRush workspace: {path}")
    return path


def _load(config_path):
    cp = Path(config_path).resolve()
    cfg = _read(cp)
    root = (cp.parent / cfg.get("workspace", "..")).resolve()
    if not cp.is_relative_to(root):
        raise GateRefused("Configuration is outside its declared workspace")
    return cp, cfg, root


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab", buffering=0) as stream:
        stream.write(encoded)
        os.fsync(stream.fileno())


def _ledger(path):
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("not an object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            raise GateRefused(f"Ledger line {n} is unreadable; repair with an appended correction: {exc}") from exc
    return rows


def _usage(rows, kind):
    opens, closes = {}, {}
    for row in rows:
        if row.get("event") == "open":
            if row["run_id"] in opens:
                raise GateRefused("Duplicate ledger open")
            opens[row["run_id"]] = row
        elif row.get("event") == "close":
            if row["run_id"] in closes:
                raise GateRefused("Duplicate ledger close")
            closes[row["run_id"]] = row
    total = 0.0
    if set(closes) - set(opens):
        raise GateRefused("Ledger close has no corresponding open")
    for rid, row in opens.items():
        if rid not in closes:
            raise GateRefused(f"Unclosed job {rid}; inspect it before another launch")
        if closes[rid].get("kind") != row.get("kind"):
            raise GateRefused("Ledger open/close job kinds differ")
        seconds = float(closes[rid]["wall_seconds"])
        if seconds < 0 or not seconds < float("inf"):
            raise GateRefused("Invalid ledger duration")
        if row.get("kind") == kind:
            total += seconds
    return total


def _pool_problems(pools):
    issues, seen_seeds, seen_levels = [], set(), set()
    for role in ("train", "validation", "heldout", "never_train"):
        p = pools.get("pools", {}).get(role, {})
        seeds, levels = p.get("seeds", []), p.get("levels", [])
        for name, values, seen in (("seeds", seeds, seen_seeds), ("levels", levels, seen_levels)):
            if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
                issues.append(f"{role} {name}: nonnegative integers required")
            if len(set(values)) != len(values) or set(values) & seen:
                issues.append(f"{role} {name}: pool overlap or duplicate")
            seen.update(values)
    if pools.get("anchors") != "train_only":
        issues.append("A1/A2 anchors must use training records only")
    if pools.get("unlisted_levels") != "never_train" or pools.get("unlisted_seeds") != "never_train":
        issues.append("Unlisted seeds and levels must be never_train")
    return issues


def audit_records(records, pools, role="train"):
    """Validate every gradient-bearing row, including A1 and A2 anchors."""
    if role not in ("train", "validation", "heldout"):
        raise GateRefused(f"Unknown data role: {role}")
    issues = _pool_problems(pools)
    p = pools["pools"][role]
    seen = set()
    for n, row in enumerate(records):
        if not isinstance(row, dict):
            issues.append(f"{role}[{n}] must be an object")
            continue
        seed, level = row.get("seed"), row.get("level", row.get("level_id"))
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in p["seeds"]:
            issues.append(f"{role}[{n}] seed {seed!r} outside {role} pool")
        if isinstance(level, bool) or not isinstance(level, int) or level not in p["levels"]:
            issues.append(f"{role}[{n}] level {level!r} outside {role} pool")
        rid = row.get("fork_id", row.get("id"))
        if rid is not None:
            if rid in seen:
                issues.append(f"{role}[{n}] repeated id {rid!r}")
            seen.add(rid)
        if row.get("anchor_set") in ("A1", "A2") and role != "train":
            issues.append(f"{role}[{n}] anchor cannot use {role} data")
    return issues


def audit_training_data(groups, anchors, pools):
    return audit_records(list(groups) + list(anchors), pools, "train")


def _dataset_rows(path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc = _read(path)
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict) and "groups" in doc:
        anchors = doc.get("anchors", [])
        if isinstance(anchors, dict):
            if set(anchors) - {"A1", "A2"}:
                raise GateRefused("Unknown anchor set in native dataset")
            anchors = list(anchors.get("A1", [])) + list(anchors.get("A2", []))
        elif not isinstance(anchors, list):
            raise GateRefused("Anchors must be A1/A2 mapping or legacy list")
        # Validation states monitor guards; they never enter gradient audit.
        return list(doc["groups"]) + list(anchors)
    raise GateRefused("Dataset must be JSONL rows, a JSON list, or groups/anchors object")


def freeze(config_path, extra_paths=()):
    """Explicit pinning step; never automatically re-freeze to bypass a refusal."""
    cp, cfg, root = _load(config_path)
    state = _inside(root, cfg["state_dir"])
    if (state / "job.lock").exists():
        raise GateRefused("Cannot change pins while a job holds the experiment lock")
    paths = set(cfg["required_pins"]) | {str(cp.relative_to(root))} | set(map(str, extra_paths))
    pins = {}
    for relative in sorted(paths):
        p = _inside(root, relative)
        if not p.is_file():
            raise GateRefused(f"Cannot pin missing file: {p}")
        pins[p.relative_to(root).as_posix()] = sha256_file(p)
    manifest = dict(schema_version=1, created_at=_utc(), files=pins)
    destination = _inside(root, cfg["pins_manifest"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, destination)
    _append(state / "pins-history.jsonl", dict(event="freeze", time=_utc(), manifest_sha256=sha256_file(destination), files=pins))
    return manifest


def preflight(config_path, kind="cpu-smoke", evidence_path=None, data_path=None, check_pins=True):
    """Read-only: no directory creation, locks, ledger writes, GPU query, or model."""
    cp, cfg, root = _load(config_path)
    issues, result = [], dict(ok=False, kind=kind, check_only=True, config_sha256=sha256_file(cp))
    state = _inside(root, cfg["state_dir"])
    if kind not in cfg.get("jobs", {}):
        issues.append(f"Unknown job kind {kind}")
        job = {}
    else:
        job = cfg["jobs"][kind]
    if kind.startswith("gpu"):
        if not cfg.get("model", {}).get("selected"):
            issues.append("GPU model is pending user selection; GPU launch is disabled")
        elif not cfg.get("model", {}).get("gpu_enabled"):
            issues.append("General GPU jobs are disabled here; use the separately bounded inference comparison launcher")
        issues.append("This stdlib CPU launcher does not dispatch GPU jobs")
    if job.get("enabled", True) is False:
        issues.append(f"{kind} is disabled")
    if "train" in kind and not cfg.get("formal_training_enabled", False):
        issues.append("Formal training is disabled; only bounded CPU smoke is configured")
    cap = job.get("max_wall_seconds", 0)
    if isinstance(cap, bool) or not isinstance(cap, (float, int)) or not 0 < cap < 60:
        issues.append("Deployment CPU jobs require 0 < max_wall_seconds < 60")
    total_cap = job.get("total_wall_seconds", 0)
    if isinstance(total_cap, bool) or not isinstance(total_cap, (float, int)) or not 0 < total_cap < float("inf"):
        issues.append("Cumulative CPU budget must be a finite positive number")
    if cfg.get("max_concurrent_jobs") != 1:
        issues.append("Deployment supports exactly one independent job at a time")
    for stop in (state / "STOP", state / "ENGINEERING-STOP"):
        if stop.exists():
            issues.append(f"Stop file exists: {stop}")
    if (state / "job.lock").exists():
        issues.append("Another AlphaRush job holds the lock (or an interrupted job needs inspection)")
    try:
        used = _usage(_ledger(state / "ledger.jsonl"), kind)
        result["used_wall_seconds"] = used
        result["max_wall_seconds"] = cap
        if used + cap > total_cap:
            issues.append("Insufficient remaining cumulative wall budget for the full job cap")
    except (GateRefused, KeyError, TypeError, ValueError) as exc:
        issues.append(str(exc))
    pp = _inside(root, cfg["pools_path"])
    try:
        pools = _read(pp)
        result["pools_sha256"] = sha256_file(pp)
        issues.extend(_pool_problems(pools))
    except (OSError, ValueError, TypeError) as exc:
        pools = None
        issues.append(f"Pool manifest unreadable: {exc}")
    manifest = _inside(root, cfg["pins_manifest"])
    if check_pins:
        try:
            pinset = _read(manifest)["files"]
            required = set(cfg["required_pins"]) | {cp.relative_to(root).as_posix()}
            if not required.issubset(pinset):
                issues.append("Pin manifest does not cover all required code/config files")
            for relative, wanted in pinset.items():
                if sha256_file(_inside(root, relative)) != wanted:
                    issues.append(f"Code/config SHA mismatch: {relative}")
            result["pins_sha256"] = sha256_file(manifest)
        except (OSError, ValueError, TypeError, KeyError, GateRefused) as exc:
            issues.append(f"No verified pin manifest; run explicit freeze first: {exc}")
    evidence = None
    if job.get("requires_native_evidence"):
        ep = _inside(root, evidence_path or cfg["native_evidence_path"])
        try:
            evidence = _read(ep)
            result["evidence_sha256"] = sha256_file(ep)
            if evidence.get("schema_version") != 1:
                issues.append("Native evidence schema_version must be 1")
            for name in cfg["required_native_gates"]:
                gate = evidence.get("gates", {}).get(name, {})
                if gate.get("status") != "verified":
                    issues.append(f"Native environment gate pending: {name}")
                    continue
                artifact = _inside(root, gate.get("artifact_path", ""))
                if not gate.get("artifact_sha256") or sha256_file(artifact) != gate["artifact_sha256"]:
                    issues.append(f"Native evidence artifact SHA mismatch: {name}")
            if "pins_sha256" in result and evidence.get("pins_sha256") not in (None, result["pins_sha256"]):
                issues.append("Native evidence belongs to another code/config pin set")
        except (OSError, ValueError, TypeError, KeyError, GateRefused) as exc:
            issues.append(f"Native evidence unavailable: {exc}")
    if data_path:
        try:
            dp = _inside(root, data_path)
            rows = _dataset_rows(dp)
            result["data_sha256"] = sha256_file(dp)
            if job.get("data_kind") == "native":
                if not rows:
                    issues.append("Native dataset contains no training rows")
                if any(row.get("data_kind") != "native" for row in rows):
                    issues.append("Native RL requires explicitly marked native rows; synthetic or unknown provenance refused")
                if pools is not None:
                    issues.extend(audit_records(rows, pools, "train"))
                document = _read(dp) if dp.suffix.lower() == ".json" else None
                if isinstance(document, dict) and document.get("pool_registry") is not None:
                    registered = json.dumps(document["pool_registry"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    expected = json.dumps(pools, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if registered != expected:
                        issues.append("Dataset pool registry differs from the frozen external pool manifest")
                    validation_rows = document.get("validation_forks", [])
                    issues.extend(audit_records(validation_rows, pools, "validation"))
                    for row in validation_rows:
                        if any(k in row for k in ("candidates", "return", "reward", "native_outcome", "outcomes")):
                            issues.append("Validation states must contain no continuation/reward fields")
                if job.get("requires_native_evidence") and evidence is not None:
                    if evidence.get("dataset_sha256") != result["data_sha256"]:
                        issues.append("Native dataset file SHA differs from the environment evidence")
                    if evidence.get("dataset_path") and _inside(root, evidence["dataset_path"]) != dp:
                        issues.append("Native dataset path differs from the environment evidence")
            elif any(row.get("data_kind") != "synthetic" for row in rows):
                issues.append("CPU optimizer sanity accepts synthetic rows only")
        except (OSError, ValueError, TypeError, KeyError, GateRefused) as exc:
            issues.append(f"Data audit failed: {exc}")
    elif kind == "cpu-rl-smoke":
        issues.append("CPU native RL smoke requires an audited dataset path")
    result["issues"] = issues
    result["ok"] = not issues
    return result


@dataclass
class RunContext:
    run_id: str
    kind: str
    output_dir: Path
    state_dir: Path
    deadline: float
    config_sha256: str

    def check(self):
        if time.monotonic() >= self.deadline:
            raise GateRefused("Job wall-clock budget exhausted")
        if any((self.state_dir / x).exists() for x in ("STOP", "ENGINEERING-STOP")):
            raise GateRefused("AlphaRush stop file was created")

    def record(self, event, **details):
        self.check()
        _append(self.output_dir / "events.jsonl", dict(event=event, run_id=self.run_id, time=_utc(), **details))


@contextmanager
def _exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise GateRefused(f"Experiment lock already exists: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid(), token=token, time=_utc()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield token
    finally:
        # Never remove a lock which another process replaced.
        if path.exists() and _read(path).get("token") == token:
            path.unlink()


@contextmanager
def job_context(config_path, kind="cpu-smoke", evidence_path=None, data_path=None):
    pf = preflight(config_path, kind, evidence_path, data_path)
    if not pf["ok"]:
        raise GateRefused("; ".join(pf["issues"]))
    _, cfg, root = _load(config_path)
    state = _inside(root, cfg["state_dir"])
    with _exclusive_lock(state / "job.lock"):
        # Recheck all gates after acquiring the lock; this job's own lock is expected.
        current = preflight(config_path, kind, evidence_path, data_path)
        current["issues"] = [s for s in current["issues"] if s != "Another AlphaRush job holds the lock (or an interrupted job needs inspection)"]
        if current["issues"]:
            raise GateRefused("; ".join(current["issues"]))
        rid = f"{kind}-{uuid.uuid4().hex}"
        output = state / "runs" / rid
        output.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        cap = float(cfg["jobs"][kind]["max_wall_seconds"])
        ctx = RunContext(rid, kind, output, state, start + cap, pf["config_sha256"])
        _append(state / "ledger.jsonl", dict(event="open", run_id=rid, kind=kind, pid=os.getpid(), time=_utc(), cap_seconds=cap, preflight=current))
        status, error = "ok", None
        try:
            yield ctx
            ctx.check()
        except BaseException as exc:
            status, error = "stopped" if isinstance(exc, GateRefused) else "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.monotonic() - start
            _append(state / "ledger.jsonl", dict(event="close", run_id=rid, kind=kind, time=_utc(), wall_seconds=elapsed, status=status, error=error))
            receipt = dict(schema_version=1, run_id=rid, status=status, wall_seconds=elapsed, config_sha256=pf["config_sha256"], pins_sha256=pf.get("pins_sha256"), data_sha256=pf.get("data_sha256"), evidence_sha256=pf.get("evidence_sha256"), verified=False, meaning="Process/budget receipt only; callback must verify its scientific result")
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def _worker(callback, context, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ALPHARUSH_FORBID_GPU"] = "1"
    try:
        context.check()
        result_queue.put((True, callback(context)))
    except BaseException as exc:
        result_queue.put((False, f"{type(exc).__name__}: {exc}"))


def launch(config_path, callback, kind="cpu-smoke", evidence_path=None, data_path=None):
    """Spawn a CPU callback with a hard deadline and STOP polling.

    Callback must be a picklable module-level function; use a guarded __main__ on
    Windows. Context.check() is still useful inside every optimizer iteration.
    """
    with job_context(config_path, kind, evidence_path, data_path) as ctx:
        mp = multiprocessing.get_context("spawn")
        results = mp.Queue()
        process = mp.Process(target=_worker, args=(callback, ctx, results), name=ctx.run_id)
        process.start()
        _append(ctx.output_dir / "events.jsonl", dict(event="child-start", pid=process.pid, time=_utc()))
        try:
            while process.is_alive():
                ctx.check()
                process.join(min(0.05, max(0.001, ctx.deadline - time.monotonic())))
            if process.exitcode != 0:
                raise GateRefused(f"Job process exited with code {process.exitcode}")
            try:
                success, result = results.get(timeout=1)
            except queue.Empty as exc:
                raise GateRefused("Job process produced no result") from exc
            if not success:
                raise GateRefused(result)
            return result
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            results.close()
            results.join_thread()


def claim_heldout_evaluation(config_path, evaluation_id, policy_sha256, records):
    """Append a once-only holdout reservation before any outcome is read.

    A stopped evaluation remains consumed. This never starts a game or reads its
    outcomes; a new reserve pool requires a separately documented protocol.
    """
    _, cfg, root = _load(config_path)
    state = _inside(root, cfg["state_dir"])
    pools = _read(_inside(root, cfg["pools_path"]))
    records = list(records)
    issues = audit_records(records, pools, "heldout")
    if not evaluation_id or not isinstance(policy_sha256, str) or len(policy_sha256) != 64 or any(c not in "0123456789abcdef" for c in policy_sha256):
        issues.append("Heldout reservation requires evaluation id and frozen policy SHA256")
    if not records:
        issues.append("Heldout reservation requires explicit seed/level records")
    if issues:
        raise GateRefused("; ".join(issues))
    audit = state / "heldout-audit.jsonl"
    with _exclusive_lock(state / "heldout.lock"):
        rows = _ledger(audit)
        if any(r.get("pool_id") == pools["pool_id"] and r.get("event") == "reserved" for r in rows):
            raise GateRefused("This heldout pool has already been reserved for its one evaluation")
        row = dict(event="reserved", time=_utc(), pool_id=pools["pool_id"], evaluation_id=evaluation_id, policy_sha256=policy_sha256, records=records)
        _append(audit, row)
    return row
