"""Append-only evidence journal; integrity is separate from engine replay parity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_data(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_state(state: dict) -> dict:
    """Sort native entity arrays, preserving every state field including tick/RNG.

    Missing native clocks/RNG stay missing. A matching digest does not certify
    that those unobserved parts of the engine matched.
    """
    result = json.loads(canonical_json(state))
    for key in ("towers", "holders", "enemies", "heroes"):
        if isinstance(result.get(key), list):
            result[key].sort(key=lambda item: (str(item.get("id", "")), canonical_json(item)))
    return result


def snapshot_hash(state: dict) -> str:
    return sha256_data(canonical_state(state))


class JournalError(ValueError):
    pass


class Journal:
    """Single-writer chain, refusing to append after any detectable tampering."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._status = self._stamp = None
        self.verify()

    def _file_stamp(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise JournalError(f"blank journal line {line_number}")
                try:
                    entries.append(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise JournalError(f"invalid journal line {line_number}") from exc
        return entries

    def verify(self, expected_tip: str | None = None) -> dict:
        previous = "0" * 64
        rows = self.entries()
        for sequence, row in enumerate(rows):
            if not isinstance(row, dict):
                raise JournalError(f"non-object journal entry {sequence}")
            if row.get("seq") != sequence or row.get("previous_sha256") != previous:
                raise JournalError(f"broken chain at entry {sequence}")
            digest = row.get("sha256")
            unsigned = {key: value for key, value in row.items() if key != "sha256"}
            try:
                correct = sha256_data(unsigned)
            except (ValueError, TypeError) as exc:
                raise JournalError(f"noncanonical journal entry {sequence}") from exc
            if digest != correct:
                raise JournalError(f"hash mismatch at entry {sequence}")
            previous = digest
        if expected_tip is not None and previous != expected_tip:
            raise JournalError("journal tip differs from pinned tip (possible truncation)")
        status = {"integrity_verified": True, "entries": len(rows), "tip_sha256": previous,
                  "engine_replay_verified": False}
        self._status, self._stamp = dict(status), self._file_stamp()
        return status

    def append(self, kind: str, payload: dict) -> dict:
        # The chain was verified in full when this writer last saw the file; re-verify only if the file
        # changed since (size or modification time), so an append costs O(1) instead of a full pass.
        status = self._status if self._status is not None and self._stamp == self._file_stamp() else self.verify()
        # Hash exactly what is stored: JSON turns integer keys into strings, which sort differently
        # once there are ten or more of them ({1: .., 10: ..} vs {"1": .., "10": ..}).
        payload = json.loads(canonical_json(payload))
        row = {"schema_version": 1, "seq": status["entries"], "kind": str(kind),
               "payload": payload, "previous_sha256": status["tip_sha256"]}
        row["sha256"] = sha256_data(row)
        encoded = canonical_bytes(row)
        with self.path.open("ab") as stream:
            stream.write(encoded + b"\n")
            stream.flush()
        self._status = {**status, "entries": status["entries"] + 1, "tip_sha256": row["sha256"]}
        self._stamp = self._file_stamp()
        return row


def verify_journal(path: str | Path, expected_tip: str | None = None) -> dict:
    return Journal(path).verify(expected_tip)
