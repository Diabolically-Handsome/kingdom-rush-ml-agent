"""Training hygiene: both level IDs and RNG seeds are disjoint across pools."""
from __future__ import annotations

import json
from pathlib import Path

from .journal import canonical_bytes, sha256_data


class HygieneError(ValueError):
    pass


def _level(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or str(value) == "":
        raise HygieneError("level requires an explicit integer or string ID")
    return str(value)


def _seed(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HygieneError("seed requires an explicit nonnegative integer")
    return value


class PoolRegistry:
    def __init__(self, manifest: dict):
        if not isinstance(manifest, dict) or not isinstance(manifest.get("pools"), dict):
            raise HygieneError("pool registry requires a pools mapping")
        self.manifest = json.loads(canonical_bytes(manifest))
        self.pools = {}
        for name, pool in manifest["pools"].items():
            levels = [_level(value) for value in pool.get("levels", [])]
            seeds = [_seed(value) for value in pool.get("seeds", [])]
            if len(set(levels)) != len(levels) or len(set(seeds)) != len(seeds):
                raise HygieneError(f"duplicate level/seed within pool {name}")
            self.pools[name] = {"levels": set(levels), "seeds": set(seeds)}
        if any(name not in self.pools for name in ("train", "validation", "heldout")):
            raise HygieneError("train, validation and heldout pools must all be declared")
        self.nevertrain_levels = {_level(value) for value in manifest.get("nevertrain_levels", [])}
        self.nevertrain_seeds = {_seed(value) for value in manifest.get("nevertrain_seeds", [])}
        # Every non-training pool is intrinsically nevertrain, even if omitted
        # from the additional permanent register in a new project manifest.
        for name, pool in self.pools.items():
            if name != "train":
                self.nevertrain_levels.update(pool["levels"])
                self.nevertrain_seeds.update(pool["seeds"])
        self.audit()

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def sha256(self):
        return sha256_data(self.manifest)

    def audit(self) -> dict:
        names = list(self.pools)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                for field in ("levels", "seeds"):
                    overlap = self.pools[left][field] & self.pools[right][field]
                    if overlap:
                        raise HygieneError(f"{field} leakage between {left} and {right}: {sorted(overlap)}")
        if self.pools["train"]["levels"] & self.nevertrain_levels:
            raise HygieneError("training levels intersect nevertrain register")
        if self.pools["train"]["seeds"] & self.nevertrain_seeds:
            raise HygieneError("training seeds intersect nevertrain register")
        return {"passed": True, "registry_sha256": self.sha256,
                "level_disjoint": True, "seed_disjoint": True, "nevertrain_audited": True}

    def check_row(self, row: dict, expected_pool: str) -> None:
        if row.get("pool") != expected_pool or expected_pool not in self.pools:
            raise HygieneError(f"row must belong to {expected_pool} pool")
        level, seed = _level(row.get("level")), _seed(row.get("seed"))
        pool = self.pools[expected_pool]
        if level not in pool["levels"] or seed not in pool["seeds"]:
            raise HygieneError(f"row level/seed absent from {expected_pool} registry")
        if expected_pool == "train" and (level in self.nevertrain_levels or seed in self.nevertrain_seeds):
            raise HygieneError("nevertrain level or seed in a training row")


def audit_dataset(data: dict, registry: PoolRegistry | None = None) -> dict:
    registry = registry or PoolRegistry(data["pool_registry"])
    result = registry.audit()
    ids = set()
    groups = data.get("groups", [])
    if not isinstance(groups, list):
        raise HygieneError("groups must be a list")
    for group in groups:
        registry.check_row(group, "train")
        fork_id = group.get("fork_id")
        if not isinstance(fork_id, str) or not fork_id or fork_id in ids:
            raise HygieneError("training fork IDs must be unique nonempty strings")
        ids.add(fork_id)
    anchors = data.get("anchors", {})
    if set(anchors) - {"A1", "A2"}:
        raise HygieneError("unknown anchor set")
    for name in ("A1", "A2"):
        for row in anchors.get(name, []):
            registry.check_row(row, "train")
    for row in data.get("validation_forks", []):
        registry.check_row(row, "validation")
        if any(key in row for key in ("candidates", "return", "reward", "native_outcome", "outcomes")):
            raise HygieneError("validation forks may contain only state/prompt/options, never continuation rewards")
    if any(data.get(key) for key in ("heldout", "heldout_forks", "heldout_groups", "heldout_outcomes")):
        raise HygieneError("heldout evidence may not be supplied to training")
    result.update({"groups": len(groups), "anchors_A1": len(anchors.get("A1", [])),
                   "anchors_A2": len(anchors.get("A2", [])),
                   "validation_forks": len(data.get("validation_forks", [])),
                   "heldout_accessed": False})
    return result


def claim_heldout_once(path: str | Path, frozen_judge_sha256: str, candidate_sha256: str) -> dict:
    """Reserve the single heldout judgment before accessing any outcomes.

    Exclusive creation makes a second judgment refuse, including after a crash.
    This records a reservation, not a successful or favorable evaluation.
    """
    for value in (frozen_judge_sha256, candidate_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise HygieneError("frozen judge and candidate SHA256 required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    claim = {"schema_version": 1, "judge_sha256": frozen_judge_sha256,
             "candidate_sha256": candidate_sha256, "status": "reserved_once"}
    try:
        with target.open("xb") as stream:
            stream.write(canonical_bytes(claim) + b"\n")
    except FileExistsError as exc:
        raise HygieneError("heldout judgment already reserved; adaptive reuse is forbidden") from exc
    return claim
