"""Versioned terminal rewards for native KR1 campaign level 1, Normal.

Victory outranks defeat; among victories more surviving lives is better.
Time, money, stars and action counts never change the return. Engineering
invalidations and unfinished episodes are rejected rather than given rewards.
"""
from __future__ import annotations

import copy
import math

from .journal import sha256_data
from .pools import audit_dataset
from .trainer import EligibilityError

CONTRACT_NAME = "kr1-level1-terminal-v2"
_CONTRACT = {
    "schema_version": 2,
    "name": CONTRACT_NAME,
    "scope": {"game": "kr1-desktop-6.4.46", "level": 1, "difficulty": 2, "mode": "campaign"},
    "native_source": "native",
    "terminal_required": True,
    "max_lives": 20,
    "victory_base": 1.0,
    "victory_lives_denominator": 20.0,
    "defeat_return": -1.0,
    "formula": "victory: 1 + lives / 20; defeat: -1",
    "ignored_fields": ["gold", "tick", "seconds", "wave", "stars", "tower_count", "action_count"],
    "engineering_invalid_flags": ["truncated", "timed_out", "timeout", "engineering_invalid", "invalid"],
    "unfinished_episode": "reject_without_return",
}


def default_contract() -> dict:
    return copy.deepcopy(_CONTRACT)


def validate_reward_contract(contract: dict) -> str:
    """This name identifies one exact protocol, not tunable reward weights."""
    if not isinstance(contract, dict) or contract != _CONTRACT:
        raise EligibilityError("reward contract differs from frozen kr1-level1-terminal-v2 protocol")
    return sha256_data(contract)


def _validate_terminal(record: dict, contract: dict, *, require_context: bool) -> tuple[bool, int]:
    if not isinstance(record, dict) or record.get("source") != "native" or record.get("terminal") is not True:
        raise EligibilityError("native terminal outcome required; unfinished episode has no reward")
    for flag in contract["engineering_invalid_flags"]:
        if flag in record and record[flag] is not False:
            raise EligibilityError(f"engineering invalidation flag {flag}; no reward is assigned")
    if record.get("error") not in (None, "", False):
        raise EligibilityError("engineering error outcome cannot receive a reward")
    won, lost = record.get("level_won"), record.get("level_lost")
    if not isinstance(won, bool) or not isinstance(lost, bool) or won == lost:
        raise EligibilityError("exactly one native win/loss flag required")
    lives = record.get("lives")
    if isinstance(lives, bool) or not isinstance(lives, (int, float)) or not math.isfinite(lives) or lives != int(lives) or not 0 <= lives <= contract["max_lives"]:
        raise EligibilityError("native lives must be an integer in 0..20")
    if (won and lives == 0) or (lost and lives != 0):
        raise EligibilityError("native victory requires surviving lives; native defeat requires zero lives")
    for name, expected in (("level", 1), ("difficulty", 2)):
        if require_context or name in record:
            actual = record.get(name)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                raise EligibilityError(f"reward requires explicit native context {name}={expected}")
    if "mode" in record and record["mode"] != "campaign":
        raise EligibilityError("reward is scoped to campaign mode")
    return won, int(lives)


def score_level1_outcome(outcome: dict, contract: dict) -> float:
    """Score an authenticated native outcome with explicit level/difficulty.

    Receipt and replay eligibility is checked by the collector/trainer. When
    native_raw is attached, the derived context must preserve its terminal
    values exactly; resources cannot be relabeled into a better reward.
    """
    validate_reward_contract(contract)
    won, lives = _validate_terminal(outcome, contract, require_context=True)
    if "native_raw" in outcome:
        raw_won, raw_lives = _validate_terminal(outcome["native_raw"], contract, require_context=False)
        if (raw_won, raw_lives) != (won, lives):
            raise EligibilityError("derived outcome differs from preserved native_raw terminal values")
    return float(contract["victory_base"] + lives / contract["victory_lives_denominator"]) if won else float(contract["defeat_return"])


def score_verified_candidate(candidate: dict, contract: dict) -> float:
    if not isinstance(candidate, dict) or candidate.get("receipt_verified") is not True or candidate.get("replay_verified") is not True:
        raise EligibilityError("verified native action receipt and continuation replay required")
    if "native_outcomes" in candidate:
        raise EligibilityError("one native terminal outcome per candidate required")
    return score_level1_outcome(candidate.get("native_outcome"), contract)


def rescore_dataset(data: dict, contract: dict) -> dict:
    """Return a new dataset version, leaving all raw native evidence untouched.

    Existing v1 outcomes without explicit level/difficulty must be recollected
    or supplied with separately verified collector context before this call.
    This function neither creates that context nor invents terminal results.
    """
    scorer_sha = validate_reward_contract(contract)
    if not isinstance(data, dict) or data.get("source") != "real_game":
        raise EligibilityError("native real_game dataset required for reward revision")
    audit_dataset(data)
    if not data.get("groups"):
        raise EligibilityError("native continuation groups required for reward revision")
    original_sha = sha256_data(data)
    revised = copy.deepcopy(data)
    raw_pins = []
    for group in revised["groups"]:
        if isinstance(group.get("level"), bool) or group.get("level") != 1:
            raise EligibilityError("reward revision cannot include another training level")
        if len(group.get("candidates", [])) < 2:
            raise EligibilityError("at least two native continuations per fork required")
        labels = {choice["label"] for choice in group.get("menu", [])}
        seen = set()
        for candidate in group["candidates"]:
            if candidate.get("label") not in labels or candidate["label"] in seen:
                raise EligibilityError("distinct legal candidate label required")
            seen.add(candidate["label"])
            reward = score_verified_candidate(candidate, contract)
            native = candidate["native_outcome"]
            if native["level"] != group["level"] or native.get("seed", group["seed"]) != group["seed"]:
                raise EligibilityError("reward context differs from native group lineage")
            raw_sha = sha256_data(native)
            raw_pins.append({"fork_id": group["fork_id"], "label": candidate["label"], "native_outcome_sha256": raw_sha})
            candidate["return"] = reward
            candidate["reward_evaluation"] = {"contract_name": CONTRACT_NAME, "scorer_sha256": scorer_sha,
                                               "native_outcome_sha256": raw_sha, "return": reward}
        group["scorer_sha256"] = scorer_sha
    revised["scoring_contract"] = copy.deepcopy(contract)
    revised["scorer_sha256"] = scorer_sha
    revised["reward_revision"] = {
        "name": CONTRACT_NAME, "source_dataset_sha256": original_sha,
        "previous_scorer_sha256": data.get("scorer_sha256"), "scorer_sha256": scorer_sha,
        "raw_native_outcomes_unchanged": True, "raw_outcome_pins": raw_pins,
        "heldout_accessed": False,
    }
    if sha256_data(data) != original_sha:
        raise RuntimeError("source dataset changed during reward revision")
    return revised
