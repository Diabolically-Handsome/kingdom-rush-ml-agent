"""What a decision policy may see; the native observation keeps privileged fields.

``policy_state`` is byte-equivalent to ``policy_state`` in
``tools/collect-level1-rewards.py`` so v1 prompts reproduce the level-1 data.
"""
from __future__ import annotations

import copy
import hashlib

from .journal import canonical_json
from .menus import decision_prompt, validate_menu

# Future wave_db group counts; never disclosed to an option policy.
PRIVILEGED_KEYS = ("level_path_wave_counts",)


def policy_state(raw: dict) -> dict:
    result = copy.deepcopy(raw)
    for key in PRIVILEGED_KEYS:
        result.pop(key, None)
    return result


def prompt_v1(state: dict, menu: list[dict]) -> str:
    """The level-1 training/request user text (schema alpharush-option-v1)."""
    return decision_prompt(policy_state(state), menu)


def compact_state(state: dict) -> dict:
    # action_catalog duplicates the legal menu, which the prompt carries anyway.
    result = policy_state(state)
    result.pop("action_catalog", None)
    return result


def prompt_v2(state: dict, menu: list[dict]) -> str:
    validate_menu(menu)
    return canonical_json({"schema": "alpharush-option-v2", "state": compact_state(state),
                           "menu": [{"label": item["label"], "text": item["text"]} for item in menu]})


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
