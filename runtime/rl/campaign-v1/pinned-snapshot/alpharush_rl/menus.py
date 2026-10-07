"""Native legality is authoritative; uncertain actions are never inferred."""
from __future__ import annotations

import math

from .journal import canonical_json, sha256_data

SUPPORTED_ACTIONS = frozenset(("wait", "build_tower", "upgrade_tower", "send_wave",
                               "upgrade_power", "sell_tower", "use_power", "point_tower", "click_entity"))
# Global spells by native button number (game_gui.power_1/power_2).
POWER_TEXT = {1: "Cast rain of fire", 2: "Call reinforcements"}


def option_label(index: int) -> str:
    if index < 0:
        raise ValueError("option index must be nonnegative")
    label = ""
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        label = chr(65 + digit) + label
    return label


def _cost(item: dict) -> float | None:
    value = item.get("cost")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build_menu(state: dict, wait_ticks: int = 30) -> list[dict]:
    """Include the entire supported native action scope, with stable labels.

    More than 26 actions use AA, AB, ... instead of dropping legal actions.
    ``action_catalog`` must already reflect native upgrades/locks/affordability.
    Secondary checks below reject malformed catalog entries, stale holder IDs,
    blocked holders and actions unaffordable in the supplied snapshot.
    Menu actions keep only the keys a native command needs (a v2 build entry's
    ``target`` is dropped), so v1 snapshots give byte-identical menus.
    """
    if isinstance(wait_ticks, bool) or not isinstance(wait_ticks, int) or wait_ticks < 1:
        raise ValueError("wait_ticks must be a positive integer")
    terminal = bool(state.get("game_over") or state.get("level_won") or state.get("level_lost"))
    actions = [{"text": f"Wait {wait_ticks} native ticks", "action": {"action": "wait", "ticks": wait_ticks},
                "cost": 0.0, "legality_source": "environment_wait"}]
    if terminal:
        actions[0]["text"] = "Terminal state; no further game action"
    holders = {item.get("id"): item for item in state.get("holders", []) if isinstance(item, dict)}
    towers = {item.get("id"): item for item in state.get("towers", []) if isinstance(item, dict)}
    gold = state.get("gold")
    gold_known = not isinstance(gold, bool) and isinstance(gold, (int, float)) and math.isfinite(gold)
    seen = {canonical_json(actions[0]["action"])}
    for native in ([] if terminal else state.get("action_catalog", [])):
        if not isinstance(native, dict) or native.get("available") is not True:
            continue
        action_name = native.get("action")
        cost = _cost(native)
        if action_name == "build_tower":
            holder = holders.get(native.get("holder_id"))
            if holder is None or holder.get("blocked"):
                continue
            if native.get("tower_type") not in ("archer", "barrack", "mage", "engineer"):
                continue
            if cost is None or not gold_known or gold < cost:
                continue
            action = {"action": action_name, "holder_id": native["holder_id"], "tower_type": native["tower_type"]}
            text = f"Build {native['tower_type']} at holder {native['holder_id']} ({cost:g} gold)"
        elif action_name == "upgrade_tower":
            tower = towers.get(native.get("tower_id"))
            if tower is None or tower.get("is_special") or not isinstance(native.get("target"), str):
                continue
            if native["target"] == tower.get("template") or native["target"] in state.get("locked_towers", []):
                continue
            if cost is None or not gold_known or gold < cost:
                continue
            action = {"action": action_name, "tower_id": native["tower_id"], "target": native["target"]}
            text = f"Upgrade tower {native['tower_id']} to {native['target']} ({cost:g} gold)"
        elif action_name == "upgrade_power":
            tower = towers.get(native.get("tower_id"))
            if tower is None or not isinstance(native.get("power"), str):
                continue
            # A level's special tower is offered only the powers it lists (e.g. the sunray beam).
            if tower.get("is_special") and not any(isinstance(p, dict) and p.get("name") == native["power"]
                                                   for p in tower.get("powers") or []):
                continue
            if not native["power"] or cost is None or not gold_known or gold < cost:
                continue
            action = {"action": action_name, "tower_id": native["tower_id"], "power": native["power"]}
            text = (f"Upgrade power {native['power']} on tower {native['tower_id']} ({tower.get('template')})"
                    f" ({cost:g} gold)")
        elif action_name == "sell_tower":
            tower = towers.get(native.get("tower_id"))
            if tower is None or tower.get("is_special"):
                continue
            action, cost = {"action": action_name, "tower_id": native["tower_id"]}, 0.0
            text = f"Sell tower {native['tower_id']} ({tower.get('template')})"
        elif action_name == "use_power":
            power, x, y, anchor = (native.get(key) for key in ("power", "x", "y", "anchor_id"))
            if not _integer(power) or power not in POWER_TEXT or not (_integer(x) and _integer(y)):
                continue
            # The anchor must be a live enemy of this snapshot (read only here, so v1 states never touch it).
            enemies = state.get("enemies")
            if not _integer(anchor) or not isinstance(enemies, list) \
                    or not any(isinstance(e, dict) and e.get("id") == anchor for e in enemies):
                continue
            action, cost = {"action": action_name, "power": power, "x": x, "y": y, "anchor_id": anchor}, 0.0
            text = f"{POWER_TEXT[power]} at ({x},{y}) near enemy {anchor}"
        elif action_name == "point_tower":
            tower = towers.get(native.get("tower_id"))
            x, y, anchor = (native.get(key) for key in ("x", "y", "anchor_id"))
            if tower is None or not tower.get("is_special") or not (_integer(x) and _integer(y)):
                continue
            enemies = state.get("enemies")
            if not _integer(anchor) or not isinstance(enemies, list) \
                    or not any(isinstance(e, dict) and e.get("id") == anchor for e in enemies):
                continue
            action, cost = {"action": action_name, "tower_id": native["tower_id"], "x": x, "y": y,
                            "anchor_id": anchor}, 0.0
            text = f"Fire tower {native['tower_id']} ({tower.get('template')}) at enemy {anchor} ({x},{y})"
        elif action_name == "click_entity":
            entity, x, y = (native.get(key) for key in ("entity_id", "x", "y"))
            if not (_integer(entity) and _integer(x) and _integer(y)):
                continue
            action, cost = {"action": action_name, "entity_id": entity, "x": x, "y": y}, 0.0
            what = native.get("template") if isinstance(native.get("template"), str) else "entity"
            why = {"downed_boss": " to finish the downed boss", "tower_trap": " to free the trapped tower"}
            text = f"Click {what} {entity} at ({x},{y})" + why.get(native.get("kind"), "")
        elif action_name == "send_wave":
            if state.get("wave_ready") is not True:
                continue
            action, text, cost = {"action": "send_wave"}, "Send the next wave", 0.0
        else:
            continue
        key = canonical_json(action)
        if key not in seen:
            actions.append({"text": text, "action": action, "cost": cost,
                            "legality_source": "native_catalog"})
            seen.add(key)
    if not terminal and state.get("wave_ready") is True and canonical_json({"action": "send_wave"}) not in seen:
        actions.append({"text": "Send the next wave", "action": {"action": "send_wave"},
                        "cost": 0.0, "legality_source": "native_wave_ready"})
    # Keep wait first; native entity traversal order must not change prompt bytes.
    actions[1:] = sorted(actions[1:], key=lambda item: canonical_json(item["action"]))
    return [{"label": option_label(index), **item} for index, item in enumerate(actions)]


def validate_menu(menu: list[dict]) -> list[str]:
    if not isinstance(menu, list) or not menu:
        raise ValueError("menu must contain legal options")
    labels = []
    action_keys = []
    for item in menu:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ValueError("each option requires a label")
        label = item["label"]
        if not label or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in label):
            raise ValueError("option labels must be uppercase letters")
        action = item.get("action")
        if not isinstance(action, dict) or action.get("action") not in SUPPORTED_ACTIONS:
            raise ValueError("unsupported or missing native action")
        if item.get("available", True) is not True:
            raise ValueError("illegal choices must not be included in a legal menu")
        labels.append(label)
        action_keys.append(canonical_json(action))
    if len(set(labels)) != len(labels) or len(set(action_keys)) != len(action_keys):
        raise ValueError("duplicate label or action")
    return labels


def decision_prompt(state: dict, menu: list[dict]) -> str:
    validate_menu(menu)
    return canonical_json({"schema": "alpharush-option-v1", "state": state, "menu": menu})


def prompt_sha256(state: dict, menu: list[dict]) -> str:
    # The prompt itself is UTF-8 canonical JSON, not a hash of lossy summaries.
    import hashlib
    return hashlib.sha256(decision_prompt(state, menu).encode("utf-8")).hexdigest()


def menu_sha256(menu: list[dict]) -> str:
    validate_menu(menu)
    return sha256_data(menu)
