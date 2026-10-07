"""Fixed-tick native environment, completion receipts, and cold-process replay."""

import copy
import math
import time

from .engine import ACTION_SCOPES, Worker
from .journal import sha256_data
from .menus import build_menu

SCOPE_V1 = ("wait", "build_tower", "send_wave")
SCOPES = {"v1": SCOPE_V1, "v2": SCOPE_V1 + ("upgrade_tower", "upgrade_power", "sell_tower", "use_power",
                                           "point_tower", "click_entity")}
# Native ticks advanced after each verified action before its receipt is checked.
BUILD_TICKS = 180
UPGRADE_TICKS = 30
SELL_TICKS = 30
CLICK_TICKS = 30
ACTION_TICKS = {"build_tower": BUILD_TICKS, "upgrade_tower": UPGRADE_TICKS, "sell_tower": SELL_TICKS,
                "click_entity": CLICK_TICKS}


def observation(raw):
    state = copy.deepcopy(raw)
    for key in ("timestamp", "save_directory", "controlled", "life_lost_events"):
        state.pop(key, None)
    for key in ("towers", "holders", "enemies", "heroes", "action_catalog"):
        if state.get(key) == {}:
            state[key] = []
        if key != "action_catalog":
            state[key] = sorted(state.get(key, []), key=lambda e: e["id"])
    return state


def _tower(state, tower_id):
    return next((t for t in state["towers"] if t.get("id") == tower_id), None)


def _mesh(tower):
    """Holder mesh id under a tower, compared as text like the build receipt; None if unknown."""
    mesh = tower.get("holder_id") if tower else None
    return None if mesh is None else str(mesh)


def _power_level(state, tower_id, power):
    powers = (_tower(state, tower_id) or {}).get("powers")
    for entry in powers if isinstance(powers, list) else []:
        if isinstance(entry, dict) and entry.get("name") == power:
            return entry.get("level")
    return None


def _power_mode(state, power):
    panel = state.get("powers_ui")
    for button in panel if isinstance(panel, list) else []:
        if isinstance(button, dict) and button.get("id") == power:
            return button.get("mode")
    return None


def _spent(state, tower_id):
    """The tower's native ``spent`` if it is a finite number, else None."""
    value = (_tower(state, tower_id) or {}).get("spent")
    finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return value if finite else None


def _upgrade_tower_receipt(before, after, action, cost):
    # Native upgrades replace the entity: find the target template on the same holder mesh.
    mesh = _mesh(_tower(before, action["tower_id"]))
    upgraded = [t for t in after["towers"] if mesh is not None and str(t.get("holder_id")) == mesh
                and t.get("template") == action["target"]]
    return {"executed": bool(upgraded), "tower_id_after": upgraded[0]["id"] if upgraded else None}


def _upgrade_power_receipt(before, after, action, cost):
    # The GUI path levels the power once and counts exactly its price as spent on that tower.
    level_before = _power_level(before, action["tower_id"], action["power"])
    level_after = _power_level(after, action["tower_id"], action["power"])
    integers = all(isinstance(v, int) and not isinstance(v, bool) for v in (level_before, level_after))
    spent_before, spent_after = _spent(before, action["tower_id"]), _spent(after, action["tower_id"])
    paid = spent_before is not None and spent_after is not None and spent_after - spent_before == cost
    return {"executed": integers and level_after == level_before + 1 and paid,
            "power_level_before": level_before, "power_level_after": level_after,
            "spent_before": spent_before, "spent_after": spent_after}


def _sell_tower_receipt(before, after, action, cost):
    mesh = _mesh(_tower(before, action["tower_id"]))
    standing = [t for t in after["towers"] if str(t.get("holder_id")) == mesh]
    restored = [h for h in after["holders"] if str(h.get("mesh_id")) == mesh]
    return {"executed": mesh is not None and not standing and bool(restored),
            "holder_id_after": restored[0]["id"] if restored else None}


def _use_power_receipt(before, after, action, cost):
    mode = _power_mode(after, action["power"])
    return {"executed": mode == "cooldown", "power_mode_before": _power_mode(before, action["power"]),
            "power_mode_after": mode}


def _point_tower_receipt(before, after, action, cost):
    """The aimed tower fired: it recharges, so the catalog no longer offers to aim it."""
    def aims(state):
        return [item for item in state.get("action_catalog") or []
                if isinstance(item, dict) and item.get("action") == "point_tower"
                and item.get("tower_id") == action["tower_id"]]
    return {"executed": bool(aims(before)) and not aims(after), "aim_offers_after": len(aims(after))}


def _click_entity_receipt(before, after, action, cost):
    """A GUI click only sets the entity's ui.clicked flag (the host replied ok) and its script counts the
    clicks itself, so the evidence is that the click was offered before, and whether it is offered after."""
    def offered(state):
        return any(isinstance(item, dict) and item.get("action") == "click_entity"
                   and item.get("entity_id") == action["entity_id"] for item in state.get("action_catalog") or [])
    return {"executed": offered(before), "offered_after": offered(after)}


# v2 completion receipts: each returns {"executed": bool, ...evidence} from the before/after
# snapshots, the verified action and its menu cost.
V2_RECEIPTS = {"upgrade_tower": _upgrade_tower_receipt, "upgrade_power": _upgrade_power_receipt,
               "sell_tower": _sell_tower_receipt, "use_power": _use_power_receipt,
               "point_tower": _point_tower_receipt, "click_entity": _click_entity_receipt}


class NativeEnv:
    scope = list(SCOPE_V1)

    def __init__(self, seed=1001, level=1, port=9879, difficulty=2, identity=None, rng_mode="",
                 action_scope="v1", profile=None):
        if action_scope not in ACTION_SCOPES:
            raise ValueError(f"action_scope must be one of {ACTION_SCOPES}")
        # The v1 Worker call is unchanged (its default scope is v1); other scopes are passed through.
        scoped = {} if action_scope == "v1" else {"action_scope": action_scope}
        if profile is not None:
            scoped["profile"] = profile
        self.profile = profile
        self.worker = Worker(seed=seed, level=level, port=port, difficulty=difficulty, identity=identity,
                             rng_mode=rng_mode, **scoped)
        self.action_scope = action_scope
        self.scope = list(SCOPES[action_scope])
        self.seed, self.level, self.difficulty = seed, level, difficulty
        self.trace = []
        self.plan = []

    def reset(self):
        self.worker.close()
        self.worker.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            raw = self.worker.rpc("state")
            if raw.get("type") == "game_state":
                if self.profile is not None:
                    # The level must have loaded exactly the written campaign progress.
                    from .campaign import profile_matches
                    problems = profile_matches(self.profile, self.worker.meta())
                    if problems:
                        raise RuntimeError("Campaign profile not loaded: " + "; ".join(problems))
                self.state = observation(raw)
                self.trace = [{"kind": "reset", "state_sha256": sha256_data(self.state),
                               "tick": self.state["tick"], "seed": self.seed, "level": self.level}]
                self.plan = []
                return self.state
            time.sleep(0.05)
        raise RuntimeError("Native level initialization timed out")

    def advance(self, ticks, *, record_plan=True):
        before = self.state
        result = self.worker.rpc("step", ticks=ticks)
        after = observation(result["state"])
        advanced = result["tick_after"] - result["tick_before"]
        if advanced != ticks and not (result.get("terminated") and 0 <= advanced < ticks):
            raise RuntimeError("Native step count mismatch")
        self.state = after
        self.trace.append({"kind": "step", "ticks": ticks, "advanced_ticks": advanced, "tick": after["tick"],
                           "state_sha256": sha256_data(after)})
        if record_plan:
            self.plan.append({"ticks": ticks})
        return after

    def act(self, action):
        action = dict(action)
        before = self.state
        name = action["action"]
        if name == "wait":
            after = self.advance(action.get("ticks", 30), record_plan=False)
            receipt = {"accepted": True, "executed": True, "action": action,
                       "tick_before": before["tick"], "tick_after": after["tick"]}
        else:
            menu = build_menu(before)
            valid = [item for item in menu if item["action"] == action]
            if not valid:
                raise ValueError("Action is not in the native legal menu")
            if name not in self.scope:
                raise ValueError(f"Action is outside the {self.action_scope} action scope")
            cost = valid[0]["cost"]
            reply = self.worker.rpc(**action)
            if reply.get("type") != "ok":
                raise RuntimeError(f"Native action rejected: {reply}")
            after = self.advance(ACTION_TICKS.get(name, 2), record_plan=False)
            if name == "build_tower":
                holder = next(h for h in before["holders"] if h["id"] == action["holder_id"])
                towers = [t for t in after["towers"] if str(t["holder_id"]) == str(holder["mesh_id"])]
                executed = bool(towers and towers[0]["template"] == f"tower_{action['tower_type']}_1")
                gold_delta = before["gold"] - after["gold"]
                # Initial preparation receipts require exact prices, without enemy income.
                if before["wave"] == 0:
                    executed = executed and gold_delta == cost
                receipt = {"accepted": True, "executed": executed, "action": action,
                           "expected_cost": cost, "gold_delta": gold_delta,
                           "tower_id": towers[0]["id"] if towers else None,
                           "tick_before": before["tick"], "tick_after": after["tick"]}
            elif name == "send_wave":
                receipt = {"accepted": True, "executed": after["wave"] > before["wave"],
                           "action": action, "tick_before": before["tick"], "tick_after": after["tick"]}
            elif name in V2_RECEIPTS:
                receipt = {"accepted": True, **V2_RECEIPTS[name](before, after, action, cost), "action": action,
                           "expected_cost": cost, "gold_delta": before["gold"] - after["gold"],
                           "tick_before": before["tick"], "tick_after": after["tick"]}
            else:
                raise RuntimeError("Unverified action scope")
        if not receipt["executed"] and name != "wait" and (self.state.get("level_won") or self.state.get("level_lost")):
            # The native host accepted the action but the level ended inside the receipt window,
            # so its effect can no longer be observed; the game's own end is the outcome.
            receipt["executed"] = True
            receipt["terminal_during_receipt"] = True
        if not receipt["executed"]:
            raise RuntimeError(f"Native completion receipt failed: {receipt}")
        receipt["before_sha256"] = sha256_data(before)
        receipt["after_sha256"] = sha256_data(self.state)
        self.trace.append({"kind": "action", "receipt": receipt})
        self.plan.append({"action": action})
        return receipt

    def terminal(self):
        s = self.state
        if not (s.get("level_won") or s.get("level_lost")):
            return None
        return {"source": "native", "terminal": True, "level_won": bool(s["level_won"]),
                "level_lost": bool(s["level_lost"]), "lives": s["lives"],
                "wave": s["wave"], "tick": s["tick"], "state_sha256": sha256_data(s)}

    def rng_audit(self):
        """Cumulative native random() call counts per calling chunk (diagnostic RNG modes only)."""
        return self.worker.rng_audit()

    def meta(self):
        """Read-only level/campaign facts from the native host (not part of the trace)."""
        return self.worker.meta()

    def replay(self, plan):
        for command in plan:
            if "action" in command:
                self.act(command["action"])
            else:
                self.advance(command["ticks"])
        return self.state

    def close(self):
        self.worker.close()

    def __enter__(self):
        try:
            self.reset()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()
