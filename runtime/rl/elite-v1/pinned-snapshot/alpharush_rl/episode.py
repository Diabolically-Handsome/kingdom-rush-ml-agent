"""Whole-episode decision loop over a native env; no rewards, no rule fallback.

The policy is asked only at decision events: a non-trivial menu that is new,
whose option set differs from the last decision's (a wait-only menu in between
clears that set), or that has not been re-asked for ``max_interval_ticks``.
A spell option counts by its kind only (``option_key``): its moving anchor alone
never makes a new decision event.
Otherwise the loop waits one fixed block. A policy
error or illegal choice voids the episode; it is never replaced by a scripted
action, so no decision is credited to the wrong source.

Callers own the env lifecycle (``run_episode`` resets but never closes)::

    env = NativeEnv(seed=1001, level=1, difficulty=2)
    try:
        result = run_episode(env, policy, EpisodeProtocol(), seed=1001, level=1)
    finally:
        env.close()

``contextlib.closing(env)`` works too. ``with NativeEnv(...)`` does not fit,
because its ``__enter__`` already resets and the episode would reset again.
"""
from __future__ import annotations

import copy
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from .journal import canonical_json, sha256_data
from .menus import build_menu, menu_sha256

SCHEMA = "alpharush-episode-v1"
STATUSES = ("terminal", "stalled", "void")


def _positive_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


@dataclass(frozen=True)
class EpisodeProtocol:
    name: str = "kr1-episode-v1"
    wait_ticks: int = 60          # forced wait block and the menu's wait option
    max_interval_ticks: int = 600 # re-ask at least this often while the menu is non-trivial
    max_ticks: int = 60000        # native ticks since reset; reaching it -> stalled
    max_decisions: int = 400
    first_wave_deadline_ticks: int | None = None  # wave still 0 after this many ticks -> stalled
    # A wait-only menu forgets the last decided option set, so options that come
    # back after a gap (e.g. the next wave_ready) are asked about again.
    reset_on_trivial_menu: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("protocol name must be a nonempty string")
        for field in ("wait_ticks", "max_interval_ticks", "max_ticks", "max_decisions"):
            if not _positive_int(getattr(self, field)):
                raise ValueError(f"{field} must be a positive integer")
        deadline = self.first_wave_deadline_ticks
        if deadline is not None and not _positive_int(deadline):
            raise ValueError("first_wave_deadline_ticks must be None or a positive integer")
        if not isinstance(self.reset_on_trivial_menu, bool):
            raise ValueError("reset_on_trivial_menu must be a boolean")


class Policy(Protocol):
    name: str

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict: ...


def _json_copy(value: Any, what: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be finite JSON-serializable data") from exc


def _validate_distribution(distribution: Any, labels: list[str]) -> dict:
    if not isinstance(distribution, dict):
        raise ValueError("distribution must be None or a dict")
    given = distribution.get("labels")
    if not isinstance(given, (list, tuple)) or list(given) != labels:
        raise ValueError("distribution labels must equal the menu labels in menu order")
    p = distribution.get("p")
    if not isinstance(p, (list, tuple)) or len(p) != len(labels):
        raise ValueError("distribution p must give exactly one probability per label")
    values = []
    for value in p:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError("distribution p must be finite and nonnegative")
        values.append(float(value))
    if abs(math.fsum(values) - 1.0) > 1e-6:
        raise ValueError("distribution p must sum to 1 within 1e-6")
    # Extra fields (logp, model, prompt hash, ...) are kept but must be plain JSON.
    rest = _json_copy({k: v for k, v in distribution.items() if k not in ("labels", "p")}, "distribution")
    return {**rest, "labels": list(labels), "p": values}


def validate_choice(choice: Any, menu: list[dict]) -> dict:
    """Normalized copy of a policy reply, or ValueError. Never repairs a reply."""
    if not isinstance(choice, dict):
        raise ValueError("policy reply must be a dict")
    labels = [item["label"] for item in menu]
    label = choice.get("label")
    if not isinstance(label, str) or label not in labels:
        raise ValueError(f"label {label!r} is not in the legal menu")
    provenance = choice.get("provenance")
    if not isinstance(provenance, str) or not provenance:
        raise ValueError("provenance must be a nonempty string")
    distribution = choice.get("distribution")
    if distribution is not None:
        distribution = _validate_distribution(distribution, labels)
    meta = choice.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    return {"label": label, "provenance": provenance, "distribution": distribution,
            "meta": _json_copy(meta, "meta")}


def option_key(action: dict) -> str:
    """Identity of a menu option when deciding whether the option set changed.

    A spell's anchor (x, y, anchor_id) follows the moving enemies almost every tick, so a
    use_power option counts only as {action, power}; the executed action keeps its anchor.
    No other action is reduced, so v1 option sets are unchanged.
    """
    if action.get("action") == "use_power":
        action = {"action": "use_power", "power": action.get("power")}
    return canonical_json(action)


def _act(env, action: dict, tick: int) -> str | None:
    """Execute one native action; return a void reason instead of raising."""
    try:
        env.act(copy.deepcopy(action))
    except Exception as exc:
        return f"native_action_error: {type(exc).__name__}: {exc}"
    # Every native action advances the clock; a frozen clock would loop forever.
    if env.terminal() is None and env.state["tick"] <= tick:
        return "native_action_error: native tick did not advance"
    return None


def run_episode(env, policy: Policy, protocol: EpisodeProtocol, *, seed: int, level: int,
                difficulty: int = 2, episode_id: str | None = None,
                check: Callable[[], None] = lambda: None,
                clock: Callable[[], float] = time.monotonic, observe_meta: bool = False) -> dict:
    """Play one episode from ``env.reset()`` until terminal, stalled or void.

    ``check()`` is the caller's STOP/deadline hook; its exceptions propagate.
    No reward is computed here and the env is not closed here. With
    ``observe_meta`` the env's read-only ``meta()`` is recorded once after reset
    as ``level_meta`` (a failure is recorded, never fatal).
    """
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    policy_name = getattr(policy, "name", None)
    if not isinstance(policy_name, str) or not policy_name:
        raise TypeError("policy must have a nonempty string name")
    started = clock()
    episode_id = uuid.uuid4().hex if episode_id is None else str(episode_id)
    check()  # do not even start the native level after a STOP
    state = env.reset()
    start_tick = state["tick"]
    level_meta = None
    if observe_meta:
        reader = getattr(env, "meta", None)
        try:
            level_meta = _json_copy(reader(), "level meta") if callable(reader) else {"error": "env has no meta()"}
        except Exception as exc:
            level_meta = {"error": f"{type(exc).__name__}: {exc}"}
    wait = {"action": "wait", "ticks": protocol.wait_ticks}
    decisions: list[dict] = []
    forced_waits = 0
    status = stall_reason = void_reason = None
    # Option set and tick of the last decision. A non-trivial menu never equals
    # the empty set, so the first one is always a decision event.
    decided_keys, decided_tick = frozenset(), None
    while True:
        check()
        state = env.state
        tick = state["tick"]
        if env.terminal():
            status = "terminal"
            break
        elapsed = tick - start_tick
        if elapsed >= protocol.max_ticks:
            status, stall_reason = "stalled", "max_ticks"
            break
        deadline = protocol.first_wave_deadline_ticks
        if deadline is not None and state.get("wave", 0) == 0 and elapsed >= deadline:
            status, stall_reason = "stalled", "first_wave_not_started"
            break
        menu = build_menu(state, wait_ticks=protocol.wait_ticks)
        option_keys = frozenset(option_key(item["action"]) for item in menu[1:])
        if not option_keys and protocol.reset_on_trivial_menu:
            # Nothing decided is still offered: options that reappear later
            # (e.g. the next wave_ready with no affordable build) are new.
            decided_keys = frozenset()
        due = bool(option_keys) and (option_keys != decided_keys
                                     or tick - decided_tick >= protocol.max_interval_ticks)
        if not due:
            void_reason = _act(env, wait, tick)
            if void_reason:
                status = "void"
                break
            forced_waits += 1
            continue
        if len(decisions) >= protocol.max_decisions:
            status, stall_reason = "stalled", "max_decisions"
            break
        context = {"episode_id": episode_id, "decision_index": len(decisions), "level": level,
                   "seed": seed, "difficulty": difficulty, "tick": tick, "protocol": protocol.name}
        state_sha256 = sha256_data(state)
        # Copies keep a misbehaving policy from mutating the env's state or menu.
        asked = clock()
        try:
            reply = policy.choose(copy.deepcopy(state), copy.deepcopy(menu), dict(context))
        except Exception as exc:
            status, void_reason = "void", f"policy_error: {type(exc).__name__}: {exc}"
            break
        latency = clock() - asked
        try:
            choice = validate_choice(reply, menu)
        except ValueError as exc:
            status, void_reason = "void", f"invalid_choice: {exc}"
            break
        item = next(option for option in menu if option["label"] == choice["label"])
        decisions.append({"index": len(decisions), "tick": tick, "wave": state.get("wave"),
                          "gold": state.get("gold"), "lives": state.get("lives"),
                          "state_sha256": state_sha256, "menu_sha256": menu_sha256(menu),
                          "labels": [option["label"] for option in menu], "label": choice["label"],
                          "action": copy.deepcopy(item["action"]), "provenance": choice["provenance"],
                          "distribution": choice["distribution"], "latency_seconds": latency,
                          "meta": choice["meta"]})
        decided_keys, decided_tick = option_keys, tick
        void_reason = _act(env, item["action"], tick)
        if void_reason:
            status = "void"
            break
    outcome = env.terminal()
    final = env.state
    result = {"schema": SCHEMA, "protocol": asdict(protocol), "policy": policy_name,
              "episode_id": episode_id, "seed": seed, "level": level, "difficulty": difficulty,
              "status": status, "stall_reason": stall_reason, "void_reason": void_reason,
              "outcome": copy.deepcopy(outcome) if outcome else None,
              "final_tick": final["tick"], "final_state_sha256": sha256_data(final),
              "final_summary": final_summary(final),
              "decisions": decisions, "n_decisions": len(decisions), "n_forced_waits": forced_waits,
              "plan": copy.deepcopy(env.plan), "trace_sha256": sha256_data(env.trace),
              "wall_seconds": clock() - started}
    if observe_meta:
        result["level_meta"] = level_meta
    return result


def final_summary(state: dict) -> dict:
    """Small readable digest of the last state, also for stalled/void episodes."""
    towers = state.get("towers") or []
    return {key: state.get(key) for key in ("wave", "wave_total", "lives", "gold", "level_won", "level_lost")} | {
        "towers": sorted(t.get("template") for t in towers if isinstance(t, dict) and isinstance(t.get("template"), str))}


def replay_unsupported(result: dict) -> bool:
    return result.get("status") == "void" and str(result.get("void_reason") or "").startswith("native_action_error")


def replay_verify(make_env: Callable[[], Any], result: dict) -> dict:
    """Cold replay of ``result["plan"]`` in a fresh env; compares trace and outcome.

    A plan the env refuses (e.g. a tampered illegal action) is reported as not
    verified with ``replay_error`` rather than raised. A void episode whose native
    action failed after advancing the game has trace steps with no plan entry, so it
    cannot be replayed; it is reported as skipped (``replay_verified`` is None).
    """
    if replay_unsupported(result):
        return {"replay_verified": None, "replay_trace_sha256": None, "outcome_match": None,
                "replay_error": None, "replay_skipped": "void_after_native_side_effect"}
    env = make_env()
    try:
        env.reset()
        try:
            env.replay(copy.deepcopy(result["plan"]))
        except Exception as exc:
            return {"replay_verified": False, "replay_trace_sha256": sha256_data(env.trace),
                    "outcome_match": False, "replay_error": f"{type(exc).__name__}: {exc}"}
        replay_sha256 = sha256_data(env.trace)
        outcome_match = env.terminal() == result["outcome"]
        return {"replay_verified": replay_sha256 == result["trace_sha256"] and outcome_match,
                "replay_trace_sha256": replay_sha256, "outcome_match": outcome_match,
                "replay_error": None}
    finally:
        env.close()
