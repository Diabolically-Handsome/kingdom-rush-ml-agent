"""Action scope v2: menus, Worker scope handshake and NativeEnv receipts.

A fake host (FakeWorker) replaces engine.Worker inside the real NativeEnv and
models host.lua's v2 catalog, upgrade_to/sell/power/spell semantics and
powers_ui. Nothing here starts the game, a process or a socket. The v1 menu
regression reads the real level-1 dataset read-only and skips when it is absent.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from alpharush_rl import engine, env as env_module
from alpharush_rl.episode import EpisodeProtocol, replay_verify, run_episode
from alpharush_rl.journal import canonical_json, sha256_data
from alpharush_rl.menus import SUPPORTED_ACTIONS, build_menu, menu_sha256, validate_menu
from alpharush_rl.scripted_policies import make_policy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runtime/rl/level1-24b-phase1/data"
V2_ONLY = ("upgrade_tower", "upgrade_power", "sell_tower", "use_power")

# Fake game data in the shape of data.tower_menus_data and the entity templates.
TOWERS = {  # template -> (tower type, tower level, price)
    "tower_archer_1": ("archer", 1, 70), "tower_archer_2": ("archer", 2, 110), "tower_archer_3": ("archer", 3, 160),
    "tower_ranger": ("ranger", 1, 230), "tower_musketeer": ("musketeer", 1, 230),
    "tower_barrack_1": ("barrack", 1, 70), "tower_mage_1": ("mage", 1, 100), "tower_engineer_1": ("engineer", 1, 125),
}
KINDS = ("archer", "barrack", "mage", "engineer")
BUILD_NAMES = {f"tower_build_{kind}": f"tower_{kind}_1" for kind in KINDS}
SELL = ("tw_sell", None)
MENUS = {
    "holder": {1: [("tw_upgrade", f"tower_build_{kind}") for kind in KINDS]},
    "archer": {1: [("tw_upgrade", "tower_archer_2"), SELL], 2: [("tw_upgrade", "tower_archer_3"), SELL],
               3: [("tw_upgrade", "tower_ranger"), ("tw_upgrade", "tower_musketeer"), SELL]},
    "barrack": {1: [SELL]}, "mage": {1: [SELL]}, "engineer": {1: [SELL]},
    "ranger": {1: [("upgrade_power", "poison"), ("upgrade_power", "thorn"), SELL]},
    "musketeer": {1: [("upgrade_power", "sniper"), SELL]},
}
# name -> (max_level, price_base, price_inc). The GUI charges price_base for the first level and
# price_inc for each later one (poison: 250 then 250, where base + inc * level would be 500).
POWERS = {"ranger": {"poison": (2, 250, 250), "thorn": (1, 300, 0)},
          "musketeer": {"sniper": (2, 250, 150)}}
# Per-tick damage; the special elf tower is idle so spells always have targets in the tests.
DAMAGE = {"archer": 1, "barrack": 1, "mage": 1, "engineer": 1, "elf": 0, "ranger": 2, "musketeer": 2}
BLAST_DAMAGE = 30
MATCH_KEYS = ("action", "holder_id", "tower_id", "tower_type", "target", "power", "x", "y", "anchor_id")
COOLDOWN_TICKS = 300
PATH_TICKS = 900


def price_of(arg):
    return TOWERS[BUILD_NAMES.get(arg, arg)][2]


def power_price(kind, name, level):
    """The GUI's (and host.lua's) price of a power's next level: price_base first, then price_inc."""
    _, base, inc = POWERS[kind][name]
    return base if level == 0 else inc


def bridge_next_cost(kind, name, level):
    """What the bridge's state still reports as next_cost (base + inc * level); never charged."""
    _, base, inc = POWERS[kind][name]
    return base + inc * level


class FakeWorker:
    """host.lua RPC semantics for both scopes over a small deterministic level.

    Commands only set native flags (upgrade_to/sell) or act at once (power
    upgrade, spell, wave), exactly like the host; flags resolve on the next tick.
    ``faults`` injects native misbehaviour after an accepted command.
    """

    def __init__(self, seed=1001, port=9879, identity=None, level=1, difficulty=2, rng_mode="",
                 action_scope="v1", *, gold=2000, waves=3, locked=(), modes=None, faults=(), leak_v2=False):
        self.kwargs = {"seed": seed, "port": port, "identity": identity, "level": level,
                       "difficulty": difficulty, "rng_mode": rng_mode, "action_scope": action_scope}
        self.seed, self.action_scope = seed, action_scope
        self.config = {"gold": gold, "waves": waves, "locked": list(locked),
                       "modes": dict(modes or {1: "ready", 2: "ready"})}
        self.faults, self.leak_v2 = set(faults), leak_v2
        self.calls = []
        self.starts = self.closes = 0

    # -- lifecycle -------------------------------------------------------
    def start(self):
        self.starts += 1
        c = self.config
        holders = [{"id": i, "template": "tower_holder", "blocked": False, "mesh_id": str(i),
                    "x": 200 * i, "y": 300, "path_score": 10 - i} for i in range(1, 5)]
        holders.append({"id": 5, "template": "tower_holder_elf", "blocked": False, "mesh_id": "5",
                        "x": 900, "y": 500, "path_score": 0})
        holders.append({"id": 7, "template": "tower_holder", "blocked": True, "mesh_id": "7",
                        "x": 950, "y": 500, "path_score": 9})
        special = {"id": 6, "template": "tower_elf", "type": "elf", "level": 1, "holder_id": "6",
                   "is_special": True, "spent": 0, "x": 600, "y": 500, "path_score": 1}
        self.s = {"tick": 1, "gold": c["gold"], "lives": 20, "wave": 0, "wave_total": c["waves"],
                  "level_won": False, "level_lost": False, "game_over": False,
                  "locked_towers": list(c["locked"]), "towers": [special], "holders": holders,
                  "enemies": [], "heroes": []}
        self.modes = dict(c["modes"])
        self.pending, self.steps, self.cooldowns, self.blasts, self.queue = {}, {}, {}, [], []
        self.next_id = 100

    def close(self):
        self.closes += 1

    # -- host catalog ----------------------------------------------------
    def _wave_ready(self):
        s = self.s
        return (not s["game_over"] and s["wave"] < s["wave_total"] and not self.queue
                and (s["wave"] == 0 or not s["enemies"]))

    def _catalog_v1(self):
        out = [{"action": "build_tower", "holder_id": h["id"], "tower_type": kind, "target": f"tower_build_{kind}",
                "cost": price_of(f"tower_{kind}_1"), "available": True, "legal": True}
               for h in self.s["holders"] if not h["blocked"] for kind in KINDS
               if price_of(f"tower_{kind}_1") <= self.s["gold"]]
        if self._wave_ready():
            out.append({"action": "send_wave", "cost": 0, "available": True, "legal": True})
        return out

    def _anchors(self):
        alive = sorted((e for e in self.s["enemies"] if e.get("path_progress") is not None),
                       key=lambda e: (-e["path_progress"], e["id"]))
        chosen = []
        for enemy in alive:
            if all(math.hypot(enemy["x"] - c["x"], enemy["y"] - c["y"]) >= 60 for c in chosen):
                chosen.append(enemy)
                if len(chosen) == 3:
                    break
        return chosen

    def _catalog_v2(self):
        s, out = self.s, []
        gold, locked = s["gold"], set(s["locked_towers"])
        if s["game_over"]:
            return out
        for h in s["holders"]:
            if h["blocked"] or h["template"] != "tower_holder" or h["id"] in self.pending:
                continue
            for _, arg in MENUS["holder"][1]:
                kind = arg[len("tower_build_"):]
                if arg in locked or f"tower_{kind}_1" in locked or price_of(arg) > gold:
                    continue
                out.append({"action": "build_tower", "holder_id": h["id"], "tower_type": kind, "target": arg,
                            "cost": price_of(arg)})
        for t in s["towers"]:
            if t["id"] in self.pending or t["type"] not in MENUS:
                continue
            for kind, arg in MENUS[t["type"]].get(t["level"], []):
                if kind == "tw_upgrade" and arg not in locked and price_of(arg) <= gold:
                    out.append({"action": "upgrade_tower", "tower_id": t["id"], "target": arg, "cost": price_of(arg)})
                elif kind == "upgrade_power":
                    pw = next((p for p in t.get("powers", []) if p["name"] == arg), None)
                    if pw and pw["level"] < pw["max_level"]:
                        price = power_price(t["type"], arg, pw["level"])
                        if price <= gold:
                            out.append({"action": "upgrade_power", "tower_id": t["id"], "power": arg,
                                        "cost": price})
                elif kind == "tw_sell":
                    out.append({"action": "sell_tower", "tower_id": t["id"], "cost": 0})
        for power in (1, 2):
            if self.modes.get(power) not in (None, "locked", "cooldown"):
                for enemy in self._anchors():
                    out.append({"action": "use_power", "power": power, "x": math.floor(enemy["x"] + 0.5),
                                "y": math.floor(enemy["y"] + 0.5), "anchor_id": enemy["id"], "cost": 0})
        if self._wave_ready():
            out.append({"action": "send_wave", "cost": 0})
        for item in out:
            item.update(available=True, legal=True)
        out.sort(key=lambda a: (a["action"], a.get("holder_id") or a.get("tower_id") or 0,
                                str(a.get("tower_type") or a.get("target") or a.get("power") or ""),
                                a.get("x") or 0, a.get("y") or 0))
        return out

    def catalog(self):
        if self.action_scope == "v2" or self.leak_v2:
            return self._catalog_v2()
        return self._catalog_v1()

    def _wire(self):
        state = copy.deepcopy(self.s)
        state.update(type="game_state", timestamp=0.5, save_directory="fake", controlled=True,
                     enemy_count=len(self.s["enemies"]), wave_spawning=bool(self.queue),
                     wave_ready=self._wave_ready(), action_catalog=self.catalog())
        if self.s["wave"] < self.s["wave_total"]:
            state["next_wave"] = {"group_idx": self.s["wave"] + 1}
        if self.action_scope == "v2":
            state["powers_ui"] = [{"id": p, "mode": self.modes.get(p, "missing")} for p in (1, 2)]
            state["action_scope"] = "v2"
        if "spent_hidden" in self.faults:
            for tower in state["towers"]:
                tower.pop("spent")  # a tower whose native spent is nil (absent from the bridge state)
        for key in ("towers", "holders", "enemies", "heroes", "action_catalog"):
            if state[key] == []:
                state[key] = {}  # the Lua encoder writes empty tables as objects
        return state

    # -- native simulation ----------------------------------------------
    def _new_id(self):
        self.next_id += 1
        return self.next_id

    def _new_tower(self, template, mesh, x, y, path_score, spent):
        kind, level, _ = TOWERS[template]
        tower = {"id": self._new_id(), "template": template, "type": kind, "level": level, "holder_id": mesh,
                 "is_special": False, "spent": spent, "x": x, "y": y, "path_score": path_score}
        if kind in POWERS:
            tower["powers"] = [{"name": name, "level": 0, "max_level": m, "next_cost": bridge_next_cost(kind, name, 0)}
                               for name, (m, _, _) in sorted(POWERS[kind].items())]
        return tower

    def _resolve(self, entity_id, op):
        s = self.s
        if op[0] == "build":
            holder = next(h for h in s["holders"] if h["id"] == entity_id)
            s["holders"].remove(holder)
            s["gold"] -= price_of(op[1])
            s["towers"].append(self._new_tower(op[1], holder["mesh_id"], holder["x"], holder["y"],
                                               holder["path_score"], price_of(op[1])))
            return
        tower = next(t for t in s["towers"] if t["id"] == entity_id)
        if op[0] == "upgrade":
            s["towers"].remove(tower)
            s["gold"] -= price_of(op[1])
            s["towers"].append(self._new_tower(op[1], tower["holder_id"], tower["x"], tower["y"],
                                               tower["path_score"], tower["spent"] + price_of(op[1])))
        elif op[0] == "sell":
            if "sell_keeps_tower" in self.faults:
                return
            s["towers"].remove(tower)
            s["gold"] += tower["spent"] if s["wave"] == 0 else math.floor(tower["spent"] * 0.6)
            if "sell_no_holder" not in self.faults:
                s["holders"].append({"id": self._new_id(), "template": "tower_holder", "blocked": False,
                                     "mesh_id": tower["holder_id"], "x": tower["x"], "y": tower["y"],
                                     "path_score": tower["path_score"]})

    def _tick(self):
        s = self.s
        s["tick"] += 1
        for entity_id, op in sorted(self.pending.items()):
            self._resolve(entity_id, op)
        self.pending.clear()
        for x, y in self.blasts:
            for enemy in s["enemies"]:
                if math.hypot(enemy["x"] - x, enemy["y"] - y) <= 60:
                    enemy["hp"] -= BLAST_DAMAGE
        self.blasts.clear()
        for power, until in sorted(self.cooldowns.items()):
            if s["tick"] >= until:
                self.modes[power] = "ready"
                del self.cooldowns[power]
        while self.queue and self.queue[0][0] <= s["tick"]:
            s["enemies"].append(self.queue.pop(0)[1])
        for tower in sorted(s["towers"], key=lambda t: t["id"]):
            alive = [e for e in s["enemies"] if e["hp"] > 0]
            if alive:
                bonus = sum(p["level"] for p in tower.get("powers", []))
                max(alive, key=lambda e: (e["path_progress"], -e["id"]))["hp"] -= DAMAGE[tower["type"]] + bonus
        survivors = []
        for enemy in s["enemies"]:
            if enemy["hp"] <= 0:
                s["gold"] += 5
                continue
            self.steps[enemy["id"]] += 1
            step = self.steps[enemy["id"]]
            if step >= PATH_TICKS:
                s["lives"] -= 1
                continue
            enemy.update(path_progress=step / PATH_TICKS, x=100.3 + step)
            survivors.append(enemy)
        s["enemies"] = survivors
        if s["lives"] <= 0:
            s["lives"], s["level_lost"] = 0, True
        elif s["wave"] >= s["wave_total"] and not s["enemies"] and not self.queue:
            s["level_won"] = True
        s["game_over"] = s["level_won"] or s["level_lost"]

    def _spawn(self):
        s = self.s
        s["wave"] += 1
        for index in range(5 + s["wave"]):
            enemy_id = self._new_id()
            self.steps[enemy_id] = 0
            self.queue.append((s["tick"] + 1 + 45 * index,
                               {"id": enemy_id, "template": "enemy_goblin", "hp": 150 + 50 * s["wave"],
                                "x": 100.3, "y": 400.6, "path_progress": 0.0, "path_index": 1}))

    # -- RPC ---------------------------------------------------------------
    def rpc(self, action, **arguments):
        self.calls.append((action, copy.deepcopy(arguments)))
        if action == "state":
            return self._wire()
        if action == "step":
            before = self.s["tick"]
            for _ in range(arguments["ticks"]):
                if self.s["game_over"]:
                    break
                self._tick()
            return {"tick_before": before, "tick_after": self.s["tick"], "terminated": self.s["game_over"],
                    "state": self._wire()}
        command = {"action": action, **arguments}
        allowed = ("build_tower", "send_wave") + (V2_ONLY if self.action_scope == "v2" else ())
        if action not in allowed:
            raise RuntimeError("Action not enabled in verified experiment scope")
        # As in host.lua: a build's target is implied by tower_type when omitted (menus never send it).
        keys = [k for k in MATCH_KEYS if not (k == "target" and action == "build_tower" and "target" not in command)]
        if not any(all(entry.get(k) == command.get(k) for k in keys) for entry in self.catalog()):
            raise RuntimeError("Action is outside the native legal catalog")
        if f"reject:{action}" in self.faults:
            return {"type": "error", "message": f"injected {action} rejection"}
        if action == "build_tower":
            self.pending[command["holder_id"]] = ("build", f"tower_{command['tower_type']}_1")
        elif action == "send_wave":
            self._spawn()
        elif action == "upgrade_tower":
            if "upgrade_noop" not in self.faults:
                self.pending[command["tower_id"]] = ("upgrade", command["target"])
        elif action == "sell_tower":
            self.pending[command["tower_id"]] = ("sell",)
        elif action == "upgrade_power":
            # As host.lua's GUI path: charge the GUI price, count it as spent, level + 1.
            tower = next(t for t in self.s["towers"] if t["id"] == command["tower_id"])
            pw = next(p for p in tower["powers"] if p["name"] == command["power"])
            if "power_noop" not in self.faults:
                price = power_price(tower["type"], pw["name"], pw["level"])
                self.s["gold"] -= price
                if "power_unspent" not in self.faults:
                    # power_bridge_spent: the bridge's old base + inc * level reaches spent instead.
                    tower["spent"] += (bridge_next_cost(tower["type"], pw["name"], pw["level"])
                                       if "power_bridge_spent" in self.faults else price)
                pw["level"] += 2 if "power_double" in self.faults else 1
                pw.pop("next_cost")
                if pw["level"] < pw["max_level"]:
                    pw["next_cost"] = bridge_next_cost(tower["type"], pw["name"], pw["level"])
        elif action == "use_power":
            if "cast_no_cooldown" not in self.faults:
                self.modes[command["power"]] = "cooldown"
                self.cooldowns[command["power"]] = self.s["tick"] + COOLDOWN_TICKS
            if command["power"] == 1:
                self.blasts.append((command["x"], command["y"]))
        return {"type": "ok", "action": action}


def worker_class(**config):
    """A Worker stand-in class whose instances simulate a level configured by ``config``."""

    class Configured(FakeWorker):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, **config)
            self.passed = dict(kwargs)  # exactly what NativeEnv handed to Worker(...)

    return Configured


def actions_of(env, name):
    return [item["action"] for item in build_menu(env.state) if item["action"]["action"] == name]


def option(env, name, **match):
    for action in actions_of(env, name):
        if all(action.get(k) == v for k, v in match.items()):
            return action
    raise AssertionError(f"no legal {name} {match} in the menu")


def native_calls(env):
    return [call for call in env.worker.calls if call[0] not in ("state", "step")]


class GreedyV2:
    """Test-only deterministic policy: spells first, then power and tower upgrades, builds, waves."""
    name = "unit_greedy_v2"
    ORDER = ("use_power", "upgrade_power", "upgrade_tower", "build_tower", "send_wave")

    def choose(self, state, menu, context):
        validate_menu(menu)
        for name in self.ORDER:
            options = [item for item in menu if item["action"]["action"] == name]
            if options:
                return {"label": options[0]["label"], "provenance": f"scripted:{self.name}", "distribution": None}
        return {"label": menu[0]["label"], "provenance": f"scripted:{self.name}", "distribution": None}


class StepClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


class V1MenuRegressionTests(unittest.TestCase):
    """v1 snapshots must give byte-identical menus after the v2 menu entries were added."""

    def pairs(self):
        found = []

        def walk(value, path):
            if isinstance(value, dict):
                if isinstance(value.get("state"), dict) and isinstance(value.get("menu"), list):
                    found.append((path, value["state"], value["menu"]))
                if isinstance(value.get("fork_state"), dict) and isinstance(value.get("menu"), list):
                    found.append((path, value["fork_state"], value["menu"]))
                for key, item in value.items():
                    walk(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        for name in ("dataset.json", "branches.json"):
            path = DATA / name
            if path.is_file():
                walk(json.loads(path.read_text(encoding="utf-8")), name)
        return found

    @unittest.skipUnless((DATA / "dataset.json").is_file(), "level-1 phase-1 dataset is not present")
    def test_real_dataset_menus_are_byte_identical(self):
        pairs = self.pairs()
        self.assertGreaterEqual(len(pairs), 4)
        builds_with_target = 0
        for path, state, menu in pairs:
            with self.subTest(path=path):
                rebuilt = build_menu(copy.deepcopy(state))
                self.assertEqual(canonical_json(menu), canonical_json(rebuilt))
                self.assertEqual(menu_sha256(menu), menu_sha256(rebuilt))
                self.assertLessEqual({item["action"]["action"] for item in rebuilt},
                                     {"wait", "build_tower", "send_wave"})
                builds_with_target += sum(1 for entry in state["action_catalog"]
                                          if entry.get("action") == "build_tower" and "target" in entry)
        # The native build entries carry a target that the menu action must drop.
        self.assertGreater(builds_with_target, 0)


class MenuV2Tests(unittest.TestCase):
    def state(self, catalog, gold=300):
        return {"gold": gold, "wave_ready": False, "locked_towers": ["tower_musketeer"],
                "holders": [{"id": 1, "blocked": False, "mesh_id": "1"}],
                "towers": [{"id": 10, "template": "tower_ranger", "is_special": False},
                           {"id": 11, "template": "tower_sunray", "is_special": True},
                           {"id": 12, "template": "tower_archer_1", "is_special": False}],
                "enemies": [{"id": 50, "x": 400.4, "y": 300.6}], "action_catalog": catalog}

    def test_v2_entries_texts_actions_and_order(self):
        catalog = [
            {"action": "use_power", "power": 2, "x": 400, "y": 301, "anchor_id": 50, "cost": 0},
            {"action": "use_power", "power": 1, "x": 400, "y": 301, "anchor_id": 50, "cost": 0},
            {"action": "upgrade_power", "tower_id": 10, "power": "poison", "cost": 250},
            {"action": "upgrade_tower", "tower_id": 12, "target": "tower_archer_2", "cost": 110},
            {"action": "sell_tower", "tower_id": 10, "cost": 0},
            {"action": "build_tower", "holder_id": 1, "tower_type": "archer", "target": "tower_build_archer",
             "cost": 70},
        ]
        menu = build_menu(self.state([dict(entry, available=True, legal=True) for entry in catalog]))
        self.assertEqual([(m["label"], m["text"], m["action"], m["cost"], m["legality_source"]) for m in menu[1:]], [
            ("B", "Build archer at holder 1 (70 gold)",
             {"action": "build_tower", "holder_id": 1, "tower_type": "archer"}, 70.0, "native_catalog"),
            ("C", "Sell tower 10 (tower_ranger)", {"action": "sell_tower", "tower_id": 10}, 0.0, "native_catalog"),
            ("D", "Upgrade power poison on tower 10 (tower_ranger) (250 gold)",
             {"action": "upgrade_power", "tower_id": 10, "power": "poison"}, 250.0, "native_catalog"),
            ("E", "Upgrade tower 12 to tower_archer_2 (110 gold)",
             {"action": "upgrade_tower", "tower_id": 12, "target": "tower_archer_2"}, 110.0, "native_catalog"),
            ("F", "Cast rain of fire at (400,301) near enemy 50",
             {"action": "use_power", "power": 1, "x": 400, "y": 301, "anchor_id": 50}, 0.0, "native_catalog"),
            ("G", "Call reinforcements at (400,301) near enemy 50",
             {"action": "use_power", "power": 2, "x": 400, "y": 301, "anchor_id": 50}, 0.0, "native_catalog"),
        ])
        self.assertEqual(validate_menu(menu), list("ABCDEFG"))
        self.assertTrue({"upgrade_power", "sell_tower", "use_power"} <= SUPPORTED_ACTIONS)
        # Native catalog order never changes the menu bytes.
        shuffled = build_menu(self.state([dict(entry, available=True) for entry in reversed(catalog)]))
        self.assertEqual(canonical_json(menu), canonical_json(shuffled))

    def test_secondary_checks_drop_malformed_or_stale_v2_entries(self):
        bad = [
            {"action": "upgrade_power", "tower_id": 99, "power": "poison", "cost": 1},     # unknown tower
            {"action": "upgrade_power", "tower_id": 11, "power": "ray", "cost": 1},        # special tower
            {"action": "upgrade_power", "tower_id": 10, "power": 3, "cost": 1},            # power not a name
            {"action": "upgrade_power", "tower_id": 10, "power": "", "cost": 1},
            {"action": "upgrade_power", "tower_id": 10, "power": "thorn", "cost": 301},    # above gold
            {"action": "upgrade_power", "tower_id": 10, "power": "thorn", "cost": -1},
            {"action": "upgrade_power", "tower_id": 10, "power": "thorn", "cost": float("nan")},
            {"action": "upgrade_power", "tower_id": 10, "power": "thorn", "cost": True},
            {"action": "upgrade_power", "tower_id": 10, "power": "thorn"},                 # no cost
            {"action": "sell_tower", "tower_id": 11, "cost": 0},                           # special tower
            {"action": "sell_tower", "tower_id": 98, "cost": 0},
            {"action": "sell_tower", "cost": 0},
            {"action": "use_power", "power": 3, "x": 1, "y": 1, "anchor_id": 50},
            {"action": "use_power", "power": True, "x": 1, "y": 1, "anchor_id": 50},
            {"action": "use_power", "power": 1.0, "x": 1, "y": 1, "anchor_id": 50},
            {"action": "use_power", "power": 1, "x": 1.5, "y": 1, "anchor_id": 50},
            {"action": "use_power", "power": 1, "x": 1, "y": "1", "anchor_id": 50},
            {"action": "use_power", "power": 1, "x": False, "y": 1, "anchor_id": 50},
            {"action": "use_power", "power": 1, "x": 1, "y": 1, "anchor_id": 77},          # not a live enemy
            {"action": "use_power", "power": 1, "x": 1, "y": 1},
            {"action": "upgrade_tower", "tower_id": 12, "target": "tower_musketeer", "cost": 1},  # locked
            {"action": "upgrade_tower", "tower_id": 11, "target": "tower_sunray_2", "cost": 1},   # special
        ]
        for entry in bad:
            with self.subTest(entry=entry):
                menu = build_menu(self.state([dict(entry, available=True)]))
                self.assertEqual([m["action"]["action"] for m in menu], ["wait"])
        unavailable = {"action": "sell_tower", "tower_id": 10, "cost": 0, "available": False}
        self.assertEqual(len(build_menu(self.state([unavailable]))), 1)
        terminal = dict(self.state([{"action": "sell_tower", "tower_id": 10, "available": True}]), level_lost=True)
        self.assertEqual(len(build_menu(terminal)), 1)
        # A sell entry needs no cost field, and a nonzero native cost never leaks into the menu.
        menu = build_menu(self.state([{"action": "sell_tower", "tower_id": 10, "cost": 5, "available": True}]))
        self.assertEqual((menu[1]["action"], menu[1]["cost"]), ({"action": "sell_tower", "tower_id": 10}, 0.0))

    def test_v2_build_entries_give_the_v1_build_option(self):
        v1 = {"action": "build_tower", "holder_id": 1, "tower_type": "mage", "cost": 100, "available": True}
        v2 = dict(v1, target="tower_build_mage", legal=True)
        a, b = build_menu(self.state([v1])), build_menu(self.state([v2]))
        self.assertEqual(canonical_json(a), canonical_json(b))
        self.assertEqual(set(b[1]["action"]), {"action", "holder_id", "tower_type"})


class WorkerScopeTests(unittest.TestCase):
    """Worker passes the scope to the host and requires the hello reply to confirm it."""

    def start(self, worker, hello):
        tmp = tempfile.TemporaryDirectory(prefix="alpharush-scope-")
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        hello = {"save_directory": f"C:/save/{worker.identity}", "rng_mode": "", **hello}
        with mock.patch.object(engine, "ROOT", base), \
                mock.patch.object(engine, "prepare", return_value=base / "rl-engine/Kingdom Rush.exe"), \
                mock.patch.object(engine.subprocess, "Popen") as popen, \
                mock.patch.object(engine.socket, "create_connection"), \
                mock.patch.object(engine.Worker, "rpc", return_value=hello) as rpc:
            popen.return_value.poll.return_value = None
            try:
                return worker.start(), popen.call_args.kwargs["env"]
            finally:
                worker.close()
                # hello, then at most the graceful quit request of close().
                self.assertEqual(mock.call("hello"), rpc.call_args_list[0])
                self.assertTrue(all(c == mock.call("quit") for c in rpc.call_args_list[1:]))

    def test_hello_must_confirm_the_requested_scope(self):
        with mock.patch.dict(os.environ, {"ALPHARUSH_ACTION_SCOPE": "v2"}):
            # An inherited value never leaks into a v1 worker; old hosts without the field are v1.
            result, environment = self.start(engine.Worker(identity="unit_scope_v1"), {})
            self.assertEqual(environment["ALPHARUSH_ACTION_SCOPE"], "v1")
            self.assertNotIn("action_scope", result)
        result, environment = self.start(engine.Worker(identity="unit_scope_v2", action_scope="v2"),
                                         {"action_scope": "v2"})
        self.assertEqual((environment["ALPHARUSH_ACTION_SCOPE"], result["action_scope"]), ("v2", "v2"))
        self.start(engine.Worker(identity="unit_scope_v1b"), {"action_scope": "v1"})
        for scope, hello in (("v2", {}), ("v2", {"action_scope": "v1"}), ("v1", {"action_scope": "v2"}),
                             ("v1", {"action_scope": None})):
            with self.subTest(scope=scope, hello=hello):
                with self.assertRaisesRegex(RuntimeError, "Native action scope"):
                    self.start(engine.Worker(identity="unit_scope_bad", action_scope=scope), hello)


class NativeEnvV2Tests(unittest.TestCase):
    def make(self, action_scope="v2", **config):
        cls = worker_class(**config)
        patcher = mock.patch.object(env_module, "Worker", cls)
        patcher.start()
        self.addCleanup(patcher.stop)
        kwargs = {} if action_scope is None else {"action_scope": action_scope}
        native = env_module.NativeEnv(seed=1001, level=1, identity="unit_v2", **kwargs)
        native.reset()
        return native

    def scripted(self, env):
        """Build, upgrade to tier 4, buy powers, sell, send a wave, cast both spells, sell under fire."""
        receipts = [env.act(option(env, "build_tower", holder_id=1, tower_type="archer"))]
        for target in ("tower_archer_2", "tower_archer_3", "tower_ranger"):
            tower = next(t for t in env.state["towers"] if t["template"].startswith(("tower_archer", "tower_ranger")))
            receipts.append(env.act(option(env, "upgrade_tower", tower_id=tower["id"], target=target)))
        ranger = receipts[-1]["tower_id_after"]
        receipts.append(env.act(option(env, "upgrade_power", tower_id=ranger, power="poison")))
        receipts.append(env.act(option(env, "upgrade_power", tower_id=ranger, power="poison")))
        receipts.append(env.act(option(env, "build_tower", holder_id=2, tower_type="mage")))
        receipts.append(env.act(option(env, "sell_tower", tower_id=receipts[-1]["tower_id"])))
        receipts.append(env.act({"action": "send_wave"}))
        env.advance(150)
        receipts.append(env.act(option(env, "use_power", power=1)))
        receipts.append(env.act(option(env, "use_power", power=2)))
        receipts.append(env.act(option(env, "sell_tower", tower_id=ranger)))
        return receipts

    def test_scope_attribute(self):
        self.assertEqual(env_module.NativeEnv.scope, ["wait", "build_tower", "send_wave"])
        v1, v2 = self.make(None), self.make("v2")
        self.assertEqual(v1.scope, ["wait", "build_tower", "send_wave"])
        self.assertEqual(v2.scope, ["wait", "build_tower", "send_wave", "upgrade_tower", "upgrade_power",
                                    "sell_tower", "use_power", "point_tower", "click_entity"])
        self.assertEqual((v1.action_scope, v2.action_scope), ("v1", "v2"))
        self.assertNotIn("action_scope", v1.worker.passed)
        self.assertEqual((v1.worker.kwargs["action_scope"], v2.worker.passed["action_scope"]), ("v1", "v2"))
        with self.assertRaises(ValueError):
            env_module.NativeEnv(action_scope="v9")

    def test_every_v2_receipt_succeeds_and_is_recorded(self):
        env = self.make()
        self.assertEqual(env.state["powers_ui"], [{"id": 1, "mode": "ready"}, {"id": 2, "mode": "ready"}])
        menu = build_menu(env.state)
        # Special holder 5, blocked holder 7 and the special elf tower 6 never get options.
        self.assertFalse([m for m in menu
                          if m["action"].get("holder_id") in (5, 7) or m["action"].get("tower_id") == 6])
        receipts = self.scripted(env)
        self.assertTrue(all(r["executed"] and r["accepted"] for r in receipts))
        build, up2, up3, up4, pow1, pow2, mage, sell_mage, wave, fire, reinforce, sell_ranger = receipts
        self.assertEqual((build["expected_cost"], build["gold_delta"], build["tick_after"] - build["tick_before"]),
                         (70.0, 70, 180))
        for receipt, target, cost in ((up2, "tower_archer_2", 110), (up3, "tower_archer_3", 160),
                                      (up4, "tower_ranger", 230)):
            self.assertEqual(receipt["action"]["target"], target)
            self.assertEqual(receipt["tick_after"] - receipt["tick_before"], env_module.UPGRADE_TICKS)
            self.assertEqual((receipt["expected_cost"], receipt["gold_delta"]), (cost, cost))
        self.assertEqual([up2["tower_id_after"], up3["tower_id_after"], up4["tower_id_after"]],
                         [up3["action"]["tower_id"], up4["action"]["tower_id"], pow1["action"]["tower_id"]])
        self.assertNotEqual(up3["tower_id_after"], up4["tower_id_after"])  # native upgrades replace the entity
        # GUI power prices: price_base 250 for the first level, price_inc 250 for the second
        # (the bridge's base + inc * level would be 500), each counted exactly as spent.
        self.assertEqual((pow1["power_level_before"], pow1["power_level_after"], pow1["gold_delta"]), (0, 1, 250))
        self.assertEqual((pow2["power_level_before"], pow2["power_level_after"], pow2["gold_delta"]), (1, 2, 250))
        self.assertEqual((pow1["expected_cost"], pow1["spent_before"], pow1["spent_after"]), (250, 570, 820))
        self.assertEqual((pow2["expected_cost"], pow2["spent_before"], pow2["spent_after"]), (250, 820, 1070))
        self.assertEqual(pow1["tick_after"] - pow1["tick_before"], 2)
        self.assertEqual((sell_mage["tick_after"] - sell_mage["tick_before"], sell_mage["gold_delta"]), (30, -100))
        self.assertEqual(sell_mage["holder_id_after"],
                         next(h["id"] for h in env.state["holders"] if h["mesh_id"] == "2"))
        self.assertEqual((fire["power_mode_before"], fire["power_mode_after"]), ("ready", "cooldown"))
        self.assertEqual((reinforce["power_mode_before"], reinforce["power_mode_after"]), ("ready", "cooldown"))
        self.assertEqual(fire["tick_after"] - fire["tick_before"], 2)
        # Sold during the wave: native refund factor (plus any kill bounty), tower gone, holder mesh restored.
        refund = math.floor((70 + 110 + 160 + 230 + 250 + 250) * 0.6)
        self.assertLessEqual(sell_ranger["gold_delta"], -refund)
        self.assertEqual((sell_ranger["gold_delta"] + refund) % 5, 0)
        self.assertEqual(sell_ranger["holder_id_after"], next(h["id"] for h in env.state["holders"]
                                                              if h["mesh_id"] == "1"))
        self.assertFalse([t for t in env.state["towers"] if t["holder_id"] == "1"])
        # Cooling spells and the sold tower are no longer offered.
        self.assertFalse(actions_of(env, "use_power"))
        self.assertFalse([a for a in actions_of(env, "sell_tower") if a["tower_id"] == pow1["action"]["tower_id"]])
        self.assertEqual({r["action"]["action"] for r in receipts}, {"build_tower", "send_wave", *V2_ONLY})
        recorded = [t["receipt"] for t in env.trace if t["kind"] == "action"]
        self.assertEqual(recorded, receipts)
        self.assertEqual([p["action"] for p in env.plan if "action" in p], [r["action"] for r in receipts])
        for receipt in receipts:
            self.assertLessEqual({"accepted", "executed", "action", "tick_before", "tick_after",
                                  "before_sha256", "after_sha256"}, set(receipt))
        # Each action entry follows the native step it verified, as for v1 actions.
        for index, entry in enumerate(env.trace):
            if entry["kind"] == "action":
                self.assertEqual(env.trace[index - 1]["kind"], "step")
                self.assertEqual(env.trace[index - 1]["state_sha256"], entry["receipt"]["after_sha256"])
        self.assertEqual(set(up2) - set(build), {"tower_id_after"})
        json.dumps(env.trace, allow_nan=False)

    def test_receipt_failures_raise_runtime_error(self):
        def to_ranger(env):
            env.act(option(env, "build_tower", holder_id=1, tower_type="archer"))
            for target in ("tower_archer_2", "tower_archer_3", "tower_ranger"):
                tower = env.state["towers"][-1]
                env.act(option(env, "upgrade_tower", tower_id=tower["id"], target=target))
            return env.state["towers"][-1]["id"]

        def wave_running(env):
            env.act({"action": "send_wave"})
            env.advance(150)

        def poison_twice(env):
            ranger = to_ranger(env)
            for _ in range(2):
                env.act(option(env, "upgrade_power", tower_id=ranger, power="poison"))

        cases = {
            "upgrade_noop": lambda env: env.act(option(env, "upgrade_tower", tower_id=env.act(
                option(env, "build_tower", holder_id=1, tower_type="archer"))["tower_id"], target="tower_archer_2")),
            "power_noop": lambda env: env.act(option(env, "upgrade_power", tower_id=to_ranger(env), power="poison")),
            "power_double": lambda env: env.act(option(env, "upgrade_power", tower_id=to_ranger(env), power="poison")),
            "power_unspent": lambda env: env.act(option(env, "upgrade_power", tower_id=to_ranger(env), power="poison")),
            "power_bridge_spent": poison_twice,  # the first level agrees (250); the second counts 500
            "spent_hidden": lambda env: env.act(option(env, "upgrade_power", tower_id=to_ranger(env), power="poison")),
            "sell_keeps_tower": lambda env: env.act(option(env, "sell_tower", tower_id=env.act(
                option(env, "build_tower", holder_id=1, tower_type="archer"))["tower_id"])),
            "sell_no_holder": lambda env: env.act(option(env, "sell_tower", tower_id=env.act(
                option(env, "build_tower", holder_id=1, tower_type="archer"))["tower_id"])),
            "cast_no_cooldown": lambda env: (wave_running(env), env.act(option(env, "use_power", power=1))),
        }
        for fault, run in cases.items():
            with self.subTest(fault=fault):
                env = self.make(faults=[fault])
                with self.assertRaisesRegex(RuntimeError, "Native completion receipt failed"):
                    run(env)
                # The failed action advanced the game but is in neither the plan nor the action trace.
                self.assertEqual(env.trace[-1]["kind"], "step")
        for name in ("upgrade_tower", "sell_tower", "upgrade_power", "use_power"):
            with self.subTest(rejected=name):
                env = self.make()
                ranger = to_ranger(env)
                wave_running(env)
                archer = env.act(option(env, "build_tower", holder_id=2, tower_type="archer"))["tower_id"]
                action = option(env, name, **{"upgrade_tower": {"tower_id": archer}, "use_power": {}}.get(
                    name, {"tower_id": ranger}))
                env.worker.faults.add(f"reject:{name}")
                tick, plan, trace = env.state["tick"], copy.deepcopy(env.plan), len(env.trace)
                with self.assertRaisesRegex(RuntimeError, "Native action rejected"):
                    env.act(action)
                self.assertEqual((env.state["tick"], env.plan, len(env.trace)), (tick, plan, trace))

    def test_power_receipt_requires_the_exact_spent_increment(self):
        """Level + 1 and the right gold are not enough: the same tower's spent must grow by the cost."""
        def ranger(env):
            env.act(option(env, "build_tower", holder_id=1, tower_type="archer"))
            for target in ("tower_archer_2", "tower_archer_3", "tower_ranger"):
                env.act(option(env, "upgrade_tower", tower_id=env.state["towers"][-1]["id"], target=target))
            return env.state["towers"][-1]["id"]

        env = self.make(faults=["power_unspent"])
        tower = ranger(env)
        gold = env.state["gold"]
        with self.assertRaisesRegex(RuntimeError, r"'power_level_after': 1, 'spent_before': 570, 'spent_after': 570"):
            env.act(option(env, "upgrade_power", tower_id=tower, power="poison"))
        self.assertEqual(gold - env.state["gold"], 250)  # the gold was charged; the spent evidence is missing
        env = self.make(faults=["power_bridge_spent"])
        tower = ranger(env)
        first = env.act(option(env, "upgrade_power", tower_id=tower, power="poison"))
        self.assertEqual((first["spent_before"], first["spent_after"], first["expected_cost"]), (570, 820, 250))
        with self.assertRaisesRegex(RuntimeError, r"'spent_before': 820, 'spent_after': 1320"):
            env.act(option(env, "upgrade_power", tower_id=tower, power="poison"))
        self.assertEqual([p["action"]["action"] for p in env.plan if "action" in p][-1], "upgrade_power")
        self.assertEqual(len([p for p in env.plan if p.get("action", {}).get("action") == "upgrade_power"]), 1)
        # Without a native spent on either side the receipt cannot be verified: it fails closed.
        env = self.make(faults=["spent_hidden"])
        tower = ranger(env)
        with self.assertRaisesRegex(RuntimeError, r"'spent_before': None, 'spent_after': None"):
            env.act(option(env, "upgrade_power", tower_id=tower, power="poison"))

    def test_illegal_actions_raise_value_error_without_native_calls(self):
        env = self.make(gold=250)
        env.act(option(env, "build_tower", holder_id=1, tower_type="archer"))
        archer = env.state["towers"][-1]["id"]
        env.act(option(env, "upgrade_tower", tower_id=archer, target="tower_archer_2"))
        archer = env.state["towers"][-1]["id"]
        illegal = [
            {"action": "upgrade_tower", "tower_id": archer, "target": "tower_archer_3"},   # 160 > gold left
            {"action": "upgrade_tower", "tower_id": archer, "target": "tower_ranger"},     # not on this menu level
            {"action": "upgrade_tower", "tower_id": 6, "target": "tower_elf_2"},           # special tower
            {"action": "upgrade_power", "tower_id": archer, "power": "poison"},            # no such power
            {"action": "sell_tower", "tower_id": 6},                                       # special tower
            {"action": "sell_tower", "tower_id": 999},
            {"action": "sell_tower", "tower_id": archer, "refund": 1},                     # extra key
            {"action": "use_power", "power": 1, "x": 100, "y": 401, "anchor_id": 1},       # no enemies yet
            {"action": "build_tower", "holder_id": 5, "tower_type": "archer"},             # special holder
            {"action": "build_tower", "holder_id": 7, "tower_type": "archer"},             # blocked holder
            {"action": "build_tower", "holder_id": 2, "tower_type": "archer", "target": "tower_build_archer"},
            {"action": "move_hero", "x": 1, "y": 1},
        ]
        self.assertLess(env.state["gold"], 160)
        for action in illegal:
            with self.subTest(action=action):
                calls, state = len(native_calls(env)), copy.deepcopy(env.state)
                with self.assertRaisesRegex(ValueError, "native legal menu"):
                    env.act(action)
                self.assertEqual((len(native_calls(env)), env.state), (calls, state))
        # A spell on cooldown, a maxed power and a locked upgrade are not offered.
        env = self.make(locked=["tower_musketeer"], modes={1: "ready", 2: "locked"})
        env.act(option(env, "build_tower", holder_id=1, tower_type="archer"))
        for target in ("tower_archer_2", "tower_archer_3"):
            env.act(option(env, "upgrade_tower", tower_id=env.state["towers"][-1]["id"], target=target))
        self.assertEqual([a["target"] for a in actions_of(env, "upgrade_tower")], ["tower_ranger"])
        env.act(option(env, "upgrade_tower", tower_id=env.state["towers"][-1]["id"], target="tower_ranger"))
        ranger = env.state["towers"][-1]["id"]
        env.act(option(env, "upgrade_power", tower_id=ranger, power="thorn"))
        ranger_menu = actions_of(env, "upgrade_power")
        self.assertEqual(ranger_menu, [{"action": "upgrade_power", "tower_id": ranger, "power": "poison"}])
        with self.assertRaises(ValueError):
            env.act({"action": "upgrade_power", "tower_id": ranger, "power": "thorn"})
        env.act({"action": "send_wave"})
        env.advance(150)
        self.assertEqual({a["power"] for a in actions_of(env, "use_power")}, {1})
        fire = option(env, "use_power", power=1)
        env.act(fire)
        self.assertEqual(env.state["powers_ui"][0]["mode"], "cooldown")
        with self.assertRaises(ValueError):
            env.act(fire)

    def test_spell_anchors_follow_the_host_rule(self):
        env = self.make()
        env.act({"action": "send_wave"})
        env.advance(200)
        enemies = sorted(env.state["enemies"], key=lambda e: (-e["path_progress"], e["id"]))
        self.assertGreaterEqual(len(enemies), 5)
        anchors = [a for a in actions_of(env, "use_power") if a["power"] == 1]
        self.assertEqual(len(anchors), 3)
        by_id = {e["id"]: e for e in enemies}
        chosen = [by_id[a["anchor_id"]] for a in anchors]
        for a in anchors:
            enemy = by_id[a["anchor_id"]]
            self.assertEqual((a["x"], a["y"]), (math.floor(enemy["x"] + 0.5), math.floor(enemy["y"] + 0.5)))
            self.assertIsInstance(a["x"], int)
        for i, first in enumerate(chosen):
            for second in chosen[i + 1:]:
                self.assertGreaterEqual(math.hypot(first["x"] - second["x"], first["y"] - second["y"]), 60)
        self.assertIn(enemies[0]["id"], {a["anchor_id"] for a in anchors})

    def test_v1_is_unaffected(self):
        env = self.make(None, gold=2000)
        self.assertNotIn("action_scope", env.worker.passed)  # the v1 Worker call is unchanged
        self.assertEqual(env.worker.kwargs["action_scope"], "v1")
        self.assertNotIn("powers_ui", env.state)
        self.assertNotIn("action_scope", env.state)
        build = env.act(option(env, "build_tower", holder_id=1, tower_type="archer"))
        self.assertEqual(set(build), {"accepted", "executed", "action", "expected_cost", "gold_delta", "tower_id",
                                      "tick_before", "tick_after", "before_sha256", "after_sha256"})
        self.assertEqual(build["tick_after"] - build["tick_before"], 180)
        self.assertEqual({m["action"]["action"] for m in build_menu(env.state)}, {"wait", "build_tower", "send_wave"})
        tower = env.state["towers"][-1]["id"]
        for action in ({"action": "upgrade_tower", "tower_id": tower, "target": "tower_archer_2"},
                       {"action": "sell_tower", "tower_id": tower}):
            calls = len(native_calls(env))
            with self.assertRaisesRegex(ValueError, "native legal menu"):
                env.act(action)
            self.assertEqual(len(native_calls(env)), calls)
        wave = env.act({"action": "send_wave"})
        self.assertEqual(set(wave), {"accepted", "executed", "action", "tick_before", "tick_after",
                                     "before_sha256", "after_sha256"})
        self.assertEqual(wave["tick_after"] - wave["tick_before"], 2)
        # A v1 env never forwards a v2 action, even if a host leaked one into its catalog.
        leaky = self.make(None, leak_v2=True)
        leaky.act(option(leaky, "build_tower", holder_id=1, tower_type="archer"))
        upgrade = option(leaky, "upgrade_tower")
        calls = len(native_calls(leaky))
        with self.assertRaisesRegex(ValueError, "outside the v1 action scope"):
            leaky.act(upgrade)
        self.assertEqual(len(native_calls(leaky)), calls)

    def test_v1_episodes_are_identical_with_explicit_or_default_scope(self):
        results = []
        for scope in (None, "v1"):
            env = self.make(scope)
            try:
                results.append(run_episode(env, make_policy("pressure_greedy"), EpisodeProtocol(), seed=1001,
                                           level=1, episode_id="ep-v1", clock=StepClock()))
            finally:
                env.close()
        self.assertEqual(results[0]["status"], "terminal")
        self.assertEqual(results[0]["plan"], results[1]["plan"])
        self.assertEqual(results[0]["trace_sha256"], results[1]["trace_sha256"])
        self.assertTrue(all(d["action"]["action"] in ("wait", "build_tower", "send_wave")
                            for d in results[0]["decisions"]))

    def test_cold_replay_reproduces_the_v2_trace(self):
        env = self.make()
        self.scripted(env)
        env.advance(60)
        plan, trace, state = copy.deepcopy(env.plan), copy.deepcopy(env.trace), copy.deepcopy(env.state)
        replay = env_module.NativeEnv(seed=1001, level=1, identity="unit_v2_replay", action_scope="v2")
        replay.reset()
        self.assertEqual(replay.replay(json.loads(json.dumps(plan))), state)
        self.assertEqual(sha256_data(replay.trace), sha256_data(trace))
        # A v1 env cannot replay a v2 plan.
        v1 = env_module.NativeEnv(seed=1001, level=1, identity="unit_v1_replay")
        v1.reset()
        with self.assertRaises(ValueError):
            v1.replay(plan)

    def test_v2_episodes_replay_from_their_plans(self):
        seen = set()
        for policy in (GreedyV2(), make_policy("random:3"), make_policy("random:11")):
            with self.subTest(policy=policy.name):
                env = self.make()
                protocol = EpisodeProtocol(max_ticks=12000)
                try:
                    result = run_episode(env, policy, protocol, seed=1001, level=1, episode_id="ep-v2",
                                         clock=StepClock())
                finally:
                    env.close()
                self.assertIn(result["status"], ("terminal", "stalled"), result["void_reason"])
                seen |= {d["action"]["action"] for d in result["decisions"]}
                result = json.loads(json.dumps(result, allow_nan=False))
                def make():
                    return env_module.NativeEnv(seed=1001, level=1, identity="unit_v2_ep", action_scope="v2")

                self.assertTrue(replay_verify(make, result)["replay_verified"])
                tampered = copy.deepcopy(result)
                tampered["plan"].insert(0, {"action": {"action": "sell_tower", "tower_id": 6}})
                self.assertFalse(replay_verify(make, tampered)["replay_verified"])
        self.assertTrue({"build_tower", "send_wave", *V2_ONLY} <= seen, seen)



class PointTowerReceiptTests(unittest.TestCase):
    def test_receipt_requires_the_aim_offer_to_be_consumed(self):
        aim = {"action": "point_tower", "tower_id": 57, "x": 1, "y": 2, "anchor_id": 9}
        offered = {"action_catalog": [{**aim, "cost": 0, "available": True}]}
        consumed = {"action_catalog": [{"action": "send_wave", "cost": 0}]}
        self.assertTrue(env_module._point_tower_receipt(offered, consumed, aim, 0)["executed"])
        self.assertFalse(env_module._point_tower_receipt(offered, offered, aim, 0)["executed"])
        self.assertFalse(env_module._point_tower_receipt(consumed, consumed, aim, 0)["executed"])


class ClickEntityTests(unittest.TestCase):
    def test_receipt_requires_the_click_to_have_been_offered(self):
        click = {"action": "click_entity", "entity_id": 500, "x": 10, "y": 21}
        offered = {"action_catalog": [{**click, "kind": "tower_trap", "cost": 0}]}
        gone = {"action_catalog": [{"action": "send_wave", "cost": 0}]}
        self.assertEqual({"executed": True, "offered_after": True},
                         env_module._click_entity_receipt(offered, offered, click, 0))
        self.assertEqual({"executed": True, "offered_after": False},
                         env_module._click_entity_receipt(offered, gone, click, 0))
        self.assertFalse(env_module._click_entity_receipt(gone, gone, click, 0)["executed"])

    def test_menu_offers_the_click_as_its_own_option(self):
        from alpharush_rl.menus import build_menu
        state = {"wave": 3, "gold": 0, "holders": [], "towers": [], "enemies": [], "wave_ready": False,
                 "action_catalog": [{"action": "click_entity", "entity_id": 500, "x": 10, "y": 21,
                                     "template": "mod_jt_tower", "kind": "tower_trap", "cost": 0, "available": True,
                                     "legal": True}]}
        menu = build_menu(state)
        clicks = [item for item in menu if item["action"]["action"] == "click_entity"]
        self.assertEqual([{"action": "click_entity", "entity_id": 500, "x": 10, "y": 21}],
                         [item["action"] for item in clicks])
        self.assertIn("mod_jt_tower", clicks[0]["text"])
        self.assertIn("free the trapped tower", clicks[0]["text"])


if __name__ == "__main__":
    unittest.main()
