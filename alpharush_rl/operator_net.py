"""The small operator network: scores every legal menu option of one decision.

The operator sees the native state, the legal menu and (optionally) the current
instruction from the strategy level (one build-order step plus the plan's
spell threshold and wave-call flag). It picks one menu option per decision;
it never invents actions outside the menu. Implemented in numpy so it trains
and runs in the game-side environment without extra dependencies.
"""

import json
import math

import numpy as np

from .menus import validate_menu
from .search import BRANCHES, KINDS, BuildOrderPolicy, check_genome, tower_kind_level

ACTIONS = ("wait", "build_tower", "upgrade_tower", "upgrade_power", "sell_tower", "use_power", "send_wave",
           "point_tower", "click_entity", "set_rally")
# Rows recorded before click_entity existed lack its one-hot column (index 8 of the "v3" layout).
LEGACY_ACTIONS = 8
RALLY_OPTIONS = ("entry", "center", "exit", "boss")
# Feature layouts. "v3": the main-campaign layout (levels 1-12, actions without set_rally, plan ops b/u/k);
# the campaign-v1 operators were trained on it and keep computing exactly the same numbers. "v4": the
# elite-stage layout (levels 1-26, rally actions and steps, boss / dormant / flying enemy features).
# Instruction features stay the trailing columns of both vectors in every layout.
LAYOUTS = {
    "v3": {"levels": 12, "actions": ACTIONS[:9], "ops": ("b", "u", "k"), "elite": False},
    "v4": {"levels": 26, "actions": ACTIONS, "ops": ("b", "u", "k", "r"), "elite": True},
}
LAYOUT = "v4"  # new data and new networks
MAX_LEVEL = LAYOUTS[LAYOUT]["levels"]
OPS = LAYOUTS[LAYOUT]["ops"]
OP_ACTION = {"b": "build_tower", "u": "upgrade_tower", "k": "upgrade_power", "r": "set_rally"}


def _n(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value)


def _one_hot(index, size):
    out = [0.0] * size
    if index is not None and 0 <= index < size:
        out[index] = 1.0
    return out


def _power_ready(state, power):
    for button in state.get("powers_ui") or []:
        if isinstance(button, dict) and button.get("id") == power:
            return 1.0 if button.get("mode") not in (None, "locked", "cooldown") else 0.0
    return 0.0


def _towers_by_mesh(state):
    return {str(t.get("holder_id")): t for t in state.get("towers", [])
            if isinstance(t, dict) and t.get("holder_id") is not None}


def _awake_bosses(state):
    return [e for e in state.get("enemies", []) if isinstance(e, dict) and e.get("boss") and not e.get("dormant")
            and _n(e.get("hp")) > 0]


def _distance(ax, ay, bx, by, cap=500.0):
    """Distance between two points, capped (``cap`` when either point is unknown)."""
    values = [_n(v, math.nan) for v in (ax, ay, bx, by)]
    if any(math.isnan(v) for v in values):
        return cap
    return min(cap, math.hypot(values[0] - values[2], values[1] - values[3]))


def elite_global_features(state: dict) -> list[float]:
    """Layout v4: [awake boss, its health fraction and progress, dormant boss waiting, untargetable /
    flying / unblockable enemies]."""
    enemies = [e for e in state.get("enemies", []) if isinstance(e, dict)]
    bosses = _awake_bosses(state)
    boss = max(bosses, key=lambda e: (_n(e.get("hp_max")), -_n(e.get("id")))) if bosses else {}
    return [1.0 if bosses else 0.0, _n(boss.get("hp")) / max(1.0, _n(boss.get("hp_max"), 1.0)),
            min(1.0, max(0.0, _n(boss.get("path_progress")))),
            1.0 if any(e.get("dormant") for e in enemies) else 0.0,
            sum(1 for e in enemies if e.get("untargetable") and not e.get("dormant")) / 20,
            sum(1 for e in enemies if e.get("flying")) / 20, sum(1 for e in enemies if e.get("unblockable")) / 20]


def global_features(state: dict, instruction: dict | None, layout: str = LAYOUT) -> list[float]:
    spec = LAYOUTS[layout]
    level = state.get("level_idx")
    wave_total = max(1.0, _n(state.get("wave_total"), 1.0))
    enemies = [e for e in state.get("enemies", []) if isinstance(e, dict)]
    progress = [min(1.0, max(0.0, _n(e.get("path_progress")))) for e in enemies]
    bins = [0.0] * 5
    for p in progress:
        bins[min(4, int(p * 5))] += 1.0 / 20
    towers = [0.0] * 16
    building = 0.0
    for tower in state.get("towers", []):
        kind, tl = tower_kind_level(tower.get("template") if isinstance(tower, dict) else None)
        if kind is None:
            continue
        if tl == 0:
            building += 0.2
        else:
            towers[KINDS.index(kind) * 4 + tl - 1] += 0.2
    heroes = [h for h in state.get("heroes", []) if isinstance(h, dict)]
    hero_alive = 1.0 if any(not h.get("dead") for h in heroes) else 0.0
    hero_hp = (sum(_n(h.get("hp")) / max(1.0, _n(h.get("hp_max"), 1.0)) for h in heroes) / len(heroes)) if heroes else 0.0
    levels = spec["levels"]
    out = _one_hot((level - 1) if isinstance(level, int) and 1 <= level <= levels else None, levels)
    out += [_n(state.get("gold")) / 1000, _n(state.get("lives")) / 20, _n(state.get("wave")) / wave_total,
            wave_total / 20, 1.0 if state.get("wave") == 0 else 0.0,
            len(enemies) / 40, (sum(progress) / len(progress)) if progress else 0.0, max(progress, default=0.0)]
    out += bins + towers + [building]
    out += [len([h for h in state.get("holders", []) if isinstance(h, dict) and not h.get("blocked")]) / 20,
            _power_ready(state, 1), _power_ready(state, 2), hero_alive, hero_hp]
    if spec["elite"]:
        out += elite_global_features(state)
    out += instruction_features(state, instruction, layout)
    return out


def instruction_features(state: dict, instruction: dict | None, layout: str = LAYOUT) -> list[float]:
    """The strategy level's current order: [has step, op, kind, holder x/y/score, tower level, cast, early]
    (layout v4: plus the plan's boss-blocking order), then 1.0 = instruction present."""
    spec = LAYOUTS[layout]
    ops = spec["ops"]
    width = 1 + len(ops) + 4 + 4 + 2 + (11 if spec["elite"] else 0) + 1
    if not instruction:
        return [0.0] * width
    step = instruction.get("step")
    feats = [1.0 if step else 0.0]
    if step:
        mesh = step[1]
        holder = next((h for h in state.get("holders", []) if isinstance(h, dict) and str(h.get("mesh_id")) == mesh),
                      None)
        tower = _towers_by_mesh(state).get(mesh)
        kind, tl = tower_kind_level(tower.get("template")) if tower else (None, None)
        if step[0] == "b":
            kind = step[2]
        where = holder or tower or {}
        feats += _one_hot(ops.index(step[0]) if step[0] in ops else None, len(ops))
        feats += _one_hot(KINDS.index(kind) if kind in KINDS else None, 4)
        feats += [_n(where.get("x")) / 1000, _n(where.get("y")) / 1000, _n(where.get("path_score")) / 5,
                  _n(tl) / 4]
    else:
        feats += [0.0] * (len(ops) + 8)
    feats += [_n(instruction.get("cast"), 50) / 100, _n(instruction.get("early"), 1)]
    if spec["elite"]:
        # The plan's boss-blocking order and the genes of the fallback rules that make most elite decisions:
        # build ratio per kind, build-first factor, preferred level-4 branch per kind.
        ratio = str(instruction.get("ratio") or "3111")
        branches = instruction.get("branches") or {}
        feats += [_n(instruction.get("block"))]
        feats += [int(d) / 9 if d.isdigit() else 0.0 for d in ratio[:4].ljust(4, "0")]
        feats += [math.log2(max(1.0, _n(instruction.get("f"), 50)) / 50) / 3, _n(instruction.get("cap")) / 12]
        feats += [float(BRANCHES[kind].index(branches[kind])) if branches.get(kind) in BRANCHES[kind] else 0.0
                  for kind in KINDS]
    feats += [1.0]  # instruction present
    return feats


def option_features(state: dict, item: dict, instruction: dict | None, layout: str = LAYOUT) -> list[float]:
    spec = LAYOUTS[layout]
    actions = spec["actions"]
    action = item["action"]
    name = action.get("action")
    holders = {h.get("id"): h for h in state.get("holders", []) if isinstance(h, dict)}
    towers = {t.get("id"): t for t in state.get("towers", []) if isinstance(t, dict)}
    kind, level, branch, where = None, 0.0, 0.0, {}
    power, anchor_progress, power_level = None, 0.0, 0.0
    mesh = None
    anchor = {}
    if name == "build_tower":
        kind, level = action.get("tower_type"), 1.0
        where = holders.get(action.get("holder_id"), {})
        mesh = str(where.get("mesh_id")) if where.get("mesh_id") is not None else None
    elif name in ("upgrade_tower", "upgrade_power", "sell_tower"):
        tower = towers.get(action.get("tower_id"), {})
        where = tower
        mesh = str(tower.get("holder_id")) if tower.get("holder_id") is not None else None
        kind, current = tower_kind_level(tower.get("template"))
        if name == "upgrade_tower":
            _, target_level = tower_kind_level(action.get("target"))
            level = _n(target_level)
            if kind in BRANCHES and action.get("target") in BRANCHES[kind]:
                branch = float(BRANCHES[kind].index(action.get("target")))
        elif name == "upgrade_power":
            level = 4.0
            for entry in tower.get("powers") or []:
                if isinstance(entry, dict) and entry.get("name") == action.get("power"):
                    power_level = _n(entry.get("level")) / max(1.0, _n(entry.get("max_level"), 1.0))
    elif name == "point_tower":
        anchor = next((e for e in state.get("enemies", []) if isinstance(e, dict) and e.get("id") == action.get(
            "anchor_id")), {})
        anchor_progress = _n(anchor.get("path_progress"))
        power_level = _n(anchor.get("hp")) / max(1.0, _n(anchor.get("hp_max"), 1.0))
        where = {"x": action.get("x"), "y": action.get("y")}
    elif name == "click_entity":
        where = {"x": action.get("x"), "y": action.get("y")}
    elif name == "use_power":
        power = action.get("power")
        anchor = next((e for e in state.get("enemies", []) if isinstance(e, dict) and e.get("id") == action.get(
            "anchor_id")), {})
        anchor_progress = _n(anchor.get("path_progress"))
        where = {"x": action.get("x"), "y": action.get("y")}
    elif name == "set_rally":
        tower = towers.get(action.get("tower_id"), {})
        mesh = str(tower.get("holder_id")) if tower.get("holder_id") is not None else None
        kind, current = tower_kind_level(tower.get("template"))
        level = _n(current)
        where = {"x": action.get("x"), "y": action.get("y")}
    out = _one_hot(actions.index(name) if name in actions else None, len(actions))
    out += _one_hot(KINDS.index(kind) if kind in KINDS else None, 4)
    out += [level / 4, branch, _n(where.get("x")) / 1000, _n(where.get("y")) / 1000,
            _n(where.get("path_score")) / 5, _n(item.get("cost")) / 500, power_level]
    out += _one_hot((power - 1) if power in (1, 2) else None, 2) + [anchor_progress]
    if spec["elite"]:
        # Rally: which point, how far the soldiers' current rally point is from it and how far an awake
        # boss is; spells and aims: whether the anchor is a boss.
        rally = name == "set_rally"
        tower = towers.get(action.get("tower_id"), {}) if rally else {}
        bosses = _awake_bosses(state)
        near_boss = min((_distance(action.get("x"), action.get("y"), b.get("x"), b.get("y")) for b in bosses),
                        default=500.0) if rally else 500.0
        out += _one_hot(RALLY_OPTIONS.index(action.get("option")) if rally and action.get("option") in RALLY_OPTIONS
                        else None, len(RALLY_OPTIONS))
        out += [(_distance(tower.get("rally_x"), tower.get("rally_y"), action.get("x"), action.get("y")) / 100)
                if rally else 0.0, near_boss / 100, 1.0 if anchor.get("boss") else 0.0]
    step = (instruction or {}).get("step")
    if step:
        same_op = 1.0 if OP_ACTION.get(step[0]) == name else 0.0
        same_mesh = 1.0 if mesh is not None and mesh == step[1] else 0.0
        same_kind = 1.0 if (step[0] == "b" and kind == step[2]) or (
            step[0] == "r" and name == "set_rally" and action.get("option") == step[2]) else 0.0
        out += [same_op, same_mesh, same_kind, same_op * same_mesh]
    else:
        out += [0.0] * 4
    return out


def decision_arrays(state: dict, menu: list[dict], instruction: dict | None, layout: str = LAYOUT):
    """(global vector, option matrix) of one decision, in menu order."""
    g = np.asarray(global_features(state, instruction, layout), dtype=np.float32)
    o = np.asarray([option_features(state, item, instruction, layout) for item in menu], dtype=np.float32)
    return g, o


DIMS = {name: (len(global_features({}, None, name)), len(option_features({}, {"action": {"action": "wait"}}, None, name)))
        for name in LAYOUTS}
G_DIM, O_DIM = DIMS[LAYOUT]


class OptionScorer:
    """MLP over [global, option] -> score; softmax over a decision's options."""

    def __init__(self, hidden=(128, 64), seed=0, layout=LAYOUT):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(LAYOUTS)}")
        self.layout = layout
        rng = np.random.default_rng(seed)
        sizes = [sum(DIMS[layout]), *hidden, 1]
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.params.append(rng.normal(0, math.sqrt(2.0 / a), size=(a, b)).astype(np.float32))
            self.params.append(np.zeros(b, dtype=np.float32))
        self.hidden = tuple(hidden)

    def _forward(self, x):
        acts = [x]
        h = x
        layers = len(self.params) // 2
        for i in range(layers):
            h = h @ self.params[2 * i] + self.params[2 * i + 1]
            if i < layers - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h[:, 0], acts

    def scores(self, g, o):
        x = np.concatenate([np.repeat(g[None, :], len(o), axis=0), o], axis=1)
        return self._forward(x)[0]

    def loss_and_grads(self, batch):
        """Mean cross-entropy over decisions; batch = list of (g, o, chosen index, weight)."""
        xs, owners, chosen, weights = [], [], [], []
        offset = 0
        for g, o, target, weight in batch:
            xs.append(np.concatenate([np.repeat(g[None, :], len(o), axis=0), o], axis=1))
            owners.append((offset, len(o)))
            chosen.append(offset + target)
            weights.append(weight)
            offset += len(o)
        x = np.concatenate(xs, axis=0)
        s, acts = self._forward(x)
        grad_s = np.zeros_like(s)
        total, wsum = 0.0, float(sum(weights))
        for (start, count), target, weight in zip(owners, chosen, weights):
            z = s[start:start + count]
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            total += -weight * math.log(max(1e-12, float(p[target - start])))
            g = p * weight / wsum
            g[target - start] -= weight / wsum
            grad_s[start:start + count] = g
        grads = [None] * len(self.params)
        delta = grad_s[:, None]
        layers = len(self.params) // 2
        for i in reversed(range(layers)):
            a_in = acts[i]
            grads[2 * i] = a_in.T @ delta
            grads[2 * i + 1] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.params[2 * i].T) * (acts[i] > 0)
        return total / wsum, grads

    def to_json(self):
        g_dim, o_dim = DIMS[self.layout]
        data = {"hidden": list(self.hidden), "g_dim": g_dim, "o_dim": o_dim,
                "params": [p.tolist() for p in self.params]}
        if self.layout != "v3":
            data["layout"] = self.layout  # weights without a layout key are the main-campaign layout v3
        return data

    @classmethod
    def from_json(cls, data):
        layout = data.get("layout", "v3")
        if layout not in LAYOUTS:
            raise ValueError(f"unknown operator feature layout {layout!r}")
        g_dim, o_dim = DIMS[layout]
        legacy = layout == "v3" and data["g_dim"] == g_dim and data["o_dim"] == o_dim - 1
        if data["g_dim"] != g_dim or (data["o_dim"] != o_dim and not legacy):
            raise ValueError("operator feature dimensions changed since these weights were trained")
        net = cls(tuple(data["hidden"]), layout=layout)
        net.params = [np.asarray(p, dtype=np.float32) for p in data["params"]]
        if legacy:
            # Weights trained before click_entity: a zero input weight for its one-hot column.
            net.params[0] = np.insert(net.params[0], g_dim + LEGACY_ACTIONS, 0.0, axis=0)
        return net


def train(net, decisions, *, epochs=30, batch=64, lr=1e-3, seed=0, log=None):
    """Adam on the cross-entropy of the demonstrator's choices; decisions = [(g, o, target, weight)]."""
    rng = np.random.default_rng(seed)
    m = [np.zeros_like(p) for p in net.params]
    v = [np.zeros_like(p) for p in net.params]
    step = 0
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(decisions))
        losses = []
        for start in range(0, len(order), batch):
            chunk = [decisions[i] for i in order[start:start + batch]]
            loss, grads = net.loss_and_grads(chunk)
            losses.append(loss)
            step += 1
            for i, grad in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * grad
                v[i] = 0.999 * v[i] + 0.001 * grad * grad
                mh = m[i] / (1 - 0.9 ** step)
                vh = v[i] / (1 - 0.999 ** step)
                net.params[i] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        history.append(float(np.mean(losses)))
        if log:
            log(epoch, history[-1])
    return history


def accuracy(net, decisions):
    hits = sum(int(np.argmax(net.scores(g, o)) == target) for g, o, target, _ in decisions)
    return hits / max(1, len(decisions))


class PlanTracker:
    """Keeps the strategy level's current step: advances when the step is executed or impossible.

    It reuses the plan executor's feasibility rules and never chooses an action itself.
    """

    def __init__(self, genome):
        self.reader = BuildOrderPolicy(genome)
        self.genome = self.reader.genome

    def instruction(self, state, menu):
        reader = self.reader
        holders = {str(h.get("mesh_id")): h.get("id") for h in state.get("holders", [])
                   if isinstance(h, dict) and h.get("mesh_id") is not None}
        towers = _towers_by_mesh(state)
        steps = self.genome["steps"]
        while reader.cursor < len(steps):
            verdict, item = reader._step_option(steps[reader.cursor], state, menu, holders, towers)
            if verdict == "skip" or (verdict == "wait" and _n(state.get("gold")) >= 800):
                reader.cursor += 1
                continue
            return {**self._genes(), "step": steps[reader.cursor],
                    "expect": item["action"] if verdict == "take" else None}
        return {**self._genes(), "step": None, "expect": None}

    def _genes(self):
        g = self.genome
        return {"cast": g["cast"], "early": g["early"], "block": g.get("block", 0), "ratio": g.get("ratio", "3111"),
                "f": g.get("f", 50), "cap": g.get("cap", 0), "branches": dict(g["branches"])}

    def observe(self, instruction, action):
        if instruction.get("expect") is not None and action == instruction["expect"]:
            self.reader.cursor += 1


class OperatorPolicy:
    """Picks the highest-scoring legal option; with a plan, follows its instructions."""

    def __init__(self, net, genome=None, name="operator"):
        self.net = net
        self.tracker = PlanTracker(genome) if genome is not None else None
        self.name = name

    def choose(self, state, menu, context):
        validate_menu(menu)
        instruction = self.tracker.instruction(state, menu) if self.tracker else None
        g, o = decision_arrays(state, menu, instruction, self.net.layout)
        scores = self.net.scores(g, o)
        index = int(np.argmax(scores))
        if self.tracker:
            self.tracker.observe(instruction, menu[index]["action"])
        return {"label": menu[index]["label"], "provenance": f"model:{self.name}", "distribution": None,
                "meta": {"rule": "operator", "score": float(scores[index]),
                         "instruction": (instruction or {}).get("step")}}


def macro_summary(state, menu, index, instruction, rule):
    """A compact, JSON-ready view of one decision (for strategy-level language-model data)."""
    towers = sorted((str(t.get("holder_id")), t.get("template")) for t in state.get("towers", [])
                    if isinstance(t, dict) and t.get("holder_id") is not None)
    holders = sorted(str(h.get("mesh_id")) for h in state.get("holders", [])
                     if isinstance(h, dict) and h.get("mesh_id") is not None and not h.get("blocked"))
    enemies = [e for e in state.get("enemies", []) if isinstance(e, dict)]
    return {"level": state.get("level_idx"), "tick": state.get("tick"), "wave": state.get("wave"),
            "wave_total": state.get("wave_total"), "gold": state.get("gold"), "lives": state.get("lives"),
            "enemies": len(enemies), "front": max((_n(e.get("path_progress")) for e in enemies), default=0.0),
            "towers": towers, "holders": holders, "menu": [item["action"] for item in menu],
            "costs": [item.get("cost") for item in menu], "choice": index, "rule": rule,
            "step": (instruction or {}).get("step")}


class Recorder:
    """Wraps a plan executor and keeps every decision's arrays and the demonstrator's choice
    (plus a compact text-ready summary of each decision in ``macro``)."""

    def __init__(self, policy, with_instruction=True, layout=LAYOUT):
        self.policy = policy
        self.name = policy.name
        self.layout = layout
        self.tracker = PlanTracker(policy.genome) if with_instruction else None
        self.rows = []
        self.macro = []

    def choose(self, state, menu, context):
        instruction = self.tracker.instruction(state, menu) if self.tracker else None
        choice = self.policy.choose(state, menu, context)
        index = next(i for i, item in enumerate(menu) if item["label"] == choice["label"])
        if self.tracker:
            self.tracker.observe(instruction, menu[index]["action"])
        g, o = decision_arrays(state, menu, instruction, self.layout)
        self.rows.append((g, o, index))
        self.macro.append(macro_summary(state, menu, index, instruction, (choice.get("meta") or {}).get("rule")))
        return choice


class DaggerRecorder:
    """The network plays (or, with probability ``beta``, the expert); every decision is labelled with
    what the plan executor would choose in that very state, its plan cursor kept in step with the play."""

    def __init__(self, net, genome, beta=0.0, seed=0, name="dagger"):
        self.net = net
        self.tracker = PlanTracker(genome)
        self.expert = BuildOrderPolicy(genome)
        self.beta, self.seed, self.name = float(beta), seed, name
        self.rows, self.macro = [], []
        self.expert_choices = self.net_choices = self.agreements = 0

    def choose(self, state, menu, context):
        import random
        validate_menu(menu)
        instruction = self.tracker.instruction(state, menu)
        self.expert.cursor = self.tracker.reader.cursor
        expert = self.expert.choose(state, menu, context)
        label = next(i for i, item in enumerate(menu) if item["label"] == expert["label"])
        g, o = decision_arrays(state, menu, instruction, self.net.layout)
        scores = self.net.scores(g, o)
        mine = int(np.argmax(scores))
        self.agreements += mine == label
        rng = random.Random(f"dagger:{self.seed}:{context.get('decision_index')}")
        if self.beta and rng.random() < self.beta:
            index, source = label, "expert"
            self.expert_choices += 1
        else:
            index, source = mine, "network"
            self.net_choices += 1
        self.tracker.observe(instruction, menu[index]["action"])
        self.rows.append((g, o, label))
        self.macro.append(macro_summary(state, menu, label, instruction, (expert.get("meta") or {}).get("rule")))
        return {"label": menu[index]["label"], "provenance": f"model:{self.name}", "distribution": None,
                "meta": {"rule": "dagger", "source": source, "expert_label": menu[label]["label"],
                         "instruction": instruction.get("step")}}


def save_rows(path, rows, meta, layout=LAYOUT):
    """One episode's decisions as a compressed npz (ragged options flattened with offsets)."""
    offsets = np.cumsum([0] + [len(o) for _, o, _ in rows]).astype(np.int64)
    g_dim, o_dim = DIMS[layout]
    np.savez_compressed(path, g=np.stack([g for g, _, _ in rows]).astype(np.float32) if rows else np.zeros((0, g_dim)),
                        o=np.concatenate([o for _, o, _ in rows]).astype(np.float32) if rows else np.zeros((0, o_dim)),
                        offsets=offsets, target=np.asarray([t for _, _, t in rows], dtype=np.int64),
                        meta=np.asarray(json.dumps(meta)))


def rows_layout(g_dim, o_dim):
    """The feature layout of recorded arrays (v3 rows from before click_entity have one column less)."""
    for name, dims in DIMS.items():
        if (g_dim, o_dim) == dims:
            return name
    if (g_dim, o_dim) == (DIMS["v3"][0], DIMS["v3"][1] - 1):
        return "v3"
    raise ValueError(f"no operator feature layout has dimensions {(g_dim, o_dim)}")


def load_rows(path):
    """(rows, meta) of one episode; ``meta["layout"]`` is the feature layout the rows were recorded in."""
    data = np.load(path, allow_pickle=False)
    g, o, offsets, target = data["g"], data["o"], data["offsets"], data["target"]
    layout = rows_layout(g.shape[1], o.shape[1])
    if layout == "v3" and o.shape[1] == DIMS["v3"][1] - 1:
        o = np.insert(o, LEGACY_ACTIONS, 0.0, axis=1)  # recorded before click_entity existed
    rows = [(g[i], o[offsets[i]:offsets[i + 1]], int(target[i])) for i in range(len(target))]
    meta = json.loads(str(data["meta"]))
    meta["layout"] = layout
    return rows, meta
