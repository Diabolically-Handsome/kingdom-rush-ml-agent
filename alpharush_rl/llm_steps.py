"""The 8B strategy brain issuing build-order steps online; the operator network executes them.

Instead of a precomputed plan, the language model is asked for the next step whenever the current one is
done or impossible: the prompt is the state summary and every structurally possible step (builds on empty
slots, upgrades, level-4 skills, the sunray beam) as labelled options, scored by the local model worker
over the complete legal label set. The operator network then executes that step when gold allows, and
handles spells (at the plan's cast threshold), aiming and wave calls itself.
"""

import hashlib

import numpy as np

from .menus import option_label, validate_menu
from .operator_net import _towers_by_mesh, decision_arrays, macro_summary
from .search import BRANCHES, KINDS, BuildOrderPolicy, check_genome, tower_kind_level

KIND_TEXT = {"archer": "archer tower", "barrack": "barracks", "mage": "mage tower", "engineer": "artillery"}
SYSTEM = ("You are the strategy brain of a Kingdom Rush agent on Normal difficulty. A separate operator executes "
          "your build-order steps when gold allows and handles spells and wave calls. Choose the next build-order "
          "step. Reply with the option label only.")


def describe_tower(template):
    kind, level = tower_kind_level(template)
    if kind is None:
        return (template or "?").replace("tower_", "") + " (special)"
    if level == 4:
        return template.replace("tower_", "") + " (level 4)"
    if level == 0:
        return f"{KIND_TEXT[kind]} under construction"
    return f"{KIND_TEXT[kind]} level {level}"


def step_candidates(summary):
    """Every structurally possible next step (same construction as tools/build-step-dataset.py)."""
    out = []
    for mesh in summary["holders"]:
        for kind in KINDS:
            out.append((["b", mesh, kind], f"build {KIND_TEXT[kind]} at slot {mesh}"))
    for mesh, template in summary["towers"]:
        kind, level = tower_kind_level(template)
        if kind is None:
            if template == "tower_sunray":
                out.append((["k", mesh], f"charge the sunray beam at slot {mesh}"))
            continue
        if 1 <= level <= 3:
            target = "a level-4 specialization" if level == 3 else f"level {level + 1}"
            out.append((["u", mesh], f"upgrade the {KIND_TEXT[kind]} at slot {mesh} to {target}"))
        elif level == 4:
            out.append((["k", mesh], f"buy a special skill for the {template.replace('tower_', '')} at slot {mesh}"))
    return out


def step_prompt(summary):
    towers = "; ".join(f"slot {mesh}: {describe_tower(t)}" for mesh, t in summary["towers"]) or "none"
    empty = ", ".join(summary["holders"]) or "none"
    return (f"Level {summary['level']}, wave {summary['wave']} of {summary['wave_total']}. "
            f"Gold {summary['gold']:g}, lives {summary['lives']}. Enemies on the field: {summary['enemies']} "
            f"(the leading one has covered {summary['front'] * 100:.0f}% of its path).\n"
            f"Towers: {towers}.\nEmpty build slots: {empty}.")


class LanguageStepSource:
    """Instruction source for the operator: asks the strategy model for each next step."""

    def __init__(self, broker, cast=50, early=1, branches=None, max_steps=200, name="8b-steps"):
        self.broker, self.name = broker, name
        self.cast, self.early = cast, early
        self.branches = branches or {kind: BRANCHES[kind][0] for kind in KINDS}
        self.step = None
        self.asked = 0
        self.max_steps = max_steps
        self.records = []
        # Feasibility of a step is judged exactly as the plan executor judges it.
        self.reader = BuildOrderPolicy(check_genome({"hero": None, "steps": [], "cast": cast, "early": early,
                                                     "branches": self.branches}))

    def _verdict(self, state, menu, step):
        holders = {str(h.get("mesh_id")): h.get("id") for h in state.get("holders", [])
                   if isinstance(h, dict) and h.get("mesh_id") is not None}
        return self.reader._step_option(step, state, menu, holders, _towers_by_mesh(state))

    def _ask(self, state, menu):
        summary = macro_summary(state, menu, 0, None, None)
        options = step_candidates(summary)
        if not options or self.asked >= self.max_steps:
            return None
        labels = [option_label(i) for i in range(len(options))]
        user = step_prompt(summary) + "\n\nOptions:\n" + "\n".join(
            f"{label}. {text}" for label, (_, text) in zip(labels, options)) + "\n\nAnswer with one label."
        self.asked += 1
        request = {"id": f"{self.name}-{self.asked}", "system": SYSTEM, "user": user, "labels": labels}
        response = self.broker.distribution(request)
        index = labels.index(response["choice"])
        self.records.append({"tick": state.get("tick"), "step": options[index][0], "p": response["p"][index],
                             "options": len(options),
                             "prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest()})
        return options[index][0]

    def instruction(self, state, menu):
        for _ in range(4):  # an impossible answer is replaced by asking again (bounded)
            if self.step is not None:
                verdict, item = self._verdict(state, menu, self.step)
                if verdict == "skip" or (verdict == "wait" and state.get("gold", 0) >= 800):
                    self.step = None
                else:
                    return {"step": self.step, "cast": self.cast, "early": self.early,
                            "expect": item["action"] if verdict == "take" else None}
            self.step = self._ask(state, menu)
            if self.step is None:
                break
        return {"step": None, "cast": self.cast, "early": self.early, "expect": None}

    def observe(self, instruction, action):
        if instruction.get("expect") is not None and action == instruction["expect"]:
            self.step = None


class StepOperatorPolicy:
    """The operator network following the strategy model's online steps."""

    def __init__(self, net, source, name="8b-steps+operator"):
        self.net, self.source, self.name = net, source, name

    def choose(self, state, menu, context):
        validate_menu(menu)
        instruction = self.source.instruction(state, menu)
        g, o = decision_arrays(state, menu, instruction, self.net.layout)
        scores = self.net.scores(g, o)
        index = int(np.argmax(scores))
        self.source.observe(instruction, menu[index]["action"])
        return {"label": menu[index]["label"], "provenance": f"model:{self.name}", "distribution": None,
                "meta": {"rule": "operator", "instruction": instruction.get("step"), "score": float(scores[index])}}
