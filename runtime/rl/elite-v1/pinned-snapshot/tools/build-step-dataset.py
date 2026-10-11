#!/usr/bin/env python3
"""Strategy-level language data: "which build-order step next?" at every point a plan issued a new step.

Reads the decision summaries (``*.macro.json``) that native-collect saved next to each demonstration,
keeps won episodes, and writes one legal multiple-choice example per issued step: the state summary,
every structurally possible next step as a labelled option, and the plan's step as the target.
Train/holdout split is by seed (holdout seeds never enter training). No game or model is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.menus import option_label  # noqa: E402
from alpharush_rl.search import KINDS, tower_kind_level  # noqa: E402

RUNS = ROOT / "runtime/rl/campaign-v1/runs"
OUT = ROOT / "runtime/rl/campaign-v1/llm"
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


def candidates(summary):
    """Every structurally possible next step: builds on empty slots, upgrades, level-4/special skills."""
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


def prompt(summary):
    towers = "; ".join(f"slot {mesh}: {describe_tower(t)}" for mesh, t in summary["towers"]) or "none"
    empty = ", ".join(summary["holders"]) or "none"
    return (f"Level {summary['level']}, wave {summary['wave']} of {summary['wave_total']}. "
            f"Gold {summary['gold']:g}, lives {summary['lives']}. Enemies on the field: {summary['enemies']} "
            f"(the leading one has covered {summary['front'] * 100:.0f}% of its path).\n"
            f"Towers: {towers}.\nEmpty build slots: {empty}.")


def examples(macro, meta, source):
    previous = None
    for index, summary in enumerate(macro):
        step = summary.get("step")
        if not step or step == previous:
            previous = step if step else previous
            continue
        previous = step
        options = candidates(summary)
        steps = [s for s, _ in options]
        if step not in steps:
            continue
        labels = [option_label(i) for i in range(len(options))]
        user = prompt(summary) + "\n\nOptions:\n" + "\n".join(
            f"{label}. {text}" for label, (_, text) in zip(labels, options)) + "\n\nAnswer with one label."
        yield {"id": f"{source}:{index}", "level": meta["level"], "seed": meta["seed"],
               "genome_id": meta["genome_id"], "system": SYSTEM, "user": user, "labels": labels,
               "target": labels[steps.index(step)], "step": step}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=[1009, 1010])
    parser.add_argument("--name", default="steps-v1")
    args = parser.parse_args(argv)
    rows = {"train": [], "holdout": []}
    for run_id in args.runs:
        for path in sorted((RUNS / run_id / "data").glob("*.npz")):
            meta = json.loads(str(np.load(path, allow_pickle=False)["meta"]))
            if not meta.get("won") or meta.get("player", "expert") != "expert":
                continue
            macro = json.loads(path.with_suffix(".macro.json").read_text(encoding="utf-8"))
            split = "holdout" if meta["seed"] in args.holdout_seeds else "train"
            rows[split].extend(examples(macro, meta, f"{run_id}/{path.stem}"))
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "alpharush-step-dataset-v1", "runs": args.runs, "holdout_seeds": args.holdout_seeds}
    for split, items in rows.items():
        path = OUT / f"{args.name}-{split}.jsonl"
        text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
        path.write_text(text, encoding="utf-8")
        receipt[split] = {"path": path.relative_to(ROOT).as_posix(), "examples": len(items),
                          "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                          "mean_options": sum(len(i["labels"]) for i in items) / max(1, len(items))}
    (OUT / f"{args.name}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
