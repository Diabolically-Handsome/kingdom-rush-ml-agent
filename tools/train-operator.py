#!/usr/bin/env python3
"""Behaviour cloning of the operator network from native-collect demonstrations (CPU, numpy).

Reads the npz decisions of one or more native-collect runs, trains on some train
seeds and measures on held-out train seeds (evaluation seeds never appear in any
collect run). Writes the weights and a receipt with data digests and metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.operator_net import (ACTIONS, DIMS, OptionScorer, instruction_features,  # noqa: E402
                                       load_rows, train)

RUNS = ROOT / "runtime/rl/campaign-v1/runs"
MODELS = ROOT / "runtime/rl/campaign-v1/models"
WAIT = ACTIONS.index("wait")
INSTRUCTION_O = 4  # trailing option dims: matches with the instruction


def load(run_ids, only_won, runs=RUNS):
    episodes = []
    for run_id in run_ids:
        data = runs / run_id / "data"
        for path in sorted(data.glob("*.npz")):
            rows, meta = load_rows(path)
            # Expert-labelled network play (DAgger) teaches from lost games too.
            if only_won and not meta.get("won") and meta.get("player") != "network":
                continue
            episodes.append((path, rows, meta))
    return episodes


def chosen_action(o, target):
    return int(np.argmax(o[target, :len(ACTIONS)]))


def metrics(net, decisions):
    hits = acts = act_hits = 0
    for g, o, target, _ in decisions:
        pick = int(np.argmax(net.scores(g, o)))
        hits += pick == target
        if chosen_action(o, target) != WAIT:
            acts += 1
            act_hits += pick == target
    return {"decisions": len(decisions), "accuracy": hits / max(1, len(decisions)),
            "non_wait_decisions": acts, "non_wait_accuracy": act_hits / max(1, acts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="native-collect run ids")
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=[1009, 1010])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="128,64")
    parser.add_argument("--action-weight", type=float, default=3.0, help="loss weight of non-wait choices")
    parser.add_argument("--all-episodes", action="store_true", help="also learn from lost demonstrations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solo", action="store_true",
                        help="zero the instruction features: a policy that plays without the strategy level")
    parser.add_argument("--phase-state", default="runtime/rl/campaign-v1",
                        help="phase state directory holding the collect runs; the weights go to its models/")
    args = parser.parse_args(argv)
    started = time.time()
    state = (ROOT / args.phase_state).resolve()
    if not state.is_relative_to(ROOT / "runtime/rl"):
        raise SystemExit("--phase-state must be a phase directory under runtime/rl")
    models = state / "models"
    episodes = load(args.runs, only_won=not args.all_episodes, runs=state / "runs")
    layouts = {meta["layout"] for _, _, meta in episodes}
    if len(layouts) != 1:
        raise SystemExit(f"the demonstrations mix feature layouts {sorted(layouts)}")
    layout = layouts.pop()
    g_dim, o_dim = DIMS[layout]
    instruction_g = len(instruction_features({}, None, layout))  # trailing global dims that carry the instruction
    train_set, holdout = [], []
    for path, rows, meta in episodes:
        target = holdout if meta["seed"] in args.holdout_seeds else train_set
        for g, o, choice in rows:
            weight = 1.0 if chosen_action(o, choice) == WAIT else args.action_weight
            if args.solo:
                g, o = g.copy(), o.copy()
                g[-instruction_g:] = 0
                o[:, -INSTRUCTION_O:] = 0
            target.append((g, o, choice, weight))
    if not train_set:
        raise SystemExit("no training decisions")
    hidden = tuple(int(x) for x in args.hidden.split(","))
    net = OptionScorer(hidden=hidden, seed=args.seed, layout=layout)
    history = train(net, train_set, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed,
                    log=lambda e, loss: print(f"epoch {e + 1}: loss {loss:.4f}", flush=True))
    weights = net.to_json()
    weights["solo"] = args.solo  # a solo network plays without strategy-level instructions
    text = json.dumps(weights, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    models.mkdir(parents=True, exist_ok=True)
    kind = "solo" if args.solo else "operator"
    path = models / f"{kind}-{digest[:12]}.json"
    path.write_text(text, encoding="utf-8")
    receipt = {"schema": "alpharush-operator-train-v1", "weights": path.relative_to(ROOT).as_posix(),
               "weights_sha256": digest, "layout": layout, "g_dim": g_dim, "o_dim": o_dim, "hidden": list(hidden),
               "runs": args.runs, "holdout_seeds": args.holdout_seeds, "epochs": args.epochs, "batch": args.batch,
               "lr": args.lr, "action_weight": args.action_weight, "only_won": not args.all_episodes, "solo": args.solo,
               "episodes": len(episodes), "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                          for p, _, _ in episodes},
               "loss_history": history, "train": metrics(net, train_set), "holdout": metrics(net, holdout),
               "optimizer_steps": args.epochs * ((len(train_set) + args.batch - 1) // args.batch),
               "gpu": False, "wall_seconds": time.time() - started}
    (models / f"{kind}-{digest[:12]}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                                    encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("weights", "weights_sha256", "episodes", "train", "holdout",
                                              "optimizer_steps", "wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
