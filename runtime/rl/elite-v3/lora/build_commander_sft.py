"""elite-v3: commander SFT data from the fork pilot (train seeds only, development evidence).

For every consult with all branches measured, the target is the play with the best progress score R; P0 (steady) wins
any tie that includes it and any gain under MARGIN; a tie among non-P0 plays is broken uniformly at random (seeded by
the consult key, so it is reproducible and does not favour the lower index). Prompts are exactly the live commander's
(commander.SYSTEM, consult_user(briefing), options in every cyclic order as LanguageBrain._ask builds them, labels A-F),
one row per rotation. Split by seed: train seeds -> train.jsonl, holdout seeds -> holdout.jsonl.

--permute: control labels. The train targets' play ids are shuffled among train consults that offered the same set of
play ids on the same level (so every label stays a legal play and the label marginals per stratum are kept), which
destroys any consult-specific information. A LoRA must beat the one trained on these to claim it learned anything
beyond a play-id prior. Holdout rows are never permuted (they are only scored, not trained on).

usage: build_commander_sft.py <S1 run id> <fork table json> <out dir> [--train 1201,1202] [--holdout 1203,1204] [--permute]"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl import commander  # noqa: E402
from alpharush_rl.strategy_brain import ROTATIONS  # noqa: E402

RUNS = ROOT / "runtime/rl/elite-v3/runs"
MARGIN = 0.01
ap = argparse.ArgumentParser()
ap.add_argument("s1")
ap.add_argument("table")
ap.add_argument("out")
ap.add_argument("--train", default="1201,1202")
ap.add_argument("--holdout", default="1203,1204")
ap.add_argument("--permute", action="store_true")
args = ap.parse_args()
train_seeds = {int(x) for x in args.train.split(",")}
hold_seeds = {int(x) for x in args.holdout.split(",")}
assert not train_seeds & hold_seeds
table_path, out_dir = Path(args.table), Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
consults = {}
for line in (RUNS / args.s1 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
    if '"cmd_steady"' not in line:
        continue
    r = json.loads(line)["payload"]
    if r.get("player") == "cmd_steady":
        for c in r["result"]["commander"]["consults"]:
            consults[(r["level"], r["seed"], r["genome_id"], c["consult"])] = c
table_bytes = table_path.read_bytes()
table = json.loads(table_bytes)
labels_all = "ABCDEF"
items = []
for entry in table:
    key = tuple(entry["key"])
    c = consults[key]
    n = len(c["plays"])
    if n < 2 or key[1] not in train_seeds | hold_seeds:
        continue
    outcomes = {int(k): v for k, v in entry["plays"].items()}
    assert set(outcomes) == set(range(n)), key
    top = max(outcomes.values())
    tied = [i for i in range(n) if outcomes[i] >= top - 1e-12]
    group = f"{key[0]}:{key[1]}:{key[2]}:{key[3]}"
    if outcomes[0] >= top - MARGIN:
        target = 0
    else:
        target = random.Random(hashlib.sha256(group.encode()).hexdigest()).choice(tied)
    items.append({"key": key, "group": group, "c": c, "n": n, "target": target, "outcomes": outcomes,
                  "ids": [p["id"] for p in c["plays"]], "split": "train" if key[1] in train_seeds else "holdout"})
if args.permute:
    strata = collections.defaultdict(list)
    for it in items:
        if it["split"] == "train":
            strata[(it["key"][0], tuple(sorted(it["ids"])))].append(it)
    rng = random.Random(20261012)
    for members in strata.values():
        ids = [m["ids"][m["target"]] for m in members]
        rng.shuffle(ids)
        for m, pid in zip(members, ids):
            m["target"] = m["ids"].index(pid)
train, hold, meta = [], [], []
for it in items:
    c, n, target, group = it["c"], it["n"], it["target"], it["group"]
    user = commander.consult_user(commander.Briefing(c["briefing"]))
    options = [p["text"] for p in c["plays"]]
    labels = list(labels_all[:n])
    for shift in range(min(n, ROTATIONS)):
        order = [(i + shift) % n for i in range(n)]
        text = user + "\n\nOptions:\n" + "\n".join(f"{label}. {options[i]}" for label, i in zip(labels, order)) \
            + "\n\nAnswer with one label."
        row = {"id": f"{group}:r{shift}", "group": group, "shift": shift, "system": commander.SYSTEM, "user": text,
               "labels": labels, "target": labels[order.index(target)], "order": order}
        (train if it["split"] == "train" else hold).append(row)
    meta.append({"group": group, "seed": it["key"][1], "split": it["split"], "target": target, "n": n,
                 "ids": it["ids"], "outcomes": [it["outcomes"][i] for i in range(n)]})
for name, data in (("train.jsonl", train), ("holdout.jsonl", hold)):
    (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8",
                                newline="\n")
summary = {"table": str(table_path), "table_sha256": hashlib.sha256(table_bytes).hexdigest(), "s1": args.s1,
           "train_seeds": sorted(train_seeds), "holdout_seeds": sorted(hold_seeds), "permuted": args.permute,
           "margin": MARGIN, "train_rows": len(train), "holdout_rows": len(hold),
           "train_consults": sum(m["split"] == "train" for m in meta),
           "holdout_consults": sum(m["split"] == "holdout" for m in meta),
           "train_targets": dict(collections.Counter(m["ids"][m["target"]] for m in meta if m["split"] == "train"))}
(out_dir / "meta.json").write_text(json.dumps({"summary": summary, "consults": meta}, indent=1), encoding="utf-8")
print(json.dumps(summary))
