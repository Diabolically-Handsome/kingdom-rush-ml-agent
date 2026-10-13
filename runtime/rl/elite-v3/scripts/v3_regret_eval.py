"""elite-v3: held-out one-step regret of commander brains on the fork pilot (read-only, development evidence).

usage: v3_regret_eval.py <S1 run id> <fork table json> [--llm choices.json] [--lora predictions.jsonl]
                         [--train 1201,1202] [--holdout 1203,1204]
For every held-out consult with every branch measured: regret = best play's R - chosen play's R (all other consults
steady). Brains: steady (P0), rule and random (shadow records), zero-shot 8B (offline choices), LoRA 8B (label
log-likelihoods of every cyclic order -> worker softmax -> mean over orders -> argmax, ties to the lower index), and
FixedSchedule (learned on train seeds only: per level x wave the play id with the best mean gain over P0, P0 if none
gains). Paired differences vs steady with 95% normal intervals over consults. One-step regret is a proxy: it says
which single choice helps when the rest of the game stays steady, not how a brain's whole game goes."""
import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
RUNS = ROOT / "runtime/rl/elite-v3/runs"
parser = argparse.ArgumentParser()
parser.add_argument("s1")
parser.add_argument("table")
parser.add_argument("--llm")
parser.add_argument("--lora", action="append", default=[], help="name=predictions.jsonl (repeatable)")
parser.add_argument("--train", default="1201,1202")
parser.add_argument("--holdout", default="1203,1204")
parser.add_argument("--json")
args = parser.parse_args()
train_seeds = {int(x) for x in args.train.split(",")}
hold_seeds = {int(x) for x in args.holdout.split(",")}

consults = {}
for line in (RUNS / args.s1 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
    if '"cmd_steady"' not in line:
        continue
    r = json.loads(line)["payload"]
    if r.get("player") == "cmd_steady":
        for c in r["result"]["commander"]["consults"]:
            consults[f"{r['level']}:{r['seed']}:{r['genome_id']}:{c['consult']}"] = c
table = []
for t in json.loads(Path(args.table).read_text(encoding="utf-8")):
    key = ":".join(str(k) for k in t["key"])
    c = consults[key]
    if len(c["plays"]) < 2:
        continue
    plays = {int(k): v for k, v in t["plays"].items()}
    table.append({"key": key, "level": t["key"][0], "seed": t["key"][1], "wave": c["wave"], "plays": plays,
                  "ids": [p["id"] for p in c["plays"]], "best": max(plays.values()), "rule": t["rule"],
                  "random": c["shadow"].get("random")})

# FixedSchedule from train seeds only.
gain = collections.defaultdict(list)
for t in table:
    if t["seed"] in train_seeds:
        for i, pid in enumerate(t["ids"]):
            gain[(t["level"], t["wave"], pid)].append(t["plays"][i] - t["plays"][0])
schedule = {}
for (level, wave, pid), gs in gain.items():
    m = statistics.mean(gs)
    if pid != "P0" and m > 0 and m > schedule.get((level, wave), ("P0", 0.0))[1]:
        schedule[(level, wave)] = (pid, m)

llm = json.loads(Path(args.llm).read_text(encoding="utf-8")) if args.llm else {}
lora = {}
for spec in args.lora:
    name, path = spec.split("=", 1)
    totals = collections.defaultdict(lambda: None)
    counts = collections.Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        s = row["scores"]
        top = max(s)
        w = [math.exp(max(x - top, -690.0)) for x in s]
        p = [x / sum(w) for x in w]
        acc = totals[row["group"]] or [0.0] * len(p)
        for position, i in enumerate(row["order"]):
            acc[i] += p[position]
        totals[row["group"]] = acc
        counts[row["group"]] += 1
    lora[name] = {g: max(range(len(acc)), key=lambda i: (acc[i], -i)) for g, acc in totals.items()}

held = [t for t in table if t["seed"] in hold_seeds]
brains = {"oracle": lambda t: max(t["plays"], key=lambda i: (t["plays"][i], -i)),
          "steady": lambda t: 0,
          "rule": lambda t: t["rule"] if t["rule"] is not None else 0,
          "random(shadow)": lambda t: t["random"] if t["random"] is not None else 0,
          "fixed_schedule": lambda t: (t["ids"].index(schedule[(t["level"], t["wave"])][0])
                                       if (t["level"], t["wave"]) in schedule
                                       and schedule[(t["level"], t["wave"])][0] in t["ids"] else 0)}
if llm:
    brains["8b_zero_shot"] = lambda t: llm.get(t["key"])
for name, choice in lora.items():
    brains[f"lora:{name}"] = (lambda ch: lambda t: ch.get(t["key"]))(choice)


def summary(name, fn, rows):
    regrets, diffs, moved, better, worse = [], [], 0, 0, 0
    for t in rows:
        i = fn(t)
        if i is None:
            continue
        regrets.append(t["best"] - t["plays"][i])
        d = t["plays"][i] - t["plays"][0]
        diffs.append(d)
        moved += i != 0
        better += d > 1e-9
        worse += d < -1e-9
    if not regrets:
        return None
    half = 1.96 * statistics.pstdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return {"brain": name, "consults": len(regrets), "mean_regret": round(statistics.mean(regrets), 4),
            "gain_vs_steady": round(statistics.mean(diffs), 4), "ci95": round(half, 4), "non_p0": moved,
            "better": better, "worse": worse}


out = {"held_out_consults": len(held), "train_consults": sum(t["seed"] in train_seeds for t in table),
       "headroom_heldout": round(statistics.mean(t["best"] - t["plays"][0] for t in held), 4) if held else None,
       "schedule_entries": len(schedule), "brains": []}
for name, fn in brains.items():
    s = summary(name, fn, held)
    if s:
        out["brains"].append(s)
        print(f"{name:18s} n={s['consults']:4d} regret {s['mean_regret']:.4f}  gain vs steady {s['gain_vs_steady']:+.4f}"
              f" +- {s['ci95']:.4f}  non-P0 {s['non_p0']:4d}  better {s['better']:3d} worse {s['worse']:3d}")
by_level = collections.defaultdict(list)
for t in held:
    by_level[t["level"]].append(t)
out["by_level"] = {}
for level, rows in sorted(by_level.items()):
    out["by_level"][level] = {name: summary(name, fn, rows) for name, fn in brains.items()}
print(json.dumps({k: v for k, v in out.items() if k not in ("brains", "by_level")}))
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
