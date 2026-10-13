"""LoRA holdout predictions -> a choice file in the offline scorer's format, so v3_skill_tests.py scores it like the
zero-shot models. Every row's label log-likelihoods -> softmax (as the worker does) -> mapped to plays through the row's
cyclic order -> averaged over the group's rotations in shift order -> argmax, ties to the lower index (exactly
strategy_brain.LanguageBrain._ask). Refuses incomplete or duplicated rotation sets.

usage: lora_choices.py <predictions.jsonl> <out choices.json>   (also writes <out>.records.json with p per consult)"""
import collections
import json
import math
import sys
from pathlib import Path

ROTATIONS = 6
preds, out = Path(sys.argv[1]), Path(sys.argv[2])
groups = collections.defaultdict(dict)
for line in preds.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row["shift"] in groups[row["group"]]:
        raise SystemExit(f"duplicate rotation {row['group']} r{row['shift']}")
    groups[row["group"]][row["shift"]] = row
choices, records = {}, []
for group, rows in sorted(groups.items()):
    n = len(next(iter(rows.values()))["labels"])
    if sorted(rows) != list(range(min(n, ROTATIONS))):
        raise SystemExit(f"incomplete rotations for {group}: {sorted(rows)}")
    total = [0.0] * n
    for shift in sorted(rows):
        row = rows[shift]
        top = max(row["scores"])
        w = [math.exp(max(s - top, -690.0)) for s in row["scores"]]
        for position, i in enumerate(row["order"]):
            total[i] += w[position] / sum(w)
    p = [x / len(rows) for x in total]
    index = max(range(n), key=lambda i: (p[i], -i))
    choices[group] = index
    records.append({"key": group, "choice": index, "p": p, "n": n})
out.write_text(json.dumps(choices, indent=1), encoding="utf-8")
out.with_suffix(".records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
print(json.dumps({"consults": len(choices), "non_p0": sum(1 for v in choices.values() if v != 0)}))
