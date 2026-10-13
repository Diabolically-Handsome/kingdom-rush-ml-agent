"""elite-v3b calibration (owner's point 5, done by a fresh blind subagent): draw 20 consults reproducibly and, later,
score the answers. The draw never looks at branch outcomes: per level (7 levels) 3 consults (level 26: 2), alternating
between won and lost base games, within each stratum a seeded uniform choice; wave and alarm consults as they come.

usage: v3b_calibration.py draw <requests dir> <out json>
       v3b_calibration.py score <draw json> <answers json> <fork table json>   (answers: {"<key>": "P3", ...})"""
import json
import random
import statistics
import sys
from pathlib import Path

SEED = 20261012
PER_LEVEL = {15: 3, 17: 3, 19: 3, 20: 3, 21: 3, 22: 3, 26: 2}
RUNS = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v3\runs")
S1B = "native-eval-d043199d066d4c6db33991b4685ff6f3"


def base_won():
    out = {}
    for line in (RUNS / S1B / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if '"cmd_steady"' in line:
            r = json.loads(line)["payload"]
            if r["player"] == "cmd_steady":
                out[f"{r['level']}:{r['seed']}:{r['genome_id']}"] = bool((r["result"].get("outcome") or {}).get("level_won"))
    return out


def draw(requests_dir, out_path):
    rows = {json.loads(l)["key"]: json.loads(l) for l in (Path(requests_dir) / "rich_reason.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()}
    won = base_won()
    rng = random.Random(SEED)
    picked = []
    for level, k in PER_LEVEL.items():
        strata = {True: sorted(key for key in rows if key.startswith(f"{level}:") and won[key.rsplit(":", 1)[0]]),
                  False: sorted(key for key in rows if key.startswith(f"{level}:") and not won[key.rsplit(":", 1)[0]])}
        order = [True, False] if rng.random() < 0.5 else [False, True]
        for i in range(k):
            pool = strata[order[i % 2]] or strata[not order[i % 2]]
            key = rng.choice(pool)
            pool.remove(key)
            picked.append(key)
    out = [{"key": key, "system": rows[key]["system"], "prompt": rows[key]["reason_user"], "ids": rows[key]["ids"]}
           for key in picked]
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"drawn": len(out), "keys": picked}))


def score(draw_path, answers_path, table_path):
    drawn = json.loads(Path(draw_path).read_text(encoding="utf-8"))
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    table = {":".join(map(str, e["key"])): e for e in json.loads(Path(table_path).read_text(encoding="utf-8"))}
    rows = []
    for item in drawn:
        y = [table[item["key"]]["plays"][str(j)] for j in range(len(item["ids"]))]
        pid = answers.get(item["key"])
        j = item["ids"].index(pid) if pid in item["ids"] else None
        rows.append({"key": item["key"], "answer": pid, "deviates": j not in (None, 0),
                     "m1": (y[j] - statistics.mean(y[1:])) if j not in (None, 0) else None,
                     "m2": (y[j] - y[0]) if j is not None else None})
    m1 = [r["m1"] for r in rows if r["m1"] is not None]
    print(json.dumps({"answered": sum(r["answer"] is not None for r in rows), "deviation_rate":
                      round(sum(r["deviates"] for r in rows) / len(rows), 3),
                      "m1_mean_qualitative": round(statistics.mean(m1), 4) if m1 else None,
                      "m2_mean_qualitative": round(statistics.mean([r["m2"] for r in rows if r["m2"] is not None]), 4)},
                     indent=1))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "draw":
        draw(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
