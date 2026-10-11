"""elite-v2 demonstrations / DAgger for the operator networks, optionally with the human-timing host features.

usage: config_v2_collect.py kind=main|elite beta=<0..1> weights=<operator weights> portfolios=<portfolios json>
                            [features=timed_clicks,early_waves] [click_ticks=9] [plans=N] [challenge_plans=M]
                            [seeds=1001-1010] (out=<file> | apply=1)
kind=main : the main-campaign operator (layout v3, levels 1-12): the first N campaign-v1 portfolio plans of levels
            1-10, all 8 of level 12 (the operator won 32/80 against the executor's 54/80 there) and every level-11
            portfolio plan (the elite-v2 portfolio), profile [2.75, 1.0].
kind=elite: the elite operator (layout v4): the first N portfolio plans of every elite stage and the first M plans of
            every Heroic/Iron challenge, profile [3, 1.2, 13].
beta 1.0: the plan executor plays and labels every decision; 0: the network plays, the executor labels (DAgger).
apply=1 writes the live elite-v2 config and is refused while any runtime/rl/*/job.lock exists."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
args = dict(a.split("=", 1) for a in sys.argv[1:])
apply = args.get("apply") == "1"
if apply == ("out" in args):
    raise SystemExit("give exactly one of out=<file> or apply=1")
if apply and any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
kind, beta, weights = args["kind"], float(args["beta"]), args["weights"]
features = [f for f in args.get("features", "").split(",") if f]
click_ticks = int(args["click_ticks"]) if "click_ticks" in args else None
n = int(args.get("plans", 2 if kind == "main" else 3))
m = int(args.get("challenge_plans", 1))
lo, hi = (int(x) for x in args.get("seeds", "1001-1010").split("-"))
seeds = list(range(lo, hi + 1))
data = json.loads(Path(args["portfolios"]).read_text(encoding="utf-8"))
tasks = []
if kind == "main":
    v1 = json.loads((ROOT / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
    main = v1["jobs"]["native-final"]["campaign"]["plans"]
    for level in list(range(1, 11)) + [12]:
        for e in main[str(level)][:(None if level == 12 else n)]:
            genome = e["genome"] if isinstance(e, dict) and "genome" in e else e
            tasks += [{"level": level, "seed": s, "genome": genome} for s in seeds]
    for e in data["elite"]["11"]:
        tasks += [{"level": 11, "seed": s, "genome": e["genome"]} for s in seeds]
    rate = [2.75, 1.0]
    what = (f"the first {n} campaign-v1 portfolio plans of levels 1-10, all of level 12 and all {len(data['elite']['11'])} "
            "elite-v2 level-11 portfolio plans, for the main-campaign operator network (layout v3)")
elif kind == "elite":
    for level, entries in sorted(data["elite"].items(), key=lambda kv: int(kv[0])):
        if int(level) < 13:
            continue
        for e in entries[:n]:
            tasks += [{"level": int(level), "seed": s, "genome": e["genome"]} for s in seeds]
    for key, entries in sorted(data["challenges"].items()):
        level, mode = (int(x) for x in key.split(":"))
        for e in entries[:m]:
            genome = e["genome"] if isinstance(e, dict) and "genome" in e else e
            tasks += [{"level": level, "seed": s, "genome": genome, "mode": mode} for s in seeds]
    rate = [3, 1.2, 13]
    what = (f"the first {n} elite-v2 portfolio plans of every elite stage and the first {m} of every Heroic/Iron "
            "challenge, for the elite operator network (layout v4)")
else:
    raise SystemExit("kind must be main or elite")

p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-collect"]
job.update({"enabled": True, "max_wall_seconds": 3600 if kind == "main" else 9000, "total_wall_seconds": 43200,
            "max_games": len(tasks) + 20,
            "operator_weights": {"path": weights, "sha256": hashlib.sha256((ROOT / weights).read_bytes()).hexdigest()},
            "collect": {"tasks": tasks, "workers": 32, "profile_stars_per_level": rate, "dagger": {"beta": beta}}})
job.pop("host_features", None)
if features:
    job["host_features"] = features
protocol = dict(job["episode_protocol"])
protocol.pop("click_ticks", None)
if click_ticks:
    protocol["click_ticks"] = click_ticks
job["episode_protocol"] = protocol
job["purpose"] = (f"elite-v2 {kind} operator data: {what}, on train seeds {lo}-{hi}, profile {rate}; DAgger beta {beta} "
                  "(1.0: the plan executor plays and labels every decision; 0: the network plays, the executor labels)"
                  + (f"; with the human-timing host features {features}"
                     + (f" and {click_ticks}-tick trap clicks" if click_ticks else "") if features else ""))
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok", kind, len(tasks), "tasks, beta", beta, "features", features, "click_ticks", click_ticks)
