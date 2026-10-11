"""elite-v2 run 3: searches with the human-timing host features (owner-approved audit items 3-4, 2026-10-10).

usage: config_v2_run3.py kind=main|elite portfolios=<build_portfolios_v2 output> [features=timed_clicks,early_waves]
                         [click_ticks=9] [hours=H] (out=<file> | apply=1)
kind=main : main level 11 at the main-campaign profile; seeded with its portfolio plans and (with early_waves) each
            plan's early=2 twin; scored on 6 fresh train seeds 1121-1126, the best 16 validated on 1001-1010 and
            1127-1140.
kind=elite: elite stages 22, 20 and 21 at the option-A profile. 20 and 21 seeded with their portfolio plans and early=2
            twins; 22 starts from random plans only (its elite-v1/v2 seeds collapsed onto one losing mage opening).
            Scored on 4 fresh train seeds 1121-1124, the best 12 validated on 1001-1010 and 1125-1134.
apply=1 writes the live elite-v2 config and is refused while any runtime/rl/*/job.lock exists."""
import copy
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
kind = args["kind"]
features = [f for f in args.get("features", "").split(",") if f]
click_ticks = int(args["click_ticks"]) if "click_ticks" in args else None
hours = float(args.get("hours", 2 if kind == "main" else 5))
portfolios = json.loads(Path(args["portfolios"]).read_text(encoding="utf-8"))["elite"]


def with_twins(level):
    plans = [copy.deepcopy(e["genome"]) for e in portfolios.get(str(level), [])]
    if "early_waves" in features:
        plans += [{**copy.deepcopy(g), "early": 2} for g in plans if g.get("early") != 2]
    return plans


p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
job["enabled"] = True
if kind == "main":
    seed_plans = {"11": with_twins(11)}
    job["search"] = {
        "levels": [11],
        "search_seed": 1121, "search_seeds": list(range(1121, 1127)),
        "population": 40, "evaluations_per_level": 800, "min_evaluations": 150,
        "stop_lives": 19, "stop_wins": 8, "stall_evaluations": 350,
        "profile_stars_per_level": [2.75, 1.0], "workers": 48,
        "validate_seeds": list(range(1001, 1011)) + list(range(1127, 1141)), "validate_top": 16,
        "seed_plans": seed_plans,
        "rng_seed": "elite-v2-search-3-main",
    }
    games = 800 * 6 + 16 * 24 + 200
    what = ("main level 11 (the campaign's weakest link) at the main-campaign profile, seeded with its 10-plan "
            "portfolio" + (" and each plan's early=2 twin" if "early_waves" in features else "")
            + "; scored on 6 fresh train seeds (1121-1126), the best 16 validated on 24 more (1001-1010, 1127-1140)")
elif kind == "elite":
    seed_plans = {"20": with_twins(20), "21": with_twins(21)}
    seed_plans = {k: v for k, v in seed_plans.items() if v}
    job["search"] = {
        "levels": [22, 20, 21],
        "search_seed": 1121, "search_seeds": list(range(1121, 1125)),
        "population": 60, "evaluations_per_level": 1800, "min_evaluations": 300,
        "stop_lives": 19, "stop_wins": 6, "stall_evaluations": 700,
        "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
        "validate_seeds": list(range(1001, 1011)) + list(range(1125, 1135)), "validate_top": 12,
        "level_weights": {"22": 4, "20": 2, "21": 3},
        "seed_plans": seed_plans,
        "rng_seed": "elite-v2-search-3-elite",
    }
    games = 1800 * 4 * 3 + 3 * 12 * 20 + 300
    what = ("elite stages 22 (never won; a fresh random population, as its seeded plans had collapsed onto one losing "
            "mage opening), 20 and 21 (seeded with their portfolio plans"
            + (" and early=2 twins" if "early_waves" in features else "") + "); scored on 4 fresh train seeds "
            "(1121-1124), the best 12 validated on 20 more (1001-1010, 1125-1134); option-A profile [3, 1.2, 13]")
else:
    raise SystemExit("kind must be main or elite")
job["max_wall_seconds"] = int(hours * 3600)
job["time_reserve_seconds"] = 1500 if kind == "main" else 2400
job["max_games"] = games
job.pop("host_features", None)
if features:
    job["host_features"] = features
protocol = dict(job["episode_protocol"])
protocol.pop("click_ticks", None)
if click_ticks:
    protocol["click_ticks"] = click_ticks
job["episode_protocol"] = protocol
job["purpose"] = (f"elite-v2 run 3 ({kind}): " + what
                  + (f"; with the human-timing host features {features}"
                     + (f" and {click_ticks}-tick trap clicks" if click_ticks else "") if features else ""))
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok", kind, {k: len(v) for k, v in job["search"]["seed_plans"].items()}, "max_games", games,
      "hours", hours, "features", features)
