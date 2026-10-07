"""Run 11: levels 24/26/19 searched on train seeds 1005-1008 (runs 9/10 searched 1001-1004), validated on the other six,
to widen the retry portfolios of the seed-sensitive levels (level-24 plans win ~6% of the seeds they were not chosen
on). Option-A profile [3, 1.2, 13]. Apply only between jobs."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
if (ROOT / "runtime/rl/elite-v1/job.lock").exists():
    raise SystemExit("refused: a job holds the lock")
p = ROOT / "configs/phases/elite-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
runs = ["native-search-c23208d7fb2d4362956640f540c6af48", "native-search-55aeddcb01824591bbf2c0fb950c669f",
        "native-search-1d4786a9538b4ba7bce718a29f7fca4a", "native-search-08f5b21f202b4b6596dbb2b4d5a8c619"]
job["search"] = {
    "levels": [24, 13, 26, 19],
    "search_seed": 1005, "search_seeds": [1005, 1006, 1007, 1008],
    "population": 40, "evaluations_per_level": 400, "min_evaluations": 80,
    "stop_lives": 19, "stop_wins": 4, "stall_evaluations": 250,
    "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
    "validate_seeds": [1001, 1002, 1003, 1004, 1009, 1010], "validate_top": 12,
    "level_weights": {"24": 4, "13": 3, "26": 2, "19": 1},
    "warm_start": {"runs": runs, "top": 16, "reevaluate": True},
    "rng_seed": "elite-v1-search-11",
}
job["max_wall_seconds"] = 7200
job["time_reserve_seconds"] = 1200
job["max_games"] = 8000
job["purpose"] += ("; elite run 11: levels 24, 13, 26 and 19 searched on train seeds 1005-1008 (runs 9 and 10 searched "
                   "1001-1004) and validated on the other six, to widen the retry portfolios of these seed-sensitive "
                   "levels (level-24 portfolio plans won 3 of 52 games on seeds they were not chosen on) and to retry level 13, whose best plans bring the 18,000-HP boss Sarelgaz to its last 4-30% (403 run-8 games on train seeds reached the final wave 20 without a win); the best 16 "
                   "plans per level from runs 5, 8, 9 and 10 re-played on the new search seeds")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok")
