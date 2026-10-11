"""elite-v2 run 2 (overnight): elite stages 22, 20, 21 with all 13 heroes, plus 26, 13, 15, 24 at lower weight.
Apply only between jobs."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
if any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
seeds = json.loads((SCRATCH / "v2_elite_seed_plans.json").read_text(encoding="utf-8"))
levels = [22, 20, 21, 26, 13, 15, 24]
job["enabled"] = True
job["search"] = {
    "levels": levels,
    "search_seed": 1101, "search_seeds": [1101, 1102, 1103, 1104],
    "population": 40, "evaluations_per_level": 1500, "min_evaluations": 200,
    "stop_lives": 19, "stop_wins": 6, "stall_evaluations": 600,
    "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
    "validate_seeds": list(range(1001, 1011)) + list(range(1105, 1115)), "validate_top": 12,
    "level_weights": {"22": 4, "20": 4, "21": 2, "26": 2, "13": 1, "15": 1, "24": 1},
    "seed_plans": {str(lv): seeds[str(lv)] for lv in levels},
    "rng_seed": "elite-v2-search-2",
}
job["max_wall_seconds"] = 36000
job["time_reserve_seconds"] = 2400
job["max_games"] = 46000
job["purpose"] = ("elite-v2 run 2 (overnight): elite stages 22 and 20 (never won so far) and 21 (opens after 20), now with "
                  "all 13 heroes (the hero room offers the 7 others from level 12 on; verified natively), plus 26, 13, "
                  "15 and 24 at lower weight to see what the new heroes add. Plans scored on 4 fresh train seeds "
                  "(1101-1104), the best 12 per level validated on 20 more (1001-1010, 1105-1114); option-A star "
                  "profile [3, 1.2, 13]; seeded with the 20 best elite-v1 plans of each level and the 6 best each with "
                  "every new hero")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok", {lv: len(seeds[str(lv)]) for lv in levels})
