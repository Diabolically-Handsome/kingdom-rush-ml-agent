"""elite-v2 run 1: main level 11 robustness search (6 search seeds, 24 validation seeds). Apply only between jobs."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
if any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
seeds = json.loads((SCRATCH / "v2_l11_seed_plans.json").read_text(encoding="utf-8"))
job["enabled"] = True
job["search"] = {
    "levels": [11],
    "search_seed": 1101, "search_seeds": [1101, 1102, 1103, 1104, 1105, 1106],
    "population": 40, "evaluations_per_level": 700, "min_evaluations": 120,
    "stop_lives": 19, "stop_wins": 8, "stall_evaluations": 350,
    "profile_stars_per_level": [2.75, 1.0], "workers": 48,
    "validate_seeds": list(range(1001, 1011)) + list(range(1107, 1121)), "validate_top": 16,
    "seed_plans": {"11": seeds},
    "rng_seed": "elite-v2-search-1",
}
job["max_wall_seconds"] = 7200
job["time_reserve_seconds"] = 1500
job["max_games"] = 6000
job["purpose"] = ("elite-v2 run 1: main level 11, the campaign's weakest link (in the elite-v1 one-shot final two of five "
                  "seeds lost it with all 8 portfolio plans, which ended their campaigns). Plans are scored on 6 fresh "
                  "train seeds (1101-1106) and the best 16 validated on 24 more (1001-1010, 1107-1120), to build a "
                  "10-plan retry portfolio of plans that hold on unseen seeds. Same plan genes as campaign-v1 (the "
                  "main-campaign operator network executes main levels), main levels at 2.75 stars (about 28 stars at "
                  "level 11, as in a real campaign); seeded with the 8 portfolio plans and the best campaign-v1 plans")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok", len(seeds), "seed plans")
