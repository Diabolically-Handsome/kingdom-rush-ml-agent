"""Run 19: level 12 robust search, mean fitness over train seeds 1001-1005; best plans of runs 14-16 re-played on
all five; validation of the 10 best by fitness on train seeds 1006-1010."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
run19 = ("run 19: level 12 robust search after the evaluation-seed rehearsal cleared 5 of 10 campaigns and lost 4 at "
         "level 12 (final-wave escapes, or early waves on one seed): mean fitness over train seeds 1001-1005, the 24 "
         "best plans of runs 14-16 re-played on all five, 200 more plans, then the 10 best validated on train seeds "
         "1006-1010")
if run19 not in job["purpose"]:
    job["purpose"] += "; " + run19
job["max_wall_seconds"] = 7200
job["max_games"] = 1300
job["search"] = {"levels": [12], "search_seed": 1001, "search_seeds": [1001, 1002, 1003, 1004, 1005],
                 "population": 40, "evaluations_per_level": 224, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 10, "rng_seed": "campaign-v1-search-19",
                 "warm_start": {"runs": ["native-search-1ee38d3eb2954a689dff7a67dea035ab",
                                         "native-search-48a52f561338494fb802410be34e62b1",
                                         "native-search-f2ccede1c003432996388a5e7dbd2100"],
                                "top": 24, "reevaluate": True}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 19 configured")
