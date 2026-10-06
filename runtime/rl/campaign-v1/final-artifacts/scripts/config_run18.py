"""Run 18: level 11 robust search, mean fitness over train seeds 1001-1005; best plans of runs 15/17 re-played on
all five; validation of the 10 best by fitness on train seeds 1006-1010."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
run18 = ("run 18: level 11 robust search (its best plan won 8 of 10 train seeds with only 3-9 lives left and lost "
         "at the decisive waves 9 and 13 elsewhere, a fragile plan): mean fitness over train seeds 1001-1005, the 24 "
         "best plans of runs 15 and 17 re-played on all five, 250 more plans, then the 10 best validated on train "
         "seeds 1006-1010")
if run18 not in job["purpose"]:
    job["purpose"] += "; " + run18
job["max_wall_seconds"] = 7200
job["max_games"] = 1600
job["search"] = {"levels": [11], "search_seed": 1001, "search_seeds": [1001, 1002, 1003, 1004, 1005],
                 "population": 40, "evaluations_per_level": 274, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 10, "rng_seed": "campaign-v1-search-18",
                 "warm_start": {"runs": ["native-search-48a52f561338494fb802410be34e62b1",
                                         "native-search-a674898ff9ae4d6b9c716e66cc777e73"],
                                "top": 24, "reevaluate": True}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 18 configured")
