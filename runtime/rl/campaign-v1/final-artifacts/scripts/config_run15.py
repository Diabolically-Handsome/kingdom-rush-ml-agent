"""Run 15: robust search for levels 11/12 (mean fitness over train seeds 1001-1003), warm start from runs 12-14,
validation of the top 8 per level on train seeds 1004-1010."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
run15 = ("run 15: robust search for levels 11 and 12 (single-seed winners won only 0-4 of 9 other train seeds; "
         "level 12's wave composition varies by seed): fitness is the mean over train seeds 1001-1003, warm start "
         "from the 24 best plans per level of runs 12-14 re-played on all three, then validation of the 8 best "
         "on train seeds 1004-1010")
if run15 not in job["purpose"]:
    job["purpose"] += "; " + run15
job["max_wall_seconds"] = 9000
job["max_games"] = 1700
job["search"] = {"levels": [11, 12], "search_seed": 1001, "search_seeds": [1001, 1002, 1003], "population": 40,
                 "evaluations_per_level": 230, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 8, "rng_seed": "campaign-v1-search-15",
                 "warm_start": {"runs": ["native-search-ce87df5686784367b9a5499c8a52f66e",
                                         "native-search-e6aa0943993a4e77925310873b7d4c23",
                                         "native-search-1ee38d3eb2954a689dff7a67dea035ab"],
                                "top": 24, "reevaluate": True}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 15 configured")
