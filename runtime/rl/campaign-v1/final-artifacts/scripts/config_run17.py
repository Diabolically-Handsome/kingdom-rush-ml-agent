"""Run 17: validation pass for level 11: run 15's population adopted as measured, one new plan, then the 12
best by mean fitness over seeds 1001-1003 validated on train seeds 1004-1010 (a retry portfolio)."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
run17 = ("run 17: validation pass for level 11: run 15's population adopted as measured, one new plan, then the "
         "12 best by mean fitness over seeds 1001-1003 validated on train seeds 1004-1010 for the retry portfolio")
if run17 not in job["purpose"]:
    job["purpose"] += "; " + run17
job["max_wall_seconds"] = 3600
job["max_games"] = 120
job["search"] = {"levels": [11], "search_seed": 1001, "search_seeds": [1001, 1002, 1003], "population": 40,
                 "evaluations_per_level": 1, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 12, "rng_seed": "campaign-v1-search-17",
                 "warm_start": {"runs": ["native-search-48a52f561338494fb802410be34e62b1"], "top": 40,
                                "reevaluate": False}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 17 configured")
