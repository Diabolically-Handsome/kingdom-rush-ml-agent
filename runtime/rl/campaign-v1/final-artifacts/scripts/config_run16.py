"""Run 16: level 12 only, mean fitness over train seeds 1001-1003; run 15's population adopted as measured
(same seeds, profile and protocol); validation of the 8 best by fitness on train seeds 1004-1010."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
run16 = ("run 16: level 12 only (its losses are final-wave damage races, Vez'nan escaping with 93-800 of 6666 HP): "
         "mean fitness over train seeds 1001-1003, run 15's population adopted as measured, 300 more plans, then the "
         "8 best by fitness (not only plans that won every search seed) validated on train seeds 1004-1010")
if run16 not in job["purpose"]:
    job["purpose"] += "; " + run16
job["max_wall_seconds"] = 7200
job["max_games"] = 1100
job["search"] = {"levels": [12], "search_seed": 1001, "search_seeds": [1001, 1002, 1003], "population": 40,
                 "evaluations_per_level": 300, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 8, "rng_seed": "campaign-v1-search-16",
                 "warm_start": {"runs": ["native-search-48a52f561338494fb802410be34e62b1"], "top": 40,
                                "reevaluate": False}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 16 configured")
