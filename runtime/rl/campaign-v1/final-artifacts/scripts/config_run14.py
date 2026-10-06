"""Run 14: the click action (finish a downed boss, break tower traps) on levels 9/11/12; the best plans of
runs 12-13 re-played under it, search continued, then validation of the top 8 per level."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
run14 = ("run 14: new v2 action click_entity, the GUI click a game script waits for (J.T. knocked down waits "
         "for the finishing tap, so 3 earlier level-9 games ended at the tick cap with the boss at 4-24 HP and the "
         "level never closing; J.T.'s ice keeps a tower frozen until it is clicked 3 times); the 40 best plans per "
         "level of runs 12-13 are re-played with clicks, the search continues, then the 8 best per level are "
         "validated on the other train seeds")
if run14 not in job["purpose"]:
    job["purpose"] += "; " + run14
job["max_wall_seconds"] = 9000
job["max_games"] = 2700
job["max_jobs"] = 24
job["search"] = {"levels": [9, 11, 12], "search_seed": 1001, "search_seeds": [1001], "population": 40,
                 "evaluations_per_level": 700, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 8, "rng_seed": "campaign-v1-search-14",
                 "warm_start": {"runs": ["native-search-ce87df5686784367b9a5499c8a52f66e",
                                         "native-search-e6aa0943993a4e77925310873b7d4c23"],
                                "top": 40, "reevaluate": True}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 14 configured")
