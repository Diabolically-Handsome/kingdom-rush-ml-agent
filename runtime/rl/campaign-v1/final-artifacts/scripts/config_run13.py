"""Run 13: levels 9/11/12, population adopted exactly from run 12 (same seed, profile and protocol), every
adopted plan also queued with the Rain-of-Fire-first star allocation, then validation on seeds 1002-1010."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
run12 = sys.argv[1]
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
seen, parts = set(), []
for part in job["purpose"].split("; "):
    if part not in seen:
        seen.add(part)
        parts.append(part)
run13 = ("run 13: star allocation packages (the upgrades screen's reset refunds every star, so a plan may "
         "name the allocation it is played with); the level 9, 11 and 12 populations of run 12 are adopted as "
         "measured and each adopted plan is also played with Rain of Fire upgraded first, then validation")
if run13 not in parts:
    parts.append(run13)
job["purpose"] = "; ".join(parts)
job["max_wall_seconds"] = 9000
job["max_games"] = 3500
job["search"] = {"levels": [9, 11, 12], "search_seed": 1001, "search_seeds": [1001], "population": 40,
                 "evaluations_per_level": 1000, "min_evaluations": 0, "stop_lives": 18,
                 "profile_stars_per_level": 2.75, "workers": 48, "validate_seeds": list(range(1001, 1011)),
                 "validate_top": 5, "rng_seed": "campaign-v1-search-13",
                 "warm_start": {"runs": [run12], "top": 40, "reevaluate": False, "packages": ["rain"]}}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("run 13 configured from", run12)
