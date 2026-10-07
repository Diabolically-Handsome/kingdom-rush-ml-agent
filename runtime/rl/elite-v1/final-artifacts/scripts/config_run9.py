"""Run 9: levels 24/19/26 at the refined option-A profile [3, 1.2, 13] (about 62 stars at level 24)."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
p = ROOT / "configs/phases/elite-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
runs = ["native-search-be0c008d6cc34b19b629c7cab95676d9", "native-search-6282b533cddf4e708e3f2ff68de3cc83",
        "native-search-40c03ca16bf9414594431eac37b4ba8f", "native-search-f7491b78a5e94b4581f6abdf72f59a52",
        "native-search-08f5b21f202b4b6596dbb2b4d5a8c619", "native-search-1d4786a9538b4ba7bce718a29f7fca4a"]
job["search"] = {
    "levels": [24, 19, 26],
    "search_seed": 1001, "search_seeds": [1001, 1002],
    "population": 40, "evaluations_per_level": 700, "min_evaluations": 100,
    "stop_lives": 19, "stop_wins": 4, "stall_evaluations": 400,
    "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
    "validate_seeds": [1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010], "validate_top": 12,
    "level_weights": {"24": 4, "19": 2, "26": 1},
    "warm_start": {"runs": runs, "top": 24, "reevaluate": True},
    "rng_seed": "elite-v1-search-9",
}
job["max_wall_seconds"] = 9000
job["time_reserve_seconds"] = 1500
job["max_games"] = 6500
job["purpose"] += ("; elite run 9: levels 24 (it gates 25, whose option-A portfolio covers 10/10 seeds, and 26), 19 and 26 at "
                   "a refined option-A estimate [3, 1.2, 13] (main levels at 3 stars after replays, the ~13 challenge "
                   "stars the validated challenge portfolios cover, and the stars of the ~5 elite stages won before: "
                   "about 62 stars at level 24); the best 24 plans per level from runs 1-8 re-played at this profile")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok")
