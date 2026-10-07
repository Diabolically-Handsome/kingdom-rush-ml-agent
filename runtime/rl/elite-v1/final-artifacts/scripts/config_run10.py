"""Run 10: levels 24/19/26/17 with 4 search seeds (run 9's plans won search seeds 1001/1002 but rarely the others),
twin-free populations (patch_dedupe.py), option-A profile [3, 1.2, 13]. Apply only between jobs."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
if (ROOT / "runtime/rl/elite-v1/job.lock").exists():
    raise SystemExit("refused: a job holds the lock")
p = ROOT / "configs/phases/elite-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
runs = ["native-search-be0c008d6cc34b19b629c7cab95676d9", "native-search-6282b533cddf4e708e3f2ff68de3cc83",
        "native-search-40c03ca16bf9414594431eac37b4ba8f", "native-search-f7491b78a5e94b4581f6abdf72f59a52",
        "native-search-08f5b21f202b4b6596dbb2b4d5a8c619", "native-search-1d4786a9538b4ba7bce718a29f7fca4a",
        "native-search-c23208d7fb2d4362956640f540c6af48"]
job["search"] = {
    "levels": [24, 19, 26, 17],
    "search_seed": 1001, "search_seeds": [1001, 1002, 1003, 1004],
    "population": 40, "evaluations_per_level": 400, "min_evaluations": 80,
    "stop_lives": 19, "stop_wins": 4, "stall_evaluations": 250,
    "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
    "validate_seeds": [1005, 1006, 1007, 1008, 1009, 1010], "validate_top": 12,
    "level_weights": {"24": 4, "19": 2, "26": 2, "17": 1},
    "warm_start": {"runs": runs, "top": 16, "reevaluate": True},
    "rng_seed": "elite-v1-search-10",
}
job["max_wall_seconds"] = 7800
job["time_reserve_seconds"] = 1200
job["max_games"] = 7600
job["purpose"] += ("; elite run 10: levels 24, 19, 26 and 17 with 4 search seeds (run 9's plans won both of its 2 "
                   "search seeds but few of the 8 validation seeds: selection on more seeds favours plans that hold on "
                   "unseen ones), populations without behavioural twins (an exact repeat of a kept plan's mean "
                   "fitness), option-A profile [3, 1.2, 13]; the best 16 plans per level from runs 1-9 re-played on "
                   "the 4 search seeds")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok")
