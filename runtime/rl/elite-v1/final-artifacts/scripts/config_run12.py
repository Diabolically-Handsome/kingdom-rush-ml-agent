"""Run 12: levels 24/13/15/26/19 on train seeds 1005-1008 with a working re-play warm start (patch_warm_reeval.py)
and, for levels 13 and 15, soft-cap twins of their best capped plans (patch_softcap_safety.py). Apply between jobs."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl.search import check_genome, genome_id  # noqa: E402

if (ROOT / "runtime/rl/elite-v1/job.lock").exists():
    raise SystemExit("refused: a job holds the lock")
RUNS = ROOT / "runtime/rl/elite-v1/runs"
CHALLENGE_RUNS = {"native-search-77f73818b99d43b98815025897d07bb3", "native-search-a9ceecb077f5459291a3768e0e2c6f84"}


def capped_best(level, count):
    """The best capped plans of every campaign-mode search (mean fitness less a short-record penalty)."""
    records = {}
    for run in RUNS.glob("native-search-*"):
        if run.name in CHALLENGE_RUNS:
            continue
        for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] not in ("search_eval", "search_validate") or row["payload"]["level"] != level:
                continue
            p = row["payload"]
            entry = records.setdefault(p["genome_id"], {"genome": p["genome"], "f": []})
            entry["f"].append(p["fitness"])
    rows = [(sum(e["f"]) / len(e["f"]) - 100 / len(e["f"]), gid, e["genome"]) for gid, e in records.items()
            if e["genome"].get("cap", 0) > 0 and len(e["f"]) >= 2]
    rows.sort(key=lambda r: (-r[0], r[1]))
    out, seen = [], set()
    for _, _, genome in rows:
        soft = check_genome({**genome, "cap": -genome["cap"]})
        if genome_id(soft) not in seen:
            seen.add(genome_id(soft))
            out.append(soft)
        if len(out) >= count:
            break
    return out


p = ROOT / "configs/phases/elite-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
runs = ["native-search-08f5b21f202b4b6596dbb2b4d5a8c619", "native-search-1d4786a9538b4ba7bce718a29f7fca4a",
        "native-search-c23208d7fb2d4362956640f540c6af48", "native-search-55aeddcb01824591bbf2c0fb950c669f",
        "native-search-f7e0e33fbba849399e9bda62a0b629de"]
seeds = {str(level): capped_best(level, 8) for level in (13, 15)}
job["search"] = {
    "levels": [24, 13, 15, 26, 19],
    "search_seed": 1005, "search_seeds": [1005, 1006, 1007, 1008],
    "population": 40, "evaluations_per_level": 300, "min_evaluations": 80,
    "stop_lives": 19, "stop_wins": 4, "stall_evaluations": 200,
    "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
    "validate_seeds": [1001, 1002, 1003, 1004, 1009, 1010], "validate_top": 12,
    "level_weights": {"24": 4, "13": 3, "15": 2, "26": 2, "19": 1},
    "warm_start": {"runs": runs, "top": 24, "reevaluate": True},
    "seed_plans": seeds,
    "rng_seed": "elite-v1-search-12",
}
job["max_wall_seconds"] = 7200
job["time_reserve_seconds"] = 1200
job["max_games"] = 8000
job["purpose"] += ("; elite run 12: levels 24, 13, 15, 26 and 19 on train seeds 1005-1008 (validated on the other six); "
                   "run 11 was stopped because its warm start imported nothing (only games on the current search seeds "
                   "counted), fixed so re-played plans come from any measured train game (the best 24 per level from "
                   "runs 5, 8, 9, 10, 11); levels 13 and 15 also start from soft-cap twins of their 8 best capped plans "
                   "(new gene value cap -N: builds resume while 1000+ gold is banked; capped plans reached the last "
                   "wave with 3,000-6,000 gold unspent and half the holders empty)")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok", {k: [g.get("cap") for g in v] for k, v in seeds.items()})
