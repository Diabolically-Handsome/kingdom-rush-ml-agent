"""Set the elite-v1 native-collect job from build_portfolios.py output.

usage: config_collect.py <portfolios.json> <elite_A|elite_B> [elite plans per level] [challenge plans] [rate json]
Demonstrations of the plan executor on train seeds 1001-1010 for the elite operator (layout v4): the top plans of
each elite stage of the chosen family and of each Heroic/Iron challenge.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
family = sys.argv[2]
elite_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
challenge_n = int(sys.argv[4]) if len(sys.argv) > 4 else 2
rate = json.loads(sys.argv[5]) if len(sys.argv) > 5 else [3, 1.0, 12]
APPROVAL = ("[用户原话已省略 / user's message omitted]"
            "（2026-10-06 用户批准精英关阶段：13-26 关、战役模式、普通难度；最终评测从新存档打完全部 26 关；"
            "可加兵营集结点动作；自主推进至 2026-10-07 20:00）")
tasks, seen = [], set()
for fam in family.split("+"):
    for level, entries in sorted(data[fam].items(), key=lambda kv: int(kv[0])):
        for entry in entries[:elite_n]:
            key = (int(level), json.dumps(entry["genome"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            for seed in range(1001, 1011):
                tasks.append({"level": int(level), "seed": seed, "genome": entry["genome"]})
for key, entries in sorted(data["challenges"].items()):
    level, mode = (int(x) for x in key.split(":"))
    for entry in entries[:challenge_n]:
        for seed in range(1001, 1011):
            tasks.append({"level": level, "seed": seed, "genome": entry["genome"], "mode": mode})
p = ROOT / "configs/phases/elite-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
cfg["jobs"]["native-collect"] = {
    "enabled": True, "approval": APPROVAL, "gpu": False, "optimizer_steps": 0, "serial": False,
    "rng_mode": "isolate_sound+stable_pairs", "action_scope": "v3", "replay_fraction": 0.0,
    "time_reserve_seconds": 120,
    "episode_protocol": {"wait_ticks": 120, "max_interval_ticks": 600, "max_ticks": 240000, "max_decisions": 3000},
    "max_jobs": 8,
    "purpose": (f"record plan-executor demonstrations (state, legal menu, plan step, choice) for behaviour cloning of the "
                f"elite operator network (feature layout v4): the top {elite_n} portfolio plans of each elite stage "
                f"({family}) and the top {challenge_n} of each Heroic/Iron challenge, on train seeds 1001-1010, profile "
                f"{rate}"),
    "max_wall_seconds": 7200, "total_wall_seconds": 21600, "max_games": len(tasks) + 20,
    "collect": {"tasks": tasks, "workers": 32, "profile_stars_per_level": rate},
}
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("collect tasks", len(tasks), "elite", sum("mode" not in t for t in tasks), "challenge", sum("mode" in t for t in tasks))
