"""Create phase elite-v3 "commander" (configs/phases/elite-v3.json, configs/pools-elite-v3.json) from elite-v2.
Run once, with no job running."""
import copy
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
if any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
for name in ("configs/phases/elite-v3.json", "configs/pools-elite-v3.json"):
    if (ROOT / name).exists():
        raise SystemExit(f"refused: {name} exists")
APPROVAL_V3 = "[用户原话已省略 / user's message omitted]"

pools_v2 = json.loads((ROOT / "configs/pools-elite-v2.json").read_text(encoding="utf-8"))
pools = copy.deepcopy(pools_v2)
pools["pool_id"] = "KR1-elite-v3"
pools["supersedes"] = "KR1-elite-v2"
pools["pools"]["train"] = {"seeds": list(range(1001, 1011)) + list(range(1101, 1141)) + list(range(1201, 1301)),
                           "note": "1001-1010 and 1101-1140 as before (plan evidence); 1201-1300 added for the commander's "
                                   "probes and forks"}
pools["pools"]["evaluation"] = {"seeds": list(range(7201, 7261)), "never_in_gradients": True}
pools["pools"]["final_campaign_run"] = {"seeds": list(range(8201, 8206)), "access": "one_frozen_evaluation"}
retired = pools_v2["pools"]["retired"]["seeds"] + list(range(7101, 7121)) + list(range(8101, 8106))
pools["pools"]["retired"] = {"seeds": retired,
                             "note": "earlier phases' evaluation and final seeds (elite-v2: 7101-7120 rehearsals and "
                                     "evaluation, 8101-8105 its one-shot final); never used in elite-v3"}
all_seeds = [s for p in pools["pools"].values() for s in p["seeds"]]
assert len(all_seeds) == len(set(all_seeds)), "seed pools overlap"
(ROOT / "configs/pools-elite-v3.json").write_text(json.dumps(pools, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")

cfg = json.loads((ROOT / "configs/phases/elite-v2.json").read_text(encoding="utf-8"))
v3 = copy.deepcopy(cfg)
v3["phase_id"] = "elite-v3"
v3["state_dir"] = "runtime/rl/elite-v3"
v3["pins_manifest"] = "runtime/rl/elite-v3/pins.json"
v3["pools_path"] = "configs/pools-elite-v3.json"
v3["authorization"] = cfg["authorization"] + [APPROVAL_V3]
v3["goal"] = {**cfg["goal"],
              "summary": "指挥官阶段：给智能体补上人类能用而它没有的能力（英雄移动、火雨与援兵自由落点，动作范围 v4），并让语言模型"
                         "在局内做需要判断的战术决定；用同种子消融（8B / 规则 / 随机 / 被动）和分叉遗憾值证明或否定它的贡献；"
                         "本窗口只在训练种子上做开发实验，不做最终评测",
              "heroes": "all 13 from level 12 (hero room gate by progress; natively verified 2026-10-10)"}
for name, job in v3["jobs"].items():
    job["approval"] = APPROVAL_V3
    job["enabled"] = False  # each job is configured and enabled when it is used
(ROOT / "configs/phases/elite-v3.json").write_text(json.dumps(v3, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
print("ok", len(pools["pools"]["train"]["seeds"]), "train seeds; jobs", list(v3["jobs"]))
