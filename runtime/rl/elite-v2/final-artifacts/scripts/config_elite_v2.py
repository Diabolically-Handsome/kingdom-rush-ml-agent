"""Create phase elite-v2 (configs/phases/elite-v2.json, configs/pools-elite-v2.json) from elite-v1. Run once, no job."""
import copy
import json
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
if any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
APPROVAL_V2 = "[用户原话已省略 / user's message omitted]"

pools_v1 = json.loads((ROOT / "configs/pools-elite-v1.json").read_text(encoding="utf-8"))
pools = copy.deepcopy(pools_v1)
pools["pool_id"] = "KR1-elite-v2"
pools["supersedes"] = "KR1-elite-v1"
pools["pools"]["train"] = {"seeds": list(range(1001, 1011)) + list(range(1101, 1141)),
                           "note": "1001-1010 as before; 1101-1140 added so seed-sensitive levels can be searched and "
                                   "validated on many more training seeds"}
pools["pools"]["evaluation"] = {"seeds": list(range(7101, 7121)), "never_in_gradients": True}
pools["pools"]["final_campaign_run"] = {"seeds": list(range(8101, 8106)), "access": "one_frozen_evaluation"}
retired = pools_v1["pools"]["retired"]["seeds"] + list(range(7001, 7021)) + list(range(8001, 8006))
pools["pools"]["retired"] = {"seeds": retired,
                             "note": "earlier phases' evaluation and final seeds (elite-v1: 7001-7020 rehearsals and "
                                     "evaluation, 8001-8005 its one-shot final); never used in elite-v2"}
(ROOT / "configs/pools-elite-v2.json").write_text(json.dumps(pools, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")

cfg = json.loads((ROOT / "configs/phases/elite-v1.json").read_text(encoding="utf-8"))
v2 = copy.deepcopy(cfg)
v2["phase_id"] = "elite-v2"
v2["state_dir"] = "runtime/rl/elite-v2"
v2["pins_manifest"] = "runtime/rl/elite-v2/pins.json"
v2["pools_path"] = "configs/pools-elite-v2.json"
v2["authorization"] = cfg["authorization"] + [APPROVAL_V2]
v2["goal"] = {**cfg["goal"],
              "summary": "在 elite-v1 的基础上：补强主线第 11 关（主线失败即结束战役），攻克精英第 22、20 关（第 21 关在第 20 关可用后），"
                         "打完主线后使用全部 13 个英雄；最终仍从新存档按顺序打完全部 26 关（战役模式、普通难度、同一架构）",
              "heroes": "all 13 from level 12 (hero room gate by progress; natively verified 2026-10-10)"}
for name, job in v2["jobs"].items():
    job["approval"] = APPROVAL_V2
    job["enabled"] = False  # each job is configured and enabled when it is used
(ROOT / "configs/phases/elite-v2.json").write_text(json.dumps(v2, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
print("ok", len(pools["pools"]["train"]["seeds"]), "train seeds; jobs", list(v2["jobs"]))
