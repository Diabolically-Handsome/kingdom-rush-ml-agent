"""Read-only: holdout (seeds 1009/1010) accuracy of operator weights per level and action, plus 'same action
type' accuracy (any click counts for a click). usage: op_acc_by_level.py <weights> <phase/run> ..."""
import collections, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from alpharush_rl.operator_net import ACTIONS, OptionScorer, load_rows
import importlib.util
spec = importlib.util.spec_from_file_location("trainop", ROOT / "tools/train-operator.py")
trainop = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainop)
weights = json.loads((ROOT / sys.argv[1]).read_text(encoding="utf-8"))
net = OptionScorer.from_json(weights) if hasattr(OptionScorer, "from_json") else None
eps = trainop.load(sys.argv[2:], only_won=True, runs=ROOT / "runtime/rl/elite-v2/runs")
stat = collections.defaultdict(lambda: [0, 0, 0])
for path, rows, meta in eps:
    if meta["seed"] not in (1009, 1010):
        continue
    for g, o, target in rows:
        a = trainop.chosen_action(o, target)
        if a == trainop.WAIT:
            continue
        pick = int(np.argmax(net.scores(g, o)))
        s = stat[(meta["level"], ACTIONS[a])]
        s[0] += 1; s[1] += pick == target; s[2] += trainop.chosen_action(o, pick) == a
for k in sorted(stat):
    n, exact, kind = stat[k]
    print(k, n, f"exact {exact / n:.2f}", f"same-kind {kind / n:.2f}")
