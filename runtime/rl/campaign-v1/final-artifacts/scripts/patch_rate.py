"""collect/eval jobs accept an average star rate (1..3) like the search job."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", "C:/Users/<user>/Documents/AlphaRush"))


def patch(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


for path in ("alpharush_rl/collect_job.py", "alpharush_rl/eval_job.py"):
    patch(path, [(
        '''    if spec["profile_stars_per_level"] not in (1, 2, 3):
        issues.append("profile_stars_per_level must be 1, 2 or 3")''',
        '''    rate = spec["profile_stars_per_level"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 1 <= rate <= 3:
        issues.append("profile_stars_per_level must be a number from 1 to 3 (an average star rate)")''')])
print("rate patched")

patch("tests/test_campaign_run.py", [
    ('''        dup = {**spec, "tasks": spec["tasks"] + spec["tasks"][:1]}
        self.assertIsNone(collect_job.check_collect(dup, POOLS)[0])''',
     '''        dup = {**spec, "tasks": spec["tasks"] + spec["tasks"][:1]}
        self.assertIsNone(collect_job.check_collect(dup, POOLS)[0])
        self.assertEqual([], collect_job.check_collect({**spec, "profile_stars_per_level": 2.75}, POOLS)[1])
        for rate in (0.5, 3.5, True, "2"):
            self.assertIsNone(collect_job.check_collect({**spec, "profile_stars_per_level": rate}, POOLS)[0])'''),
    ('''        bad = {**spec, "tasks": [{"level": 1, "seed": 6001, "genome": WIN_PLAN}]}
        self.assertIsNone(eval_job.check_eval(bad, POOLS)[0])''',
     '''        bad = {**spec, "tasks": [{"level": 1, "seed": 6001, "genome": WIN_PLAN}]}
        self.assertIsNone(eval_job.check_eval(bad, POOLS)[0])
        self.assertEqual([], eval_job.check_eval({**spec, "profile_stars_per_level": 2.75}, POOLS)[1])
        self.assertIsNone(eval_job.check_eval({**spec, "profile_stars_per_level": 4}, POOLS)[0])'''),
])
print("rate tests patched")
