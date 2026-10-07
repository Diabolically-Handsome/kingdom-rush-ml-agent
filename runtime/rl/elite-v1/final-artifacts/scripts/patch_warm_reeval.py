"""Warm starts that re-play plans (reevaluate) import them whatever seeds they were measured on.

Search run 11 (search seeds 1005-1008) imported nothing: warm_entries kept only search_eval rows on the current
search seeds, while runs 8-10 had searched 1001-1004 (their 1005-1010 games are search_validate rows). Adopted
scores still need identical seeds/profile/protocol; re-played plans only need to be plans, ranked by their record
(smoothed win rate over the games they played for plans with wins, then mean fitness). Apply only without runtime/rl/elite-v1/job.lock.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", r"C:\Users\<user>\Documents\AlphaRush"))
REAL = Path(r"C:\Users\<user>\Documents\AlphaRush")
if (REAL / "runtime/rl/elite-v1/job.lock").exists() and ROOT.resolve() == REAL.resolve():
    raise SystemExit("refused: a job holds the lock")


def patch(rel, pairs):
    path = ROOT / rel
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:80])
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


patch("alpharush_rl/search_job.py", [
    ("""def warm_entries(spec, runs_dir, protocol):""",
     """def _reevaluated(warm, level):
    flag = warm.get("reevaluate", False)
    return flag is True or (isinstance(flag, list) and level in flag)


def warm_entries(spec, runs_dir, protocol):"""),
    ("""            row = json.loads(line)
            if row.get("kind") != "search_eval":
                continue
            payload = row["payload"]
            if payload["level"] not in spec["levels"] or payload["seed"] not in search_seeds:
                continue""",
     """            row = json.loads(line)
            if row.get("kind") not in ("search_eval", "search_validate"):
                continue
            payload = row["payload"]
            if payload["level"] not in spec["levels"]:
                continue
            # Adopted scores need this run's search seeds; a plan re-played here may come from any measured game.
            if not _reevaluated(warm, payload["level"]) and (row["kind"] != "search_eval"
                                                             or payload["seed"] not in search_seeds):
                continue"""),
    ("""        for entry in entries.values():
            # Mean over the search seeds, a missing seed counting 0: plans proven on every seed rank first.
            entry["fitness"] = sum(entry["seeds"].get(seed, 0.0) for seed in search_seeds) / len(search_seeds)
        ranked[level] = sorted(entries.values(), key=lambda e: (-e["fitness"], genome_id(e["genome"])))[:warm["top"]]""",
     """        for entry in entries.values():
            if _reevaluated(warm, level):
                # Re-played plans rank by their record: smoothed win rate over the games played (winners only, so a
                # short zero-win record cannot outrank a long one), then mean fitness.
                record = list(entry["seeds"].values())
                entry["fitness"] = sum(record) / len(record)
                wins = sum(f >= WIN for f in record)
                entry["rate"] = (wins + 1) / (len(record) + 4) if wins else 0.0
            else:
                # Mean over the search seeds, a missing seed counting 0: plans proven on every seed rank first.
                entry["fitness"] = sum(entry["seeds"].get(seed, 0.0) for seed in search_seeds) / len(search_seeds)
        ranked[level] = sorted(entries.values(), key=lambda e: (-e.get("rate", 0.0), -e["fitness"],
                                                                genome_id(e["genome"])))[:warm["top"]]"""),
])

patch("tests/test_elite.py", [
    ("""class JournalTests(unittest.TestCase):""",
     """class WarmStartTests(unittest.TestCase):
    def test_replayed_plans_come_from_any_measured_seed(self):
        from dataclasses import asdict
        from alpharush_rl.ops import GateRefused
        from alpharush_rl.search_job import warm_entries
        import json as _json
        strong, weak, other = (genome([["b", "01", kind]]) for kind in ("mage", "archer", "barrack"))
        with tempfile.TemporaryDirectory(prefix="alpharush-warm-") as tmp:
            run = Path(tmp) / "native-search-0001"
            run.mkdir()
            spec_old = {"search_seed": 1001, "search_seeds": [1001, 1002], "profile_stars_per_level": [3, 1.2, 13]}
            (run / "events.jsonl").write_text(_json.dumps({"kind": "search_start", "payload": {
                "spec": spec_old, "protocol": asdict(PROTOCOL)}}) + "\\n", encoding="utf-8")
            rows = [("search_eval", strong, 1001, 10500.0), ("search_validate", strong, 1005, 10400.0),
                    ("search_validate", strong, 1006, 900.0), ("search_eval", weak, 1001, 800.0),
                    ("search_validate", weak, 1005, 700.0), ("search_eval", other, 1002, 300.0)]
            (run / "episodes.jsonl").write_text("".join(_json.dumps({"kind": kind, "sha256": f"{i:064x}", "payload": {
                "level": 13, "seed": seed, "genome": g, "fitness": f}}) + "\\n"
                for i, (kind, g, seed, f) in enumerate(rows)), encoding="utf-8")
            spec = {"levels": [13], "search_seed": 1005, "search_seeds": [1005, 1006, 1007, 1008],
                    "profile_stars_per_level": [3, 1.2, 13],
                    "warm_start": {"runs": [run.name], "top": 2, "reevaluate": True}}
            ranked = warm_entries(spec, tmp, PROTOCOL)[13]
            self.assertEqual([genome_id(strong), genome_id(weak)], [genome_id(e["genome"]) for e in ranked])
            with self.assertRaises(GateRefused):  # adopting scores still needs the same seeds and profile
                warm_entries({**spec, "warm_start": {"runs": [run.name], "top": 2}}, tmp, PROTOCOL)
            same = {**spec, "search_seed": 1001, "search_seeds": [1001, 1002],
                    "warm_start": {"runs": [run.name], "top": 3}}
            adopted = {genome_id(e["genome"]): e["fitness"]
                       for e in warm_entries(same, tmp, PROTOCOL)[13]}  # search_eval rows on its seeds only
            self.assertEqual({genome_id(strong): 5250.0, genome_id(weak): 400.0, genome_id(other): 150.0}, adopted)


class JournalTests(unittest.TestCase):"""),
])
print("patched", ROOT)
