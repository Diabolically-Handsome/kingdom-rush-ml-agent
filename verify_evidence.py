#!/usr/bin/env python3
"""Verify the published evidence: the main-campaign clears (campaign-v1) and the elite-stage final (elite-v1).

campaign-v1 checks, using only files in this repository (no game needed):
  1. every hash-chained journal (episodes/events/ledger/pins-history) is intact;
  2. the ledger's opening entry of the one-shot final job commits to the published pins.json and to the
     original job configuration (whose SHA256 is listed in REDACTIONS.json, since quotes were redacted);
  3. every file pinned for that job is published unchanged, or is listed as redacted with the pinned
     original hash (Lumi_Nox's bridge.lua is an external MIT dependency, checked if ./Lumi_Nox exists);
  4. the final run: 5 campaigns on the final seeds, each clearing levels 1-12 in order on Normal, recomputed
     from the attempt records, and the supervisor receipt's journal tip matches the journal;
  5. the final seeds appear in no other published run;
  6. the video verification records (if present) match the final run's end states game by game.
The code tree has moved on since campaign-v1: its pinned files are checked against
runtime/rl/campaign-v1/pinned-snapshot/ (the files as published with that result).

elite-v1 checks (elite stages 13-26, option A, one-shot final 2026-10-07):
  E1. every elite-v1 hash-chained journal is intact;
  E2. the ledger opened exactly one native-final job; it committed to the published pins.json, the original of the
      (quote-redacted) configuration and the seed pools, whose one-shot final seeds are 8001-8005;
  E3. every file pinned for that job is the published tree file, or redacted with the pinned original hash;
  E4. the final run: receipt (journal tip, summary hash, game count); per seed, the levels won (first wins in journal
      order), stars and challenge wins recomputed from the game records match the run's summary and the published
      table (10 / 21 / 17 / 21 / 10 levels); Normal difficulty;
  E5. the final seeds appear in no other published run (any seed field of any run file) and in no other job;
  E6. the recorded comparison results of the re-execution (every game played again by the same networks; the
      comparison itself needs the game) and of the strategy-brain replay (every 8B decision asked again; needs the
      model) cover every final game and decision and report identical traces, end states, prompts and choices.

usage: python verify_evidence.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from alpharush_rl.journal import Journal  # noqa: E402

EV = ROOT / "runtime/rl/campaign-v1"
SNAPSHOT = EV / "pinned-snapshot"
FINAL = EV / "runs/native-final-1456be77e13c46598acab730468ec34c"
FINAL_SEEDS = [6001, 6002, 6003, 6004, 6005]
failures = []


def check(ok, message):
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# 1. journals
# The first rehearsal (native-campaign-961d...) stopped because of a since-fixed bug: entry 298 was hashed with
# integer dictionary keys (a 10-level save profile, {1: 3, ..., 10: 3}) but stored as JSON string keys, which sort
# differently from ten keys on. That entry is checked under the pre-fix rule instead.
KNOWN_PRE_FIX = {("native-campaign-961d1b6b26b3492db29648eafa323646", "episodes.jsonl")}


def int_keys(value):
    if isinstance(value, dict):
        keys = list(value)
        numeric = keys and all(isinstance(k, str) and k.isdigit() for k in keys)
        return {(int(k) if numeric else k): int_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [int_keys(v) for v in value]
    return value


def verify_pre_fix(path):
    previous = "0" * 64
    for sequence, row in enumerate(rows(path)):
        unsigned = {k: v for k, v in row.items() if k != "sha256"}
        digests = {hashlib.sha256(json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode("utf-8")).hexdigest()
                   for candidate in (unsigned, {**unsigned, "payload": int_keys(unsigned["payload"])})}
        if row.get("seq") != sequence or row.get("previous_sha256") != previous or row["sha256"] not in digests:
            raise ValueError(f"entry {sequence} fails even under the pre-fix rule")
        previous = row["sha256"]
    return sequence + 1


journals = sorted(p for p in EV.rglob("*.jsonl") if p.name in ("episodes.jsonl", "events.jsonl", "ledger.jsonl",
                                                                 "pins-history.jsonl"))
bad = []
for path in journals:
    try:
        Journal(path).verify()
    except Exception as exc:  # noqa: BLE001
        if (path.parent.name, path.name) in KNOWN_PRE_FIX:
            n = verify_pre_fix(path)
            print(f"NOTE {path.parent.name}/{path.name}: {n} entries intact; entry {n - 1} verified under the "
                  "pre-fix integer-key rule (the bug that stopped that rehearsal)")
        else:
            bad.append(f"{path.relative_to(ROOT)}: {exc}")
check(not bad, f"{len(journals)} hash-chained journals intact" + ("" if not bad else f": {bad}"))

# 2. ledger -> pins and config
ledger = rows(EV / "ledger.jsonl")
opened = [r["payload"] for r in ledger if r["kind"] == "open" and r["payload"].get("run_id") == FINAL.name]
check(len(opened) == 1, "ledger has exactly one opening entry for the final run")
preflight = opened[0]["preflight"] if opened else {}
check(preflight.get("pins_sha256") == sha(EV / "pins.json"), "final job committed to the published pins.json")
redactions = {r["path"]: r for r in json.loads((EV / "REDACTIONS.json").read_text(encoding="utf-8"))["files"]}
config = redactions.get("configs/phases/campaign-v1.json", {})
check(preflight.get("config_sha256") == config.get("original_sha256"),
      "final job configuration = the original of the published (quote-redacted) config")
check(preflight.get("pools_sha256") == sha(ROOT / "configs/pools-campaign-v1.json"),
      "final job seed pools = published configs/pools-campaign-v1.json")
pools = json.loads((ROOT / "configs/pools-campaign-v1.json").read_text(encoding="utf-8"))
final_pool = pools.get("pools", {}).get("final_campaign_run", {})
check(sorted(final_pool.get("seeds", [])) == FINAL_SEEDS and final_pool.get("access") == "one_frozen_evaluation",
      "final seeds 6001-6005 are the pool's one-shot final_campaign_run seeds (difficulty %s)" % pools.get("difficulty"))

# 3. pinned files
pins = json.loads((EV / "pins.json").read_text(encoding="utf-8"))["files"]
same, redacted, external, mismatched = 0, 0, [], []
for rel, digest in sorted(pins.items()):
    path = ROOT / rel
    if rel.startswith("Lumi_Nox/"):
        if path.exists():
            (same := same + 1) if sha(path) == digest else mismatched.append(rel)
        else:
            external.append(rel)
        continue
    if (SNAPSHOT / rel).exists():
        path = SNAPSHOT / rel
    if not path.exists():
        mismatched.append(rel + " (missing)")
    elif sha(path) == digest:
        same += 1
    elif rel in redactions and redactions[rel]["original_sha256"] == digest \
            and redactions[rel]["published_sha256"] == sha(path):
        redacted += 1
    else:
        mismatched.append(rel)
check(not mismatched, f"{len(pins)} pinned files: {same} identical, {redacted} redacted with pinned original hash, "
                      f"{len(external)} external (MIT Lumi_Nox bridge; clone it to ./Lumi_Nox to check)"
      + ("" if not mismatched else f"; MISMATCH {mismatched}"))

# 4. final run
episodes = rows(FINAL / "episodes.jsonl")
attempts = [r["payload"] for r in episodes if r["kind"] == "campaign_attempt"]
summary = json.loads((FINAL / "survey-summary.json").read_text(encoding="utf-8"))
receipt = json.loads((FINAL / "supervisor-receipt.json").read_text(encoding="utf-8"))
tip = Journal(FINAL / "episodes.jsonl").verify()["tip_sha256"]
check(receipt.get("status") == "survey_completed" and receipt.get("episodes_journal", {}).get("tip_sha256") == tip,
      "final supervisor receipt: completed, journal tip matches")
for seed in FINAL_SEEDS:
    mine = [a for a in attempts if a["seed"] == seed]
    won = [a["level"] for a in mine if a["won"]]
    stars = sum(a["stars"] for a in mine if a["won"])
    normal = all(a["result"]["difficulty"] == 2 for a in mine)
    check(won == list(range(1, 13)) and normal,
          f"seed {seed}: levels 1-12 cleared in order on Normal (difficulty 2), {len(mine)} games, {stars}/36 stars")
check(summary.get("completed") == 5, "final summary: 5 of 5 campaigns completed")

# 5. seed isolation
leaks = []
for path in (EV / "runs").rglob("*.jsonl"):
    if FINAL.name in path.parts:
        continue
    for row in rows(path):
        payload = row.get("payload", {})
        if isinstance(payload, dict) and payload.get("seed") in FINAL_SEEDS:
            leaks.append(str(path.relative_to(ROOT)))
            break
check(not leaks, "final seeds appear in no other published run" + ("" if not leaks else f": {leaks}"))

# 6. video verification
video = EV / "video/seed6001_verification.jsonl"
if video.exists():
    expected = {(a["seed"], a["level"]): a["result"]["final_state_sha256"] for a in attempts if a["won"]}
    records = rows(video)
    good = [r for r in records if r["identical"] and expected.get((r["seed"], r["level"])) == r["replayed_final_state_sha256"]]
    check(len(good) == len(records) == 12,
          f"video: {len(good)}/{len(records)} recorded levels re-played to the final run's exact end state")
else:
    print("SKIP video verification records not present")

# ------------------------------------------------------------------ elite-v1
EE = ROOT / "runtime/rl/elite-v1"
EFINAL = EE / "runs/native-final-0468386df8444b648525882472495279"
EFINAL_SEEDS = [8001, 8002, 8003, 8004, 8005]
EXPECTED_LEVELS = {8001: 10, 8002: 21, 8003: 17, 8004: 21, 8005: 10}
GAME_KINDS = ("campaign_attempt", "star_replay", "challenge_attempt")
print("\n== elite-v1: elite stages 13-26, one-shot final 2026-10-07 ==")

# E1. journals
ejournals = sorted(p for p in EE.rglob("*.jsonl") if p.name in ("episodes.jsonl", "events.jsonl", "ledger.jsonl",
                                                                  "pins-history.jsonl"))
bad = []
for path in ejournals:
    try:
        Journal(path).verify()
    except Exception as exc:  # noqa: BLE001
        bad.append(f"{path.relative_to(ROOT)}: {exc}")
check(not bad, f"elite-v1: {len(ejournals)} hash-chained journals intact" + ("" if not bad else f": {bad}"))

# E2. ledger -> pins, config, pools; one shot
eledger = rows(EE / "ledger.jsonl")
eopened = [r["payload"] for r in eledger if r["kind"] == "open" and r["payload"].get("run_id") == EFINAL.name]
check(len(eopened) == 1, "elite-v1 ledger has exactly one opening entry for the final run")
finals = [r for r in eledger if r["kind"] == "open" and r["payload"].get("job_kind") == "native-final"]
check(len(finals) == 1, "elite-v1 ledger opened exactly one native-final job (one shot)")
epre = eopened[0]["preflight"] if eopened else {}
check(epre.get("job_kind") == "native-final" and epre.get("pins_sha256") == sha(EE / "pins.json"),
      "elite final job committed to the published elite-v1 pins.json")
eredactions = {r["path"]: r for r in json.loads((EE / "REDACTIONS.json").read_text(encoding="utf-8"))["files"]}
econfig = eredactions.get("configs/phases/elite-v1.json", {})
check(epre.get("config_sha256") == econfig.get("original_sha256")
      and econfig.get("published_sha256") == sha(ROOT / "configs/phases/elite-v1.json"),
      "elite final job configuration = the original of the published (quote-redacted) configs/phases/elite-v1.json")
check(epre.get("pools_sha256") == sha(ROOT / "configs/pools-elite-v1.json"),
      "elite final job seed pools = published configs/pools-elite-v1.json")
epools = json.loads((ROOT / "configs/pools-elite-v1.json").read_text(encoding="utf-8"))
efinal_pool = epools.get("pools", {}).get("final_campaign_run", {})
check(sorted(efinal_pool.get("seeds", [])) == EFINAL_SEEDS and efinal_pool.get("access") == "one_frozen_evaluation",
      "elite final seeds 8001-8005 are the pool's one-shot final_campaign_run seeds")
ejob = json.loads((ROOT / "configs/phases/elite-v1.json").read_text(encoding="utf-8"))["jobs"]["native-final"]
check(ejob["campaign"]["seeds"] == EFINAL_SEEDS and ejob["campaign"]["levels"] == list(range(1, 27))
      and "start" not in ejob["campaign"] and ejob["campaign"]["policy"] == "operator"
      and ejob["campaign"]["brain"] == "8b",
      "elite final job: all 26 levels from a new save, operator networks, 8B strategy brain")

# E3. pinned files
epins = json.loads((EE / "pins.json").read_text(encoding="utf-8"))["files"]
same, redacted, external, mismatched = 0, 0, [], []
for rel, digest in sorted(epins.items()):
    path = ROOT / rel
    if rel.startswith("Lumi_Nox/"):
        if path.exists():
            (same := same + 1) if sha(path) == digest else mismatched.append(rel)
        else:
            external.append(rel)
        continue
    if not path.exists():
        mismatched.append(rel + " (missing)")
    elif sha(path) == digest:
        same += 1
    elif rel in eredactions and eredactions[rel]["original_sha256"] == digest \
            and eredactions[rel]["published_sha256"] == sha(path):
        redacted += 1
    else:
        mismatched.append(rel)
check(not mismatched, f"elite-v1: {len(epins)} pinned files: {same} identical, {redacted} redacted with pinned "
                      f"original hash, {len(external)} external" + ("" if not mismatched else f"; MISMATCH {mismatched}"))

# E4. final run, recomputed
eepisodes = rows(EFINAL / "episodes.jsonl")
egames = [r for r in eepisodes if r["kind"] in GAME_KINDS]
ereceipt = json.loads((EFINAL / "supervisor-receipt.json").read_text(encoding="utf-8"))
etip = Journal(EFINAL / "episodes.jsonl").verify()["tip_sha256"]
check(ereceipt.get("status") == "survey_completed" and ereceipt.get("episodes_journal", {}).get("tip_sha256") == etip
      and ereceipt.get("summary_sha256") == sha(EFINAL / "survey-summary.json")
      and ereceipt.get("games_played") == len(egames),
      f"elite final supervisor receipt: completed; journal tip, summary hash and {len(egames)} games match")
esummary = json.loads((EFINAL / "survey-summary.json").read_text(encoding="utf-8"))


def campaigns_of(obj):
    if isinstance(obj, dict):
        if "campaigns" in obj:
            return obj["campaigns"]
        for value in obj.values():
            found = campaigns_of(value)
            if found:
                return found
    return None


by_seed = {c["seed"]: c for c in campaigns_of(esummary) or []}
total = 0
for seed in EFINAL_SEEDS:
    mine = [r["payload"] for r in egames if r["payload"]["seed"] == seed]
    won = {}
    for p in mine:
        if p.get("won") and "mode" not in p:
            won[p["level"]] = max(won.get(p["level"], 0), p["stars"])
    main = sorted(level for level in won if level <= 12)
    elites = sorted(level for level in won if level > 12)
    chal = sorted(f"{p['level']}:{p['mode']}" for p in mine if p.get("mode") and p.get("won"))
    normal = all(p["result"]["difficulty"] == 2 for p in mine)
    summary = by_seed.get(seed, {})
    summary_won = {int(k): v for k, v in summary.get("won", {}).items()}
    first_wins = []
    for r in egames:
        p = r["payload"]
        if r["kind"] == "campaign_attempt" and p["seed"] == seed and p["won"] and p["level"] not in first_wins:
            first_wins.append(p["level"])
    summary_chal = sorted(f"{k}:{m}" for k, modes in summary.get("challenges_won", {}).items() for m in modes)
    total += len(won)
    check(won == summary_won and main == list(range(1, len(main) + 1)) and normal
          and first_wins == sorted(first_wins) == summary.get("levels_won", first_wins)
          and chal == summary_chal and len(chal) == (summary.get("challenge_stars") or 0)
          and sum(won.values()) + len(chal) == summary.get("total_stars")
          and len(won) == EXPECTED_LEVELS[seed],
          f"seed {seed}: {len(won)} levels won (main 1-{len(main)} in order, elites {elites or 'none'}), "
          f"{sum(won.values())} campaign stars + {len(chal)} challenge wins, {len(mine)} games, Normal")
check(total == sum(EXPECTED_LEVELS.values()), f"elite final: {total}/130 levels won over 5 campaigns")

# E5. seed isolation: every integer under a key containing "seed" in any other run file or job
def seed_values(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from seed_values(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from seed_values(v, key)
    elif isinstance(obj, int) and not isinstance(obj, bool) and "seed" in key.lower():
        yield obj


leaks, scanned = [], 0
for base in (EE / "runs", EV / "runs"):
    for path in sorted(base.rglob("*")):
        if EFINAL.name in path.parts or path.suffix not in (".json", ".jsonl") or not path.stat().st_size:
            continue
        scanned += 1
        docs = rows(path) if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
        if any(v in EFINAL_SEEDS for doc in docs for v in seed_values(doc)):
            leaks.append(str(path.relative_to(ROOT)))
for name, job in json.loads((ROOT / "configs/phases/elite-v1.json").read_text(encoding="utf-8"))["jobs"].items():
    if name != "native-final" and any(v in EFINAL_SEEDS for v in seed_values(job)):
        leaks.append(f"configs/phases/elite-v1.json job {name}")
check(not leaks and scanned > 0, f"elite final seeds appear in no other published run ({scanned} run files scanned) "
      "or job" + ("" if not leaks else f": {leaks}"))

# E6. re-execution and strategy-brain replay records
VER = EE / "final-artifacts/verification"
lines = [json.loads(line) for line in (VER / "final-reexecution.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]
records = [x for x in lines if "kind" in x]
header = [x for x in lines if "games" in x]
trailer = [x for x in lines if "checked" in x]
keys = set()
for r in egames:
    p = r["payload"]
    keys.add((r["kind"], p["seed"], p["level"], p.get("mode", 1), p.get("attempt", p.get("replay")), p["won"]))
matched = [x for x in records if x["trace_equal"] and x["final_equal"]
           and (x["kind"], x["seed"], x["level"], x["mode"], x["attempt"], x["won"]) in keys]
distinct = {(x["kind"], x["seed"], x["level"], x["mode"], x["attempt"]) for x in records}
check(len(matched) == len(records) == len(egames) == len(distinct) > 0
      and header == [{"games": len(egames)}]
      and trailer == [{"checked": len(egames), "identical": len(egames), "different": []}],
      f"re-execution records: {len(matched)}/{len(egames)} final games played again by the same networks, reported "
      "with identical traces and end states")
text = (VER / "final-brain-replay.txt").read_text(encoding="utf-8")
report = json.loads(text[text.rindex('{\n "decisions"'):])
decisions = sum(r["kind"] == "strategy_decision" for r in eepisodes)
per_seed = [json.loads(line) for line in text.splitlines() if line.startswith('{"seed"')]
check(sorted(x["seed"] for x in per_seed) == EFINAL_SEEDS
      and all(x["won_levels"] == EXPECTED_LEVELS[x["seed"]]
              and x["decisions"] == sum(r["kind"] == "strategy_decision" and r["payload"]["seed"] == x["seed"]
                                        for r in eepisodes) for x in per_seed),
      "strategy brain replay: every seed re-played to the same levels with the same number of decisions")
check(report["decisions"] == report["prompt_equal"] == report["choice_equal"] == decisions
      and report["max_p_diff"] == 0 and not report["differences"],
      f"strategy brain replay records: {report['choice_equal']}/{decisions} 8B decisions asked again, reported with "
      "identical prompts, choices and probabilities")

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
