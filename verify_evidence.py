#!/usr/bin/env python3
"""Verify the published evidence: the main-campaign clears (campaign-v1), the elite-stage final (elite-v1) and the
all-26-levels final of 2026-10-11 (elite-v2).

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
The code tree has moved on since elite-v1 too: its pinned files are checked against
runtime/rl/elite-v1/pinned-snapshot/ (the files as published with that result).

elite-v2 checks (all 26 levels, one-shot final 2026-10-11, seeds 8101-8105):
  V1. every elite-v2 hash-chained journal is intact;
  V2. the ledger opened exactly one native-final job, the last job of the phase; it committed to the published
      pins.json, the original of the (quote-redacted) configuration and the seed pools (one-shot seeds 8101-8105);
  V3. every file pinned for that job is the published tree file, or redacted with the pinned original hash;
  V4. the final run: receipt (journal tip, summary hash, game count); every game belongs to this run, names the job's
      operator network, used 9-tick clicks, and its won flag and stars agree with the native outcome recorded in it;
      each seed starts from an empty save; per seed, levels won (by campaign attempts), stars and challenge wins
      recomputed from the game records match the summary and the published table (26 / 25 / 22 / 25 / 22 = 120 of
      130; seed 8101 wins every level 1-26); Normal difficulty;
  V5. the final seeds appear in no seed field of any other published JSON run file of any phase, and in no other job of
      the final configuration; the elite-v2 ledger opened them to no other job (the only other jobs are searches,
      collects and evaluations, whose frozen seed gates - the published files, per pins-history - accept train and
      evaluation seeds only, and the two published rehearsals on evaluation seeds); the seed pools mark them as the
      one-shot final seeds;
  V6. the recorded re-execution (every final game played again by the same networks) and strategy-brain replay
      (every 8B decision asked again) cover every final game and decision and report identical results;
  V7. the video verification records: each of seed 8101's 26 winning games, re-played with rendering on, reached the
      final run's exact end state;
  V8. the brain ablation: the two rehearsals on evaluation seeds 7101-7105 (one with the 8B brain, whose every
      strategy decision names the final's 8B model, one with the rule brain, which logs none) played only plans of the
      final configuration, and their level counts recomputed from the game records are 123 and 122 of 130.

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
ESNAPSHOT = EE / "pinned-snapshot"
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
ECONFIG_FILE = (ESNAPSHOT if ESNAPSHOT.exists() else ROOT) / "configs/phases/elite-v1.json"
check(epre.get("config_sha256") == econfig.get("original_sha256")
      and econfig.get("published_sha256") == sha(ECONFIG_FILE),
      "elite final job configuration = the original of the published (quote-redacted) configs/phases/elite-v1.json")
check(epre.get("pools_sha256") == sha(ROOT / "configs/pools-elite-v1.json"),
      "elite final job seed pools = published configs/pools-elite-v1.json")
epools = json.loads((ROOT / "configs/pools-elite-v1.json").read_text(encoding="utf-8"))
efinal_pool = epools.get("pools", {}).get("final_campaign_run", {})
check(sorted(efinal_pool.get("seeds", [])) == EFINAL_SEEDS and efinal_pool.get("access") == "one_frozen_evaluation",
      "elite final seeds 8001-8005 are the pool's one-shot final_campaign_run seeds")
ejob = json.loads(ECONFIG_FILE.read_text(encoding="utf-8"))["jobs"]["native-final"]
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
    if (ESNAPSHOT / rel).exists():
        path = ESNAPSHOT / rel
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
for name, job in json.loads(ECONFIG_FILE.read_text(encoding="utf-8"))["jobs"].items():
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

# ------------------------------------------------------------------ elite-v2
VV = ROOT / "runtime/rl/elite-v2"
VFINAL = VV / "runs/native-final-af081b54e5f54478aa76b24bc02bdbf5"
VFINAL_SEEDS = [8101, 8102, 8103, 8104, 8105]
VEXPECTED = {8101: 26, 8102: 25, 8103: 22, 8104: 25, 8105: 22}
VREHEARSALS = {"8b": (VV / "runs/native-campaign-92b946d5503b4887aeb1affa1bdc380c", 123),
               "rule": (VV / "runs/native-campaign-94c66770717b4ca1acdb562826ef6576", 122)}
print("\n== elite-v2: all 26 levels, one-shot final 2026-10-11 ==")

# V1. journals
vjournals = sorted(p for p in VV.rglob("*.jsonl") if p.name in ("episodes.jsonl", "events.jsonl", "ledger.jsonl",
                                                                  "pins-history.jsonl"))
bad = []
for path in vjournals:
    try:
        Journal(path).verify()
    except Exception as exc:  # noqa: BLE001
        bad.append(f"{path.relative_to(ROOT)}: {exc}")
check(not bad, f"elite-v2: {len(vjournals)} hash-chained journals intact" + ("" if not bad else f": {bad}"))

# V2. ledger -> pins, config, pools; one shot, last
vledger = rows(VV / "ledger.jsonl")
vopens = [r["payload"] for r in vledger if r["kind"] == "open"]
vopened = [p for p in vopens if p.get("run_id") == VFINAL.name]
check(len(vopened) == 1, "elite-v2 ledger has exactly one opening entry for the final run")
check([p.get("job_kind") for p in vopens].count("native-final") == 1 and vopens[-1].get("run_id") == VFINAL.name,
      "elite-v2 ledger opened exactly one native-final job (one shot), and it was the phase's last job")
vpre = vopened[0]["preflight"] if vopened else {}
check(vpre.get("job_kind") == "native-final" and vpre.get("pins_sha256") == sha(VV / "pins.json"),
      "elite-v2 final job committed to the published elite-v2 pins.json")
vredactions = {r["path"]: r for r in json.loads((VV / "REDACTIONS.json").read_text(encoding="utf-8"))["files"]}
vconfig = vredactions.get("configs/phases/elite-v2.json", {})
check(vpre.get("config_sha256") == vconfig.get("original_sha256")
      and vconfig.get("published_sha256") == sha(ROOT / "configs/phases/elite-v2.json"),
      "elite-v2 final job configuration = the original of the published (quote-redacted) configs/phases/elite-v2.json")
check(vpre.get("pools_sha256") == sha(ROOT / "configs/pools-elite-v2.json"),
      "elite-v2 final job seed pools = published configs/pools-elite-v2.json")
vpools = json.loads((ROOT / "configs/pools-elite-v2.json").read_text(encoding="utf-8"))
vfinal_pool = vpools.get("pools", {}).get("final_campaign_run", {})
check(sorted(vfinal_pool.get("seeds", [])) == VFINAL_SEEDS and vfinal_pool.get("access") == "one_frozen_evaluation",
      "elite-v2 final seeds 8101-8105 are the pool's one-shot final_campaign_run seeds")
vjobs = json.loads((ROOT / "configs/phases/elite-v2.json").read_text(encoding="utf-8"))["jobs"]
vjob = vjobs["native-final"]
check(vjob["campaign"]["seeds"] == VFINAL_SEEDS and vjob["campaign"]["levels"] == list(range(1, 27))
      and "start" not in vjob["campaign"] and vjob["campaign"]["policy"] == "operator"
      and vjob["campaign"]["brain"] == "8b" and vjob["campaign"].get("detour") is True
      and vjob.get("host_features") == ["timed_clicks"] and vjob["episode_protocol"].get("click_ticks") == 9,
      "elite-v2 final job: all 26 levels from a new save, operator networks, 8B strategy brain, detour route, "
      "human-timed trap clicks (click step 9 ticks)")

# V3. pinned files
vpins = json.loads((VV / "pins.json").read_text(encoding="utf-8"))["files"]
same, redacted, external, mismatched = 0, 0, [], []
for rel, digest in sorted(vpins.items()):
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
    elif rel in vredactions and vredactions[rel].get("original_sha256") == digest \
            and vredactions[rel]["published_sha256"] == sha(path):
        redacted += 1
    else:
        mismatched.append(rel)
check(not mismatched, f"elite-v2: {len(vpins)} pinned files: {same} identical, {redacted} redacted with pinned "
                      f"original hash, {len(external)} external" + ("" if not mismatched else f"; MISMATCH {mismatched}"))

# V4. final run, recomputed
vepisodes = rows(VFINAL / "episodes.jsonl")
vgames = [r for r in vepisodes if r["kind"] in GAME_KINDS]
vreceipt = json.loads((VFINAL / "supervisor-receipt.json").read_text(encoding="utf-8"))
vtip = Journal(VFINAL / "episodes.jsonl").verify()["tip_sha256"]
check(vreceipt.get("status") == "survey_completed" and vreceipt.get("episodes_journal", {}).get("tip_sha256") == vtip
      and vreceipt.get("summary_sha256") == sha(VFINAL / "survey-summary.json")
      and vreceipt.get("games_played") == len(vgames),
      f"elite-v2 final supervisor receipt: completed; journal tip, summary hash and {len(vgames)} games match")
vsummary = json.loads((VFINAL / "survey-summary.json").read_text(encoding="utf-8"))
vby_seed = {c["seed"]: c for c in campaigns_of(vsummary) or []}
names = {"main": "operator:" + vjob["operator_weights"]["sha256"][:12],
         "elite": "operator:" + vjob["elite_operator_weights"]["sha256"][:12]}
check(all(sha(ROOT / vjob[k]["path"]) == vjob[k]["sha256"] for k in ("operator_weights", "elite_operator_weights")),
      "elite-v2 operator networks: published weights match the job's sha256")
wrong_policy = [(r["payload"]["seed"], r["payload"]["level"]) for r in vgames
                if r["payload"]["result"].get("policy")
                != names["elite" if (r["payload"]["level"] > 12 or r["kind"] == "challenge_attempt") else "main"]]
check(not wrong_policy, f"elite-v2 final: all {len(vgames)} games name the job's operator network (main levels and "
                        "star replays the main network, elite stages and challenges the elite network)"
      + ("" if not wrong_policy else f"; OTHER {wrong_policy[:5]}"))



def native_ok(r, run_id):
    """The record's wrapper fields agree with the native outcome inside it (stars: 3 from 18 lives, 2 from 6)."""
    p, res = r["payload"], r["payload"]["result"]
    out = res.get("outcome") or {}
    won = res.get("status") == "terminal" and out.get("source") == "native" and bool(out.get("level_won"))
    lives = out.get("lives") or 0
    stars = (3 if lives >= 18 else 2 if lives >= 6 else 1) if won else 0
    return p["won"] == won and p.get("run_id") == run_id and (r["kind"] == "challenge_attempt" or p["stars"] == stars)


wrong_native = [(r["payload"]["seed"], r["payload"]["level"]) for r in vgames if not native_ok(r, VFINAL.name)]
check(not wrong_native, f"elite-v2 final: all {len(vgames)} game records belong to the final run, and their won flags "
                        "and stars agree with the native outcome recorded in each game"
      + ("" if not wrong_native else f"; DISAGREE {wrong_native[:5]}"))
wrong_clicks = [(r["payload"]["seed"], r["payload"]["level"]) for r in vgames
                if (r["payload"]["result"].get("protocol") or {}).get("click_ticks") != 9]
check(not wrong_clicks, f"elite-v2 final: all {len(vgames)} games record the 9-tick click step in their episode protocol"
      + ("" if not wrong_clicks else f"; OTHER {wrong_clicks[:5]}"))


def recount(games, seed):
    """Per seed: game payloads; levels won by campaign attempts (best stars over campaign attempts and star replays);
    challenge wins; whether the first game starts from an empty save."""
    rs = [r for r in games if r["payload"]["seed"] == seed]
    mine = [r["payload"] for r in rs]
    won = {}
    for r in rs:
        if r["kind"] == "campaign_attempt" and r["payload"]["won"]:
            won.setdefault(r["payload"]["level"], 0)
    for r in rs:
        p = r["payload"]
        if r["kind"] in ("campaign_attempt", "star_replay") and p["won"] and p["level"] in won:
            won[p["level"]] = max(won[p["level"]], p["stars"])
    chal = sorted(f"{p['level']}:{p['mode']}" for r in rs if r["kind"] == "challenge_attempt"
                  for p in [r["payload"]] if p["won"])
    first = mine[0]["profile"] if mine else {}
    new_save = bool(mine) and not first.get("levels") and not first.get("challenges") \
        and not any((first.get("upgrades") or {}).values()) and first.get("hero") is None and not first.get("hero_xp")
    return mine, won, chal, new_save


vtotal = 0
for seed in VFINAL_SEEDS:
    mine, won, chal, new_save = recount(vgames, seed)
    main = sorted(level for level in won if level <= 12)
    elites = sorted(level for level in won if level > 12)
    normal = all(p["result"]["difficulty"] == 2 for p in mine)
    summary = vby_seed.get(seed, {})
    summary_won = {int(k): v for k, v in summary.get("won", {}).items()}
    summary_chal = sorted(f"{k}:{m}" for k, modes in summary.get("challenges_won", {}).items() for m in modes)
    vtotal += len(won)
    check(won == summary_won and main == list(range(1, len(main) + 1)) and normal and chal == summary_chal
          and sum(won.values()) + len(chal) == summary.get("total_stars") and len(won) == VEXPECTED[seed] and new_save,
          f"seed {seed}: new save; {len(won)} levels won (main 1-{len(main)}, elites {elites or 'none'}), "
          f"{sum(won.values())} campaign stars + {len(chal)} challenge wins, {len(mine)} games, Normal")
check(vtotal == sum(VEXPECTED.values()) == 120, f"elite-v2 final: {vtotal}/130 levels won over 5 campaigns")
check(sorted(recount(vgames, 8101)[1]) == list(range(1, 27)),
      "seed 8101 won every level 1-26 in campaign mode: a full clear of the game from a new save")

# V5. seed isolation across every published run of every phase, and every other job
leaks, scanned = [], 0
for base in (VV / "runs", EE / "runs", EV / "runs"):
    for path in sorted(base.rglob("*")):
        if VFINAL.name in path.parts or path.suffix not in (".json", ".jsonl") or not path.stat().st_size:
            continue
        scanned += 1
        docs = rows(path) if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
        if any(v in VFINAL_SEEDS for doc in docs for v in seed_values(doc)):
            leaks.append(str(path.relative_to(ROOT)))
for name, job in vjobs.items():
    if name != "native-final" and any(v in VFINAL_SEEDS for v in seed_values(job)):
        leaks.append(f"configs/phases/elite-v2.json job {name}")
check(not leaks and scanned > 0, f"elite-v2 final seeds appear in no seed field of any other published run file "
      f"({scanned} files scanned, all phases) or other job" + ("" if not leaks else f": {leaks}"))
other_kinds = {p.get("job_kind") for p in vopens if p.get("run_id") != VFINAL.name}
rehearsal_ids = {run.name for run, _ in VREHEARSALS.values()}
campaign_runs = {p.get("run_id") for p in vopens if p.get("job_kind") == "native-campaign"}
check(other_kinds <= {"native-search", "native-collect", "native-eval", "native-campaign"}
      and campaign_runs == rehearsal_ids
      and all(p.get("preflight", {}).get("pools_sha256") == sha(ROOT / "configs/pools-elite-v2.json") for p in vopens),
      f"elite-v2 ledger: the {len(vopens) - 1} other jobs are searches, collects, evaluations and the two published "
      "rehearsals, all under the published seed pools (search, collect and eval gates take train/evaluation seeds only)")
vhistory = {r["payload"]["manifest_sha256"]: r["payload"]["files"] for r in rows(VV / "pins-history.jsonl")}
SEED_GATES = ["alpharush_rl/search_job.py", "alpharush_rl/collect_job.py", "alpharush_rl/eval_job.py",
              "alpharush_rl/phase.py", "alpharush_rl/pools.py"]
ungated = [p.get("run_id") for p in vopens if p.get("run_id") != VFINAL.name
           and not ((VV / "runs" / str(p.get("run_id"))).is_dir()
                    and all(vhistory.get(p.get("preflight", {}).get("pins_sha256"), {}).get(g) == sha(ROOT / g)
                            for g in SEED_GATES))]
check(not ungated, f"elite-v2: each of the {len(vopens) - 1} other jobs has a published run directory and ran under a frozen "
      "pins manifest (pins-history.jsonl) whose seed-gate files (search_job, collect_job, eval_job, phase, pools) are "
      "the published files, which refuse the one-shot final seeds to them" + ("" if not ungated else f": {ungated}"))

# V6. re-execution and strategy-brain replay records
VVER = VV / "final-artifacts/verification"
lines = [json.loads(line) for line in (VVER / "final-reexecution.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]
records = [x for x in lines if "kind" in x]
header = [x for x in lines if "games" in x]
trailer = [x for x in lines if "checked" in x]
keys = {(r["kind"], r["payload"]["seed"], r["payload"]["level"], r["payload"].get("mode", 1),
         r["payload"].get("attempt", r["payload"].get("replay")), r["payload"]["won"]) for r in vgames}
matched = [x for x in records if x["trace_equal"] and x["final_equal"]
           and (x["kind"], x["seed"], x["level"], x["mode"], x["attempt"], x["won"]) in keys]
distinct = {(x["kind"], x["seed"], x["level"], x["mode"], x["attempt"]) for x in records}
check(len(matched) == len(records) == len(vgames) == len(distinct) > 0
      and header == [{"games": len(vgames)}]
      and trailer == [{"checked": len(vgames), "identical": len(vgames), "different": []}],
      f"elite-v2 re-execution records: {len(matched)}/{len(vgames)} final games played again by the same networks, "
      "reported with identical traces and end states")
text = (VVER / "final-brain-replay.txt").read_text(encoding="utf-8")
report = json.loads(text[text.rindex('{\n "decisions"'):])
decisions = sum(r["kind"] == "strategy_decision" for r in vepisodes)
per_seed = [json.loads(line) for line in text.splitlines() if line.startswith('{"seed"')]
check(sorted(x["seed"] for x in per_seed) == VFINAL_SEEDS
      and all(x["won_levels"] == VEXPECTED[x["seed"]]
              and x["decisions"] == sum(r["kind"] == "strategy_decision" and r["payload"]["seed"] == x["seed"]
                                        for r in vepisodes) for x in per_seed),
      "elite-v2 strategy brain replay: every seed re-played to the same levels with the same number of decisions")
check(report["decisions"] == report["prompt_equal"] == report["choice_equal"] == decisions
      and report["max_p_diff"] == 0 and not report["differences"],
      f"elite-v2 strategy brain replay records: {report['choice_equal']}/{decisions} 8B decisions asked again, "
      "reported with identical prompts, choices and probabilities")

# V7. video verification records of seed 8101
vvideo = VV / "video/seed8101_verification.jsonl"
check(vvideo.exists(), "elite-v2 video verification records are published")
vexpected = {p["level"]: p["result"]["final_state_sha256"] for r in vgames if r["kind"] == "campaign_attempt"
             for p in [r["payload"]] if p["seed"] == 8101 and p["won"]}
vrecords = rows(vvideo) if vvideo.exists() else []
good = [r for r in vrecords if r["identical"] and r["seed"] == 8101
        and r.get("operator") == names["elite" if r["level"] > 12 else "main"]
        and vexpected.get(r["level"]) == r["replayed_final_state_sha256"] == r["expected_final_state_sha256"]]
check(len(good) == len(vrecords) == 26 and sorted(r["level"] for r in good) == list(range(1, 27)),
      f"elite-v2 video: {len(good)}/26 levels of seed 8101 re-played with rendering, by the job's networks, to the final "
      "run's exact end state")


# V8. brain ablation on the evaluation seeds
def plan_key(level, mode, entry):
    genome = entry["genome"] if isinstance(entry, dict) and "genome" in entry else entry
    return level, mode, json.dumps(genome, sort_keys=True)


plan_keys = {plan_key(int(level), 1, e) for level, entries in vjob["campaign"]["plans"].items() for e in entries}
plan_keys |= {plan_key(int(key.split(":")[0]), int(key.split(":")[1]), e)
              for key, entries in vjob["campaign"]["challenges"].items() for e in entries}
for brain, (run, expected_total) in VREHEARSALS.items():
    rrows = rows(run / "episodes.jsonl")
    rgames = [r for r in rrows if r["kind"] in GAME_KINDS]
    seeds = sorted({r["payload"]["seed"] for r in rgames})
    rdecisions = [r["payload"] for r in rrows if r["kind"] == "strategy_decision"]
    rcampaigns = campaigns_of(json.loads((run / "survey-summary.json").read_text(encoding="utf-8"))) or []
    final_models = {r["payload"].get("model") for r in vepisodes if r["kind"] == "strategy_decision"}
    brain_ok = (len(rcampaigns) == 5 and all(c.get("brain") == brain for c in rcampaigns)
                and ((bool(rdecisions) and {d.get("brain") for d in rdecisions} == {"8b"}
                      and {d.get("model") for d in rdecisions} == final_models) if brain == "8b" else not rdecisions))
    foreign = [r for r in rgames if plan_key(r["payload"]["level"], r["payload"].get("mode", 1), r["payload"]["genome"])
               not in plan_keys]
    total = sum(len(recount(rgames, seed)[1]) for seed in seeds)
    check(seeds == [7101, 7102, 7103, 7104, 7105] and brain_ok and not foreign and total == expected_total,
          f"rehearsal with the {brain} brain ({len(rdecisions)} logged strategy decisions): evaluation seeds "
          f"7101-7105, only plans of the final configuration ({len(rgames)} games), {total}/130 levels won")

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
