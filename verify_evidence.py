#!/usr/bin/env python3
"""Verify the published evidence for the full-campaign clears.

Checks, using only files in this repository (no game needed):
  1. every hash-chained journal (episodes/events/ledger/pins-history) is intact;
  2. the ledger's opening entry of the one-shot final job commits to the published pins.json and to the
     original job configuration (whose SHA256 is listed in REDACTIONS.json, since quotes were redacted);
  3. every file pinned for that job is published unchanged, or is listed as redacted with the pinned
     original hash (Lumi_Nox's bridge.lua is an external MIT dependency, checked if ./Lumi_Nox exists);
  4. the final run: 5 campaigns on the final seeds, each clearing levels 1-12 in order on Normal, recomputed
     from the attempt records, and the supervisor receipt's journal tip matches the journal;
  5. the final seeds appear in no other published run;
  6. the video verification records (if present) match the final run's end states game by game.

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

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
