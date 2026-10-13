"""elite-v3b: request files for the fair re-test arms (one JSONL per arm), for every consult of the fork table with at
least two plays (the moments whose branch outcomes are measured).

usage: v3b_prompts.py <S1b run id> <replay run id> <fork table json> <out dir>
Arms (same consults, same plays, same order of rows):
  thin_old     commander.SYSTEM + Briefing.text() + the original play texts (rotation 0 must hash to the recorded
               prompt_sha256 of the consult: proves these are the prompts the models saw in elite-v3)
  thin_new     SYSTEM_V2 + thin_text_v2 (same information, official names) + honest id-tagged play texts
  rich         SYSTEM_V2 + rich_text (map, defences, enemy cards, hero, memory) + the same play texts
  rich_reason  the rich prompt asking for reasoning (canonical order) + the rich label rotations for scoring
Also writes manifest.json: counts, sha256 per file, prompt length statistics, leak check (must be empty)."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl import commander, rich_briefing as rb  # noqa: E402
from alpharush_rl.commander import Briefing, consult_user  # noqa: E402

RUNS = ROOT / "runtime/rl/elite-v3/runs"
GEO = ROOT / "runtime/rl/elite-v3b/geometry"


def rows(run_id, player):
    out = {}
    for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if '"eval_episode"' not in line[:40]:
            continue
        r = json.loads(line)["payload"]
        if r.get("player") == player:
            for c in r["result"]["commander"]["consults"]:
                out[f"{r['level']}:{r['seed']}:{r['genome_id']}:{c['consult']}"] = (r["level"], c)
    return out


def main(s1_id, replay_id, table_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base, captured = rows(s1_id, "cmd_steady"), rows(replay_id, "cmd_steady_capture")
    keys = [":".join(map(str, e["key"])) for e in json.loads(Path(table_path).read_text(encoding="utf-8"))]
    keys = sorted(k for k in keys if len(base[k][1]["plays"]) >= 2)
    cards = rb.load_cards(GEO / "enemy_cards.json")
    maps = {}
    arms = {name: [] for name in ("thin_old", "thin_new", "rich", "rich_reason")}
    mismatched, leaks, lengths = 0, [], {name: [] for name in arms}
    for key in keys:
        level, c = base[key]
        _, cap = captured[key]
        if cap["prompt_sha256"] != c["prompt_sha256"] or cap["plays"] != c["plays"]:
            raise SystemExit(f"{key}: the replay consult differs from S1b")
        data, plays = c["briefing"], c["plays"]
        ids = [p["id"] for p in plays]
        if level not in maps:
            maps[level] = rb.LevelMap.load(GEO, level)
        # thin_old: exactly the elite-v3 prompt.
        old_user = consult_user(Briefing(data))
        labels, rotations = rb.label_requests(old_user, [p["text"] for p in plays])
        if hashlib.sha256(rotations[0][1].encode("utf-8")).hexdigest() != c["prompt_sha256"]:
            mismatched += 1
        arms["thin_old"].append({"key": key, "arm": "thin_old", "system": commander.SYSTEM, "labels": labels, "ids": ids,
                                 "rotations": [{"order": o, "user": u} for o, u in rotations]})
        boss = bool((cap.get("translator") or {}).get("boss_focus"))
        options = rb.options_v2(plays, data, cards, boss=boss)
        thin_user = rb.user_text(rb.thin_text_v2(data, cards))
        rich_user = rb.user_text(rb.rich_text(data, cap["capture"], cap["ledger"], maps[level], cards))
        for name, user in (("thin_new", thin_user), ("rich", rich_user)):
            labels, rotations = rb.label_requests(user, options)
            arms[name].append({"key": key, "arm": name, "system": rb.SYSTEM_V2, "labels": labels, "ids": ids,
                               "rotations": [{"order": o, "user": u} for o, u in rotations]})
        labels, rotations = rb.label_requests(rich_user, options)
        arms["rich_reason"].append({"key": key, "arm": "rich_reason", "system": rb.SYSTEM_V2, "labels": labels,
                                    "ids": ids, "rotations": [{"order": o, "user": u} for o, u in rotations],
                                    "reason_user": rb.reason_request(rich_user, options)})
        for name in arms:
            row = arms[name][-1]
            text = row["system"] + "\n" + row["rotations"][0]["user"] + row.get("reason_user", "")
            lengths[name].append(len(text))
            found = rb.leaks(text)
            if found:
                leaks.append({"key": key, "arm": name, "found": found})
    manifest = {"s1": s1_id, "replay": replay_id, "table": str(table_path),
                "table_sha256": hashlib.sha256(Path(table_path).read_bytes()).hexdigest(), "consults": len(keys),
                "thin_old_prompt_hash_mismatches": mismatched, "leaks": leaks, "files": {}}
    for name, data_rows in arms.items():
        path = out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data_rows), encoding="utf-8",
                        newline="\n")
        manifest["files"][name] = {"rows": len(data_rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "chars_median": statistics.median(lengths[name]), "chars_max": max(lengths[name])}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))
    for name, info in manifest["files"].items():
        print(name, info)
    if mismatched or leaks:
        raise SystemExit("prompt hash mismatch or leak")


if __name__ == "__main__":
    main(*sys.argv[1:5])
