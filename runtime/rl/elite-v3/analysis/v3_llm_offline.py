"""elite-v3: offline 8B commander choices on recorded consults (no game is played). For every consult of the S1 run's
steady games (optionally only those in a fork run), the live 8B worker (RTX 5090, 127.0.0.1:12081) answers the exact
briefing and play menu the game showed; the prompt hash must equal the one recorded in the consult. A second pass on
the first N consults checks the model is deterministic.

usage: v3_llm_offline.py <S1 run id> <out json> [fork run id] [determinism sample N]
env ALPHARUSH_OFFLINE_MODEL=24b scores with the 24B worker (127.0.0.1:12082) instead."""
import hashlib
import os
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl import commander  # noqa: E402
from alpharush_rl.model_broker import LanguageModelBroker  # noqa: E402

RUNS = ROOT / "runtime/rl/elite-v3/runs"
s1_id, out_path = sys.argv[1], Path(sys.argv[2])
fork_id = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
sample = int(sys.argv[4]) if len(sys.argv) > 4 else 20


def rows(run_id):
    return [json.loads(line)["payload"] for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8")
            .splitlines() if line.strip() and '"eval_episode"' in line[:40]]


wanted = None
if fork_id:
    wanted = {(r["level"], r["seed"], r["genome_id"]) for r in rows(fork_id)}
MODEL = os.environ.get("ALPHARUSH_OFFLINE_MODEL", "8b")
PORT = {"8b": 12081, "24b": 12082}[MODEL]
brain = commander.LanguageCommander(LanguageModelBroker(f"http://127.0.0.1:{PORT}", timeout=300), name=MODEL)
choices, records, mismatched = {}, [], 0
started = time.time()
items = []
for r in rows(s1_id):
    if r["player"] != "cmd_steady":
        continue
    key3 = (r["level"], r["seed"], r["genome_id"])
    if wanted is not None and key3 not in wanted:
        continue
    for c in r["result"]["commander"]["consults"]:
        items.append((key3, c))
for key3, c in items:
    briefing = commander.Briefing(c["briefing"])
    plays = c["plays"]
    prompt = commander.request_user(commander.consult_user(briefing), [p["text"] for p in plays])
    same_prompt = hashlib.sha256(prompt.encode("utf-8")).hexdigest() == c["prompt_sha256"]
    mismatched += not same_prompt
    index, record = brain.choose(briefing, plays, {"seed": key3[1], "genome_id": key3[2], "wave": c["wave"],
                                                  "consult": c["consult"]})
    key = f"{key3[0]}:{key3[1]}:{key3[2]}:{c['consult']}"
    choices[key] = index
    records.append({"key": key, "choice": index, "play_id": plays[index]["id"], "p": record.get("p"),
                    "model": record.get("model"),
                    "same_prompt": same_prompt, "rule": c["shadow"].get("rule"), "n": len(plays)})
repeat = 0
for rec, (key3, c) in zip(records[:sample], items[:sample]):
    index, _ = brain.choose(commander.Briefing(c["briefing"]), c["plays"], {"seed": key3[1], "genome_id": key3[2],
                                                                           "wave": c["wave"], "consult": c["consult"]})
    repeat += index == rec["choice"]
out_path.write_text(json.dumps(choices, indent=1), encoding="utf-8")
(out_path.with_suffix(".records.json")).write_text(json.dumps(records, indent=1), encoding="utf-8")
agree = sum(1 for r in records if r["choice"] == r["rule"])
print(json.dumps({"consults": len(records), "prompt_mismatch": mismatched, "repeat_same": f"{repeat}/{min(sample, len(records))}",
                  "agrees_with_rule": agree, "p0_choices": sum(1 for r in records if r["choice"] == 0),
                  "seconds": round(time.time() - started, 1)}))
