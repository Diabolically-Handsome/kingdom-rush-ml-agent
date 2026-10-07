"""Brain-only replay of the elite-v1 one-shot final: the journaled game results stand in for the games, the live 8B
server answers every strategy question again; compare prompts (SHA256), choices and probabilities with the journal.

usage: brain_replay_final.py [seeds e.g. 8002,8004] ; needs the 8B server on 127.0.0.1:12081 (no game is played)
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl import campaign_run  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol  # noqa: E402
from alpharush_rl.model_broker import LanguageModelBroker  # noqa: E402
from alpharush_rl.strategy_brain import LanguageBrain  # noqa: E402

RUN = ROOT / "runtime/rl/elite-v1/runs/native-final-0468386df8444b648525882472495279"
rows = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
results = {}
for r in rows:
    p = r["payload"]
    if r["kind"] == "campaign_attempt":
        results[(p["seed"], p["level"], 1, "a", p["attempt"])] = p["result"]
    elif r["kind"] == "star_replay":
        results[(p["seed"], p["level"], 1, "r", p["replay"])] = p["result"]
    elif r["kind"] == "challenge_attempt":
        results[(p["seed"], p["level"], p["mode"], "a", p["attempt"])] = p["result"]
old = {}
for r in rows:
    if r["kind"] == "strategy_decision":
        old.setdefault(r["payload"]["seed"], []).append(r["payload"])
cfg = json.loads((ROOT / "configs/phases/elite-v1.json").read_text(encoding="utf-8"))
spec = cfg["jobs"]["native-final"]["campaign"]
seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else spec["seeds"]


def fake_episode(env, policy, protocol, *, seed, level, episode_id, **kwargs):
    tail = episode_id.split(f"-L{level:02d}-", 1)[1]
    mode = 1
    if tail.startswith("m"):
        mode_text, tail = tail.split("-", 1)
        mode = int(mode_text[1:])
    return results[(seed, level, mode, tail[0], int(tail[1:]))]


campaign_run.run_episode = fake_episode


class Ctx:
    run_id = RUN.name
    games_played = 0
    games_remaining = 10 ** 6

    def claim_game(self):
        self.games_played += 1

    def record(self, *args, **kwargs):
        pass

    def check(self):
        pass


class Out:
    def __init__(self):
        self.rows = []

    def append(self, kind, payload):
        self.rows.append((kind, payload))


class Env:
    def close(self):
        pass


broker = LanguageModelBroker("http://127.0.0.1:12081", timeout=180)
report = {"decisions": 0, "prompt_equal": 0, "choice_equal": 0, "max_p_diff": 0.0, "differences": []}
for seed in seeds:
    out = Out()
    summary = campaign_run.run_campaign({**spec, "seed": seed}, lambda *a, **k: Env(), Ctx(), out,
                                        protocol=EpisodeProtocol(), make_policy=lambda g: None,
                                        make_elite_policy=lambda g: None, brain=LanguageBrain(broker, name="8b"))
    new = [p for kind, p in out.rows if kind == "strategy_decision"]
    if len(new) != len(old[seed]):
        report["differences"].append({"seed": seed, "count": [len(old[seed]), len(new)]})
    for a, b in zip(old[seed], new):
        report["decisions"] += 1
        same_prompt = a["prompt_sha256s"] == b["prompt_sha256s"]
        report["prompt_equal"] += same_prompt
        report["choice_equal"] += a["choice"] == b["choice"]
        diff = max(abs(x - y) for x, y in zip(a["p"], b["p"]))
        report["max_p_diff"] = max(report["max_p_diff"], diff)
        if a["choice"] != b["choice"] or not same_prompt:
            report["differences"].append({"seed": seed, "level": a["level"], "attempt": a["attempt"], "kind": a["kind"],
                                          "prompt_equal": same_prompt, "old": [a["choice"], a["p"]],
                                          "new": [b["choice"], b["p"]]})
    print(json.dumps({"seed": seed, "won_levels": len(summary["won"]), "decisions": len(new)}), flush=True)
print(json.dumps(report, indent=1))
