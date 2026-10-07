"""Brain-only dry run of an elite-v1 campaign job body (read-only; no game is played).

usage: brain_dry_campaign.py <job json (out= of make_final_jobs.py)> <job name> <A|B> [seeds] [extra run ids]
The live 8B server (127.0.0.1:12081) makes every strategy decision; each game's result is looked up from the train-seed
search records of the played plan (eval seed 70xx/80xx stands in for train seed 10xx), else drawn from the plan's
recorded win rate, else (main levels) a win with the stars campaign-v1's final earned, else a loss.
Prints every attempt with the plan's evidence and the bought upgrades, then a summary per seed.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl import campaign_run  # noqa: E402
from alpharush_rl.episode import EpisodeProtocol  # noqa: E402
from alpharush_rl.model_broker import LanguageModelBroker  # noqa: E402
from alpharush_rl.search import check_genome, genome_id  # noqa: E402
from alpharush_rl.strategy_brain import LanguageBrain  # noqa: E402

RUNS = ROOT / "runtime/rl/elite-v1/runs"
FAMILY_RUNS = {"A": ["native-search-1d4786a9538b4ba7bce718a29f7fca4a"],
               "B": ["native-search-08f5b21f202b4b6596dbb2b4d5a8c619"]}
CHALLENGE_RUNS = ["native-search-77f73818b99d43b98815025897d07bb3", "native-search-a9ceecb077f5459291a3768e0e2c6f84"]
MAIN_STARS = {1: 3, 2: 3, 3: 3, 4: 2, 5: 3, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 2, 12: 2}

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["jobs"][sys.argv[2]]
family = sys.argv[3]
spec = job["campaign"]
seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else spec["seeds"]
runs = FAMILY_RUNS[family] + (sys.argv[5].split(",") if len(sys.argv) > 5 else []) + CHALLENGE_RUNS

records = {}
for run_id in runs:
    for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kind"] not in ("search_eval", "search_validate"):
            continue
        p = row["payload"]
        key = (p["level"], {CHALLENGE_RUNS[0]: 2, CHALLENGE_RUNS[1]: 3}.get(run_id, 1), p["genome_id"])  # run-level mode
        lives = (p["result"].get("outcome") or {}).get("lives") or 0
        records.setdefault(key, {})[p["seed"]] = (p["fitness"] >= 10000, lives)


def unit(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16) / 2 ** 32


log = []


def fake_episode(env, policy, protocol, *, seed, level, episode_id, **kwargs):
    genome = policy
    tail = episode_id.rsplit("-", 2)
    mode = int(tail[1][1:]) if tail[1].startswith("m") else 1
    kind = tail[2]
    gid = genome_id(check_genome(genome))
    rec = records.get((level, mode, gid))
    train_seed = 1001 + (seed - 1) % 10
    if rec and train_seed in rec:
        won, lives = rec[train_seed]
        how = f"record of train seed {train_seed}"
    elif rec:
        rate = sum(w for w, _ in rec.values()) / len(rec)
        won = unit(seed, level, mode, gid) < rate
        lives = max([l for w, l in rec.values() if w] or [0])
        how = f"drawn at the recorded rate {rate:.2f}"
    elif level <= 12 and mode == 1:
        replay = kind.startswith("r")
        stars = 3 if replay and unit(seed, level, kind) < 0.5 else MAIN_STARS[level] if not replay else MAIN_STARS[level]
        won, lives = True, {3: 18, 2: 10, 1: 3}[stars]
        how = "main-level stand-in"
    else:
        won, lives = False, 0
        how = "no record: loss"
    log.append((seed, level, mode, kind, gid, won, lives, how))
    return {"status": "terminal", "outcome": {"level_won": won, "lives": lives if won else 0},
            "final_summary": {"wave": None}}


campaign_run.run_episode = fake_episode


class Ctx:
    run_id = "brain-dry"
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
for seed in seeds:
    out, log[:] = Out(), []
    started = time.time()
    summary = campaign_run.run_campaign({**spec, "seed": seed}, lambda *a, **k: Env(), Ctx(), out,
                                        protocol=EpisodeProtocol(), make_policy=lambda g: g,
                                        make_elite_policy=lambda g: g, brain=LanguageBrain(broker, name="8b"))
    decisions = [p for kind, p in out.rows if kind == "strategy_decision"]
    Path(f"brain_dry_{family}_{seed}.decisions.json").write_text(json.dumps(decisions), encoding="utf-8")
    ups = [d for d in decisions if d["kind"] == "upgrade"]
    print(f"upgrade decisions {len(ups)}: tested allocation chosen {sum(d['choice'] == 0 for d in ups)}, "
          f"other {[(d['level'], d['attempt'], d['choice'], round(d['p'][0], 2)) for d in ups if d['choice']]}")
    games = [p for kind, p in out.rows if kind in ("campaign_attempt", "challenge_attempt", "star_replay")]
    print(f"=== seed {seed}: {len(games)} games, {len(decisions)} brain decisions, {time.time() - started:.0f} s")
    for (kind, p), entry in zip([(k, p) for k, p in out.rows if k in ("campaign_attempt", "challenge_attempt",
                                                                        "star_replay")], log):
        upgrades = p["profile"].get("upgrades")
        plan = spec["plans"][str(p["level"])] if kind != "challenge_attempt" else \
            spec["challenges"][f"{p['level']}:{p['mode']}"]
        index = p.get("plan_index", 0)
        ev = plan[index].get("evidence", "") if isinstance(plan[index], dict) else ""
        print(f"L{p['level']:02d}{'m' + str(p['mode']) if p.get('mode') else '  '} {kind[:9]:9s} "
              f"{str(p.get('attempt', p.get('replay'))):>2} plan#{index} pkg={p['genome'].get('pkg', 'balanced'):9s} "
              f"won={int(p['won'])} stars={p.get('stars', '-')} | {ev[:60]} | {entry[7]} | up={upgrades}")
    print(json.dumps({k: summary.get(k) for k in ("completed", "levels_won", "failed_levels", "blocked_levels",
                                                  "total_stars", "challenge_stars", "won")}))
