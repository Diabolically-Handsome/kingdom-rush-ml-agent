"""elite-v3 exploratory test (pre-registered in REPORT.md before running): can a language model judge WHEN to intervene?

The play-choice question showed no skill (both models ~ always P1). The one replicable judgement signal in the fork data
is timing (the rule brain deviates where deviating helps, within-game permutation p 0.002). Here the model gets the same
briefing (only what is on screen) and a binary question, with the one fact a human commander would know: the default
was tuned together with the battle plan, so changes usually hurt. Labels scored in both orders (LanguageBrain._ask).

usage: v3_llm_timing.py score <8b|24b> <S1 run id> <out json>        (needs the model worker; writes {key: p_intervene})
       v3_llm_timing.py analyse <S1 run id> <fork table json> <p json>[,<p json>...] [--perm 5000] [--boot 4000]
Analysis (no outcome is ever shown to a model):
  timing: within-game covariance of p_intervene with g = mean(non-P0 branches) - P0 at that consult; null = p shuffled
          among the consults of the same base game (one-sided p). Same statistic for the rule brain's deviate indicator.
  cross-fitted brain: intervene when p_intervene is above its own median within the level (decided per fold from the fit
          seeds), then play the best-ranked offered play id from the fit seeds; gain vs steady on the other seeds."""
import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runtime/rl/elite-v3/runs"
SYSTEM = ("You are the tactical commander of an agent playing Kingdom Rush (2011) on Normal difficulty. Towers and "
          "upgrades follow a fixed battle plan. The plan was tuned by search together with a default way of using the "
          "hero and the two spells (Rain of Fire and reinforcements), so changing that default usually makes things "
          "worse; intervene only when the situation on screen calls for it. Winning comes first, then keeping lives. "
          "Reply with exactly one option label from the list.")
OPTIONS = ["Keep the default for the coming wave (the hero holds its start point; the spells are used as tuned).",
           "Intervene for the coming wave: send the hero and/or hold the spells for a specific threat."]


def steady_consults(s1):
    out = {}
    for line in (RUNS / s1 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if '"cmd_steady"' not in line:
            continue
        r = json.loads(line)["payload"]
        if r.get("player") == "cmd_steady":
            for c in r["result"]["commander"]["consults"]:
                out[f"{r['level']}:{r['seed']}:{r['genome_id']}:{c['consult']}"] = c
    return out


def score(model, s1, out_path):
    from alpharush_rl import commander
    from alpharush_rl.model_broker import LanguageModelBroker
    from alpharush_rl.strategy_brain import LanguageBrain
    port = {"8b": 12081, "24b": 12082}[model]
    brain = LanguageBrain(LanguageModelBroker(f"http://127.0.0.1:{port}", timeout=300), name=model)
    result, models = {}, set()
    for key, c in steady_consults(s1).items():
        if len(c["plays"]) < 2:
            continue
        user = commander.Briefing(c["briefing"]).text() + "\nShould the commander intervene this wave?"
        _, record = brain._ask("timing", user, OPTIONS, SYSTEM)
        result[key] = record["p"][1]
        models.add(record.get("model"))
    Path(out_path).write_text(json.dumps({"model": sorted(m for m in models if m), "system": SYSTEM,
                                          "options": OPTIONS, "p_intervene": result}, indent=1), encoding="utf-8")
    vals = list(result.values())
    print(json.dumps({"consults": len(vals), "mean_p": round(statistics.mean(vals), 4),
                      "share_above_half": round(sum(v > 0.5 for v in vals) / len(vals), 4), "models": sorted(models - {None})}))


def analyse(s1, table_path, p_paths, perm, boots):
    consults = steady_consults(s1)
    T = []
    for e in json.loads(Path(table_path).read_text(encoding="utf-8")):
        key = ":".join(map(str, e["key"]))
        c = consults[key]
        n = len(c["plays"])
        if n < 2:
            continue
        y = [e["plays"][str(j)] for j in range(n)]
        T.append({"key": key, "game": tuple(e["key"][:3]), "level": e["key"][0], "seed": e["key"][1], "y": y,
                  "g": statistics.mean(y[1:]) - y[0], "ids": [p["id"] for p in c["plays"]],
                  "rule_dev": 1.0 if c["shadow"]["rule"] else 0.0})
    signals = {"rule (deviate indicator)": {t["key"]: t["rule_dev"] for t in T}}
    for path in p_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        signals[Path(path).stem] = data["p_intervene"]
    rng = random.Random(5)
    by_game = collections.defaultdict(list)
    for t in T:
        by_game[t["game"]].append(t)

    def cov(sig, shuffle=False):
        total, count = 0.0, 0
        for rows in by_game.values():
            xs = [sig[t["key"]] for t in rows]
            if shuffle:
                rng.shuffle(xs)
            mx, mg = statistics.mean(xs), statistics.mean(t["g"] for t in rows)
            total += sum((x - mx) * (t["g"] - mg) for x, t in zip(xs, rows))
            count += len(rows)
        return total / count

    out = {}
    print("== timing: within-game cov(signal, mean nonP0 - P0); null = signal shuffled within each base game ==")
    for name, sig in signals.items():
        missing = [t["key"] for t in T if t["key"] not in sig]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} consults unscored")
        obs = cov(sig)
        null = [cov(sig, True) for _ in range(perm)]
        p = (1 + sum(v >= obs for v in null)) / (1 + perm)
        sd = statistics.pstdev(null)
        out[name] = {"cov": round(obs, 5), "null_sd": round(sd, 5), "perm_p": round(p, 4)}
        print(f"{name:28s} cov {obs:+.5f}  (null sd {sd:.5f}, z {obs / sd if sd else 0:+.2f})  perm p {p:.4f}")

    def ranking(rows):
        eff = collections.defaultdict(list)
        for t in rows:
            m = statistics.mean(t["y"][1:])
            for j in range(1, len(t["y"])):
                eff[t["ids"][j]].append(t["y"][j] - m)
        return sorted(eff, key=lambda pid: -statistics.mean(eff[pid]))

    def boot(groups):
        games = sorted(groups)
        stat = lambda gs: sum(sum(groups[g]) for g in gs) / max(1, sum(len(groups[g]) for g in gs))
        vals = sorted(stat([rng.choice(games) for _ in games]) for _ in range(boots))
        return stat(games), vals[int(0.025 * boots)], vals[int(0.975 * boots) - 1]

    print("== cross-fitted brain: intervene when the signal is above the fit seeds' per-level median; best-ranked play ==")
    for name, sig in signals.items():
        for fit, sc in (({1201, 1202}, {1203, 1204}), ({1203, 1204}, {1201, 1202})):
            fit_rows = [t for t in T if t["seed"] in fit]
            order = ranking(fit_rows)
            med = {lv: statistics.median(sig[t["key"]] for t in fit_rows if t["level"] == lv)
                   for lv in {t["level"] for t in fit_rows}}
            gains = collections.defaultdict(list)
            dev = 0
            for t in (t for t in T if t["seed"] in sc):
                go = sig[t["key"]] > med[t["level"]]
                i = next(t["ids"].index(p) for p in order if p in t["ids"][1:]) if go else 0
                dev += go
                gains[t["game"]].append(t["y"][i] - t["y"][0])
            g = boot(gains)
            out.setdefault(name + " brain", {})[f"{sorted(sc)}"] = [round(x, 4) for x in g] + [dev]
            print(f"{name:28s} score {sorted(sc)}: gain vs steady {g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]  deviations {dev}")
    Path(table_path).with_name("llm_timing_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("score", "analyse"))
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--perm", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    if a.mode == "score":
        score(a.rest[0], a.rest[1], a.rest[2])
    else:
        analyse(a.rest[0], a.rest[1], a.rest[2].split(","), a.perm, a.boot)
