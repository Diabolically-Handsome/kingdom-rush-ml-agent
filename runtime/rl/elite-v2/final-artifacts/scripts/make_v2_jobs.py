"""Derive make_final_jobs_v2.py (elite-v2) from make_final_jobs.py (elite-v1) by exact text replacements."""
from pathlib import Path

here = Path(__file__).parent
s = (here / "make_final_jobs.py").read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


rep('''"""Set one elite-v1 job (eval | dagger | campaign | final) from build_portfolios.py output.

usage: make_final_jobs.py <kind> <portfolios.json> <A|B> [key=value ...]
options: weights=<elite operator path>''',
    '''"""Set one elite-v2 job (eval | dagger | campaign | final) from build_portfolios_v2.py output (option A).

usage: make_final_jobs_v2.py <kind> <portfolios_v2.json> A [key=value ...]
elite-v2: level 11 takes its new portfolio (elite-v2 search 1) topped up with campaign-v1's level-11 plans; fillers
come from the searches of both phases; main=<main operator path> picks the main-campaign network.
options: weights=<elite operator path>  main=<main operator path>''')
rep('''RUNS = ROOT / "runtime/rl/elite-v1/runs"
CONFIG = ROOT / "configs/phases/elite-v1.json"
APPROVAL = ("[用户原话已省略 / user's message omitted]"
            "（2026-10-06 用户批准精英关阶段：13-26 关、战役模式、普通难度；最终评测从新存档打完全部 26 关；"
            "可加兵营集结点动作；自主推进至 2026-10-07 20:00）")''',
    '''RUN_DIRS = [ROOT / "runtime/rl/elite-v1/runs", ROOT / "runtime/rl/elite-v2/runs"]
CONFIG = ROOT / "configs/phases/elite-v2.json"
APPROVAL = json.loads(CONFIG.read_text(encoding="utf-8"))["authorization"][-1]''')
rep('''    for run in sorted(RUNS.glob("native-search-*")):''',
    '''    for run in sorted(p for d in RUN_DIRS for p in d.glob("native-search-*") if (p / "episodes.jsonl").exists()):''')
rep('''def main_plans():
    cfg = json.loads((ROOT / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
    return cfg["jobs"]["native-final"]["campaign"]["plans"]''',
    '''def main_plans(data, table, size=10):
    """campaign-v1's frozen main-campaign portfolio; level 11 from elite-v2 search 1, topped up with the old plans,
    whose evidence is their elite-v2 re-measurement (their campaign-v1 record came from other seeds)."""
    cfg = json.loads((ROOT / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
    plans = dict(cfg["jobs"]["native-final"]["campaign"]["plans"])
    new11 = list(data.get("elite", {}).get("11", []))
    ids = {genome_id(check_genome(e["genome"])) for e in new11}
    def remeasured(e):
        g = check_genome(e["genome"] if isinstance(e, dict) and "genome" in e else e)
        rec = table.get((11, 1), {}).get(genome_id(g), {}).get("seeds", {})
        return -sum(w for w, _, _ in rec.values()) / max(1, len(rec))
    for e in sorted(plans["11"], key=remeasured):  # the best re-measured older plans first
        g = check_genome(e["genome"] if isinstance(e, dict) and "genome" in e else e)
        if len(new11) < size and genome_id(g) not in ids:
            ids.add(genome_id(g))
            rec = table.get((11, 1), {}).get(genome_id(g), {}).get("seeds", {})
            wins, n = sum(w for w, _, _ in rec.values()), len(rec)
            new11.append({"genome": g, "evidence": (f"won {wins} of {n} practice seeds (older plan, re-measured)" if n
                                                    else "older plan, not re-measured")})
    plans["11"] = new11
    return plans''')
rep('''    data = json.loads(source.read_text(encoding="utf-8"))
    profile = PROFILES[family]''',
    '''    data = json.loads(source.read_text(encoding="utf-8"))
    data.setdefault("elite_A", {k: v for k, v in data.get("elite", {}).items() if int(k) >= 13})
    profile = PROFILES[family]
    main_weights = opts.get("main", MAIN_WEIGHTS)''')
rep('''                 list(range(7001, 7011)) if kind == "eval" else list(range(1001, 1011)))''',
    '''                 list(range(7101, 7111)) if kind == "eval" else list(range(1001, 1011)))''')
rep('''        seeds = [int(s) for s in opts.get("seeds", "8001,8002,8003,8004,8005" if final else
                                          "7001,7002,7003,7004,7005").split(",")]''',
    '''        seeds = [int(s) for s in opts.get("seeds", "8101,8102,8103,8104,8105" if final else
                                          "7101,7102,7103,7104,7105").split(",")]''')
rep('''        plans = {**main_plans(), **elite_plans(data, family, int(opts.get("plans", 10)), table)}''',
    '''        plans = {**main_plans(data, table), **elite_plans(data, family, int(opts.get("plans", 10)), table)}''')
rep('''                                   "attempt's plan and star allocation; the campaign-v1 operator network (levels 1-12, "
                                   "action scope v2) and the elite operator network (elite stages and challenges, v3) "''',
    '''                                   "attempt's plan and star allocation (all 13 heroes from level 12 on); the main-"
                                   "campaign operator network (levels 1-12, action scope v2) and the elite operator "
                                   "network (elite stages and challenges, v3) "''')
rep('''               "operator_weights": weights_of(MAIN_WEIGHTS), "elite_operator_weights": weights_of(opts["weights"]),''',
    '''               "operator_weights": weights_of(main_weights), "elite_operator_weights": weights_of(opts["weights"]),''')
rep('''        if (ROOT / "runtime/rl/elite-v1/job.lock").exists():
            raise SystemExit("refused: a job holds runtime/rl/elite-v1/job.lock (the config is pinned)")''',
    '''        if any((ROOT / "runtime/rl").glob("*/job.lock")):
            raise SystemExit("refused: a job holds a runtime/rl/*/job.lock (the config is pinned)")''')
(here / "make_final_jobs_v2.py").write_bytes(s.encode("utf-8"))
print("ok")
