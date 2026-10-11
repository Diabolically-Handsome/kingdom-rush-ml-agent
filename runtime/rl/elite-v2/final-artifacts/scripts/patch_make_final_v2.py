from pathlib import Path
p = Path(__file__).parent / "make_final_jobs_v2.py"
s = p.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
rep('''         levels=13,15 challenges=0 (eval/dagger/collect: only these elite levels, no challenge tasks)  apply=1''',
    '''         levels=13,15 challenges=0 (eval/dagger/collect: only these elite levels, no challenge tasks)  apply=1
         brain=8b|rule  detour=1 (campaign/final: the stuck-player route, owner-approved 2026-10-10)
         features=timed_clicks,early_waves  click_ticks=N (every job kind: the human-timing host features)''')
rep('''        if family == "A":
            campaign["challenges"] = challenge_plans(data, table)
            campaign["challenge_attempts"] = challenge_attempts
        games = len(levels) * attempts + (12 * star_attempts if star_attempts else 0) + (
            len(campaign["challenges"]) * challenge_attempts if family == "A" else 0)''',
    '''        if family == "A":
            campaign["challenges"] = challenge_plans(data, table)
            campaign["challenge_attempts"] = challenge_attempts
        detour = opts.get("detour") == "1"
        if detour:
            campaign["detour"] = True
        games = len(levels) * attempts + (12 * star_attempts if star_attempts else 0) + (
            len(campaign["challenges"]) * challenge_attempts if family == "A" else 0)
        if detour:  # each main level may be retried once after a detour (more replays and challenge attempts)
            games += 12 * attempts + 12 * star_attempts + (
                len(campaign["challenges"]) * challenge_attempts if family == "A" else 0)''')
rep('''        if family == "A":
            extra += ("; then the Heroic/Iron challenges of the 3-star main levels (each win adds a star to the "
                      "upgrade budget) before the elite stages")''',
    '''        if family == "A":
            extra += ("; then the Heroic/Iron challenges of the 3-star main levels (each win adds a star to the "
                      "upgrade budget) before the elite stages")
        if detour:
            extra += ("; a main level lost with every plan is retried after a stuck player's detour (replays of the "
                      "earlier levels for stars and their open challenges) if the detour raised the star total")
        who = ("the 8B strategy brain picks" if campaign["brain"] == "8b" else
               "the fixed rule brain (ablation of the 8B strategy brain) picks")''')
rep('''on Normal{extra}: the 8B strategy brain picks each "''', '''on Normal{extra}: {who} each "''')
rep('''    cfg["jobs"][name] = job
    # The live config''', '''    features = [f for f in opts.get("features", "").split(",") if f]
    if features:
        job["host_features"] = features
        job["purpose"] += f"; human-timing host features {features}"
    if "click_ticks" in opts:
        job["episode_protocol"]["click_ticks"] = int(opts["click_ticks"])
        job["purpose"] += f"; {int(opts['click_ticks'])}-tick trap clicks"
    cfg["jobs"][name] = job
    # The live config''')
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
