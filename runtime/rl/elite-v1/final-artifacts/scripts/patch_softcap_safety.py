"""Audit follow-ups before the elite-v1 one-shot final (2026-10-07).

1. Soft tower cap: plan gene cap -6/-8/-10/-12 (teacher_v2 m < 0) = the hard cap's rule, except that builds resume
   while SOFT_CAP_GOLD or more is banked. Hard caps 6/8/10/12 and every validated plan are unchanged (genome ids
   too). Elite searches may draw/mutate soft caps.
2. RetryingBroker: transport failures of the 8B scoring server are retried with backoff (same request, same model:
   never a rule fallback) instead of ending the whole job.
3. A campaign that hits an exception outside a game (after the broker's retries) ends that seed only: its results
   so far stay in its summary, the other seeds play on; a failing env.close() is journaled, not fatal.
4. The 8B server's health is checked before the ledger opens a campaign/final job (a down server no longer spends
   the one-shot final).
Apply only while no job holds runtime/rl/elite-v1/job.lock (all these files are pinned).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", r"C:\Users\<user>\Documents\AlphaRush"))
REAL = Path(r"C:\Users\<user>\Documents\AlphaRush")
if (REAL / "runtime/rl/elite-v1/job.lock").exists() and ROOT.resolve() == REAL.resolve():
    raise SystemExit("refused: a job holds the lock")


def patch(rel, pairs):
    path = ROOT / rel
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:80])
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


# 1. soft cap --------------------------------------------------------------------------------------------------
patch("alpharush_rl/search.py", [
    ("""# "cap": the fallback builds no new tower once this many standard towers stand (0 = no cap).
FALLBACK_CAP = (0, 6, 8, 10, 12)
""",
     """# "cap": the fallback builds no new tower once this many standard towers stand (0 = no cap). A negative cap is
# soft: past -cap towers the fallback builds again only while SOFT_CAP_GOLD or more is banked (capped elite plans
# otherwise reach the last boss with thousands of gold and empty holders).
FALLBACK_CAP = (0, 6, 8, 10, 12)
SOFT_CAP = (-6, -8, -10, -12)
"""),
    ("""    if cap not in FALLBACK_CAP or isinstance(cap, bool):
        raise ValueError(f"cap must be one of {FALLBACK_CAP}")""",
     """    if cap not in FALLBACK_CAP + SOFT_CAP or isinstance(cap, bool):
        raise ValueError(f"cap must be one of {FALLBACK_CAP + SOFT_CAP}")"""),
    ("""        extra["cap"] = rng.choice(FALLBACK_CAP)""",
     """        extra["cap"] = rng.choice(FALLBACK_CAP + SOFT_CAP)"""),
    ("""                g["cap"] = rng.choice([c for c in FALLBACK_CAP if c != g.get("cap", OPTIONAL_GENOME["cap"])])""",
     """                g["cap"] = rng.choice([c for c in FALLBACK_CAP + SOFT_CAP
                                       if c != g.get("cap", OPTIONAL_GENOME["cap"])])"""),
])

patch("alpharush_rl/scripted_policies.py", [
    ("""#   m  tower cap (0 = none): no new build once this many standard towers stand, so gold goes to upgrades""",
     """#   m  tower cap (0 = none): no new build once this many standard towers stand, so gold goes to upgrades;
#      a negative m is soft: past -m towers builds resume while SOFT_CAP_GOLD or more is banked"""),
    ("""TEACHER_DEFAULTS = {"b": "rpab", "c": 50, "f": 50, "r": "3111", "e": 0, "s": 0, "m": 0}""",
     """TEACHER_DEFAULTS = {"b": "rpab", "c": 50, "f": 50, "r": "3111", "e": 0, "s": 0, "m": 0}
# A soft tower cap (m < 0) builds again from this much banked gold: no single upgrade costs that much.
SOFT_CAP_GOLD = 1000"""),
    ("""        else:
            if not value.isdigit() or str(int(value)) != value:""",
     """        else:
            digits = value[1:] if key == "m" and value.startswith("-") else value
            if not digits.isdigit() or str(int(value)) != value:"""),
    ("""                       for tower in towers.values()) >= cap:
            builds = []  # the tower cap is reached: keep the gold for upgrades""",
     """                       for tower in towers.values()) >= abs(cap):
            if cap > 0 or _number(state.get("gold")) < SOFT_CAP_GOLD:
                builds = []  # the tower cap is reached: keep the gold for upgrades"""),
])

patch("alpharush_rl/strategy_brain.py", [
    ("""            + (f"; never more than {genome['cap']} towers" if genome.get("cap") else "")""",
     """            + (f"; never more than {genome['cap']} towers" if genome.get("cap", 0) > 0 else
               f"; at most {-genome['cap']} towers until 1000 gold is banked" if genome.get("cap") else "")"""),
])

# 2. retrying broker -------------------------------------------------------------------------------------------
patch("alpharush_rl/model_broker.py", [
    ("""class LanguageModelPolicy:
""",
     """class RetryingBroker:
    \"\"\"Retries transport failures of a deterministic scoring server: the same request to the same model, so a
    retry can only reproduce the answer (never a rule fallback). Invalid answers still raise at once.\"\"\"
    DELAYS = (5, 15, 30, 60, 120, 240)

    def __init__(self, broker, delays=DELAYS, on_retry=None, sleep=None):
        import time
        self.broker, self.delays, self.on_retry = broker, tuple(delays), on_retry
        self.sleep = time.sleep if sleep is None else sleep

    def distribution(self, request: dict) -> dict:
        import http.client
        import urllib.error
        transient = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)
        for attempt, delay in enumerate((*self.delays, None)):
            try:
                return self.broker.distribution(request)
            except transient as exc:
                if delay is None:
                    raise
                if self.on_retry is not None:
                    self.on_retry(request.get("id"), attempt + 1, exc)
                self.sleep(delay)
        raise AssertionError("unreachable")


class LanguageModelPolicy:
"""),
])

# 3. per-seed exception containment ----------------------------------------------------------------------------
path = ROOT / "alpharush_rl/campaign_run.py"
text = path.read_bytes().decode("utf-8")
start = text.index('    for level in spec["levels"]:\n        if level > MAIN_LEVELS and not challenges_played[0]')
end = text.index('    played = set(spec["levels"])\n')
body = text[start:end]
assert body.rstrip().endswith("stopped_reason = play_challenges()"), body[-200:]
indented = "".join(("    " + line if line.strip() else line) for line in body.splitlines(keepends=True))
wrapped = ("    try:\n" + indented +
           "    except GateRefused:\n"
           "        raise\n"
           "    except Exception as exc:  # noqa: BLE001\n"
           "        # An engineering failure outside a game (e.g. the strategy brain still unreachable after the\n"
           "        # broker's retries) ends this seed's campaign only: its results so far stay in the summary.\n"
           "        stopped_reason = f\"exception: {type(exc).__name__}: {exc}\"[:300]\n"
           "        out.append(\"campaign_exception\", {\"run_id\": ctx.run_id, \"seed\": spec[\"seed\"],\n"
           "                                          \"error\": stopped_reason})\n")
text = text[:start] + wrapped + text[end:]
closes = text.count("if env is not None:\n")
assert closes == 3, closes
for indent in (" " * 16, " " * 20):
    text = text.replace(f"{indent}if env is not None:\n{indent}    env.close()\n",
                        f"{indent}_close(env, out, base)\n")
assert "env.close()" not in text.split("def _close", 1)[0].split("def run_campaign(", 1)[1], "unguarded close left"
text = text.replace('''def _progress(won):''', '''def _close(env, out, base):
    """Close a game; a failing close is journaled (the job's process guard reaps the game) and never ends a run."""
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:  # noqa: BLE001
        out.append("env_close_error", {**base, "error": f"{type(exc).__name__}: {exc}"[:300]})


def _progress(won):''', 1)
path.write_bytes(text.encode("utf-8"))

# 4. 8B health before the ledger opens a job, retrying broker in the worker -----------------------------------
patch("tools/campaign-survey.py", [
    ("""def supervise(config=CONFIG, job_kind=JOB_KIND):
    \"\"\"Hard deadline/STOP even during native RPC; closing the owned job kills our descendants.\"\"\"
    check = check_only(config, job_kind)
    if not check["ok"]:
        raise SurveyRefused(check)
    cp, _, root = phase.load_phase(config)
    # Environment failures refuse here, before the sole job is spent in the ledger.
    for port in _job_ports(check.get("survey", {}).get("workers")):
        _probe_port(port)
""",
     """def _brain_health():
    \"\"\"The 8B scoring server's health; GateRefused when it is unreachable or not ready.\"\"\"
    import urllib.request
    try:
        with urllib.request.urlopen(BRAIN_ENDPOINT + "/", timeout=10) as stream:
            health = json.load(stream)
    except OSError as exc:
        raise GateRefused(f"the 8B scoring server is unreachable: {exc}") from exc
    if health.get("ready") is not True:
        raise GateRefused("the 8B scoring server is not ready")
    return health


def supervise(config=CONFIG, job_kind=JOB_KIND):
    \"\"\"Hard deadline/STOP even during native RPC; closing the owned job kills our descendants.\"\"\"
    check = check_only(config, job_kind)
    if not check["ok"]:
        raise SurveyRefused(check)
    cp, cfg, root = phase.load_phase(config)
    # Environment failures refuse here, before the sole job is spent in the ledger.
    for port in _job_ports(check.get("survey", {}).get("workers")):
        _probe_port(port)
    if ((cfg.get("jobs", {}).get(job_kind) or {}).get("campaign") or {}).get("brain") == "8b":
        _brain_health()
"""),
    ("""        import urllib.request
        with urllib.request.urlopen(BRAIN_ENDPOINT + "/", timeout=10) as stream:
            health = json.load(stream)
        if health.get("ready") is not True:
            raise GateRefused("the 8B scoring server is not ready")
        ctx.record("strategy_brain", endpoint=BRAIN_ENDPOINT, model=health.get("model"))
        brain = LanguageBrain(LanguageModelBroker(BRAIN_ENDPOINT, timeout=180), name="8b")""",
     """        from alpharush_rl.model_broker import RetryingBroker
        health = _brain_health()
        ctx.record("strategy_brain", endpoint=BRAIN_ENDPOINT, model=health.get("model"))

        def retried(request_id, attempt, exc):
            print(f"8B request {request_id} failed ({exc!r}); retry {attempt}", file=sys.stderr, flush=True)
        brain = LanguageBrain(RetryingBroker(LanguageModelBroker(BRAIN_ENDPOINT, timeout=180), on_retry=retried),
                              name="8b")"""),
])

# tests ---------------------------------------------------------------------------------------------------------
patch("tests/test_elite.py", [
    ("""    def test_operators_never_add_rally_without_scope_v3(self):""",
     """    def test_soft_tower_cap_builds_again_with_banked_gold(self):
        from alpharush_rl.scripted_policies import parse_teacher_params, teacher_name
        g = genome([], cap=-8)
        self.assertEqual(-8, BuildOrderPolicy(g).fallback.params["m"])
        self.assertIn("at most 8 towers until 1000 gold is banked", strategy_brain.describe_plan(g))
        self.assertEqual(-6, parse_teacher_params("m=-6")["m"])
        self.assertEqual("teacher_v2:m=-6", teacher_name({**parse_teacher_params(""), "m": -6}))
        for bad in ("m=-0", "m=--6", "c=-5"):
            with self.assertRaises(ValueError):
                parse_teacher_params(bad)
        towers = [{"id": i, "holder_id": str(i), "template": "tower_archer_1"} for i in range(1, 9)]
        menu = [WAIT, *[option(label, {"action": "build_tower", "holder_id": h, "tower_type": "archer"}, 70.0)
                        for label, h in (("B", 50), ("D", 51), ("E", 52))],
                option("C", {"action": "upgrade_tower", "tower_id": 1, "target": "tower_archer_2"}, 110.0)]
        for gold, soft, hard in ((500, "C", "C"), (1000, "B", "C")):
            state = {"gold": gold, "wave": 1, "enemies": [], "towers": towers,
                     "holders": [{"id": h, "mesh_id": str(h), "path_score": 1} for h in (50, 51, 52)]}
            got = [BuildOrderPolicy(genome([], cap=c, f=25)).choose(state, menu, {"decision_index": 0})["label"]
                   for c in (-8, 8)]
            self.assertEqual([soft, hard], got, gold)
        rng = random.Random(5)
        caps = {random_genome(HOLDERS, [], rng, rally=True).get("cap", 0) for _ in range(300)}
        self.assertTrue({-6, -8, -10, -12} <= caps and {6, 8, 10, 12} <= caps)

    def test_operators_never_add_rally_without_scope_v3(self):"""),
    ("""class JournalTests(unittest.TestCase):""",
     """class SafetyTests(unittest.TestCase):
    def test_retrying_broker_retries_transport_errors_only(self):
        import urllib.error
        from alpharush_rl.model_broker import RetryingBroker

        class Flaky:
            def __init__(self, failures):
                self.failures, self.calls = list(failures), 0

            def distribution(self, request):
                self.calls += 1
                if self.failures:
                    raise self.failures.pop(0)
                return {"p": [1.0]}
        slept, retries = [], []
        flaky = Flaky([urllib.error.URLError("down"), ConnectionResetError("reset")])
        broker = RetryingBroker(flaky, delays=(1, 2, 3), on_retry=lambda *a: retries.append(a[1]), sleep=slept.append)
        self.assertEqual({"p": [1.0]}, broker.distribution({"id": "x"}))
        self.assertEqual(([1, 2], [1, 2], 3), (slept, retries, flaky.calls))
        with self.assertRaises(ValueError):  # an invalid answer is not a transport failure
            RetryingBroker(Flaky([ValueError("bad")]), sleep=slept.append).distribution({"id": "y"})
        with self.assertRaises(urllib.error.URLError):
            RetryingBroker(Flaky([urllib.error.URLError("x")] * 3), delays=(0, 0), sleep=slept.append) \\
                .distribution({"id": "z"})

    def test_an_exception_outside_a_game_ends_only_that_seed(self):
        from alpharush_rl.strategy_brain import RuleBrain

        class Broken(RuleBrain):
            def choose_plan(self, level, candidates, attempt, context):
                if level == 3:
                    raise ConnectionError("brain down")
                return super().choose_plan(level, candidates, attempt, context)
        with tempfile.TemporaryDirectory(prefix="alpharush-safety-") as tmp:
            run_dir = Path(tmp) / "runs" / "native-campaign-0001"
            run_dir.mkdir(parents=True)
            ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, Path(tmp), time.monotonic() + 300, 200,
                                        job_kind="native-campaign")
            spec = {"levels": [1, 2, 3], "seeds": [1001, 1002], "attempts_per_level": 1, "policy": "plan",
                    "brain": "rule", "plans": {str(level): [WIN_PLAN] for level in (1, 2, 3)}}
            out = Journal(run_dir / "episodes.jsonl")
            summary = run_campaigns(spec, lambda level, seed, profile, attempt, port: ToyEnv(seed=seed, level=level,
                                                                                                  gold=600),
                                    ctx, out, protocol=PROTOCOL, make_policy=BuildOrderPolicy, ports=[9001, 9002],
                                    brain=Broken())
            for campaign in summary["campaigns"]:
                self.assertEqual({"1", "2"}, set(campaign["won"]))
                self.assertTrue(campaign["stopped_reason"].startswith("exception: ConnectionError"))
                self.assertFalse(campaign["completed"])
            self.assertTrue(summary["stopped_reason"].startswith("exception"))
            self.assertEqual(2, sum(row["kind"] == "campaign_exception" for row in out.entries()))


class JournalTests(unittest.TestCase):"""),
])
print("patched", ROOT)
