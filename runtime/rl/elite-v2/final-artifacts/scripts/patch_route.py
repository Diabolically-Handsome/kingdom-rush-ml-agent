"""Campaign route like a player (owner-approved 2026-10-10): opt-in spec key "detour".

With detour, a main level is tried once per plan with a given save (a lost plan replays its lost game on a
deterministic engine); when every plan has lost, the campaign detours like a stuck player: replays of the earlier
main levels won with fewer than 3 stars and the open Heroic/Iron challenges of the earlier 3-star levels (skipping
challenges already won, or already lost with as many stars), then retries the level with fresh attempts if the
detour raised the star total. Without "detour" every path is byte-for-byte the old one (same attempt indices,
episode ids and decisions). Apply only while no runtime/rl/*/job.lock exists.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", r"C:\Users\<user>\Documents\AlphaRush"))
REAL = Path(r"C:\Users\<user>\Documents\AlphaRush")
if ROOT.resolve() == REAL.resolve() and any((REAL / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")

path = ROOT / "alpharush_rl/campaign_run.py"
s = path.read_bytes().decode("utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:90]
    s = s.replace(old, new)


# --- spec ------------------------------------------------------------------------------------------------------
rep('''OPTIONAL_CAMPAIGN_KEYS = {"start", "challenges", "challenge_attempts", "star_attempts"}''',
    '''# "detour" (bool): when every plan of a main level has lost, replay earlier levels for stars and play the open
# challenges, then retry the level with the new upgrades (as a stuck player does); each plan is tried once per save.
OPTIONAL_CAMPAIGN_KEYS = {"start", "challenges", "challenge_attempts", "star_attempts", "detour"}''')
rep('''    replays = spec.get("star_attempts", 0)''',
    '''    if not isinstance(spec.get("detour", False), bool):
        issues.append("detour must be true or false")
    replays = spec.get("star_attempts", 0)''')

# --- state ------------------------------------------------------------------------------------------------------
rep('''    won_challenges, challenge_log, replay_log = {}, [], []''',
    '''    won_challenges, challenge_log, replay_log = {}, [], []
    detour = bool(spec.get("detour"))
    detours, replay_counts, challenge_counts, challenge_tried = [], {}, {}, {}''')

# --- replays: an optional level limit, attempt numbers unique across calls --------------------------------------
rep('''    def replay_for_stars():
        """Replays of main levels won with fewer than 3 stars (their portfolio plans in order, now with the
        upgrades of the whole main campaign); a re-win with more lives raises the stored stars."""
        nonlocal owned
        for level in range(1, MAIN_LEVELS + 1):
            if won.get(level, 3) >= 3 or str(level) not in spec["plans"]:
                continue
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in spec["plans"][str(level)]]
            for attempt in range(spec.get("star_attempts", 0)):''',
    '''    def replay_for_stars(limit=MAIN_LEVELS + 1):
        """Replays of main levels (below ``limit``) won with fewer than 3 stars (their portfolio plans in order,
        now with the upgrades earned so far); a re-win with more lives raises the stored stars. Replay numbers
        continue across calls, so a later pass tries the next plans."""
        nonlocal owned
        for level in range(1, min(limit, MAIN_LEVELS + 1)):
            if won.get(level, 3) >= 3 or str(level) not in spec["plans"]:
                continue
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in spec["plans"][str(level)]]
            first = replay_counts.get(level, 0)
            replay_counts[level] = first + spec.get("star_attempts", 0)
            for attempt in range(first, first + spec.get("star_attempts", 0)):''')

# --- challenges: optional limit (a detour), skip won / already-lost-with-as-many-stars ---------------------------
rep('''    def play_challenges():
        """The 3-star main levels' Heroic/Iron challenges, in level order; returns a stop reason or None."""
        nonlocal owned
        challenges_played[0] = True
        stop = replay_for_stars() if spec.get("star_attempts") else None''',
    '''    def play_challenges(limit=None):
        """The 3-star main levels' Heroic/Iron challenges, in level order; returns a stop reason or None.
        With ``limit`` (a detour before main level ``limit``) only the levels before it; a challenge already won,
        or already lost with as many stars as now, is skipped; attempt numbers continue across calls."""
        nonlocal owned
        if limit is None:
            challenges_played[0] = True
        stop = replay_for_stars(limit or MAIN_LEVELS + 1) if spec.get("star_attempts") else None''')
rep('''            if won.get(level) != 3:
                continue  # the map keeps a level's challenges locked below 3 campaign stars
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in spec["challenges"][key]]
            failed = []
            for attempt in range(spec["challenge_attempts"]):''',
    '''            if won.get(level) != 3:
                continue  # the map keeps a level's challenges locked below 3 campaign stars
            if (limit is not None and level >= limit) or mode in won_challenges.get(level, []):
                continue
            if challenge_tried.get(key, -1) >= total_stars():
                continue  # lost before with as many stars: the same games would be lost again
            challenge_tried[key] = total_stars()
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in spec["challenges"][key]]
            failed = []
            first = challenge_counts.get(key, 0)
            challenge_counts[key] = first + spec["challenge_attempts"]
            for attempt in range(first, first + spec["challenge_attempts"]):''')

# --- the level attempts become a function; the detour wraps them --------------------------------------------------
start = s.index('''            victory, failed = False, []
            for attempt in range(spec["attempts_per_level"]):
''')
body_start = s.index("\n", s.index('for attempt in range(spec["attempts_per_level"]):', start)) + 1
end = s.index('''            if stopped_reason:
                break
            if not victory:
                failed_levels.append(level)''', body_start)
body = s[body_start:end]
old_stop = '''                if ctx.games_remaining < 1:
                    stopped_reason = "max_games"
                    break
'''
assert body.count(old_stop) == 1
body = body.replace(old_stop, '''                if ctx.games_remaining < 1:
                    return victory, "max_games"
''')
dedented = "".join((line[4:] if line.startswith("    ") else line) for line in body.splitlines(keepends=True))
function = ('''    def play_attempts(level, candidates, attempts, failed):
        """Attempts with the given indices at one level; returns (victory, stop reason or None). With detour, a
        main level stops once every plan has lost with this save (a retry would replay a lost game)."""
        nonlocal owned
        victory = False
        for attempt in attempts:
            if detour and level <= MAIN_LEVELS and len(set(failed)) >= len(candidates):
                break
''' + dedented + '''        return victory, None

    def detour_level(level, candidates, failed):
        """A stuck player's detour before retrying main level ``level``: replays and open challenges of the
        earlier levels; a retry only if they raised the star total. Returns (victory, stop reason or None)."""
        before = total_stars()
        stop = play_challenges(limit=level)
        after = total_stars()
        detours.append({"level": level, "stars_before": before, "stars_after": after})
        ctx.record("campaign_detour", seed=spec["seed"], level=level, stars_before=before, stars_after=after)
        if stop or after <= before:
            return False, stop
        tried = sum(1 for a in attempts_log if a.get("level") == level and "attempt" in a)
        failed.clear()
        return play_attempts(level, candidates, range(tried, tried + spec["attempts_per_level"]), failed)

''')
call = '''            failed = []
            victory, stop = play_attempts(level, candidates, range(spec["attempts_per_level"]), failed)
            if not stop and not victory and detour and level <= MAIN_LEVELS:
                victory, stop = detour_level(level, candidates, failed)
            if stop:
                stopped_reason = stop
'''
s = s[:start] + call + s[end:]
anchor = '''    try:
        for level in spec["levels"]:'''
assert s.count(anchor) == 1
s = s.replace(anchor, function + anchor)
rep('''    if spec.get("star_attempts"):
        summary.update(star_replays=replay_log)''',
    '''    if spec.get("star_attempts"):
        summary.update(star_replays=replay_log)
    if detour:
        summary.update(detours=detours)''')
path.write_bytes(s.encode("utf-8"))

# --- tests ------------------------------------------------------------------------------------------------------
tp = ROOT / "tests/test_elite.py"
t = tp.read_bytes().decode("utf-8")
anchor = '''class JournalTests(unittest.TestCase):'''
assert t.count(anchor) == 1
t = t.replace(anchor, '''class DetourTests(unittest.TestCase):
    """The campaign route of a stuck player (spec key detour)."""

    def run(self, result=None):
        with tempfile.TemporaryDirectory(prefix="alpharush-detour-") as tmp:
            self.tmp = Path(tmp)
            return super().run(result)

    def play(self, spec, lose):
        class Toy(LoseOn):
            pass
        Toy.lose = tuple(lose)
        made = []

        def factory(level, seed, profile, attempt, port, mode=1):
            made.append((level, mode, attempt))
            return Toy(seed=seed, level=level, gold=600)
        run_dir = self.tmp / "runs" / f"native-campaign-{len(list(self.tmp.glob('**/episodes.jsonl')))}"
        run_dir.mkdir(parents=True)
        ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, self.tmp, time.monotonic() + 300, 400,
                                    job_kind="native-campaign")
        out = Journal(run_dir / "episodes.jsonl")
        summary = run_campaigns(spec, factory, ctx, out, protocol=PROTOCOL, make_policy=BuildOrderPolicy,
                                ports=[9001])
        return summary["campaigns"][0], made, out

    def spec(self, **extra):
        return {"levels": [1, 2, 3], "seeds": [1001], "attempts_per_level": 3, "policy": "plan", "brain": "rule",
                "plans": {"1": [WIN_PLAN], "2": [WIN_PLAN], "3": [WIN_PLAN, genome([["b", "02", "mage"]])]},
                **extra}

    def test_without_detour_a_lost_main_level_ends_the_campaign_as_before(self):
        campaign, made, _ = self.play(self.spec(), lose=(3,))
        self.assertEqual(("level_failed", 3), (campaign["stopped_reason"], campaign["failed_level"]))
        self.assertEqual([(3, 1, 0), (3, 1, 1), (3, 1, 2)], [m for m in made if m[0] == 3])  # repeats plan 0
        self.assertNotIn("detours", campaign)

    def test_detour_plays_open_challenges_then_retries_with_fresh_attempts(self):
        spec = self.spec(detour=True, challenges={"1:2": [WIN_PLAN]}, challenge_attempts=1)
        spec["levels"] = list(range(1, 13))
        spec["plans"] = {str(level): [WIN_PLAN, genome([["b", "02", "mage"]])] for level in range(1, 13)}
        campaign, made, out = self.play(spec, lose=(3,))
        level3 = [m for m in made if m[0] == 3]
        # each plan once, the challenge of 3-star level 1 (the detour), then each plan once more with +1 star
        self.assertEqual([(3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3)], level3)
        challenge = made.index((1, 2, 0))
        self.assertLess(made.index((3, 1, 1)), challenge)
        self.assertLess(challenge, made.index((3, 1, 2)))
        self.assertEqual([{"level": 3, "stars_before": 6, "stars_after": 7}], campaign["detours"])
        self.assertEqual(("level_failed", 3), (campaign["stopped_reason"], campaign["failed_level"]))
        self.assertEqual({"1": [2]}, campaign["challenges_won"])

    def test_a_detour_without_new_stars_does_not_retry(self):
        campaign, made, _ = self.play(self.spec(detour=True), lose=(3,))
        self.assertEqual([(3, 1, 0), (3, 1, 1)], [m for m in made if m[0] == 3])
        self.assertEqual([{"level": 3, "stars_before": 6, "stars_after": 6}], campaign["detours"])

    def test_detour_must_be_a_bool(self):
        self.assertIsNone(check_campaign(self.spec(detour="yes"), POOLS, phase)[0])
        self.assertEqual([], check_campaign(self.spec(detour=True), POOLS, phase)[1])


class JournalTests(unittest.TestCase):''')
tp.write_bytes(t.encode("utf-8"))
print("patched", ROOT)
