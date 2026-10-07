"""Elite searches keep one plan per search result: a plan scoring exactly the same mean fitness as one already in
the population (the same game on every search seed: same waves, lives and final tick) is a behavioural twin and
would only crowd out diversity and validation slots (run 9 validated up to 9 twins of one plan per level).
Apply only while no job holds runtime/rl/elite-v1/job.lock (search.py and tests are pinned)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", r"C:\Users\<user>\Documents\AlphaRush"))
if (ROOT / "runtime/rl/elite-v1/job.lock").exists():
    raise SystemExit("refused: a job holds the lock")


def patch(rel, pairs):
    path = ROOT / rel
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:60])
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


patch("alpharush_rl/search.py", [
    ("""        if self.best_seen is None or score > self.best_seen:
            self.best_seen = score
        self.population.append((score, key, check_genome(genome)))
        self.population.sort(key=lambda entry: (-entry[0], entry[1]))
        del self.population[self.size:]

    def report(""",
     """        if self.best_seen is None or score > self.best_seen:
            self.best_seen = score
        self._insert(score, key, genome)

    def _insert(self, score: float, key: str, genome: dict):
        if self.rally and any(entry[0] == score for entry in self.population):
            # Elite searches: the exact same mean fitness means the same game on every search seed (waves,
            # lives and final tick); such a twin of a kept plan would only crowd out diversity and validation.
            return
        self.population.append((score, key, check_genome(genome)))
        self.population.sort(key=lambda entry: (-entry[0], entry[1]))
        del self.population[self.size:]

    def report("""),
    ("""        if self.best_seen is None or score > self.best_seen:
            self.best_seen, self.improved_at = score, self.evaluations
        self.population.append((score, key, check_genome(genome)))
        self.population.sort(key=lambda entry: (-entry[0], entry[1]))
        del self.population[self.size:]
""",
     """        if self.best_seen is None or score > self.best_seen:
            self.best_seen, self.improved_at = score, self.evaluations
        self._insert(score, key, genome)
"""),
])

patch("tests/test_elite.py", [
    ("""    def test_improvement_tracking(self):""",
     """    def test_elite_population_keeps_one_plan_per_result(self):
        search = LevelSearch(13, HOLDERS, [], "s", population=8, rally=True)
        g = [search.propose() for _ in range(4)]
        for genome, score in zip(g, (5.0, 5.0, 3.0, 5.0)):
            search.report(genome, score)
        self.assertEqual([5.0, 3.0], [entry[0] for entry in search.population])
        self.assertEqual(genome_id(g[0]), search.population[0][1])  # the first plan with that result stays
        plain = LevelSearch(5, HOLDERS, [], "s", population=8)
        g = [plain.propose() for _ in range(3)]
        for genome in g:
            plain.report(genome, 5.0)
        self.assertEqual(3, len(plain.population))  # main-campaign searches are unchanged

    def test_improvement_tracking(self):"""),
])
print("patched")
