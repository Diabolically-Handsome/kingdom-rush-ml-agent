"""LanguageBrain: score every menu in all cyclic orders and average, so position bias cannot decide."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", "C:/Users/<user>/Documents/AlphaRush"))


def patch(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


patch("alpharush_rl/strategy_brain.py", [
    ('''LABELS = [chr(ord("A") + i) for i in range(26)]''',
     '''LABELS = [chr(ord("A") + i) for i in range(26)]
# The 8B's label scores carry a position bias (in probes it favoured option B whatever B said), so every
# menu is scored in up to ROTATIONS cyclic orders and each option's probability is averaged over them.
ROTATIONS = 6'''),
    (r'''    def _ask(self, kind, user, options):
        labels = LABELS[:len(options)]
        self.counter += 1
        request = {"id": f"strategy-{kind}-{self.counter}", "system": SYSTEM,
                   "user": user + "\n\nOptions:\n" + "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
                   + "\n\nAnswer with one label.", "labels": labels}
        response = self.broker.distribution(request)
        index = labels.index(response["choice"])
        record = {"kind": kind, "request_id": request["id"],
                  "prompt_sha256": hashlib.sha256(request["user"].encode("utf-8")).hexdigest(),
                  "options": options, "choice": index, "p": response["p"], "model": response.get("model")}
        return index, record''',
     r'''    def _ask(self, kind, user, options):
        """The option with the highest probability averaged over cyclic orders of the menu (with at most
        ROTATIONS options every option is scored once in every position)."""
        n = len(options)
        labels = LABELS[:n]
        total, ids, hashes, model = [0.0] * n, [], [], None
        shifts = range(min(n, ROTATIONS))
        for shift in shifts:
            order = [(i + shift) % n for i in range(n)]
            self.counter += 1
            request = {"id": f"strategy-{kind}-{self.counter}", "system": SYSTEM,
                       "user": user + "\n\nOptions:\n" + "\n".join(f"{label}. {options[i]}"
                                                                 for label, i in zip(labels, order))
                       + "\n\nAnswer with one label.", "labels": labels}
            response = self.broker.distribution(request)
            for position, i in enumerate(order):
                total[i] += float(response["p"][position])
            ids.append(request["id"])
            hashes.append(hashlib.sha256(request["user"].encode("utf-8")).hexdigest())
            model = response.get("model")
        p = [x / len(shifts) for x in total]
        index = max(range(n), key=lambda i: (p[i], -i))
        record = {"kind": kind, "request_id": ids[0], "request_ids": ids, "prompt_sha256": hashes[0],
                  "prompt_sha256s": hashes, "options": options, "choice": index, "p": p, "model": model,
                  "rotations": len(shifts)}
        return index, record'''),
])
patch("tests/test_strategy_brain.py", [(
    '''    def test_plan_choice_and_descriptions(self):''',
    '''    def test_choices_are_averaged_over_menu_orders(self):
        class Biased:
            """Always 0.6 on label B, the rest spread evenly; option 'good' gets +0.2 wherever it is."""
            def __init__(self):
                self.requests = []

            def distribution(self, request):
                self.requests.append(request)
                lines = [l for l in request["user"].splitlines() if l[:2] in {f"{x}." for x in request["labels"]}]
                n = len(lines)
                p = [0.4 / (n - 1)] * n
                p[1] = 0.6
                good = next(i for i, line in enumerate(lines) if "good" in line)
                p = [x + (0.2 if i == good else 0.0) for i, x in enumerate(p)]
                p = [x / sum(p) for x in p]
                return {"p": p, "choice": request["labels"][max(range(n), key=p.__getitem__)], "model": "biased"}
        broker = Biased()
        index, record = LanguageBrain(broker)._ask("plan", "pick", ["bad one", "bad two", "good", "bad three"])
        self.assertEqual(2, index)  # a single unrotated request would have answered B ("bad two")
        self.assertEqual(4, len(broker.requests))
        self.assertEqual(4, record["rotations"])
        self.assertAlmostEqual(1.0, sum(record["p"]))
        self.assertEqual(len(set(record["prompt_sha256s"])), 4)

    def test_plan_choice_and_descriptions(self):''')])
print("debias patched")
