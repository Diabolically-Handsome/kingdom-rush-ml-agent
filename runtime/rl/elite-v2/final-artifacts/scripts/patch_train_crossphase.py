"""tools/train-operator.py: a --runs entry "<phase>/<run id>" reads that run from runtime/rl/<phase>/runs (e.g. the
campaign-v1 demonstrations the main operator was trained on, together with new elite-v2 runs). Plain run ids still
come from --phase-state. Apply only while no runtime/rl/*/job.lock exists."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", r"C:\Users\<user>\Documents\AlphaRush"))
REAL = Path(r"C:\Users\<user>\Documents\AlphaRush")
if ROOT.resolve() == REAL.resolve() and any((REAL / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")


def patch(rel, pairs):
    path = ROOT / rel
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:80])
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


patch("tools/train-operator.py", [
    ("""    for run_id in run_ids:
        data = runs / run_id / "data\"""",
     """    for run_id in run_ids:
        # "<phase>/<run id>": a run of another phase under runtime/rl (e.g. campaign-v1's demonstrations).
        phase_name, _, plain = run_id.rpartition("/")
        base = (ROOT / "runtime/rl" / phase_name / "runs") if phase_name else runs
        if phase_name and not base.resolve().is_relative_to((ROOT / "runtime/rl").resolve()):
            raise SystemExit(f"run {run_id!r} is outside runtime/rl")
        data = base / plain / "data\""""),
])
print("patched", ROOT)
