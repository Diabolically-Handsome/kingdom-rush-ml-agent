"""warm_start.packages: each imported plan is also queued with these star allocations."""
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


patch("alpharush_rl/search_job.py", [
    ('''# Optional: warm_start {"runs": [earlier native-search run ids], "top": N, "reevaluate": bool} takes each
# level's N best journaled plans (adopted as measured, or re-played first);''',
     '''# Optional: warm_start {"runs": [earlier native-search run ids], "top": N, "reevaluate": bool,
# "packages": [star allocations]} takes each level's N best journaled plans (adopted as measured, or
# re-played first; each is also queued with every listed allocation);'''),
    ('''    if warm is not None and (not isinstance(warm, dict) or not {"runs", "top"} <= set(warm) <= {"runs", "top", "reevaluate"}
                             or warm.get("reevaluate", False) not in (True, False)''',
     '''    from .campaign import PACKAGES
    if warm is not None and (not isinstance(warm, dict)
                             or not {"runs", "top"} <= set(warm) <= {"runs", "top", "reevaluate", "packages"}
                             or warm.get("reevaluate", False) not in (True, False)
                             or not isinstance(warm.get("packages", []), list)
                             or not set(warm.get("packages", [])) <= set(PACKAGES)'''),
    ('''        issues.append("warm_start must be {runs: [native-search run ids], top: 1..100}")''',
     '''        issues.append("warm_start must be {runs: [native-search run ids], top: 1..100} "
                      "(optionally reevaluate: bool, packages: [star allocations])")'''),
    ('''                out.append("search_import", {"run_id": ctx.run_id, "level": task["level"],
                                             "genome_id": genome_id(entry["genome"]), "genome": entry["genome"],
                                             "fitness": entry["fitness"], "source_run": entry["source_run"],
                                             "source_sha256": entry["source_sha256"]})''',
     '''                out.append("search_import", {"run_id": ctx.run_id, "level": task["level"],
                                             "genome_id": genome_id(entry["genome"]), "genome": entry["genome"],
                                             "fitness": entry["fitness"], "source_run": entry["source_run"],
                                             "source_sha256": entry["source_sha256"]})
                for package in spec["warm_start"].get("packages", []):
                    variant = check_genome({**entry["genome"], "pkg": package})
                    if genome_id(variant) != genome_id(entry["genome"]):
                        search.queue(variant)
                        out.append("search_variant", {"run_id": ctx.run_id, "level": task["level"],
                                                      "genome_id": genome_id(variant), "genome": variant,
                                                      "parent_id": genome_id(entry["genome"])})'''),
])
print("warm packages patched")
