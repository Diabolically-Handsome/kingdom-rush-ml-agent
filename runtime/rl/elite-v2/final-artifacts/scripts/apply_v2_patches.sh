#!/usr/bin/env bash
# elite-v2: apply the owner-approved campaign route (detour) and human-timing patches to the real repo, between jobs.
set -euo pipefail
S="<local scratch directory>"
R="C:/Users/<user>/Documents/AlphaRush"
PY="$R/.venv/Scripts/python.exe"
cd "$R"
if ls runtime/rl/*/job.lock >/dev/null 2>&1; then echo "job lock present; refusing"; exit 1; fi
B="$S/main-before-v2patches"
if [ -e "$B" ]; then echo "backup $B exists; refusing to overwrite it"; exit 1; fi
mkdir -p "$B"
cp -r alpharush_rl tools tests "$B/"
cp configs/phases/elite-v2.json "$B/elite-v2.json"
export PATCH_ROOT="$(cygpath -w "$R")"
"$PY" -B "$S/patch_route.py"
"$PY" -B "$S/patch_human_timing.py"
"$PY" -B -m unittest discover -s tests 2>&1 | grep -E "^(ERROR|FAIL):|^Ran|^OK|^FAILED" || true
