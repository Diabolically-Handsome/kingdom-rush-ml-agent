#!/usr/bin/env bash
# elite-v3: copy the reviewed v4 host + commander from the merged development copy into the real repo, between jobs.
set -euo pipefail
S="<local scratch directory>"
R="C:/Users/<user>/Documents/AlphaRush"
M="$S/v4_merged"
PY="$R/.venv/Scripts/python.exe"
cd "$R"
if ls runtime/rl/*/job.lock >/dev/null 2>&1; then echo "job lock present; refusing"; exit 1; fi
if tasklist 2>/dev/null | grep -qi "kingdom rush"; then echo "game processes running (probes?); refusing"; exit 1; fi
B="$S/main-before-v4"
if [ -e "$B" ]; then echo "backup $B exists; refusing to overwrite it"; exit 1; fi
FILES="alpharush_rl/assets/host.lua alpharush_rl/engine.py alpharush_rl/env.py alpharush_rl/menus.py alpharush_rl/episode.py
alpharush_rl/eval_job.py alpharush_rl/commander.py tools/campaign-survey.py tests/test_host_actions_v4.py tests/test_commander.py
tests/test_engine_prepare.py tests/test_campaign_survey.py"
mkdir -p "$B"
for f in $FILES; do
  if [ -e "$R/$f" ]; then mkdir -p "$B/$(dirname "$f")"; cp "$R/$f" "$B/$f"; fi
done
for f in $FILES; do cp "$M/$f" "$R/$f"; done
# nothing else in the merged copy may differ from the real repo
for d in alpharush_rl tools tests configs; do
  diff -rq "$R/$d" "$M/$d" | grep -v __pycache__ | grep -v "configs/phases/elite-v3.json\|pools-elite-v3.json\|test_host_golden.py" || true
done
"$PY" -B -m unittest discover -s tests 2>&1 | grep -E "^(ERROR|FAIL):|^Ran|^OK|^FAILED" || true
