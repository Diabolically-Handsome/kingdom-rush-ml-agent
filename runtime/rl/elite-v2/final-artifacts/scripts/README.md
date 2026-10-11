# As-run scripts (provenance copies)

These are the helper scripts as they were run during the elite-v2 phase (phase setup, search / collect / eval /
campaign job configuration, code patches, probes, evaluation comparisons, the final's verification and the
video recording). They are kept for provenance: the workspace root is masked as
`C:\Users\<user>\Documents\AlphaRush` and they read their inputs from a local scratch directory, so set the
root (and inputs) before re-running any of them. Inputs published elsewhere in this repository:

* `v2_l11_portfolio.json` is `../l11-portfolio.json`; the final's portfolio file (the argument of
  `make_final_jobs_v2.py`) is `../portfolios.json`;
* the challenge plans `build_portfolios_v2.py` copies from `portfolios.json` are elite-v1's
  `../../../elite-v1/final-artifacts/portfolios.json`, and the `make_final_jobs.py` that `make_v2_jobs.py` patches
  is `../../../elite-v1/final-artifacts/scripts/make_final_jobs.py`;
* `video_v2/seed8101_verification.jsonl` (read by `assemble_video_v2.py`) is `../../video/seed8101_verification.jsonl`;
* `verify_final_v2.out` and `brain_replay_final_v2.out` are `../verification/final-reexecution.jsonl` and
  `../verification/final-brain-replay.txt`.

The code patches they apply are already in the code tree; the recording patch (`patch_recording_v2.py`) is
applied only to a separate recording copy, never to the code tree.
