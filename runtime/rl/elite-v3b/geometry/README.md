# Static game facts used by the rich briefings

Read once from a scratch development copy of the game with read-only probes (muted, headless, port >= 9950; neither the game files nor the published code were changed). Published here:

* `towers.json`: for each tower and hero template, its range and each attack's raw target-flag masks (`flags`, `bans`, as the engine reports them; a hero attack also has its range), with the derived `bans_flying` (which attacks can target flying enemies);
* `enemy_keys.json`: each enemy template's in-game string-table key and whether the template is flagged flying;
* `enemy_cards.json`: each enemy's official English name and in-game card bullets, quoted from the game's string table (see `NOTICE.md`).

Not redistributed: the probe-read level files (`level_<L>.json`: path nodes, connections, exits, ice, build spots, the hero start point) and the game's full English string table. Note that the capture-replay journal (`../../elite-v3/runs/native-eval-f18114d98c9044228f1fb1d5ae5160a2/episodes.jsonl`) records the observed state at every consult (enemy positions with path progress, built towers' positions, the hero's start point), from which path shapes and build spots can be approximately recovered.

## Regenerating from your own copy of the game

The scripts in `provenance/` contain no game data. They expect a scratch development copy of this repository in a folder `v4_merged/` next to them and `portfolios_v2_r3.json` next to them (it is `../../elite-v2/final-artifacts/portfolios.json`); their output folder is the workspace path written in the scripts.

1. Patch the copy's `alpharush_rl/assets/host.lua` with `patch_geo_host.py` (a read-only `paths` RPC), `patch_geo_host2.py` (a read-only `tower_info` RPC) and then `patch_geo_host3.py` (names as one comma-separated string, hero attacks, and a read-only `template_keys` RPC). Applied in this order to the published `host.lua`, they reproduce the probe copy byte for byte.
2. `geo_probe.py <copy> [port]` writes `level_<L>.json`; `tower_probe.py <copy> [port]` writes `towers.json` and `enemy_keys.json`.
3. `extract_enemy_cards.py <prepared engine exe> enemy_keys.json <out>` writes `enemy_cards.json` from the game's English string table; on the probe machine it reproduces the published file exactly.

`patch_geo_host3.py` and `extract_enemy_cards.py` were written up after the run from the session's inline commands (the run first dumped the string table to an unpublished `strings_en.json`); the other probe scripts are the files that were run.

The prompts built from these files are published in `../requests/` exactly as the models saw them, and `tools/verify_evidence.py` (check C8) recomputes the M1 results from the published scores without the level files. Re-running `../scripts/v3b_prompts.py` or `../scripts/v3b_analysis.py` (which also computes the rule_rich baseline and the secondary metrics) needs the regenerated `level_<L>.json` files.
