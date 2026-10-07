# Kingdom Rush ML Agent

**A machine-learning agent that clears the entire 12-level main campaign of the original *Kingdom Rush* (2011),
on Normal difficulty, starting from a brand-new save.** In a one-shot final evaluation on 5 held-out seeds
(2026-10-06) it cleared the full campaign **5 out of 5 times**.

To our knowledge (searched on 2026-10-06: GitHub, arXiv / ACM / Semantic Scholar, YouTube, Bilibili), this is
the **first publicly documented machine-learning agent to clear the full main campaign** of the original game.
Earlier work is credited [below](#prior-work); if you know of an earlier full-campaign clear by an ML agent,
please open an issue and we will correct this claim.

**Elite stages (2026-10-07):** in a second one-shot final over all 26 levels (main campaign, then the 14 elite
stages) the agent cleared 79 of 130 levels; two of five seeds won 21 levels including 9 elite stages, while on two
others the unchanged main-campaign agent lost level 11, so the main campaign was cleared on 3 of these 5 new seeds
([details](#elite-stages-2026-10-07)).

[中文说明 / Chinese README](README.zh-CN.md)

## Results

| Run | Seeds | Full campaign cleared | Notes |
|---|---|---|---|
| Rehearsal 1 | evaluation 5001–5010 | 5 / 10 | 148 games; failures at levels 11–12 |
| Rehearsal 2 | evaluation 5001–5010 | 9 / 10 | 145 games; after more robust level-12 plans |
| **Final (one-shot)** | **final 6001–6005** | **5 / 5** | **75 games, 13 min; 31–34 of 36 stars** |

Per final seed (levels 1–12 in order, Normal, new save):

| Seed | Games played | Stars | Retries |
|---|---|---|---|
| 6001 | 13 | 32 / 36 | level 12 won on the 2nd attempt |
| 6002 | 15 | 31 / 36 | level 11 won on the 4th attempt |
| 6003 | 13 | 34 / 36 | level 11 won on the 2nd attempt |
| 6004 | 19 | 34 / 36 | level 11 on the 7th, level 12 on the 2nd attempt |
| 6005 | 15 | 32 / 36 | level 11 on the 2nd, level 12 on the 3rd attempt |

Levels 1–10 were won on the first attempt in every final campaign. Retrying a level is allowed, as for a human
player: up to 8 attempts per level, each retry with a plan not tried yet (levels 11–12 have 8 different plans,
the others 3–5).

**Video:** a rendered replay of the full campaign of final seed 6001: each level re-played by the same operator
network and verified to reach exactly the same end state (SHA256) as the logged final run. Highlights (1.8 min):
[`media/kingdom-rush-ml-agent_seed6001_highlights.mp4`](media/kingdom-rush-ml-agent_seed6001_highlights.mp4);
full campaign (25 min, about 8–19× game speed) and a 4× faster cut: [GitHub release](../../releases).

## Elite stages (2026-10-07)

The second phase takes on the 14 elite stages (levels 13–26), still as a campaign on Normal from a brand-new save.
As a human player does, the agent first replays main levels won with fewer than 3 stars, then plays the Heroic and
Iron challenges of the 3-star main levels (each win adds one star to the upgrade budget: the main campaign gives at
most 36 stars, the game recommends 50+ for the elite stages), then the elite stages as they unlock.

**One-shot final evaluation on 5 held-out seeds, all 26 levels from a new save:**

| Seed | Levels cleared (of 26) | Elite stages won | Stars: campaign + challenges |
|---|---|---|---|
| 8001 | 10 | — (main level 11 lost on all 10 attempts, i.e. all 8 of its plans; a lost main level ends the campaign) | 28 + 0 |
| 8002 | **21** | 14, 15, 16, 17, 18, 19, 23, 24, 25 | 51 + 10 |
| 8003 | 17 | 14, 16, 18, 19, 23 | 46 + 10 |
| 8004 | **21** | 14, 15, 16, 17, 18, 19, 23, 24, 25 | 54 + 10 |
| 8005 | 10 | — (as 8001) | 27 + 0 |

79 of 130 levels; no seed cleared all 26. Nine different elite stages were won in the final; 13, 20, 21, 22 and 26
were not. Two rehearsals on evaluation seeds (system checks only, never used to select plans) averaged 18.0 and
19.4 levels. We know of no earlier machine-learning agent results on the elite stages (same search as above).

* **Reproducible:** all 336 final games re-executed by the same networks reached identical traces and end states,
  and all 650 strategy decisions asked again of the 8B model were identical; `verify_evidence.py` checks that these
  recorded results cover every final game and decision (re-running the comparison itself needs the game and the model).
* **What was added:** action scope v3 (barracks rally points as the GUI allows them; enemy flags such as dormant,
  untargetable, boss, flying, unblockable; per-lane path progress), an elite operator network trained by behaviour
  cloning + DAgger (an earlier version, after 1 DAgger round, won as many evaluation-seed games as the plan executor
  it imitates, 159 vs 159 of 240, with the same outcome in 208 of the 240 paired games; the network used in the final
  adds 3 DAgger rounds for the newest plans, 1,830 games in all, and was checked in a second rehearsal), 13 search
  runs on the training seeds with retry portfolios chosen by seed coverage, and a *soft tower cap* plan gene
  (the fallback stops building at N towers but resumes once 1,000 gold is banked) that turned levels 13 and 15 from
  0 wins into winnable levels.
* **Pre-final audit:** 65 agents reviewed the first rehearsal and the draft final configuration; confirmed findings were fixed
  before the final (retries for the strategy-model server, per-seed exception containment, a health check before the
  one-shot job opens, de-duplicated plan portfolios).
* **Weak points:** main level 11 (its 8 plans win 5–8 of 10 training seeds each; it often needs retries, and on two
  final seeds it lost all 10 attempts: all 8 plans, then the first plan again, which a deterministic game loses again)
  and elite stages 20–22, which the agent loses early to mechanics its action set does not
  yet handle. A drafted plan for them: [`docs/elite/l20-22-practice-plan-2026-10-07.md`](docs/elite/l20-22-practice-plan-2026-10-07.md)
  (Chinese). Full log: [`docs/DEV-LOG.zh-CN.md`](docs/DEV-LOG.zh-CN.md); evidence: [`runtime/rl/elite-v1/`](runtime/rl/elite-v1/).

Seed pools of this phase: train 1001–1010, evaluation 7001–7020, final 8001–8005 (6001–6005 retired).

## Verify it yourself (no game needed)

```bash
python verify_evidence.py
```

This checks every hash-chained journal, that the one-shot final job was frozen on exactly the published code and
configuration, recomputes the 5/5 result from the attempt records, and confirms the final seeds appear in no
other run; it does the same for the elite-stage final (and checks its re-execution and strategy-replay records).
Expected output ends with `ALL CHECKS PASSED`.

## What the agent is

```
            ┌────────────────────────── strategy brain ──────────────────────────┐
 campaign → │ Ministral-3-8B-Instruct (4-bit, zero-shot): before each level, pick  │
 progress   │ a battle plan (hero, build order, spell timing, wave calls, ...)    │
            │ and how to allocate the earned stars (the upgrade screen can reset) │
            └─────────────────────────────────┬──────────────────────────────────┘
                                              │ plan = sequence of instructions
            ┌─────────────────────────────────▼──────────────────────────────────┐
 game   →   │ operator network (MLP 128×64, behaviour cloning + DAgger): scores   │ → one legal
 state      │ every legal action at every decision point and executes the plan:  │   game action
            │ build / upgrade / skills / spells (aimed) / sunray aim / call wave  │
            │ / click (finish a downed boss, break ice off a tower) / wait        │
            └────────────────────────────────────────────────────────────────────┘
```

* **Plans** come from an evolutionary search (steady-state genetic algorithm, 20,794 games over 22 search runs) on
  the 10 *training* seeds only, scored over several seeds for the hard levels and validated on others. Each level
  gets a small retry portfolio of plans chosen to cover different seeds.
* **The 8B model is not fine-tuned** in the final agent. It scores the legal options of each strategic choice;
  each menu is scored in every cyclic order and averaged, because the raw model strongly prefers option "B".
* **The game is the real one**: an isolated copy of the Steam version's LÖVE/LuaJIT engine with an RPC host.
  Actions go through the game's own paths (as the GUI buttons do); nothing edits gold, health or waves. The
  engine runs headless with a deterministic random-number mode, so every game can be replayed exactly.
* **Fair-play rules**: new save; stars earned by the game's own rule (lives left) and spent through the upgrade
  screen's rules; heroes as unlocked; Normal difficulty; the game is not modified.

## Evaluation protocol

| Seed pool | Seeds | Used for |
|---|---|---|
| train | 1001–1010 | plan search, validation, operator training data |
| evaluation | 5001–5020 | evaluation and rehearsals only; never used to train or to select plans |
| final_campaign_run | 6001–6005 | one frozen, one-shot final evaluation |

Every job ran under a phase gate: code and configuration frozen (SHA256 pins), budget caps, a hash-chained
ledger and hash-chained per-run journals. All of those are published in [`runtime/rl/campaign-v1/`](runtime/rl/campaign-v1/).

## Things we learned

1. **Level 9 needs a click.** The boss J.T., when beaten, collapses and only dies when the player *clicks* him;
   his ice keeps a tower frozen until it is clicked three times. Without a click action level 9 is unwinnable
   (4,613 search games, 0 wins; in the closest ones the boss sat at 4 HP until the time cap); with it, 72/72
   validation wins.
2. **Stars can be re-allocated.** The upgrade screen has a free reset, so each plan carries the star allocation
   it was tested with (e.g. Rain of Fire first for J.T.).
3. **Single-seed search overfits.** Plans that win one seed often fail on others; scoring the hard levels over
   3–5 seeds and picking retry plans by seed coverage took the rehearsal from 5/10 to 9/10.
4. **LLM option-letter bias.** The 8B model picked option "B" regardless of its content; averaging over all
   option orders fixed its choices.
5. **A journal bug** (integer dictionary keys sort differently from their JSON string form once there are ten)
   stopped the first rehearsal; the published verifier checks that entry under the pre-fix rule.

The full development log (Chinese) is in [`docs/DEV-LOG.zh-CN.md`](docs/DEV-LOG.zh-CN.md).

## Prior work

* **LevelupAI, 2018–2019** — *Automatic generation of tower defense levels using PCG* (FDG 2019,
  [doi:10.1145/3337722.3337723](https://doi.org/10.1145/3337722.3337723)): flat Monte-Carlo / MCTS agents play
  single levels of a tower-defense **simulator modeled on *Kingdom Rush: Frontiers*** to assess the difficulty of
  generated levels. Not the original game, no campaign.
* **Lumi_Nox, 2026** ([MIO-456/Lumi_Nox](https://github.com/MIO-456/Lumi_Nox), MIT): AI VTubers that play the
  original Kingdom Rush on stream with a **rule-based** tower-defense bot plus a per-wave **LLM strategist**,
  through a LuaJIT bridge. We found no public record of it clearing the full main campaign. **This project's engine
  host is built on Lumi_Nox's `bridge.lua`** — thank you.
* Related tower-defense AI: EA SEED, *RL for High-Level Strategic Control in Tower Defense Games* (CoG 2024,
  Plants vs. Zombies); *TowerMind* (AAAI 2026, a custom tower-defense benchmark for LLM agents).

## Limitations

* Random seeds are fixed per game (deterministic engine); a level lost with one plan is retried with another plan.
* Plans are prepared per level in advance; the agent does not change its overall strategy mid-level.
* It is not an end-to-end reinforcement-learning policy: plans come from evolutionary search, the operator is
  trained by imitation, and the 8B model is used zero-shot.
* Observations come from the game's internal state (through the RPC host), not from pixels.
* Elite stages: 13, 20, 21, 22 and 26 were not won in the final; main level 11 is the weakest link of the
  campaign. Heroic/Iron challenges are played only as a source of upgrade stars (not all of them are won).
* Normal difficulty only; the six heroes available without purchases.

## Reproducing

You need your own copy of *Kingdom Rush* (Steam, Windows), Python 3.12, and Lumi_Nox's bridge:

```bash
git clone https://github.com/MIO-456/Lumi_Nox Lumi_Nox
git -C Lumi_Nox checkout 77f973431c49f5ab72d3b2501c0e89286997ec46
python -m alpharush_rl.engine --prepare        # builds an isolated engine copy from your game install
python tools/campaign-survey.py --check-only --job native-campaign
```

No game files are included in this repository. The strategy brain needs the 8B model served locally
(`tools/model-worker.py --model 8b --serve`, GPU with ~10 GB); see [`docs/`](docs/).

## Redactions

The authorization messages quoted from the project owner's chat, local user-name paths and the name of an
unrelated private project were removed from some text files before publishing. Every such file is listed in
[`runtime/rl/campaign-v1/REDACTIONS.json`](runtime/rl/campaign-v1/REDACTIONS.json) and
[`runtime/rl/elite-v1/REDACTIONS.json`](runtime/rl/elite-v1/REDACTIONS.json) with its published SHA256 and, for
files pinned by a frozen job, its original SHA256 (the hash that job used). Hash-chained journals were not modified. The code tree has moved on
since the main-campaign result; the files that result's job pinned are kept in
[`runtime/rl/campaign-v1/pinned-snapshot/`](runtime/rl/campaign-v1/pinned-snapshot/).

## License and notices

Code: [MIT](LICENSE). *Kingdom Rush* is © Ironhide Game Studio; this project is not affiliated with or endorsed
by Ironhide and contains no game files. See [NOTICE.md](NOTICE.md).
