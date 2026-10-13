# Kingdom Rush ML Agent

**A machine-learning agent that clears the entire 12-level main campaign of the original *Kingdom Rush* (2011),
on Normal difficulty, starting from a brand-new save.** In a one-shot final evaluation on 5 held-out seeds
(2026-10-06) it cleared the full campaign **5 out of 5 times**.

To our knowledge (searched on 2026-10-06: GitHub, arXiv / ACM / Semantic Scholar, YouTube, Bilibili), this is
the **first publicly documented machine-learning agent to clear the full main campaign** of the original game.
Earlier work is credited [below](#prior-work); if you know of an earlier full-campaign clear by an ML agent,
please open an issue and we will correct this claim.

**All 26 levels (2026-10-11):** in a third one-shot final over all 26 levels, on 5 new seeds, the agent cleared
**120 of 130 levels**, and on seed 8101 it cleared **all 26 levels** (the 12-level main campaign and all 14 elite
stages) from a new save on Normal; all five seeds cleared the main campaign. We know of no earlier machine-learning
agent result that clears all 26 levels (same search as above). Video and details:
[below](#all-26-levels-cleared-2026-10-11).

**Elite stages (2026-10-07):** in a second one-shot final over all 26 levels (main campaign, then the 14 elite
stages) the agent cleared 79 of 130 levels; two of five seeds won 21 levels including 9 elite stages, while on two
others the unchanged main-campaign agent lost level 11, so the main campaign was cleared on 3 of these 5 new seeds
([details](#elite-stages-2026-10-07)).

**Project status (2026-10-12): archived.** After the full clear we tested whether a local language model could make
the agent's in-battle tactical decisions. In a pre-registered fair re-test, with richer information and reasoning, the
two local models we tried (8B and 24B, 4-bit) still showed no measurable skill at this decision layer and stayed below a
fixed ranking of the plays (measured one decision at a time, on training seeds; [details](#an-in-battle-language-model-commander-a-pre-registered-null-result-2026-10-12)).
The project is archived here.

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
full campaign (25 min, about 4–10× game speed; its captions say 8–20× because they assumed 30 game ticks per second, the game runs 60) and a 4× faster cut: [GitHub release](../../releases).

## All 26 levels cleared (2026-10-11)

The third phase (elite-v2) targeted the weak points of the 2026-10-07 final: main level 11, and elite stages 20–22,
which had never been won. Everything ran under the same phase gate, with new seed pools: train 1001–1010 and
1101–1140, evaluation 7101–7120, one-shot final 8101–8105 (all earlier evaluation and final seeds retired).

**One-shot final evaluation on 5 new seeds, all 26 levels from a new save, Normal:**

| Seed | Levels cleared (of 26) | Not cleared | Games | Stars: campaign + challenges |
|---|---|---|---|---|
| **8101** | **26** | — | 63 | 61 + 11 |
| 8102 | 25 | 22 | 69 | 62 + 11 |
| 8103 | 22 | 22, 24 (so 25, 26 stayed locked) | 84 | 50 + 10 |
| 8104 | 25 | 22 | 73 | 58 + 11 |
| 8105 | 22 | 17, 19, 22, 26 | 110 | 52 + 10 |

120 of 130 levels (the 2026-10-07 final: 79). Every seed cleared the main campaign (level 11 in 1–2 attempts,
level 12 in 1–3), and elite stages 13–16, 18, 20, 21 and 23 were won on every seed. Level 22 (Fungal Forest) is
still the hardest: only seed 8101 won it (on its 2nd attempt); the level's 10-plan portfolio wins on about 14%
(7/50) of the training seeds.

**Video:** seed 8101's winning game of every level, re-played by the same networks with rendering on, each verified
to reach exactly the final run's end state (SHA256), 26/26. Highlights (2.8 min):
[`media/kingdom-rush-ml-agent_seed8101_highlights.mp4`](media/kingdom-rush-ml-agent_seed8101_highlights.mp4);
the full game (77 min, about 5.6–10× game speed) and a 4× cut (19 min): [GitHub release v2.0](../../releases/tag/v2.0). Captions name each level with the game's own English
and Chinese titles.

**What changed** (details in [`docs/DEV-LOG.zh-CN.md`](docs/DEV-LOG.zh-CN.md)):

* **Project rule: the agent's action set should match a human Steam player's, no more and no less (remaining gaps
  below and under Limitations).** A read-only audit listed 32 things a
  human Steam player can do and compared each with the agent
  ([`docs/elite/human-capability-audit-2026-10-10.md`](docs/elite/human-capability-audit-2026-10-10.md), Chinese).
  From it:
  * all **13 heroes** (the hero room unlocks the 7 others by campaign progress; verified natively; no purchases). Shop
    items stay unused: on the Steam desktop build the shop and item bag are unreachable for players;
  * **a human campaign route:** when every plan of a main level has lost, the agent first replays earlier levels
    for stars and plays the open Heroic/Iron challenges, then retries with the extra upgrades. In a rehearsal this
    rescued a campaign at level 12 (27 → 42 stars, then a win);
  * **human-timed trap clicks:** a tower held by Vez'nan (level 12) is clicked only inside the game's own click
    window, one tap every 9 ticks (0.15 s at the game's 60 ticks per second), as a quick human does. On train seeds
    1111–1120 (never used for level-12 search or training) this raised the plan executor's level-12 wins from 54/80
    to 80/80 with the same 8 plans (73/80 in an earlier run that also had the early-wave menu on);
  * **early wave calls with enemies on the field** were implemented and measured but *not* used: they lowered the
    existing plans' level-11 wins (38/70 → 30/70; 2/70 for plans that call every flag). Free spell targets, hero
    movement and rally points on main levels are left for later.
* **Search:** five new search runs on training seeds. Elite stages 21 and 22 had collapsed onto one losing opening
  each (e.g. 1,041 of 1,276 level-22 plans opened with the same mage tower); restarting them from random populations,
  then warm-starting from the best, produced the first wins on both. Best validated plans: level 20 18/20,
  level 21 16/20, level 22 2/20 (validation on 20 other training seeds).
* **Operators:** the main network was retrained with the new level-11 plans, all 8 (unchanged) level-12 plans and
  human-timed clicks (train seeds 1111–1120, not used to train any operator: level 11 35/70, level 12 79/80,
  levels 1–10 99/100, the one loss being a level-6 game the plan executor also lost); the elite network with the
  new heroes and plans plus DAgger (train seeds 1121–1127, not used to train any operator: 193 elite and 127
  challenge wins, vs 198 and 125 for the plan executor it imitates). Levels 11 and 20–22 had been searched or validated
  on these seeds, see Verification.

**The 8B strategy model's contribution, measured.** The final's configuration was rehearsed twice on evaluation
seeds 7101–7105, once with the 8B model choosing plans and star upgrades and once with fixed rules instead
(everything else identical): **123 vs 122 of 130 levels**. The single difference (level 26 of seed 7104, same plan
for both) was a close game: the two campaigns had already diverged in stars (63 for the 8B side, 66 for the rule
side; maxing every upgrade costs 65), so the rule side maxed every upgrade and lost, while the 8B side could not
afford the last reinforcement level and won with 1 life left. So in this agent the 8B model's measured
contribution is about zero; the capability comes from the searched plans, the operator networks (which make every
in-game decision) and the campaign route. Giving the language model decisions that need judgement is future work.

**Verification:**
* all 399 final games re-executed by the same networks: identical traces and end states; all 745 8B decisions
  asked again: identical prompts and choices (maximum probability difference 0.0);
* a read-only adversarial audit after the final (15 agents, 38 checks, all passed; nine minor caveats, none
  affecting the result: [`final-audit.json`](runtime/rl/elite-v2/final-artifacts/verification/final-audit.json)).
  Two worth stating here: the train seeds used to choose between operator networks (1111–1127) had all been used
  by plan searches (as search or validation seeds) for levels 11 and 20–22, so the win rates quoted at selection time for those levels are
  optimistic (the final seeds were never used anywhere before); and no game is entered through the game's menus:
  before each game the harness writes a save slot from the run's own earned progress (stars, won challenges,
  star upgrades and the chosen hero), then starts the game with its launch flags (`-level/-mode/-diff`) on Normal.
  Every level, mode, hero and upgrade used had been earned (audit: 0 violations in 399 games), so this is the
  equivalent of picking the level on the map, not a way around unlocks, though strictly it bypasses the menus;
* `verify_evidence.py` checks V1–V8 (journals, one-shot ledger entry, pinned files, the recomputed 120/130 and
  8101's 26/26, seed isolation across all phases, the re-execution / replay / video records, and the ablation
  counts). Evidence: [`runtime/rl/elite-v2/`](runtime/rl/elite-v2/).

**Correction to the 2026-10-06 video:** its captions give speeds of "≈8–20×", computed with 30 game ticks per
second; the game runs 60 ticks per second, so the replay runs at about 4–10× game speed.

## An in-battle language-model commander: a pre-registered null result (2026-10-12)

After the full clear we asked whether a local language model could make the in-battle judgement calls a human player
makes. (In the final above, the 8B model only chooses plans and star upgrades, and its measured contribution is about
zero.) On the evidence below the answer is **no, not with the models and the decision layer we built**. This is
development evidence on training seeds only: no evaluation or final seed was used, and nothing here changes the agent
above.

**What we built (action scope v4).** Two human abilities the agent lacked: moving the hero (to its start point, an exit, or upstream
on a path), and aiming Rain of Fire and reinforcements at more than the leading enemies (up to six candidate points per
spell: threat, lead and cluster points for Rain of Fire; block and cluster points for reinforcements). Both are offered only where the game's own GUI rules allow them (hero rally points through the
game's path-finding, spell targets through its cast-validity check), and golden transcripts prove the earlier action
scopes byte-for-byte unchanged. On top of the plan executor, a *commander* is consulted when a next-wave flag appears
(plus at most one alarm per wave) and picks one of at most six *plays*: P0 *steady* (the executor's own behaviour, which
the plans were searched with), counter one of the two biggest threats (the hero intercepts them upstream, Rain of Fire is
saved for them, reinforcements are called at them when they are on the field), guard an exit, or focus both spells on the top threat. A deterministic translator executes the play. The game is paused while
the model thinks (a declared deviation from real-time play).

**How we measured.** 56 base games were played steady (7 hard elite stages x 2 plans x train seeds 1201-1204). At each
of their 985 consults with a choice, the game was replayed taking each other play once, everything else steady: 3,026
fork games, each identical to its base game up to the branch point. A play's value at a consult is the outcome of its
branch: 1 for a win + 0.5 x the share of waves reached + 0.25 x lives left / 20 (at most 1.75). The main metric, **M1**, is the outcome of the
model's preferred deviation minus the mean of all deviations at that consult (0 = a random deviation), with 95%
intervals from a cluster bootstrap over base games.

**Results.**
* Taking over every wave hurts. Over whole games on the same 56 tasks: steady 24/56 wins, a rule commander 14/56,
  random plays 5/56.
* Zero-shot, both the 8B model (Ministral 3 8B) and a 24B model (Mistral Small 3.2 24B, both 4-bit) behave like
  "always counter the biggest threat": P1 in 955 and 937 of 985 consults. Both choose slightly worse than random among
  the deviations. A LoRA fine-tune of the 8B on fork labels collapsed to "always steady".
* The owner then asked whether the models simply got too little information. In a pre-registered fair re-test on the
  same consults and the same measured outcomes, each play was described exactly as executed, a neutral note said that P0 is the default the
  plans were tuned with, and the basic rules were stated. Then we added what a player sees on screen: tower coverage of every quarter of every path, the map, each
  enemy's in-game card, the hero's class and level, and the lives lost per path so far. Last, the models reasoned
  before answering. M1 (95% CI):

| Prompt | 8B | 24B |
|---|---|---|
| original prompt | -0.032 [-0.052, -0.010] | -0.030 [-0.054, -0.008] |
| honest play texts + P0 note + basic rules | +0.027 [+0.002, +0.052] | -0.011 [-0.036, +0.012] |
| + on-screen information | +0.009 [-0.016, +0.033] | -0.002 [-0.025, +0.021] |
| + reasoning first | +0.001 [-0.019, +0.022] | +0.009 [-0.011, +0.029] |
| *reference: a fixed ranking of the play types, learned on the other half of the seeds* | +0.063 [+0.035, +0.092] | |

All three pre-registered tiers failed:
* the on-screen information gave no detectable improvement for either model;
* the primary arm (24B, rich information, reasoning) reached neither zero with confidence nor half of the fixed ranking;
* that arm was significantly worse than the fixed ranking (-0.054 [-0.088, -0.019]).

The models still deviated from the plan's default on 83-99% of consults. They almost never chose to guard an exit,
which is the best deviation on level 19, and in their reasoning they often called the default "passive". A blind
calibration by a larger general-purpose model (Claude Fable 5.1, shown only 20 of the rich prompts) kept the default in
17 of 20.

**Our reading.**
* The plans were searched together with the default play, and the game is chaotic at this scale. Delaying every
  command by one tick changed the winner in 8 of 56 games, and a single branch outcome varies with an SD of about 0.46,
  while the play effects are about 0.1.
* The structure that replicates across seeds is per level: guard the exit on level 19, focus the spells on level 15. A
  lookup table captures it.
* The extra value of reading the situation is small; we estimate at most about +0.02-0.04 beyond a per-level lookup.
  The extra information changed about a third (24B) to half (8B) of the models' preferred deviations, but not for
  the better; the 24B's reasoning rarely cited coverage (14%), exits (8%) or lives lost (3%). One pre-registered
  secondary test did find a signal: with the rich prompt, the 8B leaned towards deviating more at the moments when
  deviating paid off more (timing z 3.30, p 0.001, the only one of 16 timing tests that survives a Bonferroni
  correction). It still deviated 98% of the time, and its actual choices were not better than steady (choice minus
  steady: -0.096 [-0.223, +0.032]).
* The in-battle commander line is therefore closed. A more promising use of a language model here may be proposing
  plan changes for the search to test.

**Caveats.**
* The measure is one-step (one changed decision per fork), not whole games with the model in control, and the models
  are 4-bit.
* Batched generation is not bit-reproducible: regenerating 20 consults one at a time changed most reasoning texts.
* In the reasoning arms, 133 of the 24B's 985 final answers (79 named a letter instead of a play id, 54 were cut off
  at 250 tokens) and 230 of the 8B's (mostly bold-formatted answers that agree with its scores) could not be parsed;
  their stated choice fell back to the scored distribution (the deviation rates use the stated choices). M1 always uses
  the scored distribution, so it is unaffected, but the primary arm's +0.009 sits almost entirely in those 133 consults
  (+0.054 there, +0.002 on the 852 parsed).
* The pre-registered same-information rule baseline (rule_rich, M1 +0.049) turned out to encode the play ranking found
  on the same 56 games and was not cross-validated. Departing from the pre-registration, its secondary contrasts (both
  models lower) are not used here.

Evidence:
* [`runtime/rl/elite-v3/`](runtime/rl/elite-v3/): runs, fork table, analyses, LoRA receipts;
* [`runtime/rl/elite-v3b/`](runtime/rl/elite-v3b/): the exact prompts, the model outputs, the analysis;
* the plan and the pre-registration in [`docs/elite/`](docs/elite/).

`verify_evidence.py` checks C1-C8:
* hash-chained journals;
* the four evaluation jobs and the sealed configuration;
* training seeds only (and none of the earlier phases' final or rehearsal seeds);
* the recomputed whole-game wins;
* the fork table recomputed from its journal, each fork game identical to its base game up to the branch point;
* the replay;
* the zero-shot 8B/24B choice files (every consult; P1 955 and 937 times);
* the frozen pre-registered files, the parity gate, the M1 of every language-model arm and of the fixed ranking, and the
  tiers (T2, T3 and the 8B's T1 fail on recomputed values; the 24B's T1 on the published bootstrap interval).

**Project status: archived on 2026-10-12.**

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
other run; it does the same for the 2026-10-07 elite-stage final (and checks its re-execution and strategy-replay
records), and runs checks V1–V8 on the 2026-10-11 final (the recomputed 120/130 and seed 8101's 26/26, seed
isolation across all phases, the re-execution, replay and video records, and the 123 vs 122 ablation counts).
It then runs checks C1–C8 on the 2026-10-12 language-model experiments. These decompress the fork pilot's journal,
recompute the fork table and the M1 of every language-model arm and of the fixed ranking, confirm the pre-registered
files, and recompute the tier verdicts (the 24B's T1 interval is taken from the published bootstrap).
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

* **Plans** come from an evolutionary search (steady-state genetic algorithm; 158,928 games over 40 search runs:
  22 runs / 20,794 games for the main campaign, 13 / 69,564 in the 2026-10-07 elite phase, 5 / 68,570 in the
  2026-10-11 phase) on *training* seeds only, scored over several seeds for the hard levels and validated on others. Each level
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
6. **More information did not make the language models better commanders.** The new prompt format (honest play
   texts, a neutral note that P0 is the tuned default, basic rules) moved the 8B from mostly P1 to mostly P4, still
   slightly below always choosing P4. Adding what a player sees on screen, then reasoning, gave no detectable gain, and a fixed ranking of the play
   types beat every model arm ([details](#an-in-battle-language-model-commander-a-pre-registered-null-result-2026-10-12)).
7. **Evaluate one-step decisions carefully.** Our first held-out evaluation was confounded: the held-out seeds'
   base games happened to be all losses, and after a loss almost any change looks good. An adversarial review caught
   it. The re-test then used a pre-registered metric that does not depend on whether the base game was won, with
   game-clustered intervals.

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
* Elite stages (2026-10-11 final): level 22 was won on 1 of 5 seeds; levels 17, 19, 24 and 26 were each lost on
  one seed, and on seed 8103 levels 25 and 26 stayed locked (never played) after level 24 was lost. Heroic/Iron challenges are played only as a source of upgrade stars (not all of them are won).
* Normal difficulty only; the heroes the game's hero room offers by campaign progress (all 13 from level 12 on),
  no purchases or shop items.
* Spells and reinforcements target only the leading enemies; the hero is not moved during a level; early wave
  calls with enemies on the field are implemented but unused (see above). (The 2026-10-12 commander experiments added
  hero moves and more spell targets, up to six candidate points per spell, under the GUI rules, without a measurable
  gain.)
* The language-model experiments are development evidence on training seeds, measured one decision at a time.

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

Text quoted from the project owner's chat messages (authorizations, requests and other remarks), local user-name
paths and scratch directories, and the name of an unrelated private project were
removed from some text files before publishing. Every such file is listed in
[`runtime/rl/campaign-v1/REDACTIONS.json`](runtime/rl/campaign-v1/REDACTIONS.json),
[`runtime/rl/elite-v1/REDACTIONS.json`](runtime/rl/elite-v1/REDACTIONS.json),
[`runtime/rl/elite-v2/REDACTIONS.json`](runtime/rl/elite-v2/REDACTIONS.json) and
[`runtime/rl/elite-v3/REDACTIONS.json`](runtime/rl/elite-v3/REDACTIONS.json) with its published SHA256 and, for
files pinned by a frozen job, its original SHA256 (the hash that job used; the elite-v3 list also gives the originals
of the re-test's files and of the LoRA adapter configs, which the pre-registration and the receipts cite). Hash-chained journals were not modified. The code tree has moved on
since the earlier results; the files their jobs pinned are kept in
[`runtime/rl/campaign-v1/pinned-snapshot/`](runtime/rl/campaign-v1/pinned-snapshot/),
[`runtime/rl/elite-v1/pinned-snapshot/`](runtime/rl/elite-v1/pinned-snapshot/) and
[`runtime/rl/elite-v2/pinned-snapshot/`](runtime/rl/elite-v2/pinned-snapshot/).

Not published:
* The language-model experiments' probe-read level files (`level_<L>.json`: path nodes, exits, ice, build spots) and
  the game's full English string table. Published instead, with the probe scripts: small probe outputs, namely tower and
  hero ranges with the raw target-flag masks and the derived flying reach (`towers.json`), enemy string-table keys
  (`enemy_keys.json`), and enemy names and short card bullets quoted from the game (`enemy_cards.json`, see
  [NOTICE.md](NOTICE.md)). The capture-replay journal
  (`runtime/rl/elite-v3/runs/native-eval-f18114d98c9044228f1fb1d5ae5160a2/episodes.jsonl`) records the observed state
  at every consult (enemy positions with path progress, built towers' positions, the hero's start point), from which
  path shapes and build spots can be approximately recovered.
* The LoRA adapter weights (4 x 178 MB); their SHA256s are in the training receipts.

The fork pilot's 263 MB journal is published as a deterministic gzip; its original SHA256 is in
[`runtime/rl/elite-v3/COMPRESSED.json`](runtime/rl/elite-v3/COMPRESSED.json).

## License and notices

Code: [MIT](LICENSE). *Kingdom Rush* is © Ironhide Game Studio; this project is not affiliated with or endorsed
by Ironhide and contains no game files. See [NOTICE.md](NOTICE.md).
