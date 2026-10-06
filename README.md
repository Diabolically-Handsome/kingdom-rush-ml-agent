# Kingdom Rush ML Agent

**A machine-learning agent that clears the entire 12-level main campaign of the original *Kingdom Rush* (2011),
on Normal difficulty, starting from a brand-new save.** In a one-shot final evaluation on 5 held-out seeds
(2026-10-06) it cleared the full campaign **5 out of 5 times**.

To our knowledge (searched on 2026-10-06: GitHub, arXiv / ACM / Semantic Scholar, YouTube, Bilibili), this is
the **first publicly documented machine-learning agent to clear the full main campaign** of the original game.
Earlier work is credited [below](#prior-work); if you know of an earlier full-campaign clear by an ML agent,
please open an issue and we will correct this claim.

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

## Verify it yourself (no game needed)

```bash
python verify_evidence.py
```

This checks every hash-chained journal, that the one-shot final job was frozen on exactly the published code and
configuration, recomputes the 5/5 result from the attempt records, and confirms the final seeds appear in no
other run. Expected output ends with `ALL CHECKS PASSED`.

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
* Only the 12-level main campaign on Normal; the 14 elite stages and Heroic/Iron challenges are not attempted.

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
[`runtime/rl/campaign-v1/REDACTIONS.json`](runtime/rl/campaign-v1/REDACTIONS.json) with its original SHA256 (the
hash the frozen job used) and its published SHA256. Hash-chained journals were not modified.

## License and notices

Code: [MIT](LICENSE). *Kingdom Rush* is © Ironhide Game Studio; this project is not affiliated with or endorsed
by Ironhide and contains no game files. See [NOTICE.md](NOTICE.md).
