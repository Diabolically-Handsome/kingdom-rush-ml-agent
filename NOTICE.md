# Notices

## Kingdom Rush

*Kingdom Rush* is a game by Ironhide Game Studio. This repository is an independent research project, not affiliated with or endorsed by Ironhide, and contains no game files, assets or game source code. To run anything, use your own legally obtained copy of the game; the engine builder reads it from your local install.

The videos (`media/` and the GitHub releases) are muted screen recordings of the agent's own games, made to document the results; the game artwork visible in them is © Ironhide Game Studio.

The language-model experiments of 2026-10-11/12 publish the prompts the models were shown and their outputs (`runtime/rl/elite-v3b/`, with `geometry/enemy_cards.json`). These quote enemy, tower and hero names and short in-game card descriptors from the game's English text, © Ironhide Game Studio, only to document what the models saw; the game's string table and the probe-read level files are not redistributed (the capture-replay journal records the observed positions of enemies, towers and the hero; see the README's Redactions section).

## Lumi_Nox (MIT)

The engine host of this project builds on the in-game TCP bridge `games/kingdom_rush/bridge.lua` of [Lumi_Nox](https://github.com/MIO-456/Lumi_Nox) (commit 77f973431c49f5ab72d3b2501c0e89286997ec46), which is not redistributed here; clone it to `./Lumi_Nox`. Its license:

```
MIT License

Copyright (c) 2026 Mio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Models

The strategy brain uses Mistral AI's Ministral-3-8B-Instruct, which is not included; download it under its own license.

The language-model commander experiments (2026-10-11/12) also used Mistral AI's Mistral-Small-3.2-24B-Instruct-2506, likewise not included; the LoRA adapters trained on the 8B model are not published (their SHA256s are in the training receipts).
