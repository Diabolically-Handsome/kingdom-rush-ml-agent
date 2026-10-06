# Rendered replays for the video

The evaluated runs are headless: once the agent takes control, the host replaces the game's draw function with
an empty one. For the video, a separate recording copy of the host (`recording-host.patch`, applied to
`alpharush_rl/assets/host.lua` and `wrapper.lua`, active only with `ALPHARUSH_RECORD=1`) keeps drawing:

* random numbers drawn while rendering come from their own generator, so drawing cannot change the game;
* tutorial pop-ups (which wait for a human "OK") and the dark overlay they put behind them are not drawn;
* steps arrive pipelined, a few ticks per rendered frame; a step queued after the level has ended does nothing.

`record_final.py` re-plays each winning attempt of a final seed with the same operator network, plan, save
profile, seed and protocol as the logged final run, captures only the game window (Windows Graphics Capture via
ffmpeg `gfxcapture`), and compares the replay's end-state hash with the final run's journal. The comparison is
written to `runtime/rl/campaign-v1/video/seed6001_verification.jsonl` and checked by `verify_evidence.py`.
