"""Assemble the recorded levels of elite-v2 final seed 8101 (all 26 levels) into captioned videos.

Outputs (video_v2/): the full game (every level's winning game, captioned), a 4x faster cut, and highlights (the
decisive last seconds of every level). Level names are the game's own English and Simplified Chinese titles (read
from its string tables). Speed captions use the game's 60 ticks per second (state tick_length = 1/60).
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
VIDEO = HERE / "video_v2"
BIN = Path(r"C:\Users\<user>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
           r"\ffmpeg-9.0.1-full_build\bin")
FFMPEG, FFPROBE = str(BIN / "ffmpeg.exe"), str(BIN / "ffprobe.exe")
FONT = "C\\:/Windows/Fonts/msyhbd.ttc"
SEED = 8101
TICKS_PER_SECOND = 60
NAMES = {1: ("Southport", "南港"), 2: ("The Outskirts", "郊原"), 3: ("Pagras", "帕格拉斯"),
         4: ("Twin River Pass", "双子河畔"), 5: ("Silveroak Forest", "银橡森林"), 6: ("The Citadel", "城堡之战"),
         7: ("Coldstep Mines", "寒步矿场"), 8: ("Icewind Pass", "冰风隘口"), 9: ("Stormcloud Temple", "暴云寺院"),
         10: ("The Wastes", "废弃之地"), 11: ("Forsaken Valley", "凄凉山谷"), 12: ("The Dark Tower", "黑暗之塔"),
         13: ("Sarelgaz's Lair", "萨雷格兹的巢穴"), 14: ("Ruins Of Acaroth", "阿卡洛斯遗迹"),
         15: ("Rotten Forest", "腐朽森林"), 16: ("Hushwood", "寂静森林"), 17: ("Bandit's Lair", "强盗的巢穴"),
         18: ("Glacial Heights", "冰川高地"), 19: ("Ha'Kraj Plateau", "哈克拉吉高原"), 20: ("Pit Of Fire", "火焰之坑"),
         21: ("Pandaemonium", "万魔殿"), 22: ("Fungal Forest", "真菌森林"), 23: ("Rotwick", "腐芯镇"),
         24: ("Ancient Necropolis", "古代大墓地"), 25: ("Nightfang Swale", "夜牙沼泽"),
         26: ("Castle Blackburn", "布莱克本城堡")}
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "30"]
FINAL = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v2\runs\native-final-af081b54e5f54478aa76b24bc02bdbf5")


def run(args):
    subprocess.run(args, check=True, cwd=VIDEO)


def duration(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def text_file(name, text):
    (VIDEO / name).write_text(text, encoding="utf-8")
    return name


def card(name, lines, seconds):
    filters = []
    y0 = 384 - 26 * len(lines)
    for i, (text, size) in enumerate(lines):
        tf = text_file(f"{name}_{i}.txt", text)
        filters.append(f"drawtext=fontfile='{FONT}':textfile={tf}:fontcolor=white:fontsize={size}:"
                       f"x=(w-text_w)/2:y={y0 + 52 * i}")
    run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x101418:s=1024x768:d={seconds}:r=30",
         "-vf", ",".join(filters), *ENC, f"{name}.mp4"])
    return f"{name}.mp4"


records = {r["level"]: r for r in map(json.loads, (VIDEO / f"seed{SEED}_verification.jsonl").read_text(
    encoding="utf-8").splitlines()) if r["seed"] == SEED}
assert sorted(records) == list(range(1, 27)) and all(r["identical"] for r in records.values()), "incomplete"
games = [json.loads(line) for line in (FINAL / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
mine = [r["payload"] for r in games if r["kind"] == "campaign_attempt" and r["payload"]["seed"] == SEED]
tries = {}
for p in mine:
    tries[p["level"]] = tries.get(p["level"], 0) + 1
challenges = sum(1 for r in games if r["kind"] == "challenge_attempt" and r["payload"]["seed"] == SEED
                 and r["payload"]["won"])
all_games = sum(1 for r in games if r["kind"] in ("campaign_attempt", "star_replay", "challenge_attempt")
                and r["payload"]["seed"] == SEED)
clips = [card("title", [("Kingdom Rush (2011) · all 26 levels · 王国保卫战 全部 26 关通关", 30),
                        ("Normal difficulty · new save · 12 main levels + 14 elite stages", 24),
                        ("ML agent: searched plans + neural operator networks (+ 8B strategy model)", 22),
                        ("One-shot final evaluation 2026-10-11 · this video: seed 8101, the agent's first clear of all 26 levels", 20),
                        ("Each level re-played and verified identical to the logged final run", 20)], 6)]
speeds = []
for level in range(1, 27):
    r = records[level]
    raw = f"seed{SEED}_L{level:02d}_raw.mp4"
    speed = r["replayed_final_tick"] / TICKS_PER_SECOND / duration(VIDEO / raw)
    speeds.append(speed)
    en, zh = NAMES[level]
    attempt = r["attempt"] + 1
    part = "main campaign" if level <= 12 else "elite stage"
    line1 = text_file(f"cap{level:02d}a.txt", f"Level {level}/26 ({part}) · {en} · {zh}")
    line2 = text_file(f"cap{level:02d}b.txt",
                      f"final seed {SEED} · won on attempt {attempt}" + (f" ({attempt - 1} lost before)" if attempt > 1
                                                                         else "")
                      + f" · {r['lives']} lives left · ≈{speed:.0f}× speed")
    line3 = text_file("capc.txt", "Kingdom Rush ML agent · replay verified identical to the final run (end-state SHA256)")
    vf = (f"drawtext=fontfile='{FONT}':textfile={line1}:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.6:"
          f"boxborderw=8:x=w-text_w-14:y=52,"
          f"drawtext=fontfile='{FONT}':textfile={line2}:fontcolor=0xffe08a:fontsize=18:box=1:boxcolor=black@0.6:"
          f"boxborderw=6:x=w-text_w-14:y=92,"
          f"drawtext=fontfile='{FONT}':textfile={line3}:fontcolor=white@0.85:fontsize=14:box=1:"
          f"boxcolor=black@0.5:boxborderw=5:x=w-text_w-12:y=h-text_h-58")
    out = f"seed{SEED}_L{level:02d}_cap.mp4"
    run([FFMPEG, "-y", "-v", "error", "-i", raw, "-vf", vf, *ENC, out])
    clips.append(out)
    print("captioned", level, f"{speed:.1f}x", flush=True)
won = [p for p in mine if p["won"]]
stars = sum(p["stars"] for p in won)
retried = ", ".join(f"{lv} ({n})" for lv, n in sorted(tries.items()) if n > 1)
clips.append(card("end", [("26 / 26 levels cleared · 全部 26 关通关", 32),
                          (f"{stars} / 78 stars in first wins · {challenges} Heroic/Iron challenges won · "
                           f"{all_games} games in all", 22),
                          (f"levels needing retries (attempts): {retried}", 18),
                          ("Final evaluation: 120 of 130 levels over seeds 8101-8105", 22),
                          ("github.com/Diabolically-Handsome/kingdom-rush-ml-agent", 22)], 7))
(VIDEO / "concat.txt").write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")
run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy",
     f"kingdom-rush-ml-agent_seed{SEED}_full.mp4"])
run([FFMPEG, "-y", "-v", "error", "-i", f"kingdom-rush-ml-agent_seed{SEED}_full.mp4", "-vf", "setpts=PTS/4,fps=30",
     *ENC, f"kingdom-rush-ml-agent_seed{SEED}_4x.mp4"])
parts = ["title.mp4"]
for level in range(1, 27):
    cap = f"seed{SEED}_L{level:02d}_cap.mp4"
    start = max(0.0, duration(VIDEO / cap) - 6.0)
    out = f"hl_L{level:02d}.mp4"
    run([FFMPEG, "-y", "-v", "error", "-ss", f"{start:.2f}", "-i", cap, *ENC, out])
    parts.append(out)
parts.append("end.mp4")
(VIDEO / "concat_hl.txt").write_text("".join(f"file '{c}'\n" for c in parts), encoding="utf-8")
run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "concat_hl.txt", "-vf", "scale=768:576",
     "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p", "-r", "30",
     f"kingdom-rush-ml-agent_seed{SEED}_highlights.mp4"])
print("speed range", f"{min(speeds):.1f}-{max(speeds):.1f}x")
for name in (f"kingdom-rush-ml-agent_seed{SEED}_full.mp4", f"kingdom-rush-ml-agent_seed{SEED}_4x.mp4",
             f"kingdom-rush-ml-agent_seed{SEED}_highlights.mp4"):
    p = VIDEO / name
    print(name, f"{duration(p) / 60:.1f} min", f"{p.stat().st_size / 1e6:.0f} MB")
