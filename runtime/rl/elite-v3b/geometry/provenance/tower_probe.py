"""elite-v3b: one reset of level 15 in the scratch dev copy, then the DEV-ONLY tower_info RPC for every standard
tower template; writes runtime/rl/elite-v3b/geometry/towers.json. Muted headless, port >= 9950, no actions."""
import json
import os
import sys
import uuid
from pathlib import Path

DEV = Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9950
SCRATCH = Path(__file__).parent
OUT = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v3b\geometry")
os.environ["ALPHARUSH_HOST_FEATURES"] = "timed_clicks"
sys.dont_write_bytecode = True
sys.path.insert(0, str(DEV))
from alpharush_rl import engine  # noqa: E402
from alpharush_rl.campaign import task_profile  # noqa: E402
from alpharush_rl.engine import DETERMINISTIC_MODE  # noqa: E402
from alpharush_rl.env import NativeEnv  # noqa: E402
from alpharush_rl.search import check_genome  # noqa: E402

assert Path(engine.__file__).resolve().is_relative_to(DEV), engine.__file__
NAMES = ["tower_archer_1", "tower_archer_2", "tower_archer_3", "tower_ranger", "tower_musketeer",
         "tower_mage_1", "tower_mage_2", "tower_mage_3", "tower_arcane_wizard", "tower_sorcerer",
         "tower_engineer_1", "tower_engineer_2", "tower_engineer_3", "tower_bfg", "tower_tesla",
         "tower_barrack_1", "tower_barrack_2", "tower_barrack_3", "tower_paladin", "tower_barbarian",
         "hero_ignus", "hero_bolin", "hero_magnus", "hero_ingvar", "hero_hacksaw", "hero_elora"]
ENEMIES = ("eb_blackburn eb_greenmuck eb_kingpin eb_moloch eb_myconid eb_ulgukhai enemy_abomination enemy_bandit enemy_brigand "
           "enemy_demon enemy_demon_cerberus enemy_demon_flareon enemy_demon_gulaemon enemy_demon_imp enemy_demon_legion "
           "enemy_demon_mage enemy_demon_wolf enemy_fallen_knight enemy_gargoyle enemy_giant_rat enemy_halloween_zombie "
           "enemy_lycan enemy_lycan_werewolf enemy_marauder enemy_necromancer enemy_pillager enemy_raider enemy_rotten_lesser "
           "enemy_rotten_tree enemy_sheep_ground enemy_skeleton enemy_skeleton_big enemy_skeleton_blackburn "
           "enemy_spectral_knight enemy_spectral_knight_spawn enemy_spider_big enemy_spider_rotten enemy_spider_rotten_tiny "
           "enemy_spider_small enemy_spider_tiny enemy_swamp_thing enemy_troll enemy_troll_axe_thrower enemy_troll_brute "
           "enemy_troll_chieftain enemy_troll_skater enemy_wererat enemy_werewolf enemy_whitewolf enemy_witch enemy_wolf "
           "enemy_wolf_small enemy_zombie").split()
PORTFOLIOS = json.loads((SCRATCH / "portfolios_v2_r3.json").read_text(encoding="utf-8"))["elite"]
genome = check_genome(PORTFOLIOS["15"][0]["genome"])
profile = task_profile(15, [3, 1.2, 13], hero=genome["hero"], package=genome.get("pkg", "balanced"))
env = NativeEnv(seed=1201, level=15, port=PORT, difficulty=2, identity=f"tower_{uuid.uuid4().hex[:8]}",
                rng_mode=DETERMINISTIC_MODE, action_scope="v4", profile=profile)
try:
    env.reset()
    info = env.worker.rpc("tower_info", names=",".join(NAMES))
    keys = env.worker.rpc("template_keys", names=",".join(ENEMIES))
finally:
    env.close()
(OUT / "towers.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
(OUT / "enemy_keys.json").write_text(json.dumps(keys, indent=1), encoding="utf-8")
print(json.dumps(keys)[:3000])
for rec in info:
    print(rec.get("name"), "missing" if rec.get("missing") else
          [(a.get("i"), a.get("bans_flying"), a.get("range")) for a in rec.get("attacks", [])], "barrack" if rec.get("has_barrack") else "")
