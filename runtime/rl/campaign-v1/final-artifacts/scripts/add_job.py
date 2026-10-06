"""Add or replace one campaign-v1 job entry built from make_jobs.py output.

usage: add_job.py <kind> <jobs.json> [key=value ...]
kinds: collect | eval | campaign
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
CONFIG = ROOT / "configs/phases/campaign-v1.json"
APPROVAL = ("[用户原话已省略 / user's message omitted]"
            "（2026-10-05 用户授权自主推进至 2026-10-06 19:00）")
PROTOCOL = {"wait_ticks": 120, "max_interval_ticks": 600, "max_ticks": 240000, "max_decisions": 3000}


def main():
    kind, source = sys.argv[1], Path(sys.argv[2])
    opts = dict(arg.split("=", 1) for arg in sys.argv[3:])
    data = json.loads(source.read_text(encoding="utf-8"))
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = {"enabled": True, "approval": APPROVAL, "gpu": False, "optimizer_steps": 0, "serial": False,
            "rng_mode": "isolate_sound+stable_pairs", "action_scope": "v2", "replay_fraction": 0.0,
            "time_reserve_seconds": 120, "episode_protocol": dict(PROTOCOL), "max_jobs": 6}
    weights = None
    if "weights" in opts:
        path = ROOT / opts["weights"]
        weights = {"path": opts["weights"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if kind == "collect":
        tasks = data["collect"]
        job = {**base, "purpose": "record plan-executor demonstrations (state, legal menu, plan step, choice) on "
                                  "train seeds for operator-network behaviour cloning",
               "max_wall_seconds": 3600, "total_wall_seconds": 10800, "max_games": len(tasks) + 10,
               "collect": {"tasks": tasks, "workers": int(opts.get("workers", 32)), "profile_stars_per_level": float(opts.get("stars", 2.75))}}
        if "dagger" in opts:
            job["collect"]["dagger"] = {"beta": float(opts["dagger"])}
            job["purpose"] = ("DAgger: the operator network plays the plans on train seeds while the plan executor "
                              "labels every state it reaches")
            job["operator_weights"] = weights
        name = "native-collect"
    elif kind == "eval":
        tasks = data["eval"]
        players = opts.get("players", "plan,operator").split(",")
        job = {**base, "purpose": "per-level comparison of the scripted plan executor and the operator network on "
                                  "evaluation seeds (never used for training)",
               "max_wall_seconds": 3600, "total_wall_seconds": 14400, "max_games": len(tasks) * len(players) + 10,
               "eval": {"tasks": tasks, "workers": int(opts.get("workers", 32)), "profile_stars_per_level": float(opts.get("stars", 2.75)),
                        "players": players}}
        if weights:
            job["operator_weights"] = weights
        name = "native-eval"
    elif kind == "campaign":
        seeds = [int(s) for s in opts["seeds"].split(",")]
        brain = opts.get("brain", "rule")
        policy = opts.get("policy", "plan")
        attempts = int(opts.get("attempts", 3))
        job = {**base, "purpose": f"full campaign from a new save, levels 1-12 in order: {brain} strategy brain, "
                                  f"{policy} player, up to {attempts} attempts per level",
               "max_wall_seconds": 7200, "total_wall_seconds": 28800, "max_games": len(seeds) * 12 * attempts + 5,
               "gpu": "external-inference" if brain == "8b" else False,
               "campaign": {"levels": list(range(1, 13)), "seeds": seeds, "attempts_per_level": attempts,
                            "plans": data["plans"], "policy": policy, "brain": brain}}
        if weights:
            job["operator_weights"] = weights
        name = opts.get("job", "native-campaign")
    else:
        raise SystemExit("unknown kind")
    cfg["jobs"][name] = job
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("set", name, {k: v for k, v in job.items() if k not in ("collect", "eval", "campaign", "approval")})


if __name__ == "__main__":
    main()
