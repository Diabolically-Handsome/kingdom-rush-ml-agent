"""Read-only live verification of the Kingdom Rush bridge."""

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Lumi_Nox"))
from games.kingdom_rush.bot import KingdomRushBot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=25)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    bot = KingdomRushBot()
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "endpoint": "127.0.0.1:9878", "checks": {}}
    try:
        while True:
            try:
                bot.sock = socket.create_connection((bot.host, bot.port), timeout=1)
                bot.sock.settimeout(0.1)
                bot.connected = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Game bridge did not start on port 9878")
                time.sleep(0.5)
        welcome = bot.receive(timeout=5)
        assert welcome and welcome.get("type") == "connected", welcome
        assert welcome.get("game") == "Kingdom Rush", welcome
        report["welcome"] = welcome
        report["checks"]["welcome"] = True
        for action, expected in [("ping", "pong"), ("detect_screen", "screen_info"),
                                 ("get_state", "game_state")]:
            result = bot.send_and_receive({"action": action}, timeout=5)
            assert result and result.get("type") == expected, (action, result)
            report[action] = result
            report["checks"][action] = True
        report["in_level_state_verified"] = not bool(report["get_state"].get("error"))
        report["status"] = "passed"
        print("PASS: bridge welcome, ping, screen detection and state response")
        print("Screen:", report["detect_screen"])
        if not report["in_level_state_verified"]:
            print("No level is loaded; tower actions and battle behavior remain unverified.")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print("FAIL:", exc)
        return 1
    finally:
        bot.close()
        report_dir = ROOT / "runtime"
        report_dir.mkdir(exist_ok=True)
        (report_dir / "smoke-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
