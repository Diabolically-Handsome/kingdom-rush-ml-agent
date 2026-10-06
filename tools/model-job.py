"""Linux supervisor for one explicitly authorized inference batch; no torch import.

Own process group only. STOP or deadline terminates that worker, never another
project's GPU process. The outer Linux timeout is an independent second guard.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("8b", "24b"), required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=int, required=True)
    args = ap.parse_args()
    if not 0 < args.seconds <= 600:
        ap.error("wall budget must be 1..600 seconds")
    root = Path(__file__).resolve().parents[1]
    state = root / "runtime/rl"
    import fcntl
    lock = (state / "model-gpu-wsl.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another AlphaRush GPU batch holds the Linux lock", file=sys.stderr)
        return 9
    worker = None

    def terminate(_signum=None, _frame=None):
        raise KeyboardInterrupt("Supervisor signal/stop")

    signal.signal(signal.SIGTERM, terminate)
    try:
        if any((state / name).exists() for name in ("STOP", "ENGINEERING-STOP")):
            raise RuntimeError("Stop file is present before worker launch")
        worker = subprocess.Popen([sys.executable, "-B", str(root / "tools/model-worker.py"),
                                   "--model", args.model, "--requests", args.requests,
                                   "--output", args.output, "--max-prompt-tokens", "4096"],
                                  start_new_session=True)
        deadline = time.monotonic() + args.seconds
        while worker.poll() is None:
            if time.monotonic() >= deadline or any((state / name).exists() for name in ("STOP", "ENGINEERING-STOP")):
                raise RuntimeError("STOP or inference wall budget exhausted")
            time.sleep(0.1)
        return worker.returncode
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 124
    finally:
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=5)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
