#!/usr/bin/env python3
"""Live local check: both real harnesses share one loaded Laya process; no cloud calls."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    processes = [subprocess.Popen([str(ROOT / "bin" / name), "--", "--version"],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for name in ("codex-laya", "claude-laya")]
    pids = []
    for process in processes:
        out, err = process.communicate(timeout=150)
        if process.returncode:
            raise RuntimeError(err)
        print(out.strip())
        pids.append(re.search(r"shared pid=(\d+)", err).group(1))
    assert len(set(pids)) == 1, pids
    duplicate = subprocess.run([sys.executable, str(ROOT / "scripts/laya_service.py"), "--serve"],
                               capture_output=True, text=True, timeout=5)
    assert duplicate.returncode == 0, duplicate.stderr
    health = json.loads(subprocess.check_output([str(ROOT / "bin/laya-mlx-advisor"), "status"], text=True))
    assert health["pid"] == int(pids[0]), health
    assert health["root"] == str(ROOT), health
    print(f"PASS: both harnesses use one Laya daemon, PID {pids[0]}")


if __name__ == "__main__":
    main()
