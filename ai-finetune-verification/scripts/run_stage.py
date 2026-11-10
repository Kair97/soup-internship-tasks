"""Run one pipeline stage with raw evidence.

  * every stdout/stderr line of the command is written to logs/<stage>.log with a
    UTC timestamp prefix (raw, nothing filtered);
  * `nvidia-smi` is sampled every 500 ms for the whole stage into
    logs/<stage>.nvidia-smi.csv (raw CSV straight from nvidia-smi), and a full
    `nvidia-smi` snapshot is saved before and after;
  * the peak memory.used seen by nvidia-smi is appended to logs/stages.jsonl.

Usage: python scripts/run_stage.py <stage-name> -- <command ...>
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def main() -> int:
    if "--" not in sys.argv:
        print(__doc__)
        return 2
    i = sys.argv.index("--")
    stage, cmd = sys.argv[1], sys.argv[i + 1:]
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"{stage}.log"
    smi_path = LOGS / f"{stage}.nvidia-smi.csv"

    (LOGS / f"{stage}.nvidia-smi.before.txt").write_text(
        subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, encoding="utf-8")
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw",
         "--format=csv", "-lms", "500"],
        stdout=smi_path.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"{now()} [run_stage] stage={stage} cmd={cmd}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1,
                             env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})

        def pump() -> None:
            assert p.stdout is not None
            buf = ""
            while True:
                ch = p.stdout.read(1)
                if not ch:
                    break
                if ch in "\r\n":           # progress bars use \r: keep each update as a line
                    if buf.strip():
                        line = f"{now()} {buf}"
                        log.write(line + "\n")
                        log.flush()
                        print(line, flush=True)
                    buf = ""
                else:
                    buf += ch
            if buf.strip():
                log.write(f"{now()} {buf}\n")

        th = threading.Thread(target=pump)
        th.start()
        rc = p.wait()
        th.join()
        log.write(f"{now()} [run_stage] exit={rc} elapsed={time.time() - t0:.1f}s\n")

    time.sleep(1.0)
    smi.terminate()
    smi.wait()
    (LOGS / f"{stage}.nvidia-smi.after.txt").write_text(
        subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, encoding="utf-8")

    peak, idle = 0, None
    with smi_path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            try:
                used = int(row[1].strip().split()[0])
            except (IndexError, ValueError):
                continue
            idle = used if idle is None else idle
            peak = max(peak, used)
    rec = {"stage": stage, "exit": rc, "elapsed_s": round(time.time() - t0, 1),
           "nvidia_smi_first_mib": idle, "nvidia_smi_peak_mib": peak, "cmd": cmd, "finished": now()}
    with (LOGS / "stages.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
