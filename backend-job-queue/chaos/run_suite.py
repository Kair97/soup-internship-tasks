"""Run the full chaos suite and write results/SUMMARY.md.

  * N runs against the real code (must PASS),
  * one run per injected bug (must FAIL: proves the harness can see that bug).

Every run keeps its raw output in results/<run-name>/ (harness.log, api.log,
workers/*.log, summary.json), whatever the result.

Usage: python -m chaos.run_suite --admin-url postgresql://... [--runs 3] [--jobs 10000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-url", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=10_000)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--skip-inject", action="store_true")
    args = ap.parse_args()

    plan = [""] * args.runs + ([] if args.skip_inject else ["no_fencing", "no_lock", "ignore_cancel"])
    rows = []
    for inject in plan:
        before = set(Path(args.out).glob("run-*"))
        cmd = [sys.executable, "-m", "chaos.harness", "--admin-url", args.admin_url,
               "--jobs", str(args.jobs), "--out", args.out]
        if inject:
            cmd += ["--inject", inject]
        rc = subprocess.call(cmd, cwd=ROOT)
        new = sorted(set(Path(args.out).glob("run-*")) - before)
        summary = json.loads((new[-1] / "summary.json").read_text()) if new else {}
        expected = "FAIL" if inject else "PASS"
        rows.append({"run": new[-1].name if new else "?", "inject": inject or "-",
                     "expected": expected, "result": summary.get("result", f"rc={rc}"),
                     "violations": summary.get("violations", []),
                     "metrics": summary.get("metrics", {})})

    lines = [f"# Chaos suite results ({datetime.now():%Y-%m-%d %H:%M})", "",
             "| run | injected bug | expected | result | kills | freezes | effects dup | "
             "stale writes rejected | jobs run >1 time | elapsed s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    ok = True
    for r in rows:
        m = r["metrics"]
        good = r["result"] == r["expected"]
        ok &= good
        dup = sum(v for k, v in m.get("effects_by_status", {}).items()
                  if int(k.split(":")[1]) > 1)
        lines.append(f"| `{r['run']}` | {r['inject']} | {r['expected']} | "
                     f"{'✅' if good else '❌'} {r['result']} | {m.get('kills', '')} | "
                     f"{m.get('freezes', '')} | {dup} | {m.get('stale_writes_rejected', '')} | "
                     f"{m.get('jobs_executed_more_than_once', '')} | {m.get('elapsed_s', '')} |")
    lines += ["", "## Violations reported", ""]
    for r in rows:
        lines.append(f"- `{r['run']}`: " + ("; ".join(r["violations"]) or "none"))
    out = Path(args.out) / "SUMMARY.md"
    if out.exists():
        out.write_text(out.read_text(encoding="utf-8") + "\n\n" + "\n".join(lines), encoding="utf-8")
    else:
        out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
