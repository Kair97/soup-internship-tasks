"""Load test for the "SHIP at 100 requests per second?" question.

Open-loop: requests are fired on a fixed schedule whatever the latency (a closed loop
would slow down with the server and hide queueing). For each target rate it records
create latency percentiles, errors, and how the job backlog evolves with 8 workers.

Also measures raw worker capacity: how many zero-duration jobs 8 workers can finish
per second (the claim/finish path is the database's hot spot).

Usage: python -m chaos.load_test --admin-url postgresql://... [--rates 100 200 400]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psycopg

from chaos.harness import ROOT, Log, Workers, free_port, ts, with_db


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


async def open_loop(base: str, rate: float, seconds: float, duration_ms: int, tag: str):
    lat, errors, codes = [], 0, {}
    sem = asyncio.Semaphore(1000)
    async with httpx.AsyncClient(base_url=base, timeout=30,
                                 limits=httpx.Limits(max_connections=400)) as c:
        async def one(i):
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                try:
                    r = await c.post("/jobs", json={"duration_ms": duration_ms},
                                     headers={"Idempotency-Key": f"{tag}-{i}"})
                    codes[r.status_code] = codes.get(r.status_code, 0) + 1
                except httpx.HTTPError:
                    errors += 1
                    return
                lat.append((time.perf_counter() - t) * 1000)

        tasks, start = [], time.perf_counter()
        n = int(rate * seconds)
        for i in range(n):
            delay = start + i / rate - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i)))
        sent_in = time.perf_counter() - start
        await asyncio.gather(*tasks)
    return {"target_rps": rate, "sent": n, "achieved_send_rps": round(n / sent_in, 1),
            "p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1),
            "p99_ms": round(pct(lat, 99), 1), "max_ms": round(max(lat), 1) if lat else None,
            "mean_ms": round(statistics.mean(lat), 1) if lat else None,
            "errors": errors, "codes": codes}


def backlog(db_url: str) -> int:
    with psycopg.connect(db_url) as c:
        return c.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-url", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[100, 200, 400])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--duration-ms", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "load"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / f"load-{time.strftime('%Y%m%d-%H%M%S')}.log")
    with psycopg.connect(args.admin_url, autocommit=True) as c:
        c.execute('DROP DATABASE IF EXISTS "jobs_load" WITH (FORCE)')
        c.execute('CREATE DATABASE "jobs_load"')
    db_url = with_db(args.admin_url, "jobs_load")
    from app import db as appdb
    appdb.init_schema(db_url)
    env = os.environ | {"DATABASE_URL": db_url, "LEASE_SECONDS": "5", "LOG_LEVEL": "WARNING"}
    port = free_port()
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app", "--port", str(port),
                            "--log-level", "warning"], cwd=ROOT, env=env)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(base + "/healthz", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    (out / "workers").mkdir(exist_ok=True)
    workers = Workers(args.workers, env, log, out / "workers")
    workers.start()
    time.sleep(2)
    results = {}
    try:
        # 1. worker capacity: 5,000 zero-duration jobs inserted directly, time to drain.
        with psycopg.connect(db_url, autocommit=True) as c:
            c.execute("""INSERT INTO jobs (idempotency_key, payload, payload_hash)
                         SELECT 'cap-' || g, '{"duration_ms":0}', 'x' FROM generate_series(1,5000) g""")
        t = time.perf_counter()
        while backlog(db_url):
            time.sleep(0.1)
        cap = 5000 / (time.perf_counter() - t)
        results["worker_capacity_jobs_per_s"] = round(cap, 1)
        log(f"worker capacity ({args.workers} workers, 0 ms jobs): {cap:.1f} jobs/s")

        # 2. open-loop API load at each rate, jobs of --duration-ms.
        for rate in args.rates:
            r = asyncio.run(open_loop(base, rate, args.seconds, args.duration_ms, f"r{int(rate)}"))
            r["backlog_at_end"] = backlog(db_url)
            t = time.perf_counter()
            while backlog(db_url) and time.perf_counter() - t < 300:
                time.sleep(0.2)
            r["drain_s_after_end"] = round(time.perf_counter() - t, 1)
            results[f"{int(rate)}rps"] = r
            log(f"{ts()} rate={rate} {json.dumps(r)}")
    finally:
        workers.stop_all()
        api.kill()
    (out / "load-summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("summary " + json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
