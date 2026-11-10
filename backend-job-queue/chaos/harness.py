"""Chaos harness for the training-job queue.

What it does (defaults match the brief):
  * creates a fresh database, starts the API and N worker processes;
  * submits JOBS jobs over HTTP, while in parallel:
      - re-POSTing the same Idempotency-Keys (simultaneous pairs + delayed retries),
      - cancelling a random subset of jobs at random moments,
      - hard-killing a random worker about every KILL_EVERY seconds
        (SIGKILL on Linux, TerminateProcess on Windows: no cleanup, no finally),
      - freezing a random worker for longer than its lease and then resuming it
        (a "zombie" that comes back with a stale lease);
  * waits until every job is terminal, then checks the invariants straight in Postgres.

Exit code 0 = all invariants hold, 1 = at least one violation, 2 = harness error/timeout.

Usage:
  python -m chaos.harness --admin-url postgresql://user:pw@localhost:5432/postgres
  python -m chaos.harness --inject no_fencing   # harness self-test: MUST fail
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import psutil
import psycopg

ROOT = Path(__file__).resolve().parents[1]
LEGAL = {  # state machine: (from, to) pairs a job may take
    ("queued", "running"), ("queued", "cancelled"),
    ("running", "queued"), ("running", "succeeded"), ("running", "failed"),
    ("running", "cancelled"),
}


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


class Log:
    def __init__(self, path: Path):
        self.f = path.open("w", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{ts()} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def with_db(url: str, dbname: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "/" + dbname, p.query, p.fragment))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Record:
    """Everything the clients observed, to compare with the database afterwards."""
    ids_by_key: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    create_codes: Counter = field(default_factory=Counter)
    duplicate_posts: int = 0
    cancel_codes: Counter = field(default_factory=Counter)
    cancel_accepted: set[str] = field(default_factory=set)    # 200 cancelled / 202 requested
    cancel_too_late: dict[str, str] = field(default_factory=dict)  # id -> status in 409
    http_errors: Counter = field(default_factory=Counter)
    running_cancels_sent: int = 0
    kills: int = 0
    freezes: int = 0
    respawns: int = 0


def real_process(p: subprocess.Popen) -> psutil.Process:
    """The process that actually runs the worker code.

    On Windows, a venv's python.exe is a launcher that starts the real interpreter as
    a child (in a job object, so killing the launcher kills the child too). PIDs in
    lease_owner are the child's. On Linux there is no launcher: it's the same process.
    """
    proc = psutil.Process(p.pid)
    for _ in range(50):
        kids = proc.children()
        if kids:
            return kids[0]
        if sys.platform != "win32":
            return proc
        time.sleep(0.02)
    return proc


class Workers:
    """Keeps N worker processes alive and lets the chaos loop kill/freeze them."""

    def __init__(self, n: int, env: dict[str, str], log: Log, logdir: Path):
        self.n, self.env, self.log, self.logdir = n, env, log, logdir
        self.procs: list[subprocess.Popen] = []
        self.frozen: set[int] = set()      # Popen pids currently suspended
        self.real: dict[int, psutil.Process] = {}
        self._seq = 0

    def spawn(self) -> subprocess.Popen:
        self._seq += 1
        out = (self.logdir / f"worker-{self._seq:04d}.log").open("w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-m", "app.worker"], cwd=ROOT, env=self.env,
                             stdout=out, stderr=subprocess.STDOUT)
        self.procs.append(p)
        try:
            self.real[p.pid] = real_process(p)
        except psutil.Error:
            pass
        return p

    def worker_pid(self, p: subprocess.Popen) -> int:
        r = self.real.get(p.pid)
        return r.pid if r else p.pid

    def start(self) -> None:
        for _ in range(self.n):
            self.spawn()

    def alive(self) -> list[subprocess.Popen]:
        return [p for p in self.procs if p.poll() is None]

    def kill_random(self) -> int | None:
        candidates = [p for p in self.alive() if p.pid not in self.frozen]
        if not candidates:
            return None
        victim = random.choice(candidates)
        real = self.real.get(victim.pid)
        if real is not None:
            try:
                real.kill()                    # SIGKILL / TerminateProcess on the worker itself
            except psutil.Error:
                pass
        victim.kill()
        victim.wait()
        self.spawn()                           # a supervisor would restart it
        return self.worker_pid(victim)

    def stop_all(self) -> None:
        for p in self.alive():
            real = self.real.get(p.pid)
            for proc in (real, psutil.Process(p.pid)) if real else (psutil.Process(p.pid),):
                try:
                    proc.resume()
                    proc.kill()
                except psutil.Error:
                    pass
            p.kill()
        for p in self.procs:
            p.wait()


def busy_worker_pids(db_url: str) -> set[int]:
    """PIDs of workers that hold a lease right now (worker id = host:pid:suffix)."""
    with psycopg.connect(db_url) as c:
        rows = c.execute("SELECT lease_owner FROM jobs WHERE status = 'running'").fetchall()
    return {int(r[0].split(":")[-2]) for r in rows if r[0]}


async def chaos_loop(workers: Workers, rec: Record, log: Log, stop: asyncio.Event,
                     kill_every: float, freeze_every: float, freeze_for: float,
                     db_url: str) -> None:
    next_kill = time.monotonic() + kill_every
    next_freeze = time.monotonic() + freeze_every
    while not stop.is_set():
        await asyncio.sleep(0.05)
        now = time.monotonic()
        if kill_every > 0 and now >= next_kill:
            pid = workers.kill_random()
            if pid:
                rec.kills += 1
                rec.respawns += 1
                log(f"CHAOS kill -9 worker pid={pid} (kills={rec.kills})")
            next_kill = now + random.uniform(0.75, 1.25) * kill_every
        if freeze_every > 0 and now >= next_freeze:
            # Freeze a worker that is in the middle of a job: that is the case that
            # produces a zombie. (Freezing an idle worker proves nothing.)
            busy = await asyncio.to_thread(busy_worker_pids, db_url)
            cands = [p for p in workers.alive()
                     if p.pid not in workers.frozen and workers.worker_pid(p) in busy]
            if cands:
                victim = random.choice(cands)
                workers.frozen.add(victim.pid)
                workers.real[victim.pid].suspend()
                rec.freezes += 1
                log(f"CHAOS freeze busy worker pid={workers.worker_pid(victim)} for {freeze_for:.1f}s")
                asyncio.get_running_loop().call_later(freeze_for, _thaw, workers, victim, log)
            next_freeze = now + random.uniform(0.75, 1.25) * freeze_every


def _thaw(workers: Workers, p: subprocess.Popen, log: Log) -> None:
    try:
        workers.real[p.pid].resume()
        log(f"CHAOS resume worker pid={workers.worker_pid(p)} (now a zombie with an expired lease)")
    except psutil.Error:
        pass
    workers.frozen.discard(p.pid)


async def post_job(client: httpx.AsyncClient, rec: Record, key: str, body: dict) -> str | None:
    for attempt in range(8):
        try:
            r = await client.post("/jobs", json=body, headers={"Idempotency-Key": key})
        except httpx.HTTPError as e:   # the client retries, exactly like a real one would
            rec.http_errors[type(e).__name__] += 1
            await asyncio.sleep(0.05 * (attempt + 1))
            continue
        rec.create_codes[r.status_code] += 1
        if r.status_code in (200, 201):
            jid = r.json()["id"]
            rec.ids_by_key[key].add(jid)
            return jid
        rec.http_errors[f"create_{r.status_code}"] += 1
        await asyncio.sleep(0.05 * (attempt + 1))
    return None


async def cancel_job(client: httpx.AsyncClient, rec: Record, jid: str, delay: float) -> None:
    await asyncio.sleep(delay)
    for attempt in range(5):
        try:
            r = await client.post(f"/jobs/{jid}/cancel")
        except httpx.HTTPError as e:
            rec.http_errors[type(e).__name__] += 1
            await asyncio.sleep(0.05 * (attempt + 1))
            continue
        rec.cancel_codes[r.status_code] += 1
        if r.status_code in (200, 202):
            rec.cancel_accepted.add(jid)
        elif r.status_code == 409:
            rec.cancel_too_late[jid] = r.json()["status"]
        return


async def cancel_running_loop(base_url: str, rec: Record, stop: asyncio.Event,
                              prob: float) -> None:
    """Cancel jobs that are running right now: the cancel/complete race lives here."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        while not stop.is_set():
            await asyncio.sleep(0.25)
            try:
                page = (await client.get("/jobs", params={"status": "running", "limit": 100})).json()
            except (httpx.HTTPError, ValueError):
                continue
            for job in page["items"]:
                if random.random() < prob:
                    asyncio.create_task(cancel_job(client, rec, job["id"], 0.0))
                    rec.running_cancels_sent += 1


async def submit_all(base_url: str, rec: Record, args, log: Log) -> None:
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=10, limits=limits) as client:
        background: list[asyncio.Task] = []

        async def one(i: int) -> None:
            key = f"job-{i:06d}"
            dur = (random.randint(*args.long_ms) if random.random() < args.long_rate
                   else random.randint(*args.duration_ms))
            body = {"model": "qwen2.5-1.5b", "duration_ms": dur,
                    "fail": random.random() < args.fail_rate, "params": {"i": i}}
            async with sem:
                if random.random() < args.dup_rate:
                    # two simultaneous requests with the same key (a client retrying
                    # before the first answer arrived)
                    rec.duplicate_posts += 1
                    ids = await asyncio.gather(post_job(client, rec, key, body),
                                               post_job(client, rec, key, body))
                    jid = ids[0] or ids[1]
                else:
                    jid = await post_job(client, rec, key, body)
            if jid is None:
                return
            if random.random() < args.retry_rate:      # a late retry of the same request
                rec.duplicate_posts += 1
                background.append(asyncio.create_task(
                    _late_retry(client, rec, key, body, random.uniform(0.1, 5.0))))
            if random.random() < args.cancel_rate:
                background.append(asyncio.create_task(
                    cancel_job(client, rec, jid, random.uniform(0.0, 3.0))))

        t0 = time.monotonic()
        await asyncio.gather(*(one(i) for i in range(args.jobs)))
        log(f"submitted {args.jobs} jobs in {time.monotonic() - t0:.1f}s; "
            f"waiting for {len(background)} retries/cancels")
        await asyncio.gather(*background)


async def _late_retry(client, rec, key, body, delay):
    await asyncio.sleep(delay)
    await post_job(client, rec, key, body)


async def wait_until_drained(base_url: str, log: Log, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        last = None
        while time.monotonic() < deadline:
            try:
                s = (await client.get("/stats")).json()
            except httpx.HTTPError:
                await asyncio.sleep(1)
                continue
            if s != last:
                log(f"stats {json.dumps(s)}")
                last = s
            if s["queued"] == 0 and s["running"] == 0:
                return s
            await asyncio.sleep(1)
    raise TimeoutError(f"jobs not drained after {timeout}s: {last}")


def check_invariants(db_url: str, rec: Record, expected_jobs: int) -> tuple[list[str], dict]:
    v: list[str] = []
    m: dict = {}
    with psycopg.connect(db_url) as conn:
        q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731

        # 1. No job lost: one job per submitted key, every id a client saw exists.
        m["jobs_in_db"] = q("SELECT count(*) FROM jobs")[0][0]
        keys_ok = [k for k, ids in rec.ids_by_key.items() if ids]
        if m["jobs_in_db"] != expected_jobs:
            v.append(f"LOST/EXTRA JOBS: expected {expected_jobs}, found {m['jobs_in_db']}")
        if len(keys_ok) != expected_jobs:
            v.append(f"CLIENT NEVER GOT AN ID for {expected_jobs - len(keys_ok)} keys")
        seen_ids = {i for ids in rec.ids_by_key.values() for i in ids}
        db_ids = {str(r[0]) for r in q("SELECT id FROM jobs")}
        if missing := seen_ids - db_ids:
            v.append(f"LOST JOBS: {len(missing)} ids returned to clients are not in the DB")

        # 2. Idempotency: every POST with the same key returned the same id.
        split = {k: ids for k, ids in rec.ids_by_key.items() if len(ids) > 1}
        m["keys_with_more_than_one_id"] = len(split)
        if split:
            v.append(f"DUPLICATE ON RETRY: {len(split)} keys produced more than one job id")

        # 3. Nothing stuck.
        stuck = q("SELECT count(*) FROM jobs WHERE status IN ('queued','running')")[0][0]
        m["stuck_non_terminal"] = stuck
        if stuck:
            v.append(f"STUCK: {stuck} jobs not terminal after drain")

        # 4. Side effects: <= 1 per job; exactly 1 iff succeeded; written by the final claim.
        rows = q("""
            SELECT j.status, count(e.id) AS n
            FROM jobs j LEFT JOIN effects e ON e.job_id = j.id
            GROUP BY j.id, j.status""")
        dist = Counter((s, n) for s, n in rows)
        m["effects_by_status"] = {f"{s}:{n}": c for (s, n), c in sorted(dist.items())}
        multi = sum(c for (s, n), c in dist.items() if n > 1)
        bad_succ = sum(c for (s, n), c in dist.items() if s == "succeeded" and n != 1)
        bad_other = sum(c for (s, n), c in dist.items() if s != "succeeded" and n != 0)
        if multi:
            v.append(f"DUPLICATE SIDE EFFECT: {multi} jobs have more than one effect")
        if bad_succ:
            v.append(f"SUCCEEDED WITHOUT EXACTLY ONE EFFECT: {bad_succ}")
        if bad_other:
            v.append(f"EFFECT ON A NON-SUCCEEDED JOB: {bad_other}")
        stale_eff = q("""SELECT count(*) FROM effects e JOIN jobs j ON j.id = e.job_id
                         WHERE e.fencing_token <> j.fencing_token""")[0][0]
        m["effects_from_stale_token"] = stale_eff
        if stale_eff:
            v.append(f"ZOMBIE WRITE: {stale_eff} effects written with a superseded fencing token")

        # 5. State machine replay (from the trigger-written audit log).
        trans = q("""SELECT from_status, to_status, (from_token <> to_token) AS new_token, count(*)
                     FROM job_transitions GROUP BY 1, 2, 3""")
        m["transitions"] = {f"{a}->{b}{' (new token)' if t else ''}": n for a, b, t, n in trans}
        illegal = [(a, b, t, n) for a, b, t, n in trans
                   if (a, b) not in LEGAL or (a == b == "running" and t)]
        double_claims = sum(n for a, b, t, n in trans if a == "running" and b == "running" and t)
        if double_claims:
            v.append(f"DOUBLE CLAIM: {double_claims} claims of a job that was already running")
        other_illegal = [(a, b, n) for a, b, t, n in illegal if not (a == b == "running")]
        if other_illegal:
            v.append(f"ILLEGAL TRANSITIONS: {other_illegal}")
        # a claim must bump the token; any other transition must not
        bad_token = q("""SELECT count(*) FROM job_transitions
                         WHERE (from_status='queued' AND to_status='running'
                                AND to_token <> from_token + 1)
                            OR (NOT (from_status='queued' AND to_status='running')
                                AND to_token <> from_token
                                AND NOT (from_status='running' AND to_status='running'))""")[0][0]
        if bad_token:
            v.append(f"FENCING TOKEN MISUSE in {bad_token} transitions")

        # 6. Cancel semantics as seen by clients.
        final = {str(i): s for i, s in q("SELECT id, status FROM jobs")}
        bad_cancel = [j for j in rec.cancel_accepted if final.get(j) != "cancelled"]
        m["cancel_accepted"] = len(rec.cancel_accepted)
        if bad_cancel:
            v.append(f"CANCEL IGNORED: {len(bad_cancel)} jobs had cancel accepted (200/202) "
                     f"but finished as {Counter(final[j] for j in bad_cancel)}")
        bad_late = [j for j, st in rec.cancel_too_late.items() if final.get(j) != st]
        m["cancel_too_late"] = len(rec.cancel_too_late)
        if bad_late:
            v.append(f"CANCEL 409 INCONSISTENT: {len(bad_late)} jobs changed after a 409 "
                     f"reported them terminal")

        # 7. Row-level consistency of terminal jobs.
        incons = q("""SELECT count(*) FROM jobs WHERE status IN ('succeeded','failed','cancelled')
                      AND (lease_owner IS NOT NULL OR finished_at IS NULL
                           OR (started_at IS NOT NULL AND finished_at < started_at)
                           OR (status = 'succeeded' AND result IS NULL))""")[0][0]
        if incons:
            v.append(f"INCONSISTENT TERMINAL ROWS: {incons}")

        # Evidence (not invariants): at-least-once execution, zombies fenced off.
        m["final_status"] = dict(q("SELECT status, count(*) FROM jobs GROUP BY 1 ORDER BY 1"))
        m["executions_total"] = q("SELECT count(*) FROM executions")[0][0]
        m["jobs_executed_more_than_once"] = q(
            "SELECT count(*) FROM (SELECT job_id FROM executions GROUP BY 1 HAVING count(*) > 1) x")[0][0]
        m["max_executions_per_job"] = q(
            "SELECT coalesce(max(n),0) FROM (SELECT count(*) n FROM executions GROUP BY job_id) x")[0][0]
        m["stale_writes_rejected"] = dict(q("SELECT kind, count(*) FROM stale_writes GROUP BY 1"))
        m["requeued_after_lease_expiry"] = q(
            "SELECT count(*) FROM job_transitions WHERE from_status='running' AND to_status='queued'")[0][0]
    return v, m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--admin-url", default=os.environ.get(
        "ADMIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"))
    ap.add_argument("--db-name", default="jobs_chaos")
    ap.add_argument("--jobs", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kill-every", type=float, default=2.0)
    ap.add_argument("--freeze-every", type=float, default=10.0)
    ap.add_argument("--lease", type=float, default=3.0)
    ap.add_argument("--duration-ms", type=int, nargs=2, default=(10, 150))
    ap.add_argument("--long-rate", type=float, default=0.03,
                    help="share of long jobs (they heartbeat and get cancelled mid-run)")
    ap.add_argument("--long-ms", type=int, nargs=2, default=(500, 3000))
    ap.add_argument("--running-cancel-prob", type=float, default=0.05)
    ap.add_argument("--fail-rate", type=float, default=0.02)
    ap.add_argument("--dup-rate", type=float, default=0.15)
    ap.add_argument("--retry-rate", type=float, default=0.10)
    ap.add_argument("--cancel-rate", type=float, default=0.10)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--inject", default="", choices=["", "no_fencing", "no_lock", "ignore_cancel"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    random.seed(seed)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"run-{stamp}-{platform.system().lower()}" + (f"-inject-{args.inject}" if args.inject else "")
    outdir = Path(args.out) / name
    (outdir / "workers").mkdir(parents=True, exist_ok=True)
    log = Log(outdir / "harness.log")
    log(f"harness start: {json.dumps(vars(args) | {'seed': seed})}")
    log(f"platform: {platform.platform()} python {platform.python_version()} cpus={os.cpu_count()}")

    # Fresh database for every run.
    with psycopg.connect(args.admin_url, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{args.db_name}" WITH (FORCE)')
        c.execute(f'CREATE DATABASE "{args.db_name}"')
        log(f"postgres: {c.execute('SELECT version()').fetchone()[0]}")
    db_url = with_db(args.admin_url, args.db_name)

    env = os.environ | {"DATABASE_URL": db_url, "LEASE_SECONDS": str(args.lease),
                        "INJECT": args.inject, "PYTHONUNBUFFERED": "1", "LOG_LEVEL": "INFO"}
    from app import db as appdb  # bootstrap schema before anything starts
    appdb.init_schema(db_url)

    port = free_port()
    api_log = (outdir / "api.log").open("w", encoding="utf-8")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1",
                            "--port", str(port), "--log-level", "warning"],
                           cwd=ROOT, env=env, stdout=api_log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base_url + "/healthz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    log(f"api up at {base_url}")

    workers = Workers(args.workers, env, log, outdir / "workers")
    workers.start()
    rec = Record()
    rc = 2
    t0 = time.monotonic()
    try:
        async def run():
            stop = asyncio.Event()
            chaos = asyncio.create_task(chaos_loop(workers, rec, log, stop, args.kill_every,
                                                   args.freeze_every, args.lease * 2 + 1, db_url))
            canceller = asyncio.create_task(cancel_running_loop(base_url, rec, stop,
                                                                args.running_cancel_prob))
            await submit_all(base_url, rec, args, log)
            await wait_until_drained(base_url, log, args.timeout)
            stop.set()
            await chaos
            await canceller
            # Let every frozen worker wake up and try its stale writes BEFORE checking:
            # otherwise a zombie bug could hide just because the run ended first.
            while workers.frozen:
                await asyncio.sleep(0.2)
            log(f"all frozen workers resumed; grace period {args.lease + 2:.0f}s for zombie writes")
            await asyncio.sleep(args.lease + 2)
            return await wait_until_drained(base_url, log, 60)
        asyncio.run(run())
        elapsed = time.monotonic() - t0
        workers.stop_all()
        violations, metrics = check_invariants(db_url, rec, args.jobs)
        metrics |= {"elapsed_s": round(elapsed, 1), "seed": seed, "kills": rec.kills,
                    "freezes": rec.freezes, "running_cancels_sent": rec.running_cancels_sent,
                    "duplicate_posts": rec.duplicate_posts,
                    "create_codes": dict(rec.create_codes), "cancel_codes": dict(rec.cancel_codes),
                    "client_transport_errors": dict(rec.http_errors)}
        log("metrics " + json.dumps(metrics, default=str))
        if violations:
            log(f"RESULT: FAIL ({len(violations)} invariant violations)")
            for x in violations:
                log(f"  VIOLATION {x}")
            rc = 1
        else:
            log("RESULT: PASS (no lost jobs, no duplicate effects, nothing stuck, states consistent)")
            rc = 0
        (outdir / "summary.json").write_text(json.dumps(
            {"name": name, "result": "PASS" if rc == 0 else "FAIL", "violations": violations,
             "args": vars(args), "metrics": metrics}, indent=2, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - a harness failure is a result too
        log(f"RESULT: HARNESS ERROR {type(e).__name__}: {e}")
        (outdir / "summary.json").write_text(json.dumps(
            {"name": name, "result": "ERROR", "error": f"{type(e).__name__}: {e}"}, indent=2),
            encoding="utf-8")
    finally:
        workers.stop_all()
        api.kill()
        api.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
