"""Targeted tests for each silent failure listed in the brief.

Each test forces the dangerous interleaving deterministically (or hammers it many
times) instead of hoping the chaos run happens to hit it.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import psycopg
import pytest

from app import repo

LEASE = 30.0


def make_jobs(conn, n, duration_ms=0):
    for i in range(n):
        repo.create_job(conn, f"k-{i}", {"duration_ms": duration_ms}, max_attempts=5)


def effects(conn, job_id):
    return conn.execute("SELECT count(*) FROM effects WHERE job_id = %s", (job_id,)).fetchone()[0]


def expire_lease(conn, job_id):
    conn.execute("UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
                 (job_id,))


# ---------------------------------------------------------------- 1. double claim
def test_no_double_claim_under_concurrency(conn, new_conn):
    """16 threads race to claim 400 jobs: every job must be claimed exactly once."""
    make_jobs(conn, 400)
    claimed, lock = [], threading.Lock()

    def drain(i):
        c = new_conn()
        while (cl := repo.claim_next(c, f"w{i}", LEASE)) is not None:
            with lock:
                claimed.append(cl.job_id)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(drain, range(16)))
    assert len(claimed) == 400
    assert len(set(claimed)) == 400


def test_injected_no_lock_does_double_claim(conn, new_conn):
    """Sanity check of the test itself: the deliberately broken claim double-claims."""
    make_jobs(conn, 300)
    claimed, lock = [], threading.Lock()

    def drain(i):
        c = new_conn()
        while (cl := repo.claim_next(c, f"w{i}", LEASE, inject="no_lock")) is not None:
            with lock:
                claimed.append(cl.job_id)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(drain, range(16)))
    assert len(claimed) > len(set(claimed)), "expected the broken claim to double-claim"


# ------------------------------------------------------------------- 2. zombie worker
def test_zombie_after_lease_expiry_cannot_write(conn):
    """Worker A freezes, its lease expires, B takes over and finishes. A wakes up and
    tries to heartbeat and to complete: both must be rejected, one effect total."""
    make_jobs(conn, 1)
    a = repo.claim_next(conn, "A", LEASE)
    expire_lease(conn, a.job_id)
    assert repo.reap_expired(conn) == 1
    b = repo.claim_next(conn, "B", LEASE)
    assert b.job_id == a.job_id and b.token == a.token + 1

    assert repo.finish(conn, b, "B", "succeeded", result={}) == "succeeded"
    assert repo.heartbeat(conn, a, "A", LEASE) is None           # zombie heartbeat
    assert repo.finish(conn, a, "A", "succeeded", result={}) is None  # zombie result
    assert effects(conn, a.job_id) == 1
    kinds = [r[0] for r in conn.execute("SELECT kind FROM stale_writes ORDER BY id")]
    assert kinds == ["heartbeat", "complete"]


def test_zombie_wakes_before_takeover_finishes(conn):
    """A's lease expired and B re-claimed, but B has not finished yet. A's write must
    still be rejected (token check, not just status check)."""
    make_jobs(conn, 1)
    a = repo.claim_next(conn, "A", LEASE)
    expire_lease(conn, a.job_id)
    repo.reap_expired(conn)
    b = repo.claim_next(conn, "B", LEASE)
    assert repo.finish(conn, a, "A", "succeeded", result={}) is None
    assert conn.execute("SELECT status, lease_owner FROM jobs").fetchone() == ("running", "B")
    assert repo.finish(conn, b, "B", "succeeded", result={}) == "succeeded"
    assert effects(conn, b.job_id) == 1


def test_injected_no_fencing_lets_zombie_write_twice(conn):
    make_jobs(conn, 1)
    a = repo.claim_next(conn, "A", LEASE)
    expire_lease(conn, a.job_id)
    repo.reap_expired(conn)
    b = repo.claim_next(conn, "B", LEASE)
    repo.finish(conn, b, "B", "succeeded", result={}, inject="no_fencing")
    repo.finish(conn, a, "A", "succeeded", result={}, inject="no_fencing")
    assert effects(conn, a.job_id) == 2, "the broken version must show the duplicate"


def test_heartbeat_racing_reaper(conn, new_conn):
    """The reaper re-checks the lease after locking the row: a heartbeat that committed
    first keeps the job. Run many times to cover both orders."""
    make_jobs(conn, 200)
    claims = [repo.claim_next(conn, "W", 0.05) for _ in range(200)]
    import time
    time.sleep(0.1)  # all leases expired
    hb_conn, reap_conn = new_conn(), new_conn()

    def beat():
        return [repo.heartbeat(hb_conn, c, "W", LEASE) for c in claims]

    with ThreadPoolExecutor(2) as ex:
        f_hb = ex.submit(beat)
        f_reap = ex.submit(lambda: sum(repo.reap_expired(reap_conn, 10) for _ in range(40)))
        beats, _ = f_hb.result(), f_reap.result()
    rows = dict(conn.execute("SELECT id, status FROM jobs").fetchall())
    for c, kept in zip(claims, beats):
        # heartbeat succeeded <=> the job is still running under this worker
        assert (kept is not None) == (rows[c.job_id] == "running")


# --------------------------------------------------------- 3. cancel / complete race
def test_cancel_complete_race(conn, new_conn):
    """Cancel and complete hit the SAME row at the same instant (a barrier before every
    job), 5 x 200 times. The row lock decides the order; both orders must occur and
    each must end in a consistent state.

    (A first version started both loops once and let them run: cancel won 1000/1000
    times because its loop simply ran ahead - it never tested the race. See REPORT.md.)"""
    seen = set()
    for _ in range(5):
        conn.execute("TRUNCATE jobs, executions, effects, stale_writes, job_transitions CASCADE")
        make_jobs(conn, 200)
        claims = [repo.claim_next(conn, "W", LEASE) for _ in range(200)]
        worker_c, api_c = new_conn(), new_conn()
        barrier = threading.Barrier(2)

        def complete():
            out = []
            for c in claims:
                barrier.wait()
                out.append(repo.finish(worker_c, c, "W", "succeeded", result={}))
            return out

        def cancel():
            out = []
            for c in claims:
                barrier.wait()
                out.append(repo.cancel_job(api_c, c.job_id).outcome)
            return out

        with ThreadPoolExecutor(2) as ex:
            f1, f2 = ex.submit(complete), ex.submit(cancel)
            finals, outcomes = f1.result(), f2.result()
        for c, final, outcome in zip(claims, finals, outcomes):
            n = effects(conn, c.job_id)
            if outcome == "cancel_requested":      # cancel committed first
                assert final == "cancelled" and n == 0
            else:                                   # completion committed first
                assert outcome == "too_late" and final == "succeeded" and n == 1
            seen.add(outcome)
    assert seen == {"cancel_requested", "too_late"}, f"race not exercised: only {seen}"


def test_cancel_running_job_whose_worker_died(conn):
    """cancel_requested on a job whose worker is dead: the reaper finalises it as
    cancelled instead of re-queueing it."""
    make_jobs(conn, 1)
    a = repo.claim_next(conn, "A", LEASE)
    assert repo.cancel_job(conn, a.job_id).outcome == "cancel_requested"
    expire_lease(conn, a.job_id)
    repo.reap_expired(conn)
    assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "cancelled"
    assert repo.claim_next(conn, "B", LEASE) is None


# ------------------------------------------------------------ 4. duplicate on retry
def test_concurrent_creates_same_key_make_one_job(conn, new_conn):
    """50 simultaneous POSTs with the same key (a client retrying on timeout)."""
    barrier = threading.Barrier(50)

    def create(_):
        c = new_conn()
        barrier.wait()
        return repo.create_job(c, "same-key", {"duration_ms": 5}, 5)

    with ThreadPoolExecutor(50) as ex:
        results = list(ex.map(create, range(50)))
    assert len({r.job["id"] for r in results}) == 1
    assert sum(r.created for r in results) == 1
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_same_key_different_body_is_a_conflict(conn):
    repo.create_job(conn, "k", {"duration_ms": 5}, 5)
    assert repo.create_job(conn, "k", {"duration_ms": 6}, 5).conflict


# --------------------------------------------------------------- 5. isolation level
def test_connections_are_read_committed(conn):
    assert conn.execute("SHOW transaction_isolation").fetchone()[0] == "read committed"


def test_claim_under_repeatable_read_breaks(conn, new_conn):
    """Why the isolation level is pinned: under REPEATABLE READ, a worker whose
    snapshot is older than another worker's claim gets a serialization error instead
    of moving on. (Correctness would survive, throughput and error rates would not.)"""
    make_jobs(conn, 1)
    rr = new_conn()
    rr.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    with rr.transaction():
        rr.execute("SELECT count(*) FROM jobs").fetchone()   # snapshot taken now
        repo.claim_next(conn, "other", LEASE)                 # another worker claims
        repo.finish(conn, repo.Claim(*conn.execute(
            "SELECT id, fencing_token, payload, attempts FROM jobs").fetchone()),
            "other", "succeeded", result={})
        # job is terminal; the RR worker still sees it as queued and tries to claim it
        with pytest.raises(psycopg.errors.SerializationFailure):
            repo.claim_next(rr, "rr-worker", LEASE)


# --------------------------------------------------------------------- 6. clock skew
def test_worker_clock_is_never_used(conn):
    """A worker whose clock is 1 hour ahead (or behind) behaves identically: leases
    are computed with the database's now(), workers only send durations."""
    make_jobs(conn, 2)
    for skew in (3600, -3600):
        with mock.patch("time.time", return_value=__import__("time").time() + skew):
            c = repo.claim_next(conn, "skewed", 5.0)
            left = conn.execute("SELECT lease_expires_at - now() FROM jobs WHERE id = %s",
                                (c.job_id,)).fetchone()[0].total_seconds()
            assert 4.0 < left <= 5.0
            assert repo.heartbeat(conn, c, "skewed", 5.0) is False
            assert repo.finish(conn, c, "skewed", "succeeded", result={}) == "succeeded"
