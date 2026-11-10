"""All SQL lives here. Every function takes an open psycopg connection.

Rules this module follows (see DESIGN.md):
* Connections run at READ COMMITTED (Postgres default). The claim relies on
  FOR UPDATE SKIP LOCKED, and every guarded UPDATE re-checks its WHERE clause on the
  newest row version after taking the row lock.
* Every write a worker makes after claiming is conditioned on
  (id, fencing_token, status = 'running'). If the lease was taken over, the write
  matches 0 rows and is recorded in stale_writes instead.
* Time is always the database clock (now()). Workers never send timestamps.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

TERMINAL = ("succeeded", "failed", "cancelled")

JOB_COLUMNS = """
    id, idempotency_key, payload, status, cancel_requested, attempts, max_attempts,
    fencing_token, lease_owner, lease_expires_at, result, error,
    created_at, updated_at, started_at, finished_at
"""


def payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# --------------------------------------------------------------------------- API side

@dataclass
class CreateResult:
    job: dict[str, Any]
    created: bool          # False = an existing job was returned for this key
    conflict: bool = False  # True = same key, different payload


def create_job(conn: psycopg.Connection, key: str, payload: dict[str, Any],
               max_attempts: int) -> CreateResult:
    """Idempotent create. Concurrent calls with the same key return the same job.

    INSERT .. ON CONFLICT DO NOTHING waits for a concurrent inserter of the same key to
    commit or roll back, so the follow-up SELECT (new snapshot under READ COMMITTED)
    always sees the winner's row.
    """
    h = payload_hash(payload)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            INSERT INTO jobs (idempotency_key, payload, payload_hash, max_attempts)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING {JOB_COLUMNS}
            """,
            (key, Jsonb(payload), h, max_attempts),
        )
        row = cur.fetchone()
        if row is not None:
            return CreateResult(job=row, created=True)
        cur.execute(
            f"SELECT {JOB_COLUMNS}, payload_hash FROM jobs WHERE idempotency_key = %s", (key,)
        )
        existing = cur.fetchone()
        assert existing is not None, "ON CONFLICT fired but the row is not visible"
        stored_hash = existing.pop("payload_hash")
        return CreateResult(job=existing, created=False, conflict=stored_hash != h)


def get_job(conn: psycopg.Connection, job_id: UUID) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {JOB_COLUMNS} FROM jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def encode_cursor(row: dict[str, Any]) -> str:
    raw = f"{row['created_at'].isoformat()}|{row['id']}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    created_at, job_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
    UUID(job_id)  # validate
    return created_at, job_id


def list_jobs(conn: psycopg.Connection, limit: int, status: str | None,
              after: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Keyset pagination on (created_at, id): stable while new jobs keep arriving,
    and O(limit) per page instead of OFFSET's O(offset + limit)."""
    clauses, params = [], []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if after:
        created_at, job_id = decode_cursor(after)
        clauses.append("(created_at, id) > (%s::timestamptz, %s::uuid)")
        params += [created_at, job_id]
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {JOB_COLUMNS} FROM jobs {where} ORDER BY created_at, id LIMIT %s",
            (*params, limit + 1),
        )
        rows = cur.fetchall()
    next_cursor = encode_cursor(rows[limit - 1]) if len(rows) > limit else None
    return rows[:limit], next_cursor


@dataclass
class CancelResult:
    job: dict[str, Any] | None
    outcome: str  # cancelled | cancel_requested | already_cancelled | too_late | not_found


def cancel_job(conn: psycopg.Connection, job_id: UUID) -> CancelResult:
    """queued -> cancelled immediately. running -> cancel_requested; the worker (or the
    reaper, if the worker died) finalises it as cancelled and no effect is written.
    The UPDATE takes the row lock, so it serialises with the worker's completion:
    whichever commits first wins, and the loser sees a consistent state."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            UPDATE jobs SET
                status           = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                cancel_requested = CASE WHEN status = 'running' THEN true ELSE cancel_requested END,
                finished_at      = CASE WHEN status = 'queued' THEN now() ELSE finished_at END,
                updated_at       = now()
            WHERE id = %s AND status IN ('queued', 'running')
            RETURNING {JOB_COLUMNS}
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if row is not None:
            outcome = "cancelled" if row["status"] == "cancelled" else "cancel_requested"
            return CancelResult(job=row, outcome=outcome)
        cur.execute(f"SELECT {JOB_COLUMNS} FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    if row is None:
        return CancelResult(job=None, outcome="not_found")
    if row["status"] == "cancelled":
        return CancelResult(job=row, outcome="already_cancelled")
    return CancelResult(job=row, outcome="too_late")


def stats(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM jobs GROUP BY status")
        counts = {s: 0 for s in ("queued", "running", *TERMINAL)}
        counts.update({s: n for s, n in cur.fetchall()})
    return counts


# ------------------------------------------------------------------------ worker side

@dataclass
class Claim:
    job_id: UUID
    token: int
    payload: dict[str, Any]
    attempts: int


def claim_next(conn: psycopg.Connection, worker_id: str, lease_s: float,
               inject: str = "") -> Claim | None:
    """Atomically take the oldest queued job. SKIP LOCKED means concurrent workers
    never block on, or double-claim, the same row."""
    with conn.cursor() as cur:
        if inject == "no_lock":
            # BUG ON PURPOSE (harness self-test): read without a lock, then update
            # without re-checking the status. Two workers can claim the same job.
            cur.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            )
            found = cur.fetchone()
            if found is None:
                return None
            cur.execute(
                """
                UPDATE jobs SET status = 'running', attempts = attempts + 1,
                       fencing_token = fencing_token + 1, lease_owner = %s,
                       lease_expires_at = now() + make_interval(secs => %s),
                       started_at = coalesce(started_at, now()), updated_at = now()
                WHERE id = %s
                RETURNING id, fencing_token, payload, attempts
                """,
                (worker_id, lease_s, found[0]),
            )
        else:
            cur.execute(
                """
                WITH next AS (
                    SELECT id FROM jobs
                    WHERE status = 'queued'
                    ORDER BY created_at, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE jobs j SET
                    status = 'running',
                    attempts = j.attempts + 1,
                    fencing_token = j.fencing_token + 1,
                    lease_owner = %s,
                    lease_expires_at = now() + make_interval(secs => %s),
                    started_at = coalesce(j.started_at, now()),
                    updated_at = now()
                FROM next
                WHERE j.id = next.id
                RETURNING j.id, j.fencing_token, j.payload, j.attempts
                """,
                (worker_id, lease_s),
            )
        row = cur.fetchone()
    if row is None:
        return None
    return Claim(job_id=row[0], token=row[1], payload=row[2], attempts=row[3])


def record_execution(conn: psycopg.Connection, claim: Claim, worker_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO executions (job_id, fencing_token, worker_id) VALUES (%s, %s, %s)",
            (claim.job_id, claim.token, worker_id),
        )


def _record_stale(conn: psycopg.Connection, claim: Claim, worker_id: str, kind: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stale_writes (job_id, fencing_token, worker_id, kind) "
            "VALUES (%s, %s, %s, %s)",
            (claim.job_id, claim.token, worker_id, kind),
        )


def heartbeat(conn: psycopg.Connection, claim: Claim, worker_id: str, lease_s: float,
              inject: str = "") -> bool | None:
    """Extend the lease. Returns cancel_requested, or None if the lease was lost."""
    guard = "id = %s" if inject == "no_fencing" else \
        "id = %s AND fencing_token = %s AND status = 'running'"
    params: tuple = (claim.job_id,) if inject == "no_fencing" else (claim.job_id, claim.token)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE jobs SET lease_expires_at = now() + make_interval(secs => %s),
                            updated_at = now()
            WHERE {guard}
            RETURNING cancel_requested
            """,
            (lease_s, *params),
        )
        row = cur.fetchone()
    if row is None:
        _record_stale(conn, claim, worker_id, "heartbeat")
        return None
    return bool(row[0])


def finish(conn: psycopg.Connection, claim: Claim, worker_id: str, outcome: str,
           result: dict[str, Any] | None = None, error: str | None = None,
           inject: str = "") -> str | None:
    """Finish a claimed job in ONE transaction together with its side effect.

    outcome: 'succeeded' | 'failed' | 'cancelled'.
    If the job has cancel_requested, the final status is 'cancelled' (no effect).
    Returns the final status, or None if this worker no longer owned the job.
    """
    if inject == "ignore_cancel":
        status_expr = "%(outcome)s"                       # BUG ON PURPOSE
    else:
        # The user's cancel always wins over the worker's outcome; an error message
        # (if any) is still kept in the error column.
        status_expr = "CASE WHEN cancel_requested THEN 'cancelled' ELSE %(outcome)s END"
    if inject == "no_fencing":
        guard = "id = %(id)s"                             # BUG ON PURPOSE
    else:
        guard = "id = %(id)s AND fencing_token = %(token)s AND status = 'running'"

    params = {"id": claim.job_id, "token": claim.token, "outcome": outcome,
              "result": Jsonb(result) if result is not None else None, "error": error}
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE jobs SET
                    status = {status_expr},
                    result = %(result)s,
                    error = %(error)s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    finished_at = now(),
                    updated_at = now()
                WHERE {guard}
                RETURNING status
                """,
                params,
            )
            row = cur.fetchone()
            if row is None:
                final = None
            else:
                final = row[0]
                if final == "succeeded":
                    # The side effect commits atomically with the state change:
                    # either both happen or neither does.
                    cur.execute(
                        "INSERT INTO effects (job_id, fencing_token, worker_id) "
                        "VALUES (%s, %s, %s)",
                        (claim.job_id, claim.token, worker_id),
                    )
    if final is None:
        _record_stale(conn, claim, worker_id, "complete" if outcome == "succeeded" else outcome)
    return final


def reap_expired(conn: psycopg.Connection, limit: int = 200) -> int:
    """Recover jobs whose lease expired (worker died or froze).

    cancel requested -> cancelled; attempts exhausted -> failed; otherwise -> queued.
    FOR UPDATE SKIP LOCKED + re-checked WHERE: a heartbeat that commits first wins."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH expired AS (
                SELECT id FROM jobs
                WHERE status = 'running' AND lease_expires_at < now()
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE jobs j SET
                status = CASE
                    WHEN j.cancel_requested THEN 'cancelled'
                    WHEN j.attempts >= j.max_attempts THEN 'failed'
                    ELSE 'queued' END,
                error = CASE
                    WHEN j.cancel_requested THEN NULL
                    WHEN j.attempts >= j.max_attempts
                        THEN 'lease expired on all ' || j.attempts || ' attempts'
                    ELSE j.error END,
                finished_at = CASE
                    WHEN j.cancel_requested OR j.attempts >= j.max_attempts THEN now()
                    ELSE NULL END,
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = now()
            FROM expired
            WHERE j.id = expired.id
            """,
            (limit,),
        )
        return cur.rowcount
