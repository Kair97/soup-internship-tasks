# Design note: training-job queue

**Stack:** Python 3.12 · FastAPI · PostgreSQL 16/17 · psycopg 3. One API process, N worker processes, no broker.

## Storage: why Postgres, not SQLite
Workers are separate processes (and in production, separate machines) that compete for jobs.
Postgres gives row-level locks with `FOR UPDATE SKIP LOCKED`, so claims never block on or collide with
each other. It also gives **one authoritative clock**, `now()`, for every lease decision.
SQLite has a single database-wide write lock, no `SKIP LOCKED`, and only works on one machine.
That is fine for a demo, but the brief asks whether this survives production.

## How a worker claims a job
One statement, one transaction:
```sql
WITH next AS (SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id
              FOR UPDATE SKIP LOCKED LIMIT 1)
UPDATE jobs SET status='running', attempts=attempts+1, fencing_token=fencing_token+1,
               lease_owner=$worker, lease_expires_at=now()+$lease ...
FROM next WHERE jobs.id = next.id RETURNING id, fencing_token, payload;
```
A row locked by another claimer is skipped, not waited for. The claim returns a **fencing token**,
a counter bumped on every claim. **Every later write by that worker is conditioned on
`(id, fencing_token, status='running')`**: heartbeat, finish and cancel-finish. If the job was taken over,
the write matches 0 rows and the attempt is logged in `stale_writes`.

## How a dead worker is detected
A **lease** of 5 s (3 s in the chaos run), extended by a heartbeat every lease/3. Any worker runs the reaper
once a second: `running` jobs whose `lease_expires_at < now()` go back to `queued`. They go to `failed` once
`attempts = max_attempts`, and to `cancelled` if a cancel was requested. The reaper locks with
`SKIP LOCKED`, and Postgres re-checks the expiry on the newest row version after locking. A heartbeat
that committed first therefore keeps the job. A frozen worker that wakes up later (a **zombie**) finds its
token superseded: its heartbeat and its result are both rejected.

## What happens when a running job is cancelled
- `queued` jobs move to `cancelled` immediately.
- `running` jobs get `cancel_requested = true` and the API returns **202**. The worker sees the flag on its
  next heartbeat and finishes the job as `cancelled`. If the worker finishes first, the completion
  `UPDATE` itself turns the result into `cancelled` (`CASE WHEN cancel_requested`). If the worker is dead,
  the reaper finalises it as `cancelled` instead of re-queueing it.
- If the job is already `succeeded`/`failed`, the API returns **409** with the final status.

Cancel and completion both take the row lock, so they serialise: whichever commits first wins, and the
other sees a consistent state. A **202 always ends in `cancelled` with no side effect.**

## Idempotency
`INSERT … ON CONFLICT (idempotency_key) DO NOTHING`, then `SELECT` the existing row.
- A concurrent insert with the same key waits for the first one to commit, so all callers get the same job.
- Same key with a different body returns **409**.
- Keys are unique forever. A TTL would be a product decision.

## The guarantee: at-least-once execution, exactly-once effect
A worker can die after doing the work but before recording it, so **execution is at-least-once**. The run is
repeated after the lease expires. You can see this in the chaos runs: `executions` > jobs.

The **effect** (here a row in `effects`; for real training, "this checkpoint is the job's result") is written
**in the same transaction** as `running → succeeded`, and that transaction is fenced by the token. So the
effect commits **exactly once**, or not at all. Exactly-once *execution* is impossible without
coordination with the thing being executed. At-most-once would lose jobs whenever a worker dies, which
is worse for training jobs: re-running is expensive but recoverable, while a lost job is silent.

**Limit of the guarantee:** an effect *outside* Postgres (S3 upload, webhook, GPU bill) is still
at-least-once. It must be made idempotent by keying it on `(job_id, fencing_token)`, for example by writing
checkpoints to `…/job_id/attempt_token/` and letting only the fenced commit publish the pointer.

## Isolation level
`READ COMMITTED` is set explicitly on every connection.
- The design depends on its semantics: guarded `UPDATE`s re-evaluate their `WHERE` on the newest row
  version after waiting for a lock.
- Under `REPEATABLE READ`, a worker with an older snapshot gets serialization errors instead of moving on.
  `tests/test_silent_failures.py::test_claim_under_repeatable_read_breaks` shows this.
- Pinning the level means a server-wide default change cannot silently alter behaviour.

## Clock skew
Workers never send timestamps. All lease math is `now() + interval` on the database. Worker clocks only
schedule "when to heartbeat next", which uses `time.monotonic()`, so skew cannot affect correctness.
What remains is the DB's own clock jumping, for example an NTP step: see REPORT.md.

## Evidence built into the schema
- An `AFTER UPDATE` trigger writes every status/token change to `job_transitions`. The harness replays
  this log against the state machine to detect a double claim (`running → running` with a new token)
  or any change to a terminal job.
- `effects` deliberately has **no** `UNIQUE(job_id)`, so a duplicate effect is observable. In production you
  would add it as defence in depth.
