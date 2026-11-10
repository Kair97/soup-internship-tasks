-- Training-job queue schema (PostgreSQL 14+).
-- All time comparisons use the database clock (now()), never a worker's clock.

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() on PG < 13

CREATE TABLE IF NOT EXISTS jobs (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key  text        NOT NULL UNIQUE,
    payload          jsonb       NOT NULL,
    payload_hash     text        NOT NULL,
    status           text        NOT NULL DEFAULT 'queued'
                     CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    cancel_requested boolean     NOT NULL DEFAULT false,
    attempts         integer     NOT NULL DEFAULT 0,
    max_attempts     integer     NOT NULL DEFAULT 5,
    -- Incremented on every claim. Every write a worker makes is conditioned on it,
    -- so a worker whose lease was taken over can never write again (fencing).
    fencing_token    bigint      NOT NULL DEFAULT 0,
    lease_owner      text,
    lease_expires_at timestamptz,
    result           jsonb,
    error            text,
    created_at       timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at       timestamptz NOT NULL DEFAULT clock_timestamp(),
    started_at       timestamptz,
    finished_at      timestamptz,
    -- A job holds a lease if and only if it is running.
    CONSTRAINT lease_iff_running CHECK (
        (status = 'running') = (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
    ),
    -- Terminal jobs always have finished_at; non-terminal never do.
    CONSTRAINT finished_iff_terminal CHECK (
        (status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)
    )
);

-- Claim path: oldest queued job first.
CREATE INDEX IF NOT EXISTS jobs_queued_idx  ON jobs (created_at, id) WHERE status = 'queued';
-- Reaper path: expired leases.
CREATE INDEX IF NOT EXISTS jobs_lease_idx   ON jobs (lease_expires_at) WHERE status = 'running';
-- Keyset pagination.
CREATE INDEX IF NOT EXISTS jobs_list_idx    ON jobs (created_at, id);
CREATE INDEX IF NOT EXISTS jobs_status_list_idx ON jobs (status, created_at, id);

-- Every time a worker starts executing a job (at-least-once: may be > 1 per job).
CREATE TABLE IF NOT EXISTS executions (
    id            bigserial   PRIMARY KEY,
    job_id        uuid        NOT NULL REFERENCES jobs (id),
    fencing_token bigint      NOT NULL,
    worker_id     text        NOT NULL,
    started_at    timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS executions_job_idx ON executions (job_id);

-- The emulated side effect ("the model was trained and saved").
-- Deliberately NO unique constraint on job_id: the chaos harness must be able to
-- OBSERVE a duplicate effect if the fencing logic is wrong. In production you would
-- add UNIQUE (job_id) as defence in depth (see DESIGN.md).
CREATE TABLE IF NOT EXISTS effects (
    id            bigserial   PRIMARY KEY,
    job_id        uuid        NOT NULL REFERENCES jobs (id),
    fencing_token bigint      NOT NULL,
    worker_id     text        NOT NULL,
    written_at    timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS effects_job_idx ON effects (job_id);

-- Audit log of every status / ownership change, written by a trigger so it sees the
-- real OLD and NEW row (no snapshot games). The chaos harness replays it to find
-- illegal transitions such as running -> running with a new fencing token (= a
-- double claim) or any change to a job that was already terminal.
CREATE TABLE IF NOT EXISTS job_transitions (
    id            bigserial   PRIMARY KEY,
    job_id        uuid        NOT NULL,
    from_status   text        NOT NULL,
    to_status     text        NOT NULL,
    from_token    bigint      NOT NULL,
    to_token      bigint      NOT NULL,
    at            timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS job_transitions_job_idx ON job_transitions (job_id, id);

CREATE OR REPLACE FUNCTION log_job_transition() RETURNS trigger AS $$
BEGIN
    INSERT INTO job_transitions (job_id, from_status, to_status, from_token, to_token)
    VALUES (NEW.id, OLD.status, NEW.status, OLD.fencing_token, NEW.fencing_token);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_transition_audit ON jobs;
CREATE TRIGGER jobs_transition_audit
    AFTER UPDATE ON jobs
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status
          OR OLD.fencing_token IS DISTINCT FROM NEW.fencing_token)
    EXECUTE FUNCTION log_job_transition();

-- Writes that were rejected because the worker no longer owned the job
-- (zombie after lease expiry). Evidence that fencing is doing its job.
CREATE TABLE IF NOT EXISTS stale_writes (
    id            bigserial   PRIMARY KEY,
    job_id        uuid        NOT NULL,
    fencing_token bigint      NOT NULL,
    worker_id     text        NOT NULL,
    kind          text        NOT NULL,   -- heartbeat | complete | fail | cancel
    at            timestamptz NOT NULL DEFAULT clock_timestamp()
);
