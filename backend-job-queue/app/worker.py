"""Worker process: claim -> execute (with heartbeats) -> finish, plus lease reaping.

Run:  python -m app.worker
Each process is one worker; run several processes (docker compose scales it to 4).
"""
from __future__ import annotations

import logging
import os
import random
import signal
import socket
import time
import uuid

import psycopg

from . import config, db, repo

log = logging.getLogger("worker")


class Worker:
    def __init__(self, settings: config.Settings, worker_id: str | None = None):
        self.s = settings
        self.id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.conn: psycopg.Connection | None = None
        self.stopping = False
        self._next_reap = 0.0  # monotonic time; only schedules work, never decides correctness

    # -- lifecycle ------------------------------------------------------------------
    def connect(self) -> None:
        self.conn = db.connect(self.s.database_url)

    def stop(self, *_: object) -> None:
        log.info("%s: stop requested, finishing current job", self.id)
        self.stopping = True

    def run_forever(self) -> None:
        self.connect()
        log.info("%s: started (lease=%ss heartbeat=%ss inject=%r)", self.id,
                 self.s.lease_seconds, self.s.heartbeat_seconds, self.s.inject)
        while not self.stopping:
            try:
                if not self.step():
                    time.sleep(self.s.poll_seconds * (0.5 + random.random()))
            except psycopg.OperationalError:
                log.exception("%s: database connection problem, reconnecting", self.id)
                time.sleep(1)
                self.connect()

    # -- one iteration ------------------------------------------------------------
    def step(self) -> bool:
        """Returns True if a job was processed."""
        assert self.conn is not None
        now = time.monotonic()
        if now >= self._next_reap:
            reaped = repo.reap_expired(self.conn)
            if reaped:
                log.info("%s: reaped %d expired lease(s)", self.id, reaped)
            self._next_reap = now + self.s.reaper_every_seconds
        claim = repo.claim_next(self.conn, self.id, self.s.lease_seconds, self.s.inject)
        if claim is None:
            return False
        self.execute(claim)
        return True

    def execute(self, claim: repo.Claim) -> None:
        """Emulated training run. The real work would happen here; the durable side
        effect is written by repo.finish() in the same transaction as the status."""
        assert self.conn is not None
        repo.record_execution(self.conn, claim, self.id)
        duration = max(0, int(claim.payload.get("duration_ms", 100))) / 1000.0
        deadline = time.monotonic() + duration
        next_beat = time.monotonic() + self.s.heartbeat_seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, max(0.0, next_beat - time.monotonic()), 0.25))
            if time.monotonic() >= next_beat:
                cancel = repo.heartbeat(self.conn, claim, self.id, self.s.lease_seconds,
                                        self.s.inject)
                if cancel is None:
                    log.warning("%s: lost lease on %s (token %s), abandoning",
                                self.id, claim.job_id, claim.token)
                    return
                if cancel:
                    repo.finish(self.conn, claim, self.id, "cancelled", inject=self.s.inject)
                    log.info("%s: cancelled %s mid-run", self.id, claim.job_id)
                    return
                next_beat = time.monotonic() + self.s.heartbeat_seconds

        if claim.payload.get("fail"):
            final = repo.finish(self.conn, claim, self.id, "failed",
                                error="simulated training failure", inject=self.s.inject)
        else:
            final = repo.finish(self.conn, claim, self.id, "succeeded",
                                result={"checkpoint": f"s3://checkpoints/{claim.job_id}",
                                        "attempt": claim.attempts},
                                inject=self.s.inject)
        if final is None:
            log.warning("%s: result for %s rejected, lease was taken over (stale token %s)",
                        self.id, claim.job_id, claim.token)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    settings = config.load()
    db.init_schema(settings.database_url)
    worker = Worker(settings)
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
