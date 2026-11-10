"""Runtime settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str
    lease_seconds: float
    heartbeat_seconds: float
    max_attempts: int
    poll_seconds: float
    reaper_every_seconds: float
    # Fault injection. ONLY used by the chaos harness to prove that it detects bugs.
    # Empty in normal operation. Values: no_fencing | no_lock | ignore_cancel
    inject: str


def load() -> Settings:
    lease = _float("LEASE_SECONDS", 5.0)
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jobs"
        ),
        lease_seconds=lease,
        heartbeat_seconds=_float("HEARTBEAT_SECONDS", lease / 3),
        max_attempts=_int("MAX_ATTEMPTS", 5),
        poll_seconds=_float("POLL_SECONDS", 0.05),
        reaper_every_seconds=_float("REAPER_EVERY_SECONDS", 1.0),
        inject=os.environ.get("INJECT", "").strip(),
    )
