"""Connection helpers and schema bootstrap."""
from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _configure(conn: psycopg.Connection) -> None:
    # Explicit, so a server-wide default_transaction_isolation change cannot silently
    # break the claim / guarded-update logic (see DESIGN.md, "Isolation level").
    conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED")


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True)
    _configure(conn)
    return conn


def make_pool(url: str, size: int = 10) -> ConnectionPool:
    return ConnectionPool(
        url, min_size=2, max_size=size, open=True,
        kwargs={"autocommit": True}, configure=_configure,
    )


def init_schema(url: str) -> None:
    with connect(url) as conn:
        # Serialise concurrent bootstraps (API + several workers starting together).
        conn.execute("SELECT pg_advisory_lock(424242)")
        try:
            # Only bootstrap once: re-running the DDL (DROP/CREATE TRIGGER) would take an
            # ACCESS EXCLUSIVE lock on jobs every time a worker restarts.
            exists = conn.execute("SELECT to_regclass('public.stale_writes')").fetchone()[0]
            if exists is None:
                conn.execute(SCHEMA)
        finally:
            conn.execute("SELECT pg_advisory_unlock(424242)")
