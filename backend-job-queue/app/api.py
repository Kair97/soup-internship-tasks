"""HTTP API: create (idempotent), get, list (keyset pagination), cancel, stats."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config, db, repo

Status = Literal["queued", "running", "succeeded", "failed", "cancelled"]
settings = config.load()
pool = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pool
    db.init_schema(settings.database_url)
    pool = db.make_pool(settings.database_url, size=int(os.environ.get("API_POOL_SIZE", 20)))
    yield
    pool.close()


app = FastAPI(title="Training-job queue", version="1.0.0", lifespan=lifespan)


class JobCreate(BaseModel):
    """What a training job would carry. `duration_ms` and `fail` drive the emulated
    execution so the chaos harness can control timing and failures."""
    model: str = Field(default="qwen2.5-1.5b", max_length=200)
    duration_ms: int = Field(default=100, ge=0, le=600_000)
    fail: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    id: UUID
    idempotency_key: str
    payload: dict[str, Any]
    status: Status
    cancel_requested: bool
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    result: dict[str, Any] | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobPage(BaseModel):
    items: list[Job]
    next_cursor: str | None


def _job(row: dict[str, Any]) -> Job:
    row = dict(row)
    row.pop("fencing_token", None)
    return Job(**row)


@app.post("/jobs", status_code=201, response_model=Job,
          responses={200: {"description": "Existing job returned for this Idempotency-Key"},
                     409: {"description": "Idempotency-Key reused with a different body"}})
def create_job(body: JobCreate, response: Response,
               idempotency_key: str = Header(..., min_length=1, max_length=200)):
    with pool.connection() as conn:
        res = repo.create_job(conn, idempotency_key, body.model_dump(), settings.max_attempts)
    if res.conflict:
        raise HTTPException(409, "Idempotency-Key was already used with a different request body")
    if not res.created:
        response.status_code = 200
    return _job(res.job)


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: UUID):
    with pool.connection() as conn:
        row = repo.get_job(conn, job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    return _job(row)


@app.get("/jobs", response_model=JobPage)
def list_jobs(limit: int = Query(50, ge=1, le=500), status: Status | None = None,
              cursor: str | None = None):
    try:
        with pool.connection() as conn:
            rows, nxt = repo.list_jobs(conn, limit, status, cursor)
    except ValueError:
        raise HTTPException(400, "invalid cursor")
    return JobPage(items=[_job(r) for r in rows], next_cursor=nxt)


@app.post("/jobs/{job_id}/cancel", response_model=Job,
          responses={202: {"description": "Running job: cancellation requested"},
                     409: {"description": "Job already finished (succeeded/failed)"}})
def cancel_job(job_id: UUID):
    with pool.connection() as conn:
        res = repo.cancel_job(conn, job_id)
    if res.outcome == "not_found":
        raise HTTPException(404, "job not found")
    if res.outcome == "too_late":
        return JSONResponse(status_code=409, content={
            "detail": f"job already {res.job['status']}", "status": res.job["status"]})
    code = 202 if res.outcome == "cancel_requested" else 200
    return JSONResponse(status_code=code, content=_job(res.job).model_dump(mode="json"))


@app.get("/stats")
def stats():
    with pool.connection() as conn:
        return repo.stats(conn)


@app.get("/healthz")
def healthz():
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}
