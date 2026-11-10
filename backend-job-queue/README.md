# Training-job queue: will it survive production?

A job queue for training runs: idempotent create, status, paginated list, cancel,
`queued → running → succeeded / failed / cancelled`, PostgreSQL, 4+ workers. The main point is the
**evidence** that it holds under failure: a chaos harness that `kill -9`s and freezes workers while
duplicate POSTs and cancels race. The harness was proven against three deliberately planted bugs.

| Document | What is in it |
|---|---|
| [DESIGN.md](DESIGN.md) | one page: claim, dead-worker detection, cancel, guarantee, isolation, clock |
| [REPORT.md](REPORT.md) | harness results, silent failures and how each was checked, verdict, what surprised me |
| [results/](results/) | **raw output of every run**, including the failed ones |

## Run it

```bash
docker compose up --build            # Postgres + API on :8000 + 4 workers
curl -X POST localhost:8000/jobs -H 'Idempotency-Key: demo-1' \
     -H 'Content-Type: application/json' -d '{"model":"qwen2.5-1.5b","duration_ms":2000}'
curl localhost:8000/jobs/<id>
curl 'localhost:8000/jobs?status=succeeded&limit=50'      # next page: &cursor=<next_cursor>
curl -X POST localhost:8000/jobs/<id>/cancel
```

```bash
docker compose --profile test  run --rm test     # pytest: races, zombies, isolation, clock skew
docker compose --profile chaos run --rm chaos    # 3 chaos runs + 3 injected-bug runs (~25 min)
```

Without Docker (Python 3.12 + any PostgreSQL 14+):
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
export ADMIN_DATABASE_URL=postgresql://USER:PASS@localhost:5432/postgres
pytest -q
python -m chaos.harness --admin-url $ADMIN_DATABASE_URL                     # one run, 10k jobs
python -m chaos.harness --admin-url $ADMIN_DATABASE_URL --inject no_fencing # must FAIL
python -m chaos.run_suite --admin-url $ADMIN_DATABASE_URL --out results/mine
python -m chaos.load_test --admin-url $ADMIN_DATABASE_URL                   # 100/200/400 rps
```

## API

| Method | Path | Result |
|---|---|---|
| `POST` | `/jobs` + header `Idempotency-Key` | `201` new · `200` same key, same body (existing job) · `409` same key, different body |
| `GET` | `/jobs/{id}` | job, or `404` |
| `GET` | `/jobs?status=&limit=&cursor=` | keyset page + `next_cursor` |
| `POST` | `/jobs/{id}/cancel` | `200` cancelled (was queued, or already cancelled) · `202` running, cancel requested · `409` already succeeded/failed |
| `GET` | `/stats`, `/healthz` | counts per status, liveness |

Job body: `{"model": str, "duration_ms": int, "fail": bool, "params": {...}}`.
`duration_ms` and `fail` drive the emulated execution.

## Layout

```
app/schema.sql        tables, constraints, transition-audit trigger
app/repo.py           every SQL statement (claim, heartbeat, finish, reap, cancel, create, list)
app/worker.py         worker loop: reap -> claim -> execute with heartbeats -> fenced finish
app/api.py            FastAPI endpoints
chaos/harness.py      10k jobs, 8 workers, kill -9 ~2 s, freezes, duplicate POSTs, cancels, invariants
chaos/run_suite.py    N clean runs + injected-bug runs -> results/SUMMARY.md
chaos/load_test.py    open-loop load at 100/200/400 rps + worker capacity
tests/                targeted tests for each silent failure + HTTP contract
results/              raw logs of every run (windows/ = local, linux/ = GitHub Actions)
```
