# Report: will the job queue survive production?

**Verdict at 100 rps: DON'T SHIP yet.** The queue *semantics* held under every failure I could produce:
90,000 jobs, 890 `kill -9`, 172 freezes, 0 lost jobs, 0 duplicate effects. The 100 rps request path measured
p99 = 45 ms. What fails is **operations, within days**: nothing ever deletes jobs, and `/stats` already
takes 1.1–3.7 s after the equivalent of 8 hours at 100 rps. Fix list at the end.

## 1. Chaos harness: what it does and the raw results

`chaos/harness.py`, per run:
- **Load:** 10,000 jobs submitted over HTTP while 8 worker processes run.
- **Kills:** a random worker is hard-killed about every 2 s (SIGKILL on Linux, `TerminateProcess` on Windows), then a
  supervisor restarts it.
- **Freezes:** about every 10 s, a worker that **currently holds a job** is suspended for 2 × lease + 1 s, then
  resumed as a zombie.
- **Duplicates:** 15 % of keys are POSTed twice simultaneously and 10 % again after 0.1–5 s.
- **Cancels:** 10 % of jobs are cancelled at random moments, plus a canceller that targets jobs **running right
  now**.
- **Job mix:** 3 % of jobs are long (0.5–3 s, so they heartbeat), 2 % fail on purpose.

After the drain, and after every zombie has woken up and had a grace period, invariants are checked
**in Postgres**:
1. **Jobs:** exactly one job per key, and every id a client saw exists.
2. **Nothing stuck:** no job left `queued` or `running`.
3. **Effects:** `succeeded` ⇔ exactly 1 effect, all other states 0, and every effect carries the job's final fencing token.
4. **Transitions:** a replay of the trigger-written transition log against the state machine finds no illegal
   transition and no `running → running` claim.
5. **Cancels:** every accepted cancel (200/202) ended `cancelled`, and every 409 matches the final state.

| where | runs (10k jobs each) | result | kills | freezes | zombie writes rejected | requeued after lease expiry | jobs executed >1× |
|---|---|---|---|---|---|---|---|
| Windows 11, PG 17 ([results/windows](results/windows/SUMMARY.md)) | 3 | 3 × PASS | 378 | 74 | 64 | 215 | 204 |
| Linux (GitHub Actions, Docker, PG 16), 2 CI runs ([results/linux](results/linux)) | 6 | 6 × PASS | 512 | 98 | 96 | 328 | 327 |
| **total** | **9 = 90,000 jobs** | **0 violations** | **890** | **172** | **160** | **543** | **531** (max 3 runs of one job) |

Across those runs: 22,590 duplicate POSTs and 9,065 accepted cancels (2,195 of them aimed at running jobs).

**Is the harness able to fail?** Each run is repeated against three bugs planted with `--inject`. All 9 such runs
FAILED, as they must:

| planted bug | Windows | Linux (2 runs) |
|---|---|---|
| `no_fencing`: zombie writes not checked against the token | 10 duplicate effects | 6 / 9 |
| `no_lock`: claim without `FOR UPDATE SKIP LOCKED` | 1,503 double claims | 602 / 701 |
| `ignore_cancel`: completion ignores `cancel_requested` | 25 cancels ignored | 115 / 127 |

The first two versions of the harness **said PASS on two of these bugs.** The zombie case never happened
because frozen workers were idle, and cancels never hit running jobs. Then a Windows launcher-PID problem
meant 0 freezes. Those false negatives are kept in [results/harness-development](results/harness-development/README.md).
Note: under `no_lock`, fencing still prevented every duplicate *effect*. The double claims only wasted work,
which is the design's defence in depth working.

**What the harness does NOT catch**
- Postgres itself dying, failover or a replica promotion, a full disk, and a network partition between a worker
  and the DB mid-COMMIT. All of these are untested.
- Effects **outside** the database (S3, GPU, webhooks). The guarantee covers only the in-DB effect.
- Long-run problems: bloat, the growth of `job_transitions`/`executions`, memory leaks, connection-pool exhaustion.
  Each run lasts 3–8 minutes.
- More than one API replica, a load balancer, client retries with backoff storms, and fairness or starvation.
- A zombie waking up *exactly* between computing its result and committing it. In the clean runs zombies were
  always caught at their next heartbeat (all 160 rejections are `heartbeat`). The `complete` rejection path is
  covered deterministically by `tests/test_silent_failures.py::test_zombie_*`.
- Rare interleavings in general. Repetition reduces, but cannot remove, the chance of missing them.

## 2. Silent failures: how each one was checked

| failure (no error is raised) | defence in the code | evidence |
|---|---|---|
| **double claim** | one-statement claim with `FOR UPDATE SKIP LOCKED`; the transition trigger logs every claim | `test_no_double_claim_under_concurrency` (16 threads, 400 jobs, 400 unique claims); planted `no_lock` caught (602–1,503 double claims) |
| **zombie after lease expiry writes a stale result** | fencing token on every write; the reaper re-checks expiry after locking | `test_zombie_after_lease_expiry_cannot_write`, `test_zombie_wakes_before_takeover_finishes`; 160 heartbeat rejections in chaos; planted `no_fencing` caught |
| **cancel / complete race** | both take the row lock; completion does `CASE WHEN cancel_requested THEN 'cancelled'` | `test_cancel_complete_race`: same row, same instant, 1,000 times. Both orders occur (974 cancel-first / 26 complete-first) and both are consistent. My first version of this test **never raced** (cancel won 1000/1000), so it now asserts that both orders were seen |
| **duplicate on request retry** | `ON CONFLICT (key) DO NOTHING`, then `SELECT`; same key with another body gets 409 | `test_concurrent_creates_same_key_make_one_job` (50 simultaneous); 22,590 duplicate POSTs in chaos, 0 keys with >1 id |
| **transaction isolation level** | `READ COMMITTED` pinned on every connection | `test_claim_under_repeatable_read_breaks`: under `REPEATABLE READ` a stale-snapshot claim raises a serialization error |
| **clock skew** | workers send no timestamps; all lease math is `now()` in the DB | `test_worker_clock_is_never_used`: ±1 h worker skew, lease still 5 s. **Not covered:** the DB clock itself jumping (an NTP step). A backward step lengthens leases; a forward step expires every lease at once, causing mass re-execution with 0 duplicate effects |
| heartbeat racing the reaper | the reaper locks, then re-evaluates expiry | `test_heartbeat_racing_reaper` (200 jobs, both orders) |
| dead worker holding a cancel-requested job | the reaper finalises it as `cancelled`, not `queued` | `test_cancel_running_job_whose_worker_died` |
| tests passing locally only by accident | — | CI failed with `ModuleNotFoundError: app`, because I had only ever run `python -m pytest`. Fixed with `pytest.ini` (commit history) |

## 3. 100 requests per second

Measured with `chaos/load_test.py`. The load is **open-loop**: it fires on schedule whatever the latency. Raw files are in [results/windows/load](results/windows/load).

| | p50 | p99 | errors |
|---|---|---|---|
| `POST /jobs` at **100 rps**, 8 workers running, 1 API process | 10.5 ms | **45 ms** | 0 |
| `POST /jobs` at 200 / 400 rps, 1 API process | 9.8 s | 63 s | 0 |
| same at 200 rps with **4 API processes** | 7.9 s | 44 s | 0 |
| control: `GET /healthz` (no DB write) at 200 rps | 1.6 s | 25 s | 0 |
| worker capacity, 8 workers, 0 ms jobs | **2,250 jobs/s** | | |

The knee sits between 150 and 200 rps whatever the API does: 4× the processes don't move it, and neither
does an empty endpoint. **So it is my load generator, a single Python process on Windows, and not the service.**
This means the service's own ceiling above about 150 rps is *unmeasured*. Claiming headroom I did not measure
would be wrong, so I don't.

**Growth** ([results/windows/growth](results/windows/growth/growth-3M.log)): 3,000,000 finished jobs, which is
8.3 h at 100 rps and 1 GB of table:

| operation | median | max |
|---|---|---|
| claim | 0.9 ms | 24 ms |
| create | 0.8 ms | 8.6 ms |
| list | 2.1 ms | 78 ms |
| **`/stats`** (`count(*) GROUP BY status`, a full scan) | **1,115 ms** | **3,657 ms** |

At 100 rps the table grows by about 8.6 M rows and ≈3 GB a day, plus 2–3 audit rows per job, and none of it is ever deleted.

**Why DON'T SHIP, and what to fix first:**
1. **Retention:** partition `jobs`, `executions`, `effects` and `job_transitions` by day; archive or drop terminal
   jobs after N days. Replace `/stats` with counters kept in a small table updated in the same transactions, or
   with a partial-index count on non-terminal states only.
2. **Fence the real side effect:** checkpoints under `…/{job_id}/{fencing_token}/`, with only the fenced commit
   publishing the pointer. Without this, a zombie duplicates GPU work *and* writes.
3. **Observability before load:** export lease expiries, stale-write rejections, queue age and claim latency
   as metrics with alerts. Every silent failure above shows up there first.
4. Then: replace polling (8 workers × 20 idle polls/s) with `LISTEN/NOTIFY` plus a fallback poll, run more than one
   API replica, and add auth and rate limiting.

With 1 and 3 done, I would ship it at 100 rps.

## 4. What surprised me, and what still concerns me

**Surprises.** How easily a test can look green while testing nothing. Three of my own checks were vacuous
the first time: the harness missed two planted bugs, the race test never raced, and the freeze hit the wrong
PID. Each one was only caught by pointing the check at a version that *must* fail. I was also surprised that
`/stats` (one innocent `count(*)`) becomes the first production problem, not the concurrency logic.

**Concerns.** Exactly-once applies only to the in-database effect; the expensive part of a training job sits
outside it. Every failure scenario here is one that I imagined. Postgres failover and network partitions are
untested, and the 100 rps conclusion rests on one laptop whose load generator saturates at about 150 rps.

## 5. Short answers to the live-discussion questions (the discussion was cancelled)

- **A worker froze for 5 minutes and came back:** its lease expired after 5 s and the reaper re-queued the job
  (`attempts + 1`, new token), so another worker ran it. On waking, the zombie's next heartbeat or result
  matches 0 rows, is logged in `stale_writes`, and the zombie abandons the job: 160 such events were observed.
  The cost is **duplicate execution** (wasted GPU time). The effect is not duplicated.
- **Weakest point:** side effects outside Postgres, then retention. Then the single Postgres instance.
- **First to break at 10×** (1,000 rps): `/stats` and table growth (×10, so 30 M rows/day); then idle polling
  and connection counts (each worker holds a connection; 10× workers means `max_connections`); then a single
  API process.
- **Is 10,000 jobs enough?** For the planted bugs, by a wide margin: every one was caught in every one of
  its 3 runs, with 6 to 1,503 violations each. For bugs whose window is far rarer than "a zombie with a held
  job", no fixed number is enough. That is why the deterministic tests force those interleavings instead of
  waiting for them.
