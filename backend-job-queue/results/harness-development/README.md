# Harness development runs (kept on purpose)

These are the runs made while building the harness. Several of them are **false
negatives**: the harness said PASS against code that had a bug planted on purpose.
They are kept because they show how the harness was proven, and why its final
version looks the way it does.

| run | harness version | planted bug | result | what it taught |
|---|---|---|---|---|
| `run-…162528` | v1, 500 jobs | none | PASS | smoke test |
| `run-…162557-inject-no_fencing` | v1 | zombie writes not fenced | **PASS = missed** | short jobs (10–150 ms), so a frozen worker almost never held a job |
| `run-…162623-inject-no_lock` | v1 | claim without row lock | FAIL (326 double claims) | the transition audit log catches double claims |
| `run-…162648-inject-ignore_cancel` | v1 | completion ignores cancel | **PASS = missed** | random cancels never hit a *running* job |
| `run-…162752-inject-no_fencing` | v2: long jobs, cancels of running jobs, wait for zombies | zombie | **PASS = missed** | 0 freezes happened: see next row |
| `run-…162828-inject-ignore_cancel` | v2 | completion ignores cancel | FAIL (18 cancels ignored) | targeting running jobs works |
| `run-…162934-inject-no_fencing` | v3: freeze the *real* worker PID | zombie | FAIL (4 duplicate effects, 5 stale-token effects) | on Windows the venv `python.exe` is a launcher; PIDs in `lease_owner` belong to its child, so v2 never found a "busy" worker to freeze |

The final suite (`results/windows/`, `results/linux/`) uses harness v3.
