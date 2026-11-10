# Chaos suite results (2026-11-10 12:10)

| run | injected bug | expected | result | kills | freezes | effects dup | stale writes rejected | jobs run >1 time | elapsed s |
|---|---|---|---|---|---|---|---|---|---|
| `run-20261110-115146-linux` | - | PASS | ✅ PASS | 86 | 16 | 0 | {'complete': 1, 'heartbeat': 14} | 56 | 182.3 |
| `run-20261110-115451-linux` | - | PASS | ✅ PASS | 88 | 18 | 0 | {'heartbeat': 18} | 48 | 187.5 |
| `run-20261110-115800-linux` | - | PASS | ✅ PASS | 86 | 16 | 0 | {'heartbeat': 16} | 61 | 177.6 |
| `run-20261110-120100-linux-inject-no_fencing` | no_fencing | FAIL | ✅ FAIL | 86 | 18 | 6 | {} | 55 | 185.4 |
| `run-20261110-120408-linux-inject-no_lock` | no_lock | FAIL | ✅ FAIL | 88 | 18 | 0 | {'failed': 7, 'heartbeat': 34, 'complete': 571} | 627 | 188.5 |
| `run-20261110-120718-linux-inject-ignore_cancel` | ignore_cancel | FAIL | ✅ FAIL | 89 | 17 | 0 | {'heartbeat': 17} | 53 | 187.5 |

## Violations reported

- `run-20261110-115146-linux`: none
- `run-20261110-115451-linux`: none
- `run-20261110-115800-linux`: none
- `run-20261110-120100-linux-inject-no_fencing`: DUPLICATE SIDE EFFECT: 6 jobs have more than one effect; SUCCEEDED WITHOUT EXACTLY ONE EFFECT: 6; ZOMBIE WRITE: 6 effects written with a superseded fencing token
- `run-20261110-120408-linux-inject-no_lock`: DOUBLE CLAIM: 602 claims of a job that was already running
- `run-20261110-120718-linux-inject-ignore_cancel`: CANCEL IGNORED: 115 jobs had cancel accepted (200/202) but finished as Counter({'succeeded': 113, 'failed': 2})