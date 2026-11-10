# Chaos suite results (2026-11-10 12:15)

| run | injected bug | expected | result | kills | freezes | effects dup | stale writes rejected | jobs run >1 time | elapsed s |
|---|---|---|---|---|---|---|---|---|---|
| `run-20261110-115720-linux` | - | PASS | ✅ PASS | 85 | 16 | 0 | {'heartbeat': 16} | 58 | 176.2 |
| `run-20261110-120018-linux` | - | PASS | ✅ PASS | 85 | 16 | 0 | {'heartbeat': 16} | 58 | 178.8 |
| `run-20261110-120319-linux` | - | PASS | ✅ PASS | 82 | 16 | 0 | {'heartbeat': 15} | 46 | 175.2 |
| `run-20261110-120616-linux-inject-no_fencing` | no_fencing | FAIL | ✅ FAIL | 86 | 18 | 9 | {} | 60 | 184.6 |
| `run-20261110-120923-linux-inject-no_lock` | no_lock | FAIL | ✅ FAIL | 91 | 19 | 0 | {'failed': 12, 'heartbeat': 30, 'complete': 672} | 709 | 191.4 |
| `run-20261110-121237-linux-inject-ignore_cancel` | ignore_cancel | FAIL | ✅ FAIL | 85 | 17 | 0 | {'heartbeat': 17} | 63 | 182.0 |

## Violations reported

- `run-20261110-115720-linux`: none
- `run-20261110-120018-linux`: none
- `run-20261110-120319-linux`: none
- `run-20261110-120616-linux-inject-no_fencing`: DUPLICATE SIDE EFFECT: 9 jobs have more than one effect; SUCCEEDED WITHOUT EXACTLY ONE EFFECT: 9; ZOMBIE WRITE: 9 effects written with a superseded fencing token
- `run-20261110-120923-linux-inject-no_lock`: DOUBLE CLAIM: 701 claims of a job that was already running
- `run-20261110-121237-linux-inject-ignore_cancel`: CANCEL IGNORED: 127 jobs had cancel accepted (200/202) but finished as Counter({'succeeded': 123, 'failed': 4})