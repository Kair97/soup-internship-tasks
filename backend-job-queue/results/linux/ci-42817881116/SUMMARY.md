# Chaos suite results (2026-11-10 12:59)

| run | injected bug | expected | result | kills | freezes | effects dup | stale writes rejected | jobs run >1 time | elapsed s |
|---|---|---|---|---|---|---|---|---|---|
| `run-20261110-124150-linux` | - | PASS | ✅ PASS | 81 | 16 | 0 | {'heartbeat': 15} | 51 | 174.5 |
| `run-20261110-124446-linux` | - | PASS | ✅ PASS | 82 | 16 | 0 | {'complete': 1, 'heartbeat': 15} | 56 | 171.2 |
| `run-20261110-124739-linux` | - | PASS | ✅ PASS | 80 | 16 | 0 | {'heartbeat': 16} | 53 | 174.8 |
| `run-20261110-125035-linux-inject-no_fencing` | no_fencing | FAIL | ✅ FAIL | 77 | 16 | 6 | {} | 43 | 168.2 |
| `run-20261110-125325-linux-inject-no_lock` | no_lock | FAIL | ✅ FAIL | 88 | 16 | 0 | {'failed': 18, 'heartbeat': 38, 'complete': 654} | 707 | 186.0 |
| `run-20261110-125632-linux-inject-ignore_cancel` | ignore_cancel | FAIL | ✅ FAIL | 83 | 16 | 0 | {'heartbeat': 15} | 48 | 176.5 |

## Violations reported

- `run-20261110-124150-linux`: none
- `run-20261110-124446-linux`: none
- `run-20261110-124739-linux`: none
- `run-20261110-125035-linux-inject-no_fencing`: DUPLICATE SIDE EFFECT: 6 jobs have more than one effect; SUCCEEDED WITHOUT EXACTLY ONE EFFECT: 6; ZOMBIE WRITE: 6 effects written with a superseded fencing token
- `run-20261110-125325-linux-inject-no_lock`: DOUBLE CLAIM: 702 claims of a job that was already running
- `run-20261110-125632-linux-inject-ignore_cancel`: CANCEL IGNORED: 130 jobs had cancel accepted (200/202) but finished as Counter({'succeeded': 127, 'failed': 3})