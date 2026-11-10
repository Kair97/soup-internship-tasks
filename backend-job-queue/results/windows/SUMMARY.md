# Chaos suite results (2026-11-10 17:04)

| run | injected bug | expected | result | kills | freezes | effects dup | stale writes rejected | jobs run >1 time | elapsed s |
|---|---|---|---|---|---|---|---|---|---|
| `run-20261110-163044-windows` | - | PASS | ✅ PASS | 80 | 16 | 0 | {'heartbeat': 16} | 49 | 175.7 |
| `run-20261110-163343-windows` | - | PASS | ✅ PASS | 118 | 23 | 0 | {'heartbeat': 20} | 72 | 249.1 |
| `run-20261110-163758-windows` | - | PASS | ✅ PASS | 180 | 35 | 0 | {'heartbeat': 28} | 83 | 381.2 |
| `run-20261110-164424-windows-inject-no_fencing` | no_fencing | FAIL | ✅ FAIL | 144 | 28 | 10 | {} | 74 | 304.0 |
| `run-20261110-164932-windows-inject-no_lock` | no_lock | FAIL | ✅ FAIL | 221 | 42 | 0 | {'failed': 40, 'heartbeat': 73, 'complete': 1403} | 1396 | 471.5 |
| `run-20261110-165731-windows-inject-ignore_cancel` | ignore_cancel | FAIL | ✅ FAIL | 186 | 37 | 0 | {'complete': 1, 'heartbeat': 22} | 67 | 398.0 |

## Violations reported

- `run-20261110-163044-windows`: none
- `run-20261110-163343-windows`: none
- `run-20261110-163758-windows`: none
- `run-20261110-164424-windows-inject-no_fencing`: DUPLICATE SIDE EFFECT: 10 jobs have more than one effect; SUCCEEDED WITHOUT EXACTLY ONE EFFECT: 10; ZOMBIE WRITE: 10 effects written with a superseded fencing token
- `run-20261110-164932-windows-inject-no_lock`: DOUBLE CLAIM: 1503 claims of a job that was already running
- `run-20261110-165731-windows-inject-ignore_cancel`: CANCEL IGNORED: 25 jobs had cancel accepted (200/202) but finished as Counter({'succeeded': 25})