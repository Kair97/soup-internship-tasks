# Soup internship: take-home tasks

Kairzhan Orynbek · Astana · [github.com/Kair97](https://github.com/Kair97)

Soup asked for **one** of the tasks. Both were done. The **Backend Engineer** task is the submission;
the AI Engineer task is included as extra work.

| Task | Folder | Question | Short answer |
|---|---|---|---|
| **Backend Engineer** (submitted) | [`backend-job-queue/`](backend-job-queue/) | A training-job queue with several workers: will it survive production? | **DON'T SHIP yet.** 12 chaos runs (120,000 jobs, 1,133 `kill -9`, 220 frozen workers) gave 0 lost jobs and 0 duplicate effects, and all 12 planted-bug runs were caught. The failure is operational: no retention, and `/stats` takes 1.1–3.7 s after 8 h of 100 rps. [REPORT](backend-job-queue/REPORT.md) |
| AI Engineer | [`ai-finetune-verification/`](ai-finetune-verification/) | LoRA → merge → 4-bit: ship this artifact to users or not? | **DON'T SHIP.** The training is real: held-out NLL 1.435 → 1.017, and 4-bit keeps 93 % of the gain. No-op controls are rejected. But the model stopped refusing harmful requests: safety 0.925 → 0.125, with 33/40 refusals flipped, checked by reading the outputs. [REPORT](ai-finetune-verification/REPORT.md) |

Both follow the same rule from the briefs: **green happy-path results prove nothing.** Each check was first
pointed at a deliberately broken version, to show that it can fail. Every raw log is kept, including runs
where my own tooling was wrong.

## How AI tools were used (as the brief asks)

Most of the code and text were produced together with **Claude Code** (Anthropic's coding agent) working
on my laptop. I set the direction and made the decisions, the agent wrote most of the code, and **every
claim in the reports comes from a run on real hardware, whose raw output is in the repo.**
Places where the AI tool was wrong, and how it was caught, are part of the evidence:

| Where the tool was wrong | How it was caught | Kept as evidence |
|---|---|---|
| The first chaos harness said PASS against code with a planted zombie-write bug | running the harness against `--inject no_fencing` | `backend-job-queue/results/harness-development/` |
| …and against code that ignores cancels | `--inject ignore_cancel` | same folder |
| The "freeze a busy worker" fix still froze nothing: on Windows the venv `python.exe` is a launcher, so the PID it knew was not the worker's PID | 0 freezes in the log, then a direct PID experiment | same folder, README |
| The first cancel/complete race test never raced: cancel won 1000/1000 times | measuring the outcome distribution (now 974 / 26) | `tests/test_silent_failures.py` docstring |
| The first KL computation would have needed ~14 GB of RAM | reviewing the arithmetic before running | `scripts/verify_artifacts.py` |
| The data length filter silently did nothing (transformers 5 `apply_chat_template` returns a dict, `len()` = 2) | `train_tokens_mean: 2.0` in the manifest | `ai-finetune-verification/REPORT.md` §3 |
| First VRAM estimate wrong on 3 points (checkpointing, fp32 upcast, fp32 logits) | Soup's pre-flight refused the run with a 6.6 GB estimate | `results/memory_budget_v1_first_estimate.txt` |
| The merge check's fixed threshold would have rejected the correct merge; G1 also judged the base model; the memory probe ran in eval mode | numbers that looked wrong, then re-derived | `ai-finetune-verification/results/superseded/` |

> Note on raw logs: the local database login in the Windows run logs was replaced with `USER:PASS`.
> Nothing else in any log was edited.
