# Task Queue

Status values: `ready`, `active`, `review`, `blocked`, `done`.

| ID | Priority | Status | Owner | Task | Completion check |
| --- | --- | --- | --- | --- | --- |
| T-001 | P0 | active | Codex | Add shared coordination documents and GitHub CI. | Local validation and tests run; workflow file is reviewable. |
| T-002 | P0 | ready | Claude | Review project boundaries, CI, risks, and task ordering. | Findings recorded in `CLAUDE_FEEDBACK.md`. |
| T-003 | P0 | ready | Codex | Establish a clean baseline for all existing tests. | Failures are fixed or documented with owners. |
| T-004 | P0 | blocked | User | Provide or approve a source for real WM-811K data. | Dataset path and license/provenance are documented. |
| T-005 | P1 | ready | Codex | Make wafer-map ingestion and baseline classification reproducible. | Real-data path works; demo fallback is clearly labeled. |
| T-006 | P1 | ready | Claude | Review wafer-map leakage, split strategy, labels, and metrics. | Review covers lot/wafer grouping and class imbalance. |
| T-007 | P1 | ready | Codex | Define and implement an equipment anomaly baseline. | Time-aware split, anomaly score, threshold, and report exist. |
| T-008 | P1 | ready | Codex | Integrate the three analysis outputs into one program view. | Missing data and unrelated IDs are handled honestly. |
| T-009 | P2 | ready | Codex | Add model profiling for size, latency, memory, and operation proxies. | Repeated measurements produce machine-readable results. |
| T-010 | P2 | blocked | User | Select CPU/GPU/NPU target hardware and runtimes. | Device matrix and measurement protocol are approved. |
| T-011 | P2 | ready | Codex and Claude | Design the cloud agent bridge and review loop. | State machine, budgets, stop conditions, and permissions are documented. |

## Immediate order

1. Finish T-001 and T-003.
2. Ask Claude to perform T-002 without editing implementation files.
3. Resolve T-004 while Codex improves the demo/reproducibility path.
4. Continue T-005 through T-008.
5. Start hardware profiling and the full agent bridge only after the analysis baselines are stable.
