# Task Queue

Status values: `ready`, `active`, `review`, `blocked`, `done`.

| ID | Priority | Status | Owner | Task | Completion check |
| --- | --- | --- | --- | --- | --- |
| T-001 | P0 | done | Codex | Add shared coordination documents and GitHub CI. | Local validation and tests run; workflow file is reviewable. |
| T-002 | P0 | done | Claude | Review project boundaries, CI, risks, and task ordering. | Findings recorded in `CLAUDE_FEEDBACK.md` (verdict: approve with follow-up). |
| T-003 | P0 | done | Codex | Establish a clean baseline for all existing tests. | Failures are fixed or documented with owners. |
| T-004 | P0 | blocked | User | Provide or approve a source for real WM-811K data. | Dataset path and license/provenance are documented. |
| T-005 | P1 | ready | Codex | Make wafer-map ingestion and baseline classification reproducible. | Real-data path works; demo fallback is clearly labeled. |
| T-006 | P1 | ready | Claude | Review wafer-map leakage, split strategy, labels, and metrics. | Review covers lot/wafer grouping and class imbalance. |
| T-007 | P1 | done | Codex | Define and implement an equipment anomaly baseline. | Time-aware split, anomaly score, threshold, and report exist. |
| T-008 | P1 | done | Codex | Integrate the three analysis outputs into one program view. | Missing data and unrelated IDs are handled honestly. |
| T-009 | P2 | ready | Codex | Add model profiling for size, latency, memory, and operation proxies. | Repeated measurements produce machine-readable results. |
| T-010 | P2 | blocked | User | Select CPU/GPU/NPU target hardware and runtimes. | Device matrix and measurement protocol are approved. |
| T-011 | P2 | ready | Codex and Claude | Design the cloud agent bridge and review loop. | State machine, budgets, stop conditions, and permissions are documented. |
| T-012 | P1 | ready | Codex | Migrate the wafer-map CNN and autoencoder from TensorFlow to PyTorch (per D-009). | PyTorch model trains on demo data; CI stays green; ONNX export path works. |

## Immediate order

1. T-001, T-002, and T-003 are complete. The current suite passes 81 tests with 3 optional-dependency skips.
2. T-007 is complete in `b134b66`: time-aware CPU equipment anomaly baseline and machine-readable outputs.
3. Claude approved T-007 and T-008; minor follow-ups are recorded in `CLAUDE_FEEDBACK.md`.
4. Resolve T-004 by documenting a licensed WM-811K source and local dataset path.
5. Continue T-005 and T-006 once real wafer data is available; enforce grouped splits from D-011.
6. Install or provide PyTorch and ONNX Runtime before starting T-012.
7. Start T-009 profiling after model artifacts are stable; keep NPU measurements blocked until target hardware exists.
8. Merge `codex/collaboration-baseline` to `main` only after review and user approval.
