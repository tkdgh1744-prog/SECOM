# Task Queue

Status values: `ready`, `active`, `review`, `blocked`, `done`.

| ID | Priority | Status | Owner | Task | Completion check |
| --- | --- | --- | --- | --- | --- |
| T-001 | P0 | done | Codex | Add shared coordination documents and GitHub CI. | Local validation and tests run; workflow file is reviewable. |
| T-002 | P0 | done | Claude | Review project boundaries, CI, risks, and task ordering. | Findings recorded in `CLAUDE_FEEDBACK.md` (verdict: approve with follow-up). |
| T-003 | P0 | done | Codex | Establish a clean baseline for all existing tests. | Failures are fixed or documented with owners. |
| T-004 | P0 | blocked | User | Approve a licensed WM-811K source and exact cloud dataset path; cloud-only storage is already accepted. | Dataset path and license/provenance are documented. |
| T-005 | P1 | active | Codex | Make wafer-map ingestion and baseline classification reproducible. | Cloud preflight validates provenance, labels, groups, maps, and checksum; final real-data run remains gated on T-004. |
| T-006 | P1 | ready | Claude | Review wafer-map leakage, split strategy, labels, and metrics. | Review covers lot/wafer grouping and class imbalance. |
| T-007 | P1 | done | Codex | Define and implement an equipment anomaly baseline. | Time-aware split, anomaly score, threshold, and report exist. |
| T-008 | P1 | done | Codex | Integrate the three analysis outputs into one program view. | Missing data and unrelated IDs are handled honestly. |
| T-009 | P2 | done | Codex | Add model profiling for size, latency, memory, and operation proxies. | Repeated CPU measurements produce CSV/JSON results for PyTorch FP32, ONNX FP32, and ONNX INT8. |
| T-010 | P2 | done | User and Codex | Select the current hardware/runtime scope: CPU/ONNX target, cloud GPU training, NPU unavailable/deferred. | Device matrix and measurement protocol are approved in D-010 and D-013. |
| T-011 | P2 | done | Codex and Claude | Design the cloud agent bridge and review loop. | State machine, budgets, stop conditions, and permissions are documented in `docs/AGENT_BRIDGE.md`. |
| T-012 | P1 | done | Claude | Review the PyTorch CNN, autoencoder, grouped split, and ONNX path. | Approved in `ab8a497`; PyTorch/ONNX tests pass and CI-safe optional dependency guards were verified. |

## Immediate order

1. The current T-005 preflight branch passes 96 tests in the local AI environment; lightweight CI skips optional runtime tests.
2. T-007 is complete in `b134b66`: time-aware CPU equipment anomaly baseline and machine-readable outputs.
3. Claude approved T-007 and T-008; minor follow-ups are recorded in `CLAUDE_FEEDBACK.md`.
4. Resolve T-004 by documenting a licensed WM-811K source and cloud dataset path; a full local download is not required.
5. T-005 cloud preflight is implemented; finish T-005 and start T-006 after T-004 supplies the licensed source/path and the real preflight passes.
6. T-012 is complete and approved in `ab8a497`. CPU PyTorch, ONNX export, and ONNX Runtime inference pass in the local AI environment.
7. T-009 is complete: PyTorch FP32, ONNX Runtime FP32, and ONNX Runtime INT8 write repeated CPU profiles to CSV/JSON. NPU remains unavailable.
8. The first stable milestone was fast-forwarded to `main` at `7ee8575` after Claude approval and 92 passing tests.
9. T-011 design is complete. Implementation remains gated on P-004 hosted runtime/API authentication and must begin in dry-run mode.
