# Task Queue

Status values: `ready`, `active`, `review`, `blocked`, `done`.

| ID | Priority | Status | Owner | Task | Completion check |
| --- | --- | --- | --- | --- | --- |
| T-001 | P0 | active | Codex | Add shared coordination documents and GitHub CI. | Local validation and tests run; workflow file is reviewable. |
| T-002 | P0 | done | Claude | Review project boundaries, CI, risks, and task ordering. | Findings recorded in `CLAUDE_FEEDBACK.md` (verdict: approve with follow-up). |
| T-003 | P0 | ready | Codex | Establish a clean baseline for all existing tests. | Failures are fixed or documented with owners. |
| T-004 | P0 | blocked | User | Provide or approve a source for real WM-811K data. | Dataset path and license/provenance are documented. |
| T-005 | P1 | ready | Codex | Make wafer-map ingestion and baseline classification reproducible. | Real-data path works; demo fallback is clearly labeled. |
| T-006 | P1 | ready | Claude | Review wafer-map leakage, split strategy, labels, and metrics. | Review covers lot/wafer grouping and class imbalance. |
| T-007 | P1 | ready | Codex | Define and implement an equipment anomaly baseline. | Time-aware split, anomaly score, threshold, and report exist. |
| T-008 | P1 | ready | Codex | Integrate the three analysis outputs into one program view. | Missing data and unrelated IDs are handled honestly. |
| T-009 | P2 | ready | Codex | Add model profiling for size, latency, memory, and operation proxies. | Repeated measurements produce machine-readable results. |
| T-010 | P2 | blocked | User | Select CPU/GPU/NPU target hardware and runtimes. | Device matrix and measurement protocol are approved. |
| T-011 | P2 | ready | Codex and Claude | Design the cloud agent bridge and review loop. | State machine, budgets, stop conditions, and permissions are documented. |
| T-012 | P1 | ready | Codex | Migrate the wafer-map CNN and autoencoder from TensorFlow to PyTorch (per D-009). | PyTorch model trains on demo data; CI stays green; ONNX export path works. |

## Immediate order

1. T-001 done (baseline pushed); T-002 done (review in CLAUDE_FEEDBACK.md). Decisions D-009/D-010/D-011 locked 2026-07-17.
2. Merge the `codex/collaboration-baseline` branch to `main` after user approval.
3. Codex: T-003 (clean test baseline) mostly satisfied — wafer diff + tests committed in 2cf14a4 (72 passed / 3 skipped). Next: T-012 (TF->PyTorch migration), which needs a PyTorch/ONNX runtime installed first.
4. Resolve T-004 (WM-811K source) while Codex improves the demo/reproducibility path.
5. Continue T-005 through T-008; lock the split policy (D-011) before training any baseline.
6. Start hardware profiling and the full agent bridge only after the analysis baselines are stable.
