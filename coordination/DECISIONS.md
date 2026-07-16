# Decisions

## Accepted

| ID | Decision | Reason |
| --- | --- | --- |
| D-001 | GitHub is the shared source of truth. | Remote agents can continue while the local PC is off. |
| D-002 | GitHub Actions is the first automated referee. | Every push and pull request receives the same checks. |
| D-003 | Codex implements and verifies; Claude reviews design, risk, and changes. | Clear ownership reduces conflicting edits. |
| D-004 | Final merge requires user approval during the initial phase. | Automation remains observable and reversible. |
| D-005 | Raw datasets and generated model binaries stay out of Git unless explicitly approved. | They may be large, licensed, private, or reproducible. |
| D-006 | Cross-source joins require validated identifiers and documented cardinality. | Row order does not establish manufacturing traceability. |
| D-007 | Demo, synthetic, and real results must be visibly distinguished. | Educational outputs must not be mistaken for production evidence. |
| D-008 | CPU validation runs in the default CI; optional TensorFlow/GPU work is separated. | Fast checks should not depend on unavailable accelerators. |
| D-009 | Deep-learning framework is standardized on PyTorch; the existing TensorFlow wafer CNN/autoencoder is migrated to PyTorch. TF path stays working until the port lands. | User-confirmed 2026-07-17 (re-affirming 2026-07-15). Unifies the FLOPs / quantization / ONNX-export story and keeps the CPU/ONNX optimization path clean. |
| D-010 | Primary AI-semiconductor target is CPU / ONNX Runtime. GPU is for training convenience only; NPU comparison is documented only if concrete hardware and a runtime become available (see P-003). | User-confirmed 2026-07-17. Crown deliverable = accuracy-vs-latency Pareto (FP32 vs INT8, PyTorch vs ONNX Runtime); keeps scope achievable. |
| D-011 | Evaluation splits must prevent leakage: wafer maps grouped by lot/wafer, equipment series split time-ordered, no random row-level split. Row-order cross-source joins remain forbidden (see D-006). | Claude technical policy 2026-07-17; standard ML correctness. Prevents optimistic metrics from dies/wafers/time bleeding across train and test. |

## Pending

| ID | Question | Owner |
| --- | --- | --- |
| P-001 | What licensed source and storage location will be used for WM-811K? | User |
| P-002 | What equipment time-series dataset will be used first? | User and Claude |
| P-003 | Which NPU hardware/runtime will be available for comparison? | User |
| P-004 | Which remote service will host the future Codex-Claude agent bridge? | User and Codex |
