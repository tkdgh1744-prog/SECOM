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
| D-012 | WM-811K will be stored in a cloud-accessible dataset location, not downloaded into the repository or required on the local PC. Only code, small approved samples, provenance, and generated summaries belong in the project. | User-confirmed 2026-07-17. Keeps the repository lightweight and lets Colab/Kaggle supply training storage and GPU access. |
| D-013 | No NPU is available for the current milestone. CPU/ONNX Runtime measurements proceed now, cloud GPU is used for training, and NPU results remain explicitly unavailable until real hardware and a compatible runtime are selected. | User-confirmed 2026-07-17. Prevents estimated or simulated NPU numbers from being presented as measured results. |
| D-014 | A future Codex-Claude bridge must run on hosted infrastructure with finite budgets, isolated work branches, GitHub Actions validation, independent review, and exact-SHA human merge approval. Agents cannot push or merge `main`. | Completes the T-011 safety design while leaving the provider and hosting choice open in P-004. The workflow can continue while the local PC is off without granting unbounded repository access. |

## Pending

| ID | Question | Owner |
| --- | --- | --- |
| P-001 | Which licensed WM-811K source and exact cloud dataset path will be used? Cloud storage is already approved by D-012. | User |
| P-002 | What equipment time-series dataset will be used first? | User and Claude |
| P-003 | Which NPU hardware/runtime may become available for a future comparison? This does not block the current milestone. | User |
| P-004 | Which hosted runtime and provider API authentication will run the future Codex-Claude bridge? | User and Codex |
