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

## Pending

| ID | Question | Owner |
| --- | --- | --- |
| P-001 | What licensed source and storage location will be used for WM-811K? | User |
| P-002 | What equipment time-series dataset will be used first? | User and Claude |
| P-003 | Which NPU hardware/runtime will be available for comparison? | User |
| P-004 | Which remote service will host the future Codex-Claude agent bridge? | User and Codex |
