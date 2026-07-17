# Agent Status

Last updated: 2026-07-17

| Actor | Role | Current work | State |
| --- | --- | --- | --- |
| User | Product owner and final approver | Decide D-A (CPU/GPU/NPU scope), D-B (PyTorch vs TF), D-C (split policy) — see CLAUDE_FEEDBACK.md. | waiting on decisions |
| Codex | Implementation and verification | Committed 2cf14a4 (C-003/C-006 fixes + tests) and CODEX_FOLLOWUP.md; T-012 needs a PyTorch/ONNX runtime before it can be claimed to work. | active |
| Claude | Architecture and code review | T-002 review complete → findings in CLAUDE_FEEDBACK.md (verdict: approve with follow-up). | done |

## Repository snapshot

- Branch observed: `main`, tracking `origin/main`.
- GitHub remote is configured.
- Existing uncommitted edits in wafer-map implementation and two untracked notebooks are
  intentionally preserved and are not part of the coordination/CI change.
- `.claude/settings.local.json` is local state and must not contain committed credentials.
- Real WM-811K data is not present locally.

## Next handoff

T-002 review is complete (see `CLAUDE_FEEDBACK.md`, verdict `approve with follow-up`). The
CI + coordination baseline is safe to merge to `main` after user approval. Before Codex starts
the analysis baselines (T-005 onward), the user should resolve decision requests D-A, D-B, and
D-C and lock the split policy (finding C-002), since those shape model design.

## Current update

- T-009 CPU profiling is complete in `ea6f405` and pushed to the collaboration branch.
- T-010 is resolved: CPU/ONNX is primary, cloud GPU is training-only, and NPU is unavailable.
- T-011 bridge design is complete; hosted runtime/API authentication remains P-004.
- T-012 is approved by Claude in `ab8a497`; only non-blocking real-data metric and FLOPs follow-ups remain.
- The first stable milestone is on `main` at `7ee8575` and pushed to GitHub.
- T-005 cloud preflight work continues on `codex/wm811k-cloud-preflight` while T-004 awaits the licensed source/path.
