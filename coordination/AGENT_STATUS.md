# Agent Status

Last updated: 2026-07-17

| Actor | Role | Current work | State |
| --- | --- | --- | --- |
| User | Product owner and final approver | Confirm data sources and hardware targets. | waiting |
| Codex | Implementation and verification | T-001: CI and coordination baseline. | active |
| Claude | Architecture and code review | T-002: review after Codex handoff. | waiting |

## Repository snapshot

- Branch observed: `main`, tracking `origin/main`.
- GitHub remote is configured.
- Existing uncommitted edits in wafer-map implementation and two untracked notebooks are
  intentionally preserved and are not part of the coordination/CI change.
- `.claude/settings.local.json` is local state and must not contain committed credentials.
- Real WM-811K data is not present locally.

## Next handoff

After local checks, Codex will update `CODEX_HANDOFF.md`. Claude should then review only
the stated change set and place findings in `CLAUDE_FEEDBACK.md`.
