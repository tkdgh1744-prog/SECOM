# Agent Coordination

This directory is the shared coordination channel for the user, Codex, and Claude.
GitHub issues and pull-request comments may supplement these files, but decisions that
affect implementation must be recorded here so both agents can work from the same state.

## Working order

1. Read `GOAL.md` and `DECISIONS.md`.
2. Select one ready item from `TASK_QUEUE.md`.
3. Update `AGENT_STATUS.md` before implementation or review.
4. Codex records implementation details in `CODEX_HANDOFF.md`.
5. Claude records review findings in `CLAUDE_FEEDBACK.md`.
6. Run the required checks before requesting a merge.

Focused handoffs and future automation:

- `CLAUDE_T012_PROMPT.md`: exact review contract for the PyTorch/ONNX milestone.
- `../docs/AGENT_BRIDGE.md`: hosted Codex-Claude orchestration design.
- `agent_bridge.example.json`: machine-readable dry-run limits and permissions.

## Safety rules

- Do not push directly to `main`.
- Do not overwrite uncommitted work from another agent or the user.
- Do not commit datasets, model binaries, API keys, or credentials.
- Treat demo data and real manufacturing data as different sources.
- Do not merge sensor, wafer, and equipment records by row order unless an explicit,
  validated key contract permits it.
- The user approves the final merge until the workflow is stable.
