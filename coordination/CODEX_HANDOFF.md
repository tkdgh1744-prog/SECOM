# Codex Handoff

Date: 2026-07-17

## Change under review

Codex is establishing the first remote collaboration baseline:

- GitHub Actions validation and unit tests.
- A lightweight CI dependency list that excludes optional TensorFlow/GPU training.
- Shared goal, decisions, task queue, agent status, and review templates.

## Preserved existing work

The current local modifications to `scripts/analyze_wafer_maps.py`,
`src/wafer_map_analysis.py`, `.claude/`, and the untracked wafer-map notebooks were not
created or altered by this handoff.

## Claude review request

1. Check whether `GOAL.md` accurately separates the first integrated milestone from later
   CPU/GPU/NPU optimization.
2. Review CI for missing CPU test dependencies or unsafe permissions.
3. Review `DECISIONS.md` for data leakage, provenance, privacy, and merge-policy gaps.
4. Reorder or split tasks in `TASK_QUEUE.md` when the dependency is technically necessary.
5. Record findings in `CLAUDE_FEEDBACK.md`; do not edit implementation files during this review.

## Checks

Pending local execution. Codex will replace this line with the results before handoff.
