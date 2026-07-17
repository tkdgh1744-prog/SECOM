# Claude T-012 Review Prompt

Use this prompt in Claude Code or Claude Desktop with repository access.

---

You are the independent reviewer for T-012 in the SECOM semiconductor AI
repository. Work from the latest `codex/collaboration-baseline` branch.

## Required setup

1. Fetch and inspect the latest branch. The PyTorch migration starts at commit
   `2375c26`; later commits may add profiling or coordination documents.
2. Read `coordination/GOAL.md`, `DECISIONS.md`, `TASK_QUEUE.md`, and the existing
   `CLAUDE_FEEDBACK.md` before reviewing code.
3. Do not edit implementation files during this review.

## Review scope

- `src/wafer_torch.py`
- `src/wafer_ai_outputs.py`
- `scripts/analyze_wafer_maps.py`
- `tests/test_wafer_torch.py`
- Relevant documentation and dependency files

Verify all of the following:

1. CNN and autoencoder training run on CPU with synthetic demo data.
2. PyTorch bundles can be saved and loaded without serializing an arbitrary
   model object.
3. The ONNX export is valid and ONNX Runtime can execute it.
4. Grouped splitting prevents the same `lotName` or wafer group from crossing
   train and test boundaries.
5. Demo, synthetic, and real outputs remain visibly distinguishable.
6. Labels, class imbalance, metrics, input shape, and rare-class behavior do
   not produce misleading evaluation claims.
7. The lightweight CI remains usable when PyTorch/ONNX are not installed.
8. No raw dataset, generated model, credential, or unrelated notebook is added.

Run the most complete available test command. On this Windows checkout, prefer:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Also run a direct train -> save -> load -> ONNX -> ONNX Runtime smoke test when
the optional AI dependencies are available.

## Required output

Append a new `T-012 review` section to `coordination/CLAUDE_FEEDBACK.md` with:

- reviewed commit SHA
- tests and smoke checks run
- findings ordered by severity with file/line references
- residual risks, especially synthetic-data limitations
- one verdict: `approve`, `approve with follow-up`, or `changes required`

Only when the verdict is `approve` or `approve with follow-up`, change T-012 in
`coordination/TASK_QUEUE.md` from `review` to `done`. Commit only the review and
coordination files with message `Complete T-012 review`, then push
`codex/collaboration-baseline`. Do not merge to `main`; Codex will verify and
perform the user-approved milestone merge after reading your verdict.

---
