# Claude Feedback

Review status: `complete`
Reviewed change set: branch `codex/collaboration-baseline` (commit 5212875 — CI + coordination baseline)
Reviewer: Claude · Date: 2026-07-17
Scope note: implementation files were NOT edited. The preserved uncommitted wafer-map edits and the two untracked notebooks were treated as out-of-scope for this baseline and reviewed only for follow-up flags.

## Summary

The collaboration baseline is safe and mergeable. CI uses least-privilege permissions
(`contents: read`), pins action versions, sets a timeout, and correctly excludes TensorFlow
from the CPU test image. I verified this does not break tests: TF imports are lazy (inside
functions), no test imports TF, and the wafer-map test never calls the TF training functions.
The 3 local skips are `joblib`-gated, not TF-gated, and `joblib` is present in CI.

The main gaps are not in the CI plumbing but in **direction alignment**: `GOAL.md` and
`DECISIONS.md` do not yet capture the framework/target decisions the user locked on 2026-07-15
(PyTorch unification, wafer-CNN as the optimization "star", CPU/ONNX Runtime as the primary
target). `GOAL.md`'s "GPU training recommended / compare CPU, GPU, NPU" language may conflict
with that. This needs a user decision before deep optimization work starts — it does not block
merging the scaffolding.

Verdict: **approve with follow-up**.

## Follow-up verification (2026-07-17)

Codex acted on this review in commit `2cf14a4` ("Harden wafer map record limits") and recorded
results in `CODEX_FOLLOWUP.md`. I verified the fixes against the actual diff, not just the claim.
The branch has since advanced to `e97e8e8`; the verdict stands.

- **C-001** — resolved: framework/target decisions recorded as D-009 (PyTorch) and D-010 (CPU/ONNX); `GOAL.md` wording reconciled.
- **C-002** — resolved: split policy locked as D-011.
- **C-003** — resolved (verified in `2cf14a4`): CI now triggers on all branch pushes and adds `concurrency: cancel-in-progress`.
- **C-006** — resolved (verified in `2cf14a4`): `--max-records` is positive-only (omit = all), `--similarity-max-records 0` skips similarity, negatives are rejected at both the CLI and the internal API, EOF blank lines removed, tests added (full suite 72 passed / 3 skipped).
- **Open**: C-004 (task deps — partially addressed by adding T-012), C-005 (enforce an `integration_mode` marker — future work), C-007 (root README mojibake — separate task).

## Findings

Severity: `critical`, `high`, `medium`, `low`, `info`.

| ID | Severity | File or area | Finding | Recommended action |
| --- | --- | --- | --- | --- |
| C-001 | medium | GOAL.md / DECISIONS.md | Direction drift: locked 2026-07-15 decisions (PyTorch unify, wafer-CNN star, CPU/ONNX target, RF=baseline-only) are absent from DECISIONS.md; GOAL.md says "GPU training recommended" and "compare CPU/GPU/NPU", which may contradict the CPU/ONNX target. | Reconcile via a user decision (see Decision requests D-A/D-B), then record the outcome as new DECISIONS.md rows and align GOAL.md wording. |
| C-002 | medium | DECISIONS.md | No locked split policy to prevent leakage. WM-811K has many dies per wafer and wafers per lot; a random split leaks. Equipment time series leaks if split randomly instead of by time. T-006 reviews this but no decision locks it. | Add a decision: wafer split grouped by lot/wafer (no die/wafer bleed across train/test); equipment split is time-ordered. Lock before T-005/T-007 build baselines. |
| C-003 | low | .github/workflows/ci.yml | Feature-branch pushes do not trigger CI (only `main` push + `pull_request` do), so a branch gets no CI until a PR exists. No `concurrency:` guard, so superseded runs are not cancelled. | Optional: add the branch to push triggers or rely on PRs (acceptable); add a `concurrency` block to cancel in-progress runs. **RESOLVED in 2cf14a4 (verified).** |
| C-004 | low | TASK_QUEUE.md | Dependencies are implicit. T-008 (integration) needs T-005 and T-007 to exist; T-009 (profiling) needs trained models. No migration task exists for TF->PyTorch if that decision stands. | Add an explicit "depends-on" note per row; add a migration task and a "wafer-diff needs tests before commit" task (see C-006). |
| C-005 | low | Repo boundary | Real WM-811K absent; current wafer outputs are demo/synthetic. This is stated in GOAL.md/AGENT_STATUS.md but the code does not yet stamp an `integration_mode = demo/synthetic/real` marker on outputs. | When T-005/T-008 land, require outputs to carry an explicit mode marker so demo results cannot be mistaken for real. Reinforces D-006/D-007. |
| C-006 | low | scripts/analyze_wafer_maps.py, src/wafer_map_analysis.py (preserved, uncommitted) | New `--max-records` / `--similarity-max-records` options are untested; `--max-records 0` means "all" (falsy) not "zero"; `--similarity-max-records` uses 0=skip but negative=all (undocumented) — inconsistent semantics. Trailing blank lines added at EOF. | Before this diff is committed: add tests for both options, unify 0/negative semantics with the help text, and strip EOF blank lines. **RESOLVED in 2cf14a4 (verified): positive-only max-records, 0-skips similarity, negatives rejected, tests added.** |
| C-007 | info | README.md (root) | Several sections contain baked-in mojibake (Korean corrupted by an encoding round-trip, re-saved as valid UTF-8). Lines ~3, ~33-58, ~181-206. Codec auto-recovery (cp949) fails. | Regenerate the corrupted sections as clean UTF-8 (no BOM). Separate task; does not affect this branch. |

## Data and evaluation review

- **Data provenance**: Handled well. D-005 keeps raw data and model binaries out of Git; P-001 tracks WM-811K source/license as a User item. WM-811K confirmed not local — wafer results are demo/synthetic.
- **Leakage and split strategy**: Not yet locked (C-002). This is the highest ML-correctness risk. Wafer maps must be split by lot/wafer, equipment series by time. Recommend locking before any baseline is trained.
- **Class imbalance and metrics**: SECOM is highly imbalanced; existing code already uses class weight / oversampling / SMOTE / threshold tuning / PR curves (good). For wafer maps, defect patterns are also imbalanced — require macro-F1 / per-class recall, not raw accuracy, in T-005/T-006.
- **Demo versus real-data labeling**: Policy exists (D-007) but is not yet enforced in code (C-005). Recommend a machine-checkable `integration_mode` marker on all integrated outputs.

## Decision requests

Recommended default is given for each; user confirmation needed.

- **D-A — CPU/GPU/NPU scope.** GOAL.md/system goal #6 say "compare CPU, GPU, NPU", but the 2026-07-15 note set CPU/ONNX Runtime as the target. Recommended default: **CPU/ONNX Runtime is the primary optimization + measurement target and the crown deliverable (accuracy-vs-latency Pareto, FP32 vs INT8, PyTorch vs ONNX Runtime)**; GPU is allowed for training convenience only; NPU comparison is documented only if concrete hardware/runtime is available (ties to P-003). Owner: User.
- **D-B — Deep-learning framework.** A 2026-07-15 note locked PyTorch (migrate the TF wafer CNN to PyTorch), but current code and CI use TensorFlow. Recommended default: **standardize on PyTorch**, record it in DECISIONS.md, keep the working TF path until the PyTorch port lands, and add an explicit migration task. If the user has since reversed this, keep TF and update the plan instead. Owner: User.
- **D-C — Split policy (C-002).** Recommended default: **wafer split grouped by lot/wafer; equipment split time-ordered; no random row-level split; row-order joins remain forbidden (D-006).** Owner: User/Claude.

## Verdict

`approve with follow-up`

The CI + coordination baseline (branch `codex/collaboration-baseline`) is safe to merge to `main`
after user approval. None of the findings block the scaffolding. Before Codex starts the analysis
baselines (T-005 onward), resolve D-A, D-B, and D-C and lock the split policy (C-002), because those
shape model design and cannot be cheaply changed later.
