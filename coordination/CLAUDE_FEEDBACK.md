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

---

# T-007 / T-008 review (2026-07-17)

Reviewed commits: `b134b66` (equipment anomaly baseline, T-007) and `c4e8f95` (integrated
dashboard, T-008). Implementation files were not edited. Full suite: **80 tests pass, 3 skipped**.
Verdict for both: **approve** (minor, non-blocking follow-ups only).

## T-007 — Equipment anomaly baseline (`b134b66`)

Strengths (verified against the diff, not just the description):

- **Leakage-free, D-011-compliant time split.** `time_ordered_split_mask` splits on unique
  timestamps so equal times never straddle the boundary; the detector fits on train rows only and
  the threshold is the train-score quantile — no evaluation data touches fitting or thresholding.
- **`integration_mode` stamped everywhere** (each scored row, summary, metadata JSON, report);
  demo forces `synthetic`. This implements finding C-005.
- **Label leakage guarded**: auto sensor-selection excludes common label columns
  (`failure_label`, `is_anomaly`, `target`, ...); metrics are computed on the evaluation split only.
- Tests (6) cover the split, late-anomaly detection, label handling, and machine-readable outputs.

Follow-ups (non-blocking):

| ID | Severity | Area | Finding | Recommended action |
| --- | --- | --- | --- | --- |
| E-1 | info | src/equipment_anomaly.py | Detector is a per-row robust z-score (point anomaly), not a temporal model — no windowing/lag/changepoint. Honest as a baseline, but "anomaly-time" detection benefits from temporal structure. | Keep as the CPU baseline; consider rolling/lag features or a changepoint pass later. |
| E-2 | low | scripts/analyze_equipment_anomalies.py:106 | Model bundle persisted via `pickle`, while the SECOM track uses `joblib`. Pickle of a custom class is version-fragile and unsafe to load from untrusted sources. | Prefer `joblib`, or serialize the detector arrays to `.npz`/JSON, for consistency and safe loading. |
| E-3 | info | src/equipment_anomaly.py:271 | Threshold assumes the earliest training window is mostly normal; early anomalies or regime shift inflate it. | Document the assumption; consider a dedicated validation window for the threshold on real data. |

## T-008 — Integrated dashboard (`c4e8f95`)

This is the highest key-mismatch-risk item, and the primary concern is **cleanly avoided**:

- **No cross-source / row-order join.** The three tracks render as independent panels from separate
  output directories; there is no sample-level merge. The overview carries an explicit boundary
  notice ("does not imply sample-level linkage between SECOM, wafer, and equipment records") and the
  header reads "No row-order joins". Satisfies the GOAL.md first-milestone rule and D-006.
- Per-track `integration_mode` badges; missing tracks render as "not available"/"missing" honestly
  (tested at 0/3).
- **Standalone**: inline CSS/JS + base64-embedded PNG, no CDN — opens offline; equipment timeline is
  hand-drawn on canvas.
- **Safe rendering**: table cells/labels HTML-escaped; the JSON chart payload escapes `<` to prevent
  a `</script>` breakout.
- **Correct counting**: only evaluation-split anomalies are tallied (train excluded), verified by test.

Follow-ups (non-blocking):

| ID | Severity | Area | Finding | Recommended action |
| --- | --- | --- | --- | --- |
| D-1 | info | src/integrated_dashboard.py:53 | Booleans render as `1`/`0` in tables (Python `bool` is an `int` subclass in `_format_value`); e.g. `is_anomaly` shows 1/0. | Optional: special-case `bool` to `Yes`/`No` for readability. |
| D-2 | info | future | The "no linkage" stance is correct. When real IDs eventually exist, any linkage must still pass a validated key contract (D-006/D-011), never row order. | Keep the boundary notice; gate future joins on validated keys. |

## Verdict (T-007 / T-008)

`approve` — both. T-007 and T-008 can move to `done`. Suggested next reviews: T-006 (wafer-map
leakage/split/metrics) once the wafer classification baseline (T-005) lands; T-009 profiling should
target CPU/ONNX per D-010.

---

# PyTorch toolchain verification + `0d3460b` review (2026-07-17)

## Independent CPU toolchain check (my own scratch script, not repo code)

Ran a self-contained PyTorch CNN -> `.pt` -> ONNX -> ONNX Runtime (+INT8) smoke test in `.venv`:

- torch **2.13.0+cpu**, onnxruntime 1.27.0, onnx 1.22.0; `torch.cuda.is_available()=False` (CPU build).
- PyTorch vs ONNX Runtime outputs: max abs diff **1.0e-07** (numerical parity holds).
- INT8 dynamic quantization: ONNX **547 KB -> 143 KB (26% of FP32)**; INT8 vs torch max diff 8.7e-04.
- CPU latency (tiny model): PyTorch ~3.3 ms vs ONNX Runtime ~0.09 ms/infer.

Conclusion: the **D-010 crown-deliverable path (FP32 vs INT8, PyTorch vs ONNX Runtime) is feasible on
this machine.**

**Gotcha found:** torch 2.13's *default* `torch.onnx.export` requires `onnxscript` (not installed) and
errors; the legacy `dynamo=False` exporter works without it. **Codex already uses `dynamo=False`** in
`src/wafer_torch.py::export_torch_model_to_onnx` (uncommitted), so the T-012 export path is sound and
no extra install is needed. `requirements-ai.txt` does not pin `onnxscript`, which is fine given
`dynamo=False`; add it only if the modern dynamo exporter is ever wanted. **CI note:** any ONNX-export
test must use `dynamo=False` (or install `onnxscript`) to stay green.

## `0d3460b` (Address dashboard review) — approve

- **D-1 resolved**: `_format_value` now renders booleans as `Yes`/`No`, placed *before* the int branch
  (correct, since `bool` is an `int` subclass). README/Makefile docs and a test were added.
- **D-2** was a future-only note; no code change required now.

## Pending

- Full review of `src/wafer_torch.py` + `tests/test_wafer_torch.py` (T-012) once Codex commits them
  (currently uncommitted / in progress). Will independently verify train -> `.pt` -> `.onnx` -> ORT then.

---

# T-012 (PyTorch migration) + T-009 (profiling) review — 2026-07-17

Reviewed committed `2375c26` (T-012) and `ea6f405` (T-009). Ran their tests in `.venv`: **11/11 pass**
(torch 2.13.0+cpu, onnx 1.22.0, onnxruntime 1.27.0). Implementation files not edited. Verdict: **approve**
(minor, non-blocking follow-ups).

## CI safety (verified)

Both suites guard torch/onnx with `@unittest.skipUnless(torch_is_available() ...)`, and the availability
helpers use `find_spec` (no hard import). CI installs `requirements-ci.txt` (no torch), so the torch/ONNX
tests **skip** (not error) while the pure-numpy data/utility tests still run. CI stays green.

## T-012 — `src/wafer_torch.py` (`2375c26`) — approve

- **Grouped split (D-011) is correct**: `grouped_train_test_indices` assigns whole wafer groups
  (`group_id or wafer_id`) to train/test so a group never crosses the boundary, and guarantees every class
  appears in training (retries for rare classes). No leakage; verified by test.
- **Safe model loading**: `load_torch_model_bundle` uses `torch.load(..., weights_only=True)`, avoiding
  arbitrary-pickle execution — better than the equipment track's `pickle` (see E-2; worth aligning).
- ONNX export uses `dynamo=False` (avoids the torch-2.13 onnxscript gotcha), guarded by `onnx_is_available`.
- Device resolve handles cpu/cuda/mps (Colab GPU per D-010); seeded and reproducible.

| ID | Severity | Area | Finding | Recommended action |
| --- | --- | --- | --- | --- |
| W-1 | info | wafer_torch.py CNN trainer | Reports raw accuracy with unweighted CrossEntropy; wafer patterns are imbalanced. | When real WM-811K lands (T-006), report macro-F1 / per-class recall and consider class weights. |

## T-009 — `src/model_profiling.py` (`ea6f405`) — approve

- Profiles **PyTorch FP32, ONNX Runtime FP32, and ONNX INT8 — all on CPU** (D-010 crown-deliverable inputs);
  `quantize_onnx_dynamic_int8` supplies the INT8 leg.
- Sound methodology: warmup + N measured runs, p50/p95/min/max/std, throughput; deterministic ORT threads
  (intra=1/inter=1); RSS memory via psutil/ctypes//proc with an honest "process-level proxy" note.
- Machine-readable: flat CSV + operations CSV + lossless JSON with captured environment. Repeatable.

| ID | Severity | Area | Finding | Recommended action |
| --- | --- | --- | --- | --- |
| P-1 | info | model_profiling.py | `operation_node_count` is an operator-node proxy, not true FLOPs (honestly labeled). | If the course needs FLOPs specifically, add a FLOP estimator (onnx shape inference or `thop`/`fvcore`) later. |

## Status

T-012 and T-009 can move to `done`. The CPU PyTorch/ONNX/INT8 profiling path is real and tested — the crown
deliverable (accuracy-vs-latency Pareto, FP32 vs INT8, PyTorch vs ONNX Runtime) is ready to populate once
trained models on real data exist (gated on WM-811K, P-001).
