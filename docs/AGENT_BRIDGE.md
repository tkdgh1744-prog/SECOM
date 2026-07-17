# Cloud Agent Bridge Design

## Purpose

The bridge lets a user submit one objective and have Codex implement, Claude
review, and GitHub Actions verify repeated rounds without depending on the local
PC. GitHub remains the durable source of truth; the bridge coordinates work but
does not replace version control, CI, or human merge approval.

This document completes the design portion of T-011. It does not enable remote
agents by itself. API credentials and a hosted runtime are still required.

## Components

| Component | Responsibility | Durable state |
| --- | --- | --- |
| Hosted orchestrator | Lease tasks, call agents, enforce limits, and record transitions. | GitHub issue/comment or `coordination/runs/<run-id>.json` |
| Codex implementer | Create a branch, change scoped files, test, commit, and push. | Git branch and commit |
| Claude reviewer | Inspect the diff and checks; approve or request concrete changes. | Review comment and coordination feedback |
| GitHub Actions | Run repository-owned validation independent of both agents. | Workflow checks and logs |
| User | Define the objective, approve sensitive decisions, and merge the final PR. | Issue/PR approval |

The hosted orchestrator may run in GitHub Actions, a small cloud container, or a
serverless job. It must use provider APIs or hosted agent products; a local
desktop subscription alone cannot continue while the PC is off.

## Task State Machine

```text
queued
  -> leased
  -> implementing
  -> validating
  -> review_requested
  -> approved
  -> awaiting_human_merge
  -> complete

review_requested -> changes_requested -> implementing
any active state -> blocked | failed | cancelled
```

Transitions are append-only events. Every event records the run ID, task ID,
actor, prior state, next state, commit SHA when applicable, timestamp, attempt,
and a short reason. Only the orchestrator writes state; agents return proposed
results to it.

### Transition Guards

- `queued -> leased`: no active lease exists and a budget is present.
- `implementing -> validating`: an allowed branch has a new commit.
- `validating -> review_requested`: required local checks and GitHub Actions pass.
- `review_requested -> approved`: the reviewer supplies an allowed verdict and
  no unresolved high/critical finding remains.
- `changes_requested -> implementing`: another round remains in the budget.
- `approved -> awaiting_human_merge`: branch protection and merge policy pass.
- `awaiting_human_merge -> complete`: the user or an explicitly approved merge
  policy merges the exact reviewed SHA.

## Default Budgets

Each run must set finite limits before the first model call.

| Limit | Initial default | Purpose |
| --- | ---: | --- |
| Review/repair rounds | 3 | Prevent endless agent debate. |
| Consecutive failed validations | 2 | Stop repeated broken fixes. |
| Wall-clock runtime | 60 minutes | Bound unattended execution. |
| Changed files | 20 | Detect accidental scope expansion. |
| Changed lines | 1,500 | Keep review practical. |
| Agent calls | 8 | Bound API usage. |
| Per-run API cost | Explicitly configured | No run starts with an unlimited or missing cost cap. |

Provider token limits and cost ceilings belong in secret-backed deployment
configuration, not committed files. The run record stores consumed totals but
never API keys or full private prompts.

## Stop Conditions

The orchestrator stops and marks the run `blocked` or `failed` when any of these
conditions occurs:

1. A budget, wall-clock, changed-file, or changed-line limit is reached.
2. The same material review finding survives two consecutive repair rounds.
3. Required validation fails twice without a new diagnosis.
4. The branch conflicts with the base branch and an automatic clean merge is not
   possible.
5. Work needs a dataset license, credential, hardware device, business decision,
   or schema contract that has not been approved.
6. A proposed command is destructive, changes protected infrastructure, exposes
   secrets, or exceeds the configured permissions.
7. The agent proposes direct `main` changes, force-push, branch-protection edits,
   or merging a SHA different from the reviewed SHA.
8. The objective materially expands beyond the original task.

The user receives the last successful commit, failed check, unresolved finding,
and exact decision required to resume.

## Permissions

| Actor | Read | Write | Forbidden by default |
| --- | --- | --- | --- |
| Orchestrator | Repository, checks, task metadata | Run state, comments, agent branches | Source implementation, `main`, secrets output |
| Codex | Repository and task context | Assigned branch and implementation handoff | `main`, force-push, credentials, merge |
| Claude | Repository, diff, checks, handoff | Review feedback and verdict | Implementation changes, `main`, merge |
| GitHub Actions | Checked-out commit | Check status and approved artifacts | Repository pushes, long-lived credentials |
| User | All project state | Decisions and final merge | None, subject to GitHub policy |

Use separate least-privilege credentials for GitHub, Codex/OpenAI, and Claude.
Store them only in the hosted platform's encrypted secret store. Pull requests
from untrusted forks must never receive provider secrets.

## Execution Loop

1. The user creates an objective with acceptance checks and a budget.
2. The orchestrator creates a run record and an isolated branch.
3. Codex receives only the objective, current repository state, allowed paths,
   limits, and test commands.
4. Codex pushes a commit; GitHub Actions validates it.
5. Claude receives the objective, exact diff, check results, and prior findings.
6. Claude approves or returns actionable findings with severity and file/line.
7. The orchestrator either starts a bounded repair round or freezes the approved
   SHA for human merge.
8. The user merges; the orchestrator records completion and cost/usage totals.

## Deployment Sequence

1. Keep the current file-based coordination and manual merge policy.
2. Choose the hosted runtime and provider API authentication (P-004).
3. Implement a dry-run orchestrator that validates task envelopes and logs state
   transitions without calling an AI provider.
4. Add Codex implementation calls with branch-only GitHub permissions.
5. Add Claude review calls with read/review-only permissions.
6. Add GitHub Actions/PR gates and exact-SHA approval checks.
7. Run synthetic objectives in a sandbox repository before enabling this project.
8. Retain human merge approval until several bounded runs complete reliably.

The machine-readable defaults are in
`coordination/agent_bridge.example.json`.
