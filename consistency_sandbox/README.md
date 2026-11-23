# Consistency sandbox

**Run each unchanged prompt five times, then review the final outputs and the recorded decision paths.** The initial batch covers C3, MEF2C, and ANGPT2 for 15 attempts total. Evaluation happens after capture.

## Capture a batch

From the repository root in WSL2, use the existing MODAL-ASO Python environment.

```sh
python consistency_sandbox/run_batch.py --dry-run
python consistency_sandbox/run_batch.py
```

The default batch runs C3, MEF2C, and ANGPT2 five times each in fresh processes, one at a time. Use `--data-dir /path/to/existing/data` to explicitly select the same installed tools, models, and references for every run. The script supplies the archived MEF2C reply once if the controller asks a single biological-context question. Other clarification requests are recorded and left paused. The batch continues after a paused or failed attempt and exits nonzero if any attempt did not complete.

Each attempt lives in `captures/<batch-id>/<prompt>/repeat_<nn>/`. It contains the prompt, console log, status, any clarification transcript, and the native MODAL-ASO output tree under `output/`. That tree holds the controller trace, state, snapshots and decision log where emitted, alongside the final reports and structured exports. Both captures and the archive are ignored by Git. Installed runtime assets and shared caches continue to use their configured data locations. This script captures runs for later evaluation.

| Held fixed | How |
| --- | --- |
| Prompt and clarification | Snapshot each prompt once per batch. Replay the same saved reply under the same MEF2C rule. |
| Controller and runtime | Use the same working checkout, Python interpreter, tool configuration and runtime data directory. Record the Git revision, dirty-tree status, interpreter and platform in `batch.json`. Keep this checkout and runtime assets unchanged while the batch runs. |
| Execution options | Use agentic mode, controller budget of 120 steps, router budget of 25, and the same tracing options. Runs are sequential with no previous conversation or run state. |
| LLM configuration | Inherit one environment for the batch and record the supported provider, model, reasoning and routing overrides listed in the script. Unset options use the checkout's normal defaults. To pin a model or upstream provider explicitly, set its supported overrides before starting the batch. |

Each repeat differs in its run ID, output directory, timestamps and position in the schedule. The first prompt in each round rotates. Model sampling is allowed to vary. Backend routing within the configured policy, live tool responses, shared cache warmth and service timing can vary as well. The batch does not set a seed or override generation parameters.

## Review captured runs

Review each prompt's five attempts using the saved artifacts.

- **Final outputs.** Read the final report and structured exports. Check whether each attempt delivered what the prompt requested and whether its claims are supported by the saved evidence.
- **Decision paths and rationale.** Use the controller trace, state snapshots, and decision log to reconstruct the branches each attempt took. Review clarifications, validation gates, tool choices, retries, and fallbacks. Where the records show it, identify whether the agent chose the action or the controller enforced it, and review any stated rationale against the evidence available at that step. Each attempt records one path through the workflow. The five traces show which branches were actually observed.

For later scoring, use the versioned keys in `rubrics/`. For MEF2C, review intake and resume separately, including whether the run asked for biological context before receiving the saved reply.

For each applicable check, use **2** for full satisfaction, **1** for a secondary omission with a supported result, and **0** for a material failure. For hybrid checks, verify structured facts against artifacts first and use the judge for scope, prose, and rationale. A material artifact mismatch fixes the score at 0. Record the check ID, score, evidence location, and reason. Confirm judge-only critical flags with human review. A confirmed critical failure makes the run fail. Run-level pass requires no confirmed critical failures and at least 1 on every applicable check. Record the MEF2C resume checks as pending until that turn is run. Keep grading fixtures and keys out of the execution prompt.

Report the pass rate and outcome agreement separately. Use the outcome IDs in each key to compare runs. The historical runs are evidence for the rubric, and their candidate counts, exact rankings, and wording may vary. The original archive is local under `og_responses/`.
