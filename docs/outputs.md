# Reading a run

Start with **`report.html`**. It summarizes the request, candidate panel, calculations and limitations. Its links lead to the supporting tables and files.

Use **CSV/TSV tables** to inspect sequences and measurements. Use **GenBank files** to view candidate positions alongside the reference sequence. Chemistry descriptions belong to the individual molecules; sequence alone does not describe their modifications.

Open **`report.md`** for a text version. The **`artifacts/`** directory retains inputs, structured results and calculation details.

Keep the whole run folder together when moving it. Report links use paths within that folder.

## Predictions

Splice scores describe modeled changes at the evaluated junctions. OligoAI scores describe predicted in vitro inhibition under the reported dose, delivery and chemistry conditions. Reports separate these predictions from sequence matches, off-target evidence and experimental measurements.

## Resume a paused request

When a run needs an answer, the terminal prints its state path. Resume with:

```bash
modal-aso --agentic-state-path /path/to/run_state.json
```

The existing calculations stay with the run.
