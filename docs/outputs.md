# Reading a run

Start with **`report.html`**. It summarizes the request, candidate panel, calculations and limitations. Its links lead to the supporting tables and files.

Use **CSV/TSV tables** to inspect sequences and measurements. Use **GenBank files** to view candidate positions alongside the reference sequence. Chemistry descriptions belong to the individual molecules; sequence alone does not describe their modifications.

Open **`report.md`** for a text version. The **`artifacts/`** directory retains inputs, structured results and calculation details.

Keep the whole run folder together when moving it. Report links use paths within that folder.

The `artifacts/candidates/` directory includes `candidate_assessments.json` (the complete assessed eligible pool) and `recommended_shortlist.json`, `.csv`, and `.md` (distinct selected molecules and their evidence). Explicit panel/role objectives and hard requirements take precedence. Otherwise selection uses scientific direction, comparable off-target evidence, applicable motif flags and the native ranking. Missing or incomplete screens remain visible.

Genomic tiling evidence identifies the configured reference indexes by their contents, the searched substitution budget and the counting unit. Its genomic search does not cover spliced transcript junctions. A censored search retains unknown higher-mismatch counts; it cannot be treated as a complete clean screen. Mature-transcript searches retain their separate mapping provenance.

Reference accessions are joined to the resolved transcript only through verified identities. A preservation constraint keeps that same reference binding. An intron objective remains one program with two flanking junctions when the run resumes; those exon references do not create extra design objectives.

Published sequence motifs are context-dependent flags; explicit user motif exclusions are separate hard constraints. Reference Tm and ΔG describe an unmodified perfect-match RNA/DNA duplex at the reported conditions, not validated 2′MOE/PS affinity. For splice-evaluated candidates, `cryptic_splice_evidence` records the largest acceptor/donor score gains outside the supplied annotation and perturbed ASO span, with genomic coordinates and search scope. These computational flags do not establish toxicity, cryptic RNA products, or safety.

## Predictions

Splice scores describe modeled changes at the evaluated junctions. OligoAI scores describe predicted in vitro inhibition under the reported dose, delivery and chemistry conditions. Reports separate these predictions from sequence matches, off-target evidence and experimental measurements.

## Resume a paused request

When a run needs an answer, the terminal prints its state path. Resume with:

```bash
modal-aso --agentic-state-path /path/to/run_state.json
```

The existing calculations stay with the run. Resuming keeps its original output directory, including when you open a new terminal.
