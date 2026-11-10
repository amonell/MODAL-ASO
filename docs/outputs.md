# Reading a run

Start with **`report.html`**. It summarizes the request, candidate panel, calculations and limitations. Its links lead to the supporting tables and files.

Design reports open with a short fact strip, the goal, a binding map and a searchable table of the saved scored pool. Choose a target group to view its predictions. Panel badges identify the recommended molecules. Each row expands into its chemistry, selection evidence and splice checks. Evaluation, comparison, sequence-check and combined reports use the same layout with the evidence appropriate to their task.

Use the checkboxes or map to collect ASOs in the selection basket. The basket exports CSV, FASTA and a supplier order sheet containing the exact saved ordering notation. Scale and purification are editable. Chemistry discrepancies appear beside the order download; missing notation keeps that download disabled. Selections persist in browsers that allow local storage. The report works offline, and large tables use search and pagination.

Splice prediction colors run from grey to teal as support for the selected direction increases. The prediction is the mean of the two recorded perturbation simulations; incomplete pairs remain unavailable. Off-target counts retain their search scope and units. The zero-exact-or-one-mismatch filter requires completed screening and both counts.

Gapmer tables show saved **OligoAI RNA reduction** predictions as percentages. Expand a value for its RNA reference, binding site and conditions. A context selector appears when the run contains several references or conditions. Predictions appear first; the analysis-rank column preserves the original design ranking. Ranges cover multiple binding sites in one context, with sorting and minimum filters using the lower value. The CSV includes a row for each saved prediction, with full precision and its dose, delivery method and model provenance.

Use **CSV/TSV tables** to inspect sequences and measurements. Use **GenBank files** to view candidate positions alongside the reference sequence. Chemistry descriptions belong to the individual molecules; sequence alone does not describe their modifications.

Open **`report.md`** for a text version. The **`artifacts/`** directory retains inputs, structured results and calculation details.

Keep the whole run folder together when moving it. Report links use paths within that folder.

The `artifacts/candidates/` directory includes `candidate_assessments.json` (the complete assessed eligible pool) and `recommended_shortlist.json`, `.csv`, and `.md` (distinct selected molecules and their evidence). Explicit panel/role objectives and hard requirements take precedence. Otherwise selection uses scientific direction, comparable off-target evidence, applicable motif flags and the native ranking. Missing or incomplete screens remain visible.

Shortlists include genomic coordinates, producing programs and jobs, response-site predictions and the recorded rank with its scope. Junction-spanning ASOs retain every binding segment. Named rank components accompany the original ranking vectors for compatibility. `design_qualification.json` records filter thresholds from execution receipts, including inclusive or exclusive GC bounds where recorded. Older runs retain an explicit unavailable status for missing receipts.

An exact requested count sets both a minimum and a maximum; a minimum-only request does not cap the panel. Global totals and independently requested per-site bounds retain their own scope. When no upper bound or all-eligible preference is supplied, the final shortlist-size question remains available after scoring, with existing minimums and scientific artifacts preserved.

Genomic tiling evidence identifies the configured reference indexes by their contents, the searched substitution budget and the counting unit. Its genomic search does not cover spliced transcript junctions. A censored search retains unknown higher-mismatch counts; it cannot be treated as a complete clean screen. Mature-transcript searches retain their separate mapping provenance.

Reference accessions are joined to the resolved transcript only through verified identities. A preservation constraint keeps that same reference binding. An intron objective remains one program with two flanking junctions when the run resumes; those exon references do not create extra design objectives.

A supplied sequence can provide isoform annotations and define a genetic background. Detected differences trigger a per-file question when that background role is unresolved. Explicit annotation-only choices retain the differences as evidence without applying them; background choices apply supported substitutions before prediction. Whole-file mapping records its ungapped search scope and difference budget. Unresolved whole-file placement leaves only exon-level evidence and is reported as incomplete. File-content changes invalidate cached annotations and sequence-bound choices.

If a count is required at each nominated junction window, the final panel must cover each window separately, including when one design program searches their union. Mature-RNA candidates retain their producing job so the final panel can recheck the applicable region constraints.

Published sequence motifs are context-dependent flags; explicit user motif exclusions are separate hard constraints. Reference Tm and ΔG describe an unmodified perfect-match RNA/DNA duplex at the reported conditions, not validated 2′MOE/PS affinity. For splice-evaluated candidates, `cryptic_splice_evidence` records the largest acceptor/donor score gains outside the supplied annotation and perturbed ASO span, with genomic coordinates and search scope. These computational flags do not establish toxicity, cryptic RNA products, or safety.

## Predictions

Splice scores describe modeled changes at the evaluated junctions. OligoAI scores describe predicted in vitro inhibition under the reported dose, delivery and chemistry conditions. Reports separate these predictions from sequence matches, off-target evidence and experimental measurements.

## Resume a paused request

When a run needs an answer, the terminal prints its state path. Resume with:

```bash
modal-aso --agentic-state-path /path/to/run_state.json
```

The existing calculations stay with the run. Resuming keeps its original output directory, including when you open a new terminal.
