# Reading a run

During a run, the terminal announces each main step. A single status line shows
the active operation and counts up its elapsed time in place, for example,
`Predicting splice effects (2m 00s)`. Updates also cover loading, model waits and
resumed steps. Redirected logs and `--raw-cli` output receive a new status line
after every 30 seconds without another status message.
The timer measures time spent waiting for the current operation to return.
Updates pause while the run asks for your input and stop when execution ends.

Start with **`report.html`**. It summarizes the request, candidate panel, calculations and limitations. Its links lead to the supporting tables and files.

**What this run was asked to do** keeps the recorded biological objectives visible
as you switch targets and approaches. It includes requested programs that retained
no candidates. The selected approach's splice-site aim and ranking rationale
appear under **Targets and approaches**.

**Targets and approaches** shows each saved approach rank, confidence and reason beside target selection. **Compare all ranked approaches** expands the alternatives and their reasons, including approaches that retained zero candidates. Rankings stay within the design job that produced them. Each ASO’s details retain its saved rank within the approach; the main table opens on the biological measurements. Runs that evaluate one approach say so; unavailable comparison rankings remain explicit. These views use the saved analysis and make no new ranking calls.

For gapmer requests targeting an exon-defined transcript group, selectivity counts
all resolved members of that group. The transcript used to locate the exon is
only its coordinate reference. Membership uses a verified exon span or an
annotated junction that bypasses it; incomplete and incompatible structures stay
unknown. Equal selectivity favors coverage of more resolved target members.
Unknown matches enter a conservative selectivity lower bound and remain visible
in the saved evidence. An unresolved defining feature makes group selectivity
unavailable. `isoform_group_resolution` records the catalog membership once in
the scored table; each row retains its definition, match counts, coverage and
ranking status. These priorities describe sequence matches within the recorded catalog.
Single-transcript and whole-gene objectives retain their respective selectivity
and catalog-coverage scoring. Existing saved ranks are not recalculated.

Design reports open with a short fact strip, the goal, a binding map and a searchable table of the saved scored pool. Choose a target group to view its predictions. Panel badges identify the recommended molecules. Each row expands into its chemistry, selection evidence and splice checks. Evaluation, comparison, sequence-check and combined reports use the same layout with the evidence appropriate to their task.

Use the checkboxes or map to collect ASOs in the selection basket. The basket exports CSV, FASTA and a supplier order sheet containing the exact saved ordering notation. Scale and purification are editable. Chemistry discrepancies appear beside the order download; missing notation keeps that download disabled. Selections persist in browsers that allow local storage. The report works offline, and large tables use search and pagination.

The **Filters** sidebar sits beside **Where the ASOs bind** and stacks above it on smaller screens. **Color by** offers numeric measurements and attributes that group ASOs into a small set of shared categories. Use the OligoAI, uniqueness and conservation shortcuts when those values are available. Grey indicates unavailable values. **Add filter** opens with common measurements. Search or enable **Show all saved fields** to inspect every saved attribute, including names and ordering notation. Selected attributes also appear as sortable table columns.

The shared binding-map renderer automatically adds a transcript overview and
aligned regulatory tracks when the run has saved annotation evidence. The
overview follows RNA order, 5′ to 3′, with compressed introns. Both the overview
and detailed map distinguish **5′ UTR, CDS and 3′ UTR** using the resolved
transcript's own coding boundaries. Missing or inconsistent CDS annotation stays
gray; an exon label alone never establishes UTR membership.

Materialized RBP and miRNA constraints are scoped to their design program,
transcript, assembly, chromosome and strand. RBP peaks and predicted miRNA sites
use separate tracks aligned with the ASO intervals, plus an expandable table of
exact coordinates and source evidence. Source cell types remain visible; a peak
from another cell type is not relabeled as evidence for the requested model.
Hover a regulatory interval for its target, coordinates and source. Requested
sites remain visible when candidate filters remove ASOs, including sites without
overlapping candidates. Dense annotations use two bounded tracks and a scrolling
evidence table. Report generation uses saved records only and does not fetch
sites, rerun models or alter scientific results.

**Filtered candidates** shows recorded exclusions separately from the scored ASO table and recommendation basket. Search by sequence, binding position or reason; filter by stage or disposition; open **Source & chemistry** for the saved evidence; or download the matching records as CSV. It also appears when every candidate was excluded. The viewer paginates large pools and keeps exact saved sequences, ordering notation and binding coordinates. Missing evidence stays unavailable. Passing scoring without being recommended is not a filter failure.

This history uses the same read-only projector for design and supplied-candidate HTML reports, including steric and gapmer jobs. It reads native rejection buckets, admission receipts with `excluded_path` / `rejected_path`, `filtered_<rule>: {path, reason}` summaries, and `excluded_rows` / `rejected_rows` in evaluation artifacts. New producers can also write `artifacts/candidates/candidate_exclusions.json` with `rows` carrying explicit antisense sequence, chemistry, coordinates, job/program, stage and reason. Generic `filtered_tiles.json` files contain survivors and are never interpreted as rejection lists. Sequence-only stage traces establish a removal stage, not an unrecorded filter reason; ambiguous chemistry or binding contexts are not guessed. Report generation does not rerun filters, models or selection.

**Add filter** opens a distribution with minimum and maximum sliders and precise numeric inputs. Category filters offer checkboxes and text search. Add several filters to keep ASOs that satisfy all of them. Each distribution shows the ASOs passing the other filters, so you can adjust a range without losing the surrounding distribution. For attributes with several recorded values per ASO, choose whether every value or any value must match. Missing values have a separate inclusion checkbox. **Download matching ASOs** and the basket CSV export one row per physical ASO, in selection order. Chemistry distinguishes molecules with identical sequences. Shared measurements appear in ordinary columns; the `context_evidence_json` column preserves each binding/scoring context, common and selected measurements, simulation values, safeguard bounds and recorded OligoAI conditions at full precision. **Full context evidence (JSON)** includes every saved attribute and reference definition; the CSV names its companion evidence file. In the sidebar, **More downloads** provides full evidence for matching ASOs. Measurements that differ between contexts remain in that context evidence. Up to 32 distinct junctions also receive readable mean-effect columns. Your selection basket remains available as you change filters.

Conservation keeps phastCons and phyloP separate, including zero and negative measurements. Uniqueness keeps off-target counts with different mismatch limits or counting units separate. Combined gene counts use the saved distinct-gene total, preserving genes shared across mismatch groups as one gene. Isoform selectivity and isoform coverage have distinct labels. Large reference sequences and file paths appear as readable identities; their full contents stay in the supporting files. The HTML shares repeated evidence internally and needs no downloaded cache or online service for these controls.

**Junction filters** exposes each recorded acceptor and donor independently. Predictions show the signed mean of deletion and replacement simulations: negative values mean reduced junction use and positive values mean increased use. Teal shows reductions; ochre shows increases; grey marks unavailable values. Incomplete or conflicting pairs remain unavailable. Reduction objectives sort the strongest reductions first and offer a maximum-effect threshold. All selected junction ranges apply together and persist across response-site views. Reference, job and model identity keep distinct measurements separate.

Each ASO’s expanded splice checks show required or review-only enforcement, directional bounds, individual simulation values, reference coordinates and biological rationale. A recorded failure remains visible when an average falls inside the bound. Recommendation tables and Markdown reports retain the same check details. Off-target counts retain their search scope and units. The zero-exact-or-one-mismatch filter requires completed screening and both counts.

Gapmer tables show **OligoAI RNA reduction** predictions as percentages. OligoAI evaluates the full assessed gapmer pool before the final panel is selected. Expand a value for its RNA reference, binding site and conditions. The default view includes all RNA contexts at the selected dose and delivery method, including mature-RNA predictions for ASOs spanning exon junctions. Use the context selector to inspect a particular reference or another set of conditions. Predictions appear first; the analysis-rank column preserves the original design ranking. Ranges cover separate RNA contexts or binding sites, with sorting and minimum filters using the lower value. The CSV includes a row for each saved prediction, with full precision and its dose, delivery method and model provenance.

State prediction conditions in the request, for example, “Use 25 nM with free uptake.” Intake converts concentration units to nM and preserves the delivery category: free uptake (Gymnosis), lipid transfection (Lipofection), Electroporation or Other. Conditions can also be specified for each design program. Unspecified settings use a labelled reference of 100 nM and Lipofection. Cell type, treatment duration, medium and reagent details remain visible as experiment context outside the model inputs. An explicit dose or delivery requirement that needs clarification leaves the prediction pending resolution. Supported chemistry and available RNA references are required for a prediction; each unavailable result carries its reason.

Use **CSV/TSV tables** to inspect sequences and measurements. Use **GenBank files** to view candidate positions alongside the reference sequence. Chemistry descriptions belong to the individual molecules; sequence alone does not describe their modifications.

Open **`report.md`** for a text version. The **`artifacts/`** directory retains inputs, structured results and calculation details.

Keep the whole run folder together when moving it. Report links use paths within that folder.

The `artifacts/candidates/` directory includes `candidate_assessments.json` (the complete assessed pool), `recommendation_review.json` (the model review), and `recommended_shortlist.json`, `.csv`, and `.md` (distinct selected molecules and their reasons, tradeoffs and evidence). The configured model reviews the eligible molecules and compares the leading choices using their saved analysis. Explicit panel/role objectives and hard requirements take precedence. A separate review checks each explanation against the evidence. See [Recommended ASOs](recommended-asos.md) for coverage, validation and recovery behavior.

Shortlists include genomic coordinates, producing programs and jobs, response-site predictions and the recorded rank with its scope. Junction-spanning ASOs retain every binding segment. Named rank components accompany the original ranking vectors for compatibility. `design_qualification.json` records filter thresholds from execution receipts, including inclusive or exclusive GC bounds where recorded. Older runs retain an explicit unavailable status for missing receipts.

An exact requested count sets both a minimum and a maximum. A minimum-only request sets the required minimum. Global totals and independently requested per-site bounds retain their own scope. When the panel size is unspecified, the review chooses and explains a compact count while preserving required minimums and region coverage. An all-eligible request includes every eligible molecule. The complete assessed library stays available alongside the panel.

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
