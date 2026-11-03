# Readable HTML reports

The HTML report is an offline reading view of the saved scientific report. It opens with the design goal, the selected ASOs, available predictions and concrete gaps in the results. It then links to findings, limitations and files. Methods and administrative records are expandable. Reports for multiple targets have distinct target labels. A small glossary explains terms that appear in the report. Full tables and downloads remain available, including missing results and unsuccessful analyses.

The scientific Markdown, JSON, CSV and Excel files are unchanged by this view. Rendering does not call a model, run predictions, select candidates, change ranks, or revise a scientific assessment. Known status and exclusion codes receive plain labels only in their own table columns; their exact source values remain in HTML attributes and tooltips as well as the original exports. Numeric cells and unknown codes are untouched. This does not change existing benchmark records. Scientific validation remains a separate workflow.

## Writing rules for the reading view

- Lead with the result and its limits. Use “ASOs,” “candidates,” “results,” “files,” and “methods.” Avoid internal terms such as “native records,” “payload,” “child run,” and “artifact” in navigation.
- Keep target identity, sequence direction, chemistry, units, coordinate conventions, comparison scope, and uncertainty explicit. Never turn “unavailable” into zero or a finished calculation into biological success.
- Say directly what something is. Avoid “X, not Y” constructions and generic warnings about what a count, rank or prediction cannot prove. Explain concrete missing checks and failures.
- Define off-targets as other sequences an ASO may also bind. Describe mismatches as base differences. Keep routine search settings in methods.
- Keep unknown sections visible. Only known methods and administrative sections may be collapsed automatically. Never filter out candidate rows or silently discard columns.
- Change labels at the rendering boundary. Do not apply global word substitutions to scientific prose, sequences, paths, identifiers or machine-readable exports.
- Do not change scientific prompts to improve report style. Those prompts can affect the agent's decisions and require separate scientific qualification.

`aso_agents/readable_report.py` contains the common reading view. The design, analysis and combined-report renderers use it. Scientific prose stays verbatim; the view improves organization and tool-owned language without asking a second model to paraphrase evidence. A deeper prose rewrite would need its own meaning-preservation checks.

`aso_agents/report_orientation.py` builds the opening explanation from saved report sections and structured prediction records. Counts retain their original scope: panel, preview or assessed pool. Sequence-check summaries must agree with the complete status table and any retained flags. Multiple prediction contexts are never pooled. Missing predictions are distinguished from zero activity; an empty main candidate table is distinguished from other candidate lists elsewhere in the report. Comparison summaries count recorded entries and trade-off flags without deduplicating, reranking, or choosing a winner. Unknown or inconsistent formats receive no guessed count cards. Every table-based count links back to its source section.

The opening explanation states the purpose directly. Fully available predictions appear as results with their model and conditions; missing predictions appear as concrete gaps. Design reports put detailed analysis tabs behind “Explore the analysis”; the original request and exact target identifier remain available together. In-page links open folded explanations when needed. Essential findings and limitations remain visible.

The opening labels the intended biological effect as the **goal**. Intermediate
pool counts stay with selection details. A stated search limit describes its
settings; completion is reported separately. For an evaluation or comparison
with a main finding, that finding leads the page. Counts and coverage open on
demand. Checked status summaries and the source's limitations share one section.
Missing results never become zero, and a missing exact match is called out only
when its search is recorded as complete.

Design reports have three main reading paths: **Selected ASOs**, **Chemistry,
predictions and checks**, and **Other designs considered**. A recognized selected
panel opens as individual ASO sequences, with positions and search details under
each one. The complete table remains available below. Predictions with the same
candidate identity, sequence and assay conditions are grouped for reading; all
reference-context rows remain in the full prediction table. Differing predictions
are never averaged. Unrecognized layouts use the complete original table.

Known missing count values and motif identifiers receive plain display labels in
their own columns, with the exact value retained in `data-recorded-value`. A motif
label identifies its study context without assigning that chemistry to the ASO.
The selected molecule, its chemistry, numeric measurements, rank and scientific
assessment remain unchanged.

The title, original request and critical alerts are rendered into the HTML itself.
They remain available without JavaScript. Repeated clicks on a source link reopen
its section, including nested sections; links also restore the appropriate
analysis tab. Included downloads and other file locations share one disclosure.
Missing files are explicitly labelled unavailable.

The view works without external fonts or services. JavaScript adds table-column controls and expand-all; with JavaScript disabled, all table columns are visible and native HTML details controls still work. Printing expands details and restores the screen state afterward. Existing frozen reports should be previewed into a separate directory, never regenerated in place.
## Reading levels

The opening view contains the result, checked counts and known limitations.
Candidate counts describe their source table. Failed predictions, incomplete
off-target searches and genuinely missing requested files appear before the
expandable evidence. Routine off-target search settings stay in methods. Prediction rows are not
pooled across contexts to infer a number of distinct ASOs.

Known evidence sections open on demand: selected ASOs, candidate search pools,
splicing predictions, chemistry, sequence checks and comparisons. Their labels
explain the scope before the reader opens them. Subsections within these blocks
open separately, including individual ASO assessments. Methods and run metadata
have a separate entry point. Unfamiliar scientific sections remain visible.

Long, simple caveat lists retain each caveat's first sentence or short consequence
before a colon on screen, with the remaining exact explanation underneath. An
observation label stays attached to its consequence. Complex lists are left intact. No model
paraphrases the saved scientific source during HTML rendering. Report writing
prompts ask for a short finding and its main uncertainty, followed by detailed
evidence; new text exports put interpretation and limitations before methods.

Opening a source link reveals all its containing sections. Expand-all and print
include the complete evidence, and native details controls work without JavaScript.
All original table cells, sequence roles, chemistry, coordinates and measurement
units remain available. Presentation checks should include both ordinary outputs
and incomplete runs, on desktop and mobile, with source hashes and table-cell
preservation checks for frozen examples.

Where a native table explicitly records a target RNA strand and a synthesized
ASO strand together, the recorded ASO is displayed first. The target strand and
complete original record are expandable, and the original value is retained in
the cell's `data-recorded-value`. No complement is calculated and no missing
ASO strand is inferred. Chemistry descriptions come from the saved molecule specification.

## Small downloads

Individual HTML exports are self-contained reading copies. Small linked files
are embedded as downloads with their exact bytes and SHA-256 hashes. Each linked
file is limited to 512 KiB, with a 1 MiB total. Reference genomes, aligner indexes,
annotations and model weights stay in the working run. Larger evidence is listed
under “Files kept in the original run folder.” Rendering needs no external service.

Combined HTML reports with child frames use a bounded portable folder so each
child report keeps its links. Markdown bundles have a 2 MiB limit per linked
file and an 8 MiB total. Legacy asset directories are checked before copying,
including symlinks. Individual report text itself is retained in full.

Final delivery refreshes download links after the requested files exist and
updates the delivery hashes. Aliases of the same report are omitted from its
own download list to prevent recursive bundles. Real missing files stay visible.
Frozen scientific outputs remain intact; use a separate directory for previews.
