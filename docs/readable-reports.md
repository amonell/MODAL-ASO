# Readable HTML reports

The HTML report is an offline reading view of the saved scientific report. It opens with the design goal, the selected ASOs, available predictions and concrete gaps in the results. It then links to findings, limitations and files. Methods and administrative records are expandable. Reports for multiple targets have distinct target labels. A small glossary explains terms that appear in the report. Full tables and downloads remain available, including missing results and unsuccessful analyses.

The scientific Markdown, JSON, CSV and Excel files are unchanged by this view. Rendering does not call a model, run predictions, select candidates, change ranks, or revise a scientific assessment. Known status and exclusion codes receive plain labels in their corresponding fields; their exact source values remain in HTML attributes and the original exports. Numeric measurements and unknown codes retain their values. This does not change existing benchmark records. Scientific validation remains a separate workflow.

## Writing rules for the reading view

- Lead with the result and its limits. Use “ASOs,” “candidates,” “results,” “files,” and “methods.” Avoid internal terms such as “native records,” “payload,” “child run,” and “artifact” in navigation.
- Keep target identity, sequence direction, chemistry, units, coordinate conventions, comparison scope, and uncertainty explicit. Never turn “unavailable” into zero or a finished calculation into biological success.
- Say directly what something is. Avoid “X, not Y” constructions and generic warnings about what a count, rank or prediction cannot prove. Explain concrete missing checks and failures.
- Define off-targets as other sequences an ASO may also bind. Describe mismatches as base differences. Keep routine search settings in methods.
- Keep unknown sections visible. Only known methods and administrative sections may be collapsed automatically. Never filter out candidate rows or silently discard columns.
- Change labels at the rendering boundary using explicit vocabulary and structured fields. Preserve scientific findings, sequences, identifiers and machine-readable exports.
- Do not change scientific prompts to improve report style. Those prompts can affect the agent's decisions and require separate scientific qualification.

`aso_agents/readable_report.py` contains the common reading view. The design, analysis and combined-report renderers use it. Scientific explanations stay verbatim apart from readable labels for known internal codes and file paths. The view improves organization and tool-owned language without asking a second model to paraphrase evidence. A deeper prose rewrite would need its own meaning-preservation checks.

`aso_agents/report_orientation.py` builds the opening explanation from saved report sections and structured prediction records. Counts retain their original scope: panel, preview or assessed pool. Sequence-check summaries must agree with the complete status table and any retained flags. Multiple prediction contexts are never pooled. Missing predictions are distinguished from zero activity; an empty main candidate table is distinguished from other candidate lists elsewhere in the report. Comparison summaries count recorded entries and trade-off flags without deduplicating, reranking, or choosing a winner. Unknown or inconsistent formats receive no guessed count cards. Every table-based count links back to its source section.

The opening explanation states the purpose directly. Fully available predictions appear as results with their model and conditions; missing predictions appear as concrete gaps. Design reports put detailed analysis tabs behind “Explore the analysis”; the original request and exact target identifier remain available together. In-page links open folded explanations when needed. Essential findings and limitations remain visible.

The opening labels the intended biological effect as the **goal**. Intermediate
pool counts stay with selection details. A stated search limit describes its
settings; completion is reported separately. For an evaluation or comparison
with a main finding, that finding leads the page. Counts and coverage open on
demand. Checked status summaries and the source's limitations share one section.
Missing results never become zero, and a missing exact match is called out only
when its search is recorded as complete.

Design reports open **Selected ASOs**, with chemistry beside the sequences.
**Predictions and selection checks** and **Explore the wider search** open on
demand. A recognized selected panel shows individual ASO sequences, with positions
and search details under each one. Identical check summaries appear once above the
displayed sequences; differing summaries stay beside the affected ASO. The complete
table remains available below. Predictions with the same
candidate identity, sequence and assay conditions are grouped for reading; all
reference-context rows remain in the full prediction table. Differing predictions
are never averaged. Unrecognized layouts use the complete original table.

Known missing count values and motif identifiers receive plain display labels in
their own columns, with the exact value retained in `data-recorded-value`. A motif
label identifies its study context without assigning that chemistry to the ASO.
The selected molecule, its chemistry, numeric measurements, rank and scientific
assessment remain unchanged.

`aso_agents/report_values.py` formats structured evidence for both the static
report and its interactive tabs. Objects become labeled facts, lists become
lists, and nested evidence opens in expandable rows. Explicit BED0 positions
retain their exact endpoints and state that numbering starts at zero and
excludes the end. Missing values, empty lists, empty objects, zero and false
remain distinct. Full tables and target details use this same reading format.
The original JSON remains embedded for provenance and downloads.

The design goal retains typed intron-retention/excision events. A saved objective
preserves alternate-product and other detailed splicing requests that an exon-only
label would flatten. The renderer never infers intent from gene names or keywords.
Preview counts describe distinct molecules; repeated binding records do not create
an apparent shortfall in the sequence preview.

New shortlist Markdown names missing evidence and explains prediction-review flags
from the saved per-molecule assessment. Motif notes state when study chemistry
differs from the designed chemistry. Inapplicable safeguard columns and empty
review-note columns are omitted from this reading table; complete assessment
records remain in JSON and CSV. Joining these checks to sequence cards requires
matching complete panel order and unique sequence identities. Ambiguous, partial,
reordered and repeated-sequence panels keep separate source tables.

Incomplete model review of splice safeguards appears in the shortlist and opening
notices. The report states that recorded default checks were used. Sites approved
for observation retain their measurements with the label “Changes recorded for
context.” Their changes add no safeguard ranking penalty. Full output records
retain each review decision and its biological explanation.

The panel export includes every available mismatch bin. Older summaries explicitly
mark absent bins as missing from that summary, preserving the recorded search
completion status and count units. Missing exon-skipping consequence calculations
appear in the opening with a link to their source; they describe that calculation's
scope and never imply a failure of transcript resolution for the entire design.

Download links are present in the saved HTML. JavaScript leaves portable export
URLs intact. Primary links use the run's current outputs; copies nested inside
earlier export bundles stay out of the primary menu. Additional formats open on
demand, and file sizes appear when recorded. Requested deliverables and primary
output files receive the bounded embedding budget before supporting evidence,
even when evidence links occur earlier in the document.

Known native codes receive readable labels in structured fields and prose.
Unknown field names get spaces between words; unknown values and scientific
identifiers remain literal. Local file paths display the file name or a known
file label, with the full path retained in an HTML attribute. Only complete
structured records are parsed. Chemistry notation, interval expressions with an
unspecified convention, code examples and malformed records stay literal.

The title, original request and critical alerts are rendered into the HTML itself.
They remain available without JavaScript. Repeated clicks on a source link reopen
its section, including nested sections; links also restore the appropriate
analysis tab. Included downloads and other file locations share one disclosure.
Missing files are explicitly labelled unavailable.

The view works without external fonts or services. JavaScript adds table-column controls and expand-all; with JavaScript disabled, all table columns are visible and native HTML details controls still work. Printing expands details and restores the screen state afterward. Existing frozen reports should be previewed into a separate directory, never regenerated in place.

## Selection and interaction

Recommendation filters describe the saved recommendation panel. **In my basket**
shows the reader's chosen molecules, including choices outside that panel.
Recommendation buttons show whether each ASO is in the basket and allow removal.
Clearing the basket offers an immediate undo that restores the selection order;
another selection change ends that undo opportunity. Empty filter results offer
a clear-filters action. Exploring recommendations moves keyboard focus to the
table, and changing a checkbox preserves focus when its row remains visible.

The UX review uses the [Vercel Web Interface Guidelines](https://github.com/vercel-labs/agent-skills/tree/main/skills/web-design-guidelines)
for interaction feedback, keyboard access and recovery. Apply these principles to
the shared renderer and verify real desktop and mobile reports. Scientific
measurements, recommendation reasoning and downloads retain their recorded values.

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
ASO strand together, the recorded ASO is displayed first. The target strand is
expandable, and the complete original value is retained in
the cell's `data-recorded-value`. No complement is calculated and no missing
ASO strand is inferred. Chemistry descriptions come from the saved molecule specification.

## Exon names

Runs retrieve the NCBI RefSeqGene/LRG index and the relevant LRG XML record on
demand. Curated names require an exact transcript identifier and version, genome
assembly, chromosome, strand and complete exon-coordinate match to the source's
transcript annotation. Each name additionally requires an exact match to a
curated exon. Transcript ends with different boundaries retain annotation-order
labels while matching internal exons receive their verified names. Structured
alternative naming includes its published attribution. The resolver handles
reverse-strand alignments and explicit intronic insertions/deletions. Conflicting
names, exonic alignment gaps and incompatible references retain annotation-order
wording, such as “Acceptor of the eighth annotated exon.”

The report's **Exon names and reference** section shows the name, order within
the selected transcript, coordinates and source links. Plot labels and splice
site labels use the same evidence. A scored splice site must also match the
reference's exact splice boundary. Changing the reference invalidates its saved
naming receipt.

Small source records are fetched automatically and held briefly in process
memory. An unavailable source preserves the run's annotation-order labels. The
saved receipt contains names, coordinates, source URLs and hashes; reference
sequences stay out of the report. Report rendering and viewing work offline.

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
