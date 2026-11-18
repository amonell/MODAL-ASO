# Recommended ASOs

Design reports include an evidence-reviewed recommendation panel and retain the
full scored candidate library. Each recommendation explains why the ASO was
chosen, its main tradeoff, and what it adds to the panel. Expand a recommendation
to inspect the saved evidence behind those statements.
Adding the recommended panel keeps its order in the basket and downloads. PMO
recommendations retain their physical chemistry even without supplier notation.

To run candidate generation, screening and reporting without the model-based
shortlist review, add `--skip-shortlist`:

```bash
modal-aso --skip-shortlist --output-dir /path/to/runs "your ASO design request"
```

This skips shortlist evidence paging, model ranking, recommendation writing and
recommendation audits. Other pipeline model calls and scientific screening still
run. The full assessed candidate pool remains available; the native selector
retains an **unreviewed provisional panel**, honoring explicit counts, required
regions and spacing. With no requested count it uses the existing fallback size
of four, adjusted for required minima and available candidates.

The option is saved in run state, inherited by child design tasks and retained
when resuming without repeating the flag. It can also be supplied when resuming
before report generation. It does not automatically rebuild already completed
reports. Exported review status is `skipped`, with a receipt in
`artifacts/candidates/recommendation_review_skipped.json`; any earlier review
receipts remain intact. Normal runs without the flag keep the existing review.

The report stage uses the configured language model, including `OPENROUTER_MODEL`.
Large libraries begin with a survey of saved attributes and prediction conditions.
The model can query distributions, apply reversible filters, sort candidates,
compare individual ASOs and inspect original evidence pages. Numeric thresholds,
missing-value handling and aggregation across contexts are explicit. The queries
preserve units, dose, delivery, transcript, species and search scope. Every operation
uses validated structured arguments and exact saved values.

An independent reviewer examines alternatives outside the proposed filters,
including criterion leaders, threshold neighbors, distinct groups and candidates
with missing measurements. Detailed review then reads complete evidence for the
leading candidates and native constraint reserves. Additional candidates required
by the panel contract receive full review before publication. Material contrary
evidence can reopen selection. Unproductive exploration expands into the existing
full-library reviewer, with progress and saved receipts.

Query exploration applies above 128 eligible molecules or the normal comparison
packet limit. Small libraries and all-eligible panel requests retain direct review.
The full assessed library remains available throughout. Two fresh reviews can
choose different, similarly supported alternatives.

The model supplies the candidate preference order. Existing selectors apply requested
counts, program limits, physical spacing, region coverage, ordered roles and explicit
optimization priorities. Counts use distinct sequence-and-chemistry combinations.
An unspecified count receives a compact panel with a written explanation of its size.
Explicit counts and all-eligible requests retain their original meaning.
The canonical reference identifies the design target. A preference for sparing
other isoforms requires recorded preservation targets. Ranking responses that
introduce that preference receive a correction before panel selection.

A separate model pass checks each explanation against the evidence, including
comparisons, biological interpretation, uncertainty and clarity. Candidate identity,
review coverage, structured responses and cited source paths are checked in code.
Numerical ranking claims receive an additional arithmetic check. Repairs preserve
the fields already approved by the reviewer and focus on the cited facts; subsequent
audits retain the full evidence. Panel numbering, signed changes, change magnitudes
and strand-aware binding positions have separate arithmetic checks. Visible internal
site notation and contrastive boilerplate receive focused wording repairs. Invalid responses receive bounded
repair attempts; repeatedly incomplete rankings are retried in smaller batches.
Provider outages keep their bounded retry limit. The saved scientific scores,
sequences and chemistries remain unchanged.

If the review cannot finish, the report identifies a provisional panel from the
saved analysis and keeps the complete library accessible. If the review finds no
defensible recommendation, it explains the empty panel. Missing measurements stay
visible in the supporting evidence.

The run retains `recommendation_review.json`, component receipts, and the explained
`recommended_shortlist.json`, `.csv` and `.md`. Accepted components can resume after
an interruption. Changes to input evidence or the configured reviewer trigger a new
review. Reopening the HTML is offline and uses the saved recommendations.
An explicit retry resumes the last audited draft with another bounded correction
pass. Each resumed draft is checked against the complete evidence again. A repair
must change the flagged text or its supporting citations before it can advance.

Batch requests respect the provider's existing admission limits. With transport
retries enabled, the default OpenRouter admission resource has one slot. Operators
can set `ASO_RESOURCE_LIMIT_LLM_TRANSPORT_OPENROUTER` to a bounded concurrency suited
to their provider quota, or configure `ASO_LLM_CALL_RESOURCE` and its corresponding
`ASO_RESOURCE_LIMIT_<RESOURCE>` setting. All receipts stay inside the run.
Large evidence packets share field paths and repeated values while retaining the
original measurements. Generation that reaches its token allowance receives a
bounded retry with a larger allowance.

Individual molecule records that exceed the comparison capacity are reviewed in
bounded source pages. Query-driven review loads these pages for finalists and
constraint reserves; smaller records reach detailed comparison with all their
original facts. The paging format preserves exact values, source paths,
array order and missing values; partial strings and arrays carry explicit ranges.
Related small facts share their source prefixes. The reviewer selects source
records for comparison without generating replacement measurements or prose
summaries. Further reduction is bounded and must make progress. Ordinary-sized
records keep the existing review path.

Every original page of each selected large molecule is checked again against its
draft recommendation, including facts omitted from the comparison. Concrete
counterevidence is supplied to the existing explanation-repair loop with its exact
source records. An incomplete page review cannot approve a panel. The
`source_evidence_reviews` section of the review receipt records source hashes,
coverage, nominations and final source-page audits; component receipts retain the
inputs and responses. The original assessment pool remains unchanged, and final
citations resolve to its original values. Paging changes review preparation only;
native generation, scoring, hard filters and panel constraints are unchanged.

Query-driven receipts use version `evidence-reviewed-shortlist/3` and add
`review_coverage`, `selection_process`, `compared_ids`, `fully_reviewed_ids` and
`review_strategy`. Coverage distinguishes the local library index, query matches,
individual comparisons and complete evidence reviews. `ranked_ids` remains a
complete permutation for native selection; its `ordering_scope` describes the
reviewed preference order and the remaining library. Reports explain the actual
criteria under **How these ASOs were chosen** and read legacy receipts as before.
Exploration receipts live under `recommendation_review/exploration/`; they retain
validated queries, rejected query diagnostics, exact result hashes and model usage.
The proposed comparison IDs are separate from query-set provenance, so a broad
filter can supply a compact, explicitly chosen comparison. Ordinary measurement
gaps and objections to the filtering strategy have separate review fields.
The model can select exact slices of a stored, sorted query result by offset and
count. This preserves its chosen ordering and avoids copying long candidate-ID
lists. Final explanation audits use small concurrent batches with the full panel
as comparison context. Selected source-page audits also run concurrently; every
original page still receives its assigned check. Mismatch arrays supply a separate
arithmetic table for each bin, with missing values retained.
Reopening the report stays
offline. New installations generate their own index from their run's evidence.
