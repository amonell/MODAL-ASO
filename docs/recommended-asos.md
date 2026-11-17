# Recommended ASOs

Design reports include an evidence-reviewed recommendation panel and retain the
full scored candidate library. Each recommendation explains why the ASO was
chosen, its main tradeoff, and what it adds to the panel. Expand a recommendation
to inspect the saved evidence behind those statements.
Adding the recommended panel keeps its order in the basket and downloads. PMO
recommendations retain their physical chemistry even without supplier notation.

The report stage uses the configured language model, including `OPENROUTER_MODEL`.
Every eligible physical molecule enters a review batch with all its saved analysis
contexts and attributes. Larger libraries use bounded batches followed by comparisons
of their leading choices. Earlier rankings remain available to satisfy required
binding regions or other panel constraints. This is a hierarchical comparison;
small score differences can change the suggested order between fresh reviews.

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
