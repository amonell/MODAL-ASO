# Readable HTML reports

The HTML report is an offline reading view of the saved scientific report. It opens with a short explanation of what the report contains, a few counts from the saved records when they can be checked, and guidance on what those results mean. It then links to findings, limitations and files. Methods and administrative records are expandable. Reports for multiple targets have distinct target labels. A small glossary explains terms that appear in the report. Full tables and downloads remain available, including missing results and unsuccessful analyses.

The scientific Markdown, JSON, CSV and Excel files are unchanged by this view. Rendering does not call a model, run predictions, select candidates, change ranks, or revise a scientific assessment. Known status and exclusion codes receive plain labels only in their own table columns; their exact source values remain in HTML attributes and tooltips as well as the original exports. Numeric cells and unknown codes are untouched. This does not change existing benchmark records. Clearer presentation is not evidence of improved scientific performance, and future human or model evaluations of presentation may differ.

## Writing rules for the reading view

- Lead with the result and its limits. Use “ASOs,” “candidates,” “results,” “files,” and “methods.” Avoid internal terms such as “native records,” “payload,” “child run,” and “artifact” in navigation.
- Keep target identity, sequence direction, chemistry, units, coordinate conventions, comparison scope, and uncertainty explicit. Never turn “unavailable” into zero or a finished calculation into biological success.
- Explain necessary scientific terminology. Do not invent new terms for ordinary operations.
- Keep unknown sections visible. Only known methods and administrative sections may be collapsed automatically. Never filter out candidate rows or silently discard columns.
- Change labels at the rendering boundary. Do not apply global word substitutions to scientific prose, sequences, paths, identifiers or machine-readable exports.
- Do not change scientific prompts to improve report style. Those prompts can affect the agent's decisions and require separate scientific qualification.

`aso_agents/readable_report.py` contains the common reading view. The design, analysis and combined-report renderers use it. Scientific prose stays verbatim; the view improves organization and tool-owned language without asking a second model to paraphrase evidence. A deeper prose rewrite would need its own meaning-preservation checks.

`aso_agents/report_orientation.py` builds the opening explanation from saved report sections and structured prediction records. Counts describe the entries shown, not fulfillment of a request or biological success. Sequence-check summaries must agree with the complete status table and any retained flags. Multiple prediction contexts are never pooled. Missing predictions are distinguished from zero activity; an empty main candidate table is distinguished from other candidate lists elsewhere in the report. Comparison summaries count recorded entries and trade-off flags without deduplicating, reranking, or choosing a winner. Unknown or inconsistent formats receive no guessed count cards. Every table-based count links back to its source section.

The opening explanation defines candidates, sequence checks and predictions in ordinary language. Design reports put detailed analysis tabs behind “Explore the analysis”; the original request and exact target identifier remain available together. In-page links open folded explanations when needed. Essential findings and limitations remain visible.

The view works without external fonts or services. JavaScript adds table-column controls and expand-all; with JavaScript disabled, all table columns are visible and native HTML details controls still work. Printing expands details and restores the screen state afterward. Existing frozen reports should be previewed into a separate directory, never regenerated in place.
