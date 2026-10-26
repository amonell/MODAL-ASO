# Readable HTML reports

The HTML report is an offline reading view of the saved scientific report. It opens on results and limitations, uses short headings, and puts methods and administrative records in expandable sections. Reports for multiple targets have distinct target labels. A small glossary explains terms that appear in the report. Full tables and downloads remain available, including missing results and unsuccessful analyses.

The scientific Markdown, JSON, CSV and Excel files are unchanged by this view. Rendering does not call a model, run predictions, select candidates, change ranks, or revise a scientific assessment. Known status and exclusion codes receive plain labels only in their own table columns; their exact source values remain in HTML attributes and tooltips as well as the original exports. Numeric cells and unknown codes are untouched. This does not change existing benchmark records. Clearer presentation is not evidence of improved scientific performance, and future human or model evaluations of presentation may differ.

## Writing rules for the reading view

- Lead with the result and its limits. Use “ASOs,” “candidates,” “results,” “files,” and “methods.” Avoid internal terms such as “native records,” “payload,” “child run,” and “artifact” in navigation.
- Keep target identity, sequence direction, chemistry, units, coordinate conventions, comparison scope, and uncertainty explicit. Never turn “unavailable” into zero or a finished calculation into biological success.
- Explain necessary scientific terminology. Do not invent new terms for ordinary operations.
- Keep unknown sections visible. Only known methods and administrative sections may be collapsed automatically. Never filter out candidate rows or silently discard columns.
- Change labels at the rendering boundary. Do not apply global word substitutions to scientific prose, sequences, paths, identifiers or machine-readable exports.
- Do not change scientific prompts to improve report style. Those prompts can affect the agent's decisions and require separate scientific qualification.

`aso_agents/readable_report.py` contains the common reading view. The design, analysis and combined-report renderers use it. Scientific prose stays verbatim; this first step improves organization and tool-owned language without asking a second model to paraphrase evidence. A deeper prose rewrite would need its own meaning-preservation checks.

The view works without external fonts or services. JavaScript adds table-column controls and expand-all; with JavaScript disabled, all table columns are visible and native HTML details controls still work. Printing expands details and restores the screen state afterward. Existing frozen reports should be previewed into a separate directory, never regenerated in place.
