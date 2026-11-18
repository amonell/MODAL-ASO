# Changelog

## Unreleased

- Give the shortlist reviewer structured tools to survey, filter and compare large
  candidate libraries before detailed evidence review. Check omitted alternatives,
  preserve native panel requirements, and retain every ASO in reports and exports.
  Record separate survey, comparison and complete-review counts with resumable
  run-local receipts (#38). Recommendation membership and ordering can change.
- Add an optional browser GUI with provider setup, chat, report and file panels,
  responsive scrolling, and offline `--test` previews. Group SDK serialization
  warnings under optional technical activity. The GUI uses the existing analysis
  controller and provider transport ([#4](https://github.com/amonell/MODAL-ASO/issues/4)).
- Preserve exon-defined transcript groups through clarification, scoring and review. Resolve physical exon identity before design, keep partial transcript membership explicit, and report group coverage separately from selectivity (#31). New group-aware runs can rank candidates differently; historical scientific artifacts retain their recorded values.
- Bound generated analysis programs to five minutes by default, terminate their process groups, and retain timeout diagnostics. Configure the budget through `ASO_ANALYSIS_TIMEOUT_SECONDS` (#1).
- Keep provisional ASOs visible when recommendation review is incomplete, with review status and diagnostics carried into reports and terminal completion (#22).
- Add independent signed junction filters, consistent splice-effect colors, one-row-per-ASO basket exports with all context evidence, readable safeguard details, and searchable filter history. Simplify the default attribute chooser and table while retaining saved ranks and full evidence (#23–#29).

- Keep dense ASO binding maps readable with region navigation, spaced rows,
  collision-free labels, expandable splice-site details and optional value plots.
  Every saved ASO and prediction remains available in the table and downloads.

- Show the saved original request above design approaches, including reports
  regenerated from archived request metadata. Keep compiled objectives as the
  fallback when the original wording is unavailable ([#18](https://github.com/amonell/MODAL-ASO/issues/18)).

## 1.0.1 — 2026-11-16

Faster shortlist review and improvements to request intake, references and terminal progress.

- Accelerate shortlist review for large libraries with parallel nominations,
  compact candidate tables, shorter model-facing identifiers and bounded
  generation. Retain detailed finalist comparisons, full-evidence rationale
  checks and every candidate's availability for selection constraints. Show
  candidate and batch progress during the review
  ([#14](https://github.com/amonell/MODAL-ASO/issues/14)).
- Default all language-model roles to DeepSeek v4.1 Flash `1017` through
  OpenRouter, preserve explicit model/provider overrides, and show the resolved
  model at startup and in `doctor`
  ([#12](https://github.com/amonell/MODAL-ASO/issues/12)).
- Share original chemistry proposals, compiled linkages and native support rules
  across chemistry selection and review. Collect missing design inputs together
  and keep installed references separate from user evidence during intake
  ([#10](https://github.com/amonell/MODAL-ASO/issues/10)).
- Keep the requested biological goals visible while switching report approaches,
  preserving complete goal text and all requested programs
  ([#8](https://github.com/amonell/MODAL-ASO/issues/8)).
- Print periodic elapsed-time updates during long CLI steps, model waits and
  loading, including resumed runs and redirected logs
  ([#7](https://github.com/amonell/MODAL-ASO/issues/7)).
- Warn before transcript resolution when local reference files are missing,
  including the expected paths, the effect on mapping and setup instructions
  ([#5](https://github.com/amonell/MODAL-ASO/issues/5)).
- Recover versioned Ensembl transcript lookups from missing-ID HTTP 400 responses
  while requiring the returned reference to establish the requested version.
- Read compressed local transcript annotations consistently during ID resolution.

The revised shortlist review can change recommendation membership and ordering.
Every eligible ASO remains available for panel constraints and in the saved
candidate pool. Two fresh reviews of 4,057 saved MAPT candidates completed in
4m 50s and 8m 02s, including the final evidence audits.

## 1.0.0 — 2026-11-12

First versioned release of the current MODAL-ASO main branch.

- Natural-language ASO design, evaluation of supplied ASOs, questions about
  existing results and combined reports.
- Evidence-reviewed recommended panels alongside the full assessed candidate
  pool and recorded selection rationale.
- Interactive reports with binding maps, attribute filters, distributions,
  selection baskets and downloadable tables.
- Question and evaluation reports with complete saved evaluation records,
  refreshed download links and portable combined-report exports.
- Locked Python dependencies and installation checks for Linux, Apple silicon
  macOS and Windows through WSL2.
- GitHub issue forms, pull-request checks, protected release tags and a
  documented release process.

This release establishes the versioning baseline for the existing software.
Model predictions retain the assumptions and limitations recorded in each run.
Existing bugs remain tracked in [GitHub Issues](https://github.com/amonell/MODAL-ASO/issues).
