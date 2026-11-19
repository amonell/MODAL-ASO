# Changelog

## Unreleased

## 1.1.0 — 2026-11-19

### Browser interface

- Run `modal-aso gui` to submit requests, answer clarification questions, resume a run, and open its report and downloads in one browser tab. Browser and terminal runs share the same controller, provider settings and saved outputs. The optional `gui` extra includes a preview you can explore offline with `modal-aso gui --test`. [PR #47](https://github.com/amonell/MODAL-ASO/pull/47)
- OpenRouter credit and budget errors include guidance on what to check. Technical SDK warnings are available in the activity details. [PR #47](https://github.com/amonell/MODAL-ASO/pull/47)

### Recommendations and reports

- The shortlist reviewer can survey, filter and compare a large library before reviewing individual candidates. It counts distinct candidates across searches and checks omitted alternatives against the evidence. These changes can affect which ASOs it recommends and their order. The full candidate set stays available in reports and downloads. [PR #40](https://github.com/amonell/MODAL-ASO/pull/40) · [PR #44](https://github.com/amonell/MODAL-ASO/pull/44)
- Review resumes from saved comparisons and can correct a rationale against its original measurements. If review pauses, the report identifies provisional candidates and the terminal shows the review status. [PR #34](https://github.com/amonell/MODAL-ASO/pull/34) · [PR #44](https://github.com/amonell/MODAL-ASO/pull/44)
- Requests about exon-defined transcript groups keep the selected group through clarification, scoring and review. Reports distinguish coverage of the group from selectivity for it. This can change candidate rankings in new runs. [PR #33](https://github.com/amonell/MODAL-ASO/pull/33)
- Reports lead with the original request when available. Dense binding maps have region navigation, spaced labels and expandable splice-site details. Signed junction filters, filter history and one-row-per-ASO basket exports make it easier to compare and share candidates. [PR #19](https://github.com/amonell/MODAL-ASO/pull/19) · [PR #21](https://github.com/amonell/MODAL-ASO/pull/21) · [PR #35](https://github.com/amonell/MODAL-ASO/pull/35)

### Questions and file analysis

- Questions about supplied files use the full input records and retain the evidence for the answer. A missing input prompts for a file, an empty result can complete with a count of zero, and resumed analyses retain earlier exports. [PR #46](https://github.com/amonell/MODAL-ASO/pull/46)
- Generated analysis programs have a five-minute default time limit and save diagnostics when it expires. Set `ASO_ANALYSIS_TIMEOUT_SECONDS` to adjust it. [PR #34](https://github.com/amonell/MODAL-ASO/pull/34)

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
