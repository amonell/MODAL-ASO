# Changelog

## Unreleased

- Print periodic elapsed-time updates during long CLI steps, model waits and
  loading, including resumed runs and redirected logs
  ([#7](https://github.com/amonell/MODAL-ASO/issues/7)).
- Warn before transcript resolution when local reference files are missing,
  including the expected paths, the effect on mapping and setup instructions
  ([#5](https://github.com/amonell/MODAL-ASO/issues/5)).
- Recover versioned Ensembl transcript lookups from missing-ID HTTP 400 responses
  while requiring the returned reference to establish the requested version.
- Read compressed local transcript annotations consistently during ID resolution.

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
