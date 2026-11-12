# Contributing

Start with a [GitHub issue](https://github.com/amonell/MODAL-ASO/issues/new/choose)
for a bug, feature or usage question. Search existing issues first. Bug reports
should include a short reproduction, `modal-aso --version`, the relevant error
and, when useful, `modal-aso doctor --json`. Remove credentials and private data.

## Issues and pull requests

Develop changes on a branch from current `main`, such as `fix/issue-12`,
`feat/report-export` or `release/v1.0.1`. Open a pull request against `main` and
link its issue. Use `Closes #12` when the PR fully resolves that issue. Keep each
PR focused and explain the behavior and validation for a reader new to the work.

Main requires a pull request, resolved review conversations and successful
checks. Maintainers can merge their own PRs after those checks; independent
review is encouraged when another maintainer is available. Published version
tags stay fixed. The `benchmark` branch remains a frozen scientific archive.

## Development checks

For changes:

```bash
uv sync --frozen --extra splice --extra dev
source .venv/bin/activate
modal-aso setup --features tools
pytest
```

The regression suite also uses Node.js and Poppler's `pdftotext` to inspect generated reports. On Ubuntu, install `nodejs poppler-utils`; on macOS, install `node poppler` with Homebrew.

Keep scientific changes separate from installation or presentation changes. Add tests for affected behavior and preserve sequence, coordinate, chemistry and evidence consistency.

API and GPU integration checks are opt-in. Ordinary tests use local fixtures and mocked model calls.

## Versions and releases

Use `MAJOR.MINOR.PATCH`: patch releases fix compatible behavior, minor releases
add compatible functionality, and major releases introduce incompatible changes
to public commands, configuration or saved output contracts. Explain scientific
behavior changes explicitly in the changelog. The package version identifies the
software; each run also retains its model and reference provenance.

For each release:

1. Open a release issue and PR. Update `pyproject.toml`,
   `aso_agents/__init__.py`, `CITATION.cff`, installation examples and
   `CHANGELOG.md`. Run `uv lock` to synchronize the project version while keeping
   existing dependency pins. Run `python ci/check_version.py`.
2. Merge the PR after **Required checks** passes. Wait for **Test and install**
   to pass on that exact commit on `main`.
3. From a clean checkout of that commit, run
   `git tag -a vX.Y.Z -m "MODAL-ASO X.Y.Z"` and
   `git push origin vX.Y.Z`. Use the version validated by `check_version.py`.
4. The **Release** workflow verifies the tag, version metadata and successful
   main checks, then creates a draft GitHub release with the wheel, source
   distribution and `SHA256SUMS`. Review its notes and files, then publish with
   `gh release edit vX.Y.Z --draft=false --latest` or the GitHub release page.

If release preparation fails, rerun the failed workflow after fixing the cause.
For a manual workflow run, select the existing version tag as its ref. Published
tags and assets are retained; corrections receive a new patch version.
