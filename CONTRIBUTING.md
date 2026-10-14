# Contributing

Open an issue with a short reproduction, the relevant error and `modal-aso doctor --json` output. Remove credentials and private data before sharing files.

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
