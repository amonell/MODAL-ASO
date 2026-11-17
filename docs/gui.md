# Local browser interface

The optional GUI runs on your own computer and uses the same configuration,
models, references, controller and clarification flow as the command line.
It supports design, evaluation, analysis and question requests through one
conversation interface.

From a source checkout:

```bash
uv sync --frozen --extra gui --extra splice
uv run --no-sync modal-aso gui --output-dir /path/to/runs
```

For an installed release, run `modal-aso gui --output-dir /path/to/runs`.
The `gui` installation extra adds no dependencies: its local host uses Python's
standard library. Complete normal reference/model setup and configure your API
key before starting a run. Use `modal-aso doctor` to inspect the installation.

The command opens a local URL. Paste your request, press **Start run**, and
answer any questions shown. Requests can name local input files using absolute
paths, just as in the CLI. Progress and elapsed time remain visible while the
pipeline works. The resulting report and its supporting files open directly in
the browser. An incomplete recommendation review retains its provisional panel
and displays the same notice as the terminal.

The interface runs one pipeline at a time. **Stop run** preserves completed
outputs; **Resume saved state** continues a stopped run when a state file is
available. A browser refresh keeps the server and run alive. Recent conversations
are saved in `.modal-aso-gui` inside the output directory. The sidebar displays
the latest 50; all saved conversations and outputs remain on disk.

Keep the server terminal open. Ctrl+C stops the server and its active worker.
Use `--no-open` to print the URL or `--port 0` to choose an available port.
The host binds to `127.0.0.1` and uses a session token in its launch URL. It serves
files from registered run folders; linked external reference assets remain
accessible through the local filesystem. This interface is intended for local
use. The command-line interface remains available alongside it.

The UI lives in `aso_agents/resources/gui/`. It receives structured progress,
questions and output links through `gui_worker.py`. New pipeline questions
appear automatically without a duplicate input schema in the frontend.
