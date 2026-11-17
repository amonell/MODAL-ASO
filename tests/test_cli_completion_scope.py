"""Completion text must preserve execution limitations without changing status."""
from copy import deepcopy
import io

import pytest

from run_goal_intake import _CliNarrator


@pytest.mark.parametrize("limitation", [
    "Native design stages did not execute; only sequence tiling was completed.",
    "One comparison input was unavailable, so only the available set was analyzed.",
])
def test_completion_surfaces_reported_limitations_and_preserves_output_links(limitation):
    screen = io.StringIO()
    narrator = _CliNarrator(stream=screen, enable_stream=False, enable_spinner=False)
    result = {"status": "run_complete", "execution_limitation": limitation,
              "run_dir": "/outputs/run", "artifacts": {
                  "report_md": "/outputs/run/report.md", "report_html": "/outputs/run/report.html"}}
    before = deepcopy(result)
    narrator.show_completion(result)
    output = screen.getvalue()
    assert "finished with an execution limitation" in output
    assert limitation in output
    assert "The run is complete." not in output
    assert "Report: /outputs/run/report.md" in output
    assert "HTML summary: /outputs/run/report.html" in output
    assert result == before


@pytest.mark.parametrize("limitation", [None, "", "  "])
def test_complete_native_and_calculation_runs_keep_existing_success_behavior(limitation):
    screen = io.StringIO()
    narrator = _CliNarrator(stream=screen, enable_stream=False, enable_spinner=False)
    result = {"status": "run_complete", "execution_limitation": limitation}
    narrator.show_completion(result)
    assert screen.getvalue() == (
        "The run is complete. I saved the outputs and wrapped everything into a clean run folder.\n")
    assert result["status"] == "run_complete"
