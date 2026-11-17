"""Failure causes must be useful without exposing credentials or locals."""
from aso_agents.gui.diagnostics import diagnostic


def test_diagnostic_preserves_exception_and_location_but_redacts_credentials():
    sensitive_local = "local-value-must-not-be-recorded"
    try:
        raise TypeError("Could not serialize output; key=example-secret Bearer alternate-secret sk-other-secret")
    except TypeError as exc:
        result = diagnostic(exc, "example-secret")
    assert result["type"] == "TypeError"
    assert "Could not serialize output" in result["message"]
    assert "test_gui_diagnostics.py" in result["traceback"]
    assert "example-secret" not in str(result)
    assert "alternate-secret" not in str(result)
    assert "sk-other-secret" not in str(result)
    assert sensitive_local not in str(result)
