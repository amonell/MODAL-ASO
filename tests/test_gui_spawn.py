"""Use a real spawned process to cover Streamlit main-module re-imports."""
import subprocess
import sys
from pathlib import Path
import pytest

pytest.importorskip("streamlit")


def test_spawn_does_not_execute_streamlit_page_without_session():
    app = Path(__file__).parents[1] / "aso_agents" / "gui" / "app.py"
    # Spawn reimports Streamlit's page as __mp_main__, without a UI session.
    script = """
import __main__
import multiprocessing
import os
import sys
__main__.__file__ = sys.argv[1]
__main__.__spec__ = None
process = multiprocessing.get_context('spawn').Process(target=os.getpid)
process.start()
process.join(timeout=20)
if process.is_alive():
    process.terminate()
    process.join()
    raise RuntimeError('Spawn hung while importing GUI page')
assert process.exitcode == 0, process.exitcode
"""
    result = subprocess.run([sys.executable, "-c", script, str(app)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "missing ScriptRunContext" not in result.stderr
