"""Lazy optional GUI entry point."""
import importlib.util
from pathlib import Path
import subprocess
import sys


def launch(arguments):
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("GUI is optional. Install with: uv sync --frozen --extra splice --extra gui")
    return subprocess.call([sys.executable, "-m", "streamlit", "run",
                            str(Path(__file__).with_name("app.py")),
                            "--server.address=127.0.0.1", "--server.headless=true", "--browser.gatherUsageStats=false",
                            "--theme.base=light", "--theme.backgroundColor=#EAF4F8",
                            "--theme.secondaryBackgroundColor=#F6FAFC",
                            "--theme.textColor=#253D4B", "--theme.primaryColor=#0FA3BE", *arguments])
