"""Lazy optional GUI entry point."""
import importlib.util
from pathlib import Path
import subprocess
import sys


def launch(arguments):
    arguments = list(arguments)
    test_mode = "--test" in arguments
    arguments = [argument for argument in arguments if argument != "--test"]
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("GUI is optional. Install with: uv sync --frozen --extra splice --extra gui")
    return subprocess.call([sys.executable, "-m", "streamlit", "run",
                            str(Path(__file__).with_name("app.py")),
                            "--server.address=127.0.0.1", "--server.headless=true", "--browser.gatherUsageStats=false",
                            "--global.developmentMode=false", "--server.fileWatcherType=none",
                            "--theme.base=light", "--theme.backgroundColor=#F5F8FC",
                            "--theme.secondaryBackgroundColor=#F5F8FC",
                            "--theme.textColor=#253D4B", "--theme.primaryColor=#0FA3BE", *arguments,
                            *(["--", "--test"] if test_mode else [])])
