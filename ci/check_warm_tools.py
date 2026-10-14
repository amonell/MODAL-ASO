"""A second setup must use installed tools without a downloader or package solver."""
from unittest.mock import patch
from aso_agents.setup_tools import install_tools

with patch("aso_agents.asset_store.urlopen", side_effect=AssertionError("Unexpected download")), \
     patch("aso_agents.setup_tools.subprocess.run", side_effect=AssertionError("Unexpected package install")):
    assert install_tools()["status"] == "ready"
print("Warm tool setup: no network or package installation")
