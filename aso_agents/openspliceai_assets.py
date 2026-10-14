"""Install the release-pinned mouse OpenSpliceAI checkpoint."""

import os
from pathlib import Path

from .asset_store import download
from .paths import data_root

REVISION = "13ff432c211d0871eec27a1884d59df0a99df5eb"
SHA256 = "5498a8ff740b630d380a4d22f5180facf68f8d71c1bdfd438cfe70223ace1ff0"
URL = f"https://raw.githubusercontent.com/Kuanhao-Chao/OpenSpliceAI/{REVISION}/models/openspliceai-mouse/10000nt/model_10000nt_rs10.pt"


def install_mouse_model(*, progress=print):
    root = Path(os.environ.get("OPENSPLICEAI_MODEL_DIR", data_root() / "models/openSpliceAI"))
    return download(URL, root / "OSAI_mouse_10000nt_model1.pt", expected_sha256=SHA256, progress=progress)
