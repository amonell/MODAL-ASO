#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential git curl ca-certificates python3-venv zlib1g-dev libbz2-dev liblzma-dev libffi-dev nodejs poppler-utils
workspace=$(mktemp -d -t modal-aso-install-XXXXXX)
cp -a . "$workspace/source"
cd "$workspace/source"
export MODAL_ASO_DATA_DIR="$workspace/data"
export MODAL_ASO_CACHE_DIR="$workspace/cache"
export MODAL_ASO_CONFIG_FILE="$workspace/config.json"
export OPENAI_API_KEY=test-only
python3 -m venv "$workspace/bootstrap"
"$workspace/bootstrap/bin/python" -m pip install uv==0.13.1
"$workspace/bootstrap/bin/uv" sync --frozen --extra splice --extra dev
.venv/bin/modal-aso setup --features tools
.venv/bin/python ci/check_warm_tools.py
.venv/bin/python -m pytest -q tests/test_asset_store.py tests/test_splice_eval_assets.py tests/test_cli_public_surface.py
ASO_RUN_INSTALLED_SMOKE=1 .venv/bin/python -m pytest -q tests/integration/test_installed_runtime.py::test_packaged_tiling_from_a_directory_with_spaces
