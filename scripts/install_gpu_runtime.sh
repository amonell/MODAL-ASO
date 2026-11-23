#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  printf '%s\n' 'Activate the MODAL-ASO virtual environment first.' >&2
  exit 2
fi

python="$VIRTUAL_ENV/bin/python"
backend=auto
torch_requirement=()
doctor_args=()
nvidia_smi=$(command -v nvidia-smi || true)
if [[ -z "$nvidia_smi" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  nvidia_smi=/usr/lib/wsl/lib/nvidia-smi
fi
if [[ -n "$nvidia_smi" ]] && gpu_names=$("$nvidia_smi" --query-gpu=name --format=csv,noheader); then
  if [[ -n "$gpu_names" ]]; then
    doctor_args=(--require-cuda)
  fi
  if grep -Eq 'GeForce RTX 50[0-9]{2}' <<< "$gpu_names"; then
    # CUDA 12.8 supports these Blackwell GPUs. CUDA 13.2 can crash on WSL drivers.
    backend=cu128
    torch_requirement=(torch==2.7.1)
  fi
fi

uv pip install --python "$python" --torch-backend="$backend" --reinstall-package torch \
  "${torch_requirement[@]}" -e '.[splice]'
"$python" -m aso_agents.cli doctor --species human "${doctor_args[@]}"
