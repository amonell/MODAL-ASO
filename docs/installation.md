# Installation

Python **3.11**. Linux x86-64, WSL2, or Apple silicon macOS.

## Prerequisites

**Ubuntu / WSL2**

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates zlib1g-dev libbz2-dev liblzma-dev libffi-dev
```

On Windows, [install WSL2](https://learn.microsoft.com/windows/wsl/install) and run these commands inside Ubuntu. Keep the repository and data in the Linux filesystem for faster I/O.

**Apple silicon macOS**

```bash
xcode-select --install
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then follow the README quick start.

## Install from a ZIP

Download and extract the repository ZIP. Open a terminal in the extracted folder:

```bash
uv sync --frozen --extra splice
source .venv/bin/activate
modal-aso setup --species human --features tools references splice
```

Model source code is fetched automatically at pinned revisions.

## Install a wheel

With a release wheel downloaded locally:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "./modal_aso-0.1.0-py3-none-any.whl[splice]"
modal-aso setup --species human --features tools references splice
```

The source installation uses `uv.lock` to reproduce the full Python environment.

## Return to an installation

```bash
cd MODAL-ASO
source .venv/bin/activate
modal-aso "your request"
```

Activation reuses the installed environment. Models and references also persist across new terminals and repository upgrades.

## Models

| Model | Purpose | Setup |
| --- | --- | --- |
| SpliceTransformer | Human splice predictions | `modal-aso setup --species human --features splice` |
| OpenSpliceAI | Mouse splice predictions | `modal-aso setup --species mouse --features splice` |
| OligoAI | Gapmer efficacy predictions | `modal-aso setup --features oligoai` |

Install the corresponding species references before splice models. Splice models run on CPU or compatible NVIDIA GPUs. OligoAI requires Linux x86-64, Python 3.11, and an NVIDIA Ampere or newer GPU with a CUDA 12.4-compatible driver. Its isolated environment uses PyTorch 2.6.0 and FlashAttention 2.7.4.post1.

OligoAI supports DNA/MOE/cET sugars and PO/PS bonds. Reports identify unsupported chemistries and the conditions attached to each prediction.

## Upgrade

```bash
git pull
uv sync --frozen --extra splice
source .venv/bin/activate
modal-aso doctor
```

Run `modal-aso setup` if the new version requests updated assets. Verified assets with unchanged versions are reused.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| `modal-aso: command not found` | Activate `.venv/bin/activate`. |
| Missing compiler during installation | Install the platform prerequisites above. |
| Missing command-line tools | Run `modal-aso setup --features tools`. |
| Missing reference or index | Run `modal-aso setup --species human --features references` (or `mouse`). |
| Interrupted download | Repeat the same setup command; supported transfers resume. |
| Checksum mismatch | Repeat setup to fetch a verified replacement. |
| CUDA unavailable | Check `nvidia-smi` and GPU access in your environment. |
| Authentication error | Set the API key for the selected provider. |
| Existing references on another drive | Use [reference import](references.md#import-existing-references). |

Run `modal-aso doctor --json` for a local installation report. Include its output and the error message when opening an issue. Remove personal paths and request details before sharing logs.
