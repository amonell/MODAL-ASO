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

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then follow the README quick start. The project uses uv’s native Python 3.11 build.

## Install from a ZIP

Download and extract the source ZIP from the [1.0.0 release](https://github.com/amonell/MODAL-ASO/releases/tag/v1.0.0). Open a terminal in the extracted folder:

```bash
uv sync --frozen --extra splice
source .venv/bin/activate
modal-aso setup --species human --features tools references splice
```

Model source code is fetched automatically at pinned revisions.

## Install a wheel

Download the wheel from [GitHub Releases](https://github.com/amonell/MODAL-ASO/releases). For version 1.0.0:

```bash
uv venv --python 3.11 --managed-python
source .venv/bin/activate
uv pip install "./modal_aso-1.0.0-py3-none-any.whl[splice]"
modal-aso --version
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

OligoAI uses BF16 arithmetic on supported GPUs to prevent overflow in its RNA encoder. Each prediction receipt records the arithmetic precision and runner identity. If an individual RNA context produces an invalid estimate, its activity remains unavailable and valid estimates for other contexts are retained.

After a CUDA execution fault, splice scoring gets one retry in a fresh model process. The run keeps the original error, retry inputs, model log and outcome beside the input tile table. A failed retry remains an error. Later splice jobs in the same run also use fresh processes because a CUDA execution fault invalidates the original process's GPU context.

OligoAI supports DNA/MOE/cET sugars and PO/PS bonds. Reports identify unsupported chemistries and the conditions attached to each prediction.

Setup remembers a custom OligoAI directory across sessions:

```bash
modal-aso setup --features oligoai --oligoai-root /path/to/oligoai
```

Register an existing installation without downloading it again or using the GPU:

```bash
modal-aso assets import-model oligoai /path/to/oligoai
```

Registration verifies the released checkpoint and source and remembers their location; it does not certify current CUDA access. Before a batch or a request that requires OligoAI predictions, check the actual execution environment:

```bash
modal-aso doctor --require-model oligoai --verify
modal-aso --require-model oligoai "your request"
```

The required-model check exercises the isolated runtime and GPU before any LLM request. Run it inside the container or sandbox that will execute the workload. That environment must expose the model directory, its Python dependencies and supported GPU access; a host installation or environment variable alone cannot make hidden files or devices accessible. Missing optional predictions remain unavailable and do not establish a failed biological design.

## Upgrade

Choose a version from [GitHub Releases](https://github.com/amonell/MODAL-ASO/releases), then check out its tag. For 1.0.0:

```bash
git fetch origin --tags
git checkout v1.0.0
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
