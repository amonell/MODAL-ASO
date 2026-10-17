# Splice Evaluation Integrations

MODAL-ASO can use local splice-effect backends when a workflow requests splice scoring. Large model and reference files are local assets and should not be committed.

## SpliceTransformer

`SpliceTransformer/` is a pinned git submodule that links to the upstream SpliceTransformer repository without vendoring its source into MODAL-ASO. MODAL-ASO uses SpliceTransformer for human splice scoring.

Initialize it after cloning this repository, or let the CLI install the missing pieces when it asks:

```bash
git submodule update --init --recursive spliceAI_integration/SpliceTransformer
uv run python -m aso_agents.splice_eval_assets
```

`uv run modal-aso` now treats Enter on the SpliceTransformer preflight prompt as **install splice assets**, not as a claim that the files are already present.

The upstream SpliceTransformer repository hosts its checkpoint outside GitHub. Download the model weights from:

```text
https://drive.google.com/file/d/1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno/view?usp=drive_link
```

Place the downloaded checkpoint here:

```text
spliceAI_integration/SpliceTransformer/model/weights/SpTransformer_pytorch.ckpt
```

Example local setup after the browser download completes:

```bash
mkdir -p spliceAI_integration/SpliceTransformer/model/weights
mv /path/to/SpTransformer_pytorch.ckpt \
  spliceAI_integration/SpliceTransformer/model/weights/SpTransformer_pytorch.ckpt
```

SpliceTransformer also expects hg38 reference files in its `data/data_package` directory:

```bash
mkdir -p spliceAI_integration/SpliceTransformer/data/data_package

curl -L -o /tmp/hg38.fa.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip -c /tmp/hg38.fa.gz > \
  spliceAI_integration/SpliceTransformer/data/data_package/hg38.fa

curl -L -o \
  spliceAI_integration/SpliceTransformer/data/data_package/hg38.annotation.gtf.gz \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
```

## OpenSpliceAI

MODAL-ASO uses OpenSpliceAI for mouse splice scoring and for any configured non-human species. Install the optional Python backend when splice scoring is needed:

```bash
uv sync --extra splice
```

The default mouse model path is:

```text
spliceAI_integration/openSpliceAI/OSAI_mouse_10000nt_model1.pt
```

OpenSpliceAI released models are available from:

```text
https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models
```

For the default mouse model, use one of the 10,000 nt mouse checkpoints from:

```text
https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mouse/10000nt
```

This command installs the `rs10` mouse checkpoint at MODAL-ASO's default path:

```bash
mkdir -p spliceAI_integration/openSpliceAI
curl -L \
  -o spliceAI_integration/openSpliceAI/OSAI_mouse_10000nt_model1.pt \
  https://raw.githubusercontent.com/Kuanhao-Chao/OpenSpliceAI/main/models/openspliceai-mouse/10000nt/model_10000nt_rs10.pt
```

To use a different model file or a directory containing multiple `.pt` checkpoints, set a species-specific environment variable:

```bash
export OPENSPLICEAI_MODEL_MOUSE=/absolute/path/to/model_or_directory
```

For multiple species, create `spliceAI_integration/openSpliceAI/model_registry.json`:

```json
{
  "mouse": {
    "model_path": "spliceAI_integration/openSpliceAI/OSAI_mouse_10000nt_model1.pt",
    "source": "OpenSpliceAI mouse 10000nt rs10"
  }
}
```

You can also point to another registry file with `OPENSPLICEAI_MODEL_REGISTRY`.
