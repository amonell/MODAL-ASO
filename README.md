# MODAL-ASO · benchmark version

**Multi-mechanism Orchestration and Design by an Agentic LLM for ASOs**

Frozen tool source used to generate the 28-request benchmark runs on 2–3 October 2026.

- Source commit: `b5c31f66c483143d009fe01576e7b16e04635af7`
- Frozen: 2 October 2026, 15:47:40 UTC
- Models: DeepSeek V4 None/Max and V4.1 Max through OpenRouter
- [Current release](https://github.com/amonell/MODAL-ASO/tree/main)

## Install

Use Python 3.11 on Linux or WSL2, with [uv](https://docs.astral.sh/uv/).

```bash
git clone --branch benchmark --single-branch https://github.com/amonell/MODAL-ASO.git
cd MODAL-ASO
git submodule update --init --recursive
uv sync --extra splice
uv run python -m aso_agents.runtime_binaries
```

Set your API key, model and reference directory:

```bash
export ASO_LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_MODEL="your-model-id"
export ASO_REFERENCE_DATA_ROOT="/path/to/reference_data"
uv run modal-aso "Your design or evaluation request"
```

Genomes, indexes and model checkpoints are separate downloads. See [splice model setup](spliceAI_integration/README.md). Reports and generated files are written under `runs/`.

## Source integrity

All included Python implementation files, the tiling workflow and dependency lock are unchanged from the frozen source. `BENCHMARK_SOURCE.json` records their SHA-256 hashes and the original SpliceTransformer revision.

This standalone archive has fresh Git history. Publication packaging omits private orchestration, internal notes and manuscript assets; removes the internal evaluation CLI entry point; and adds this README, license, acknowledgments and source manifest. The native MODAL-ASO implementation is unchanged.

This branch preserves the original generation version. Later report-only corrections used source revisions `b503a8153f4f8b62f393b1f6d773a2b831499b07` and `e7b51b55a86951f9606c56a7549a82be71d32626`.
