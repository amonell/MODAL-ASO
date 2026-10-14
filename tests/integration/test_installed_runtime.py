"""Opt-in checks for the installed native workflow and released GPU models."""

import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.real_runtime,
              pytest.mark.skipif(os.environ.get("ASO_RUN_INSTALLED_SMOKE") != "1",
                                 reason="Set ASO_RUN_INSTALLED_SMOKE=1 after setup")]


def test_packaged_tiling_from_a_directory_with_spaces(tmp_path, monkeypatch):
    from aso_agents import tiling_agent
    from aso_agents.runtime_binaries import refresh_runtime_path
    refresh_runtime_path()
    reference = tmp_path / "reference with spaces"
    reference.mkdir()
    randomizer = random.Random(401)
    sequence = "".join(randomizer.choices("ACGT", k=3000))
    fasta = reference / "neutral.fa"
    fasta.write_text(">chr1\n" + sequence + "\n")
    gtf = reference / "neutral.gtf"
    gtf.write_text('chr1\tfixture\texon\t1\t3000\t.\t+\t.\tgene_id "gene1"; transcript_id "tx1"; gene_name "NEUTRAL";\n')
    sizes = reference / "neutral.sizes"
    sizes.write_text("chr1\t3000\n")
    prefix = reference / "neutral"
    subprocess.run(["bowtie-build", str(fasta), str(prefix)], check=True, capture_output=True)
    cfg = SimpleNamespace(genome_fasta=fasta, phastcons_path=reference/"unused.bw",
                          phylop_path=reference/"unused2.bw", chrom_sizes_path=sizes,
                          bowtie_prefix=prefix, gtf_path=gtf, conservation_mode="disabled", ucsc_build="hg38")
    monkeypatch.setattr(tiling_agent, "get_species_config", lambda *_: cfg)
    output = tmp_path / "results with spaces"
    manifest = tiling_agent.emit_pipeline_config(
        json.dumps({"gene_symbol": "NEUTRAL", "species": "human", "gc_min": 0, "gc_max": 1}),
        json.dumps({"rows": [["chr1", 400, 500, "neutral_region", 0, "+"]]}),
        json.dumps({"chemistry": "5-10-5"}), output_dir=str(output))
    completed = subprocess.run(manifest["snakemake_cmd"], shell=True, text=True, capture_output=True)
    (tmp_path / "native.stdout").write_text(completed.stdout)
    (tmp_path / "native.stderr").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    results = tiling_agent.collect_results_json(str(output), species="human")
    assert not results.get("error"), results
    rows = json.loads(Path(results["json_path"]).read_text())["rows"]
    assert len(rows) == 81
    assert (output / "NEUTRAL.scored.idt.tsv").is_file()
    assert (output / "NEUTRAL.trackhub.txt").is_file()
    assert not (Path(tiling_agent.PIPELINE_ROOT) / ".snakemake").exists()


@pytest.mark.skipif(os.environ.get("ASO_RUN_GPU_SMOKE") != "1", reason="Set ASO_RUN_GPU_SMOKE=1 on a supported NVIDIA GPU")
def test_released_oligoai_inference_and_result_reuse(tmp_path):
    from aso_agents.oligoai import (evaluate_candidates, reverse_complement,
                                   MODEL_SHA256, SOURCE_REVISION)
    from aso_chemistry import encode_idt_oligo
    from aso_molecule import molecule_record
    sequence = "ACGTCAGACTTAGCACTGCA"
    notation = encode_idt_oligo(sequence, "5-10-5")
    record = molecule_record(sequence, "5-10-5", notation, assumption_status="configured_intent_unknown")
    row = {"sequence": reverse_complement(sequence), "synthesized_antisense_sequence": sequence,
           "chemistry_idt": notation, "source_modality": "gapmer", "molecule_specification": record}
    reference = {"label": "neutral", "transcript_id": "fixture", "species": "human", "kind": "pre_mrna",
                 "sequence": "G"*50 + reverse_complement(sequence) + "C"*50,
                 "genomic_start0": 100, "genomic_end0": 220, "strand": "+", "chrom": "chr1",
                 "site_start0": 50, "site_end0": 70, "provenance": {"source": "synthetic_smoke_fixture"}}
    result = evaluate_candidates([row], {}, tmp_path, references=[reference])
    assert result["status"] == "complete", result
    assert result["predicted_count"] == 1
    value = result["candidates"][0]["contexts"][0]["predicted_inhibition_percent"]
    assert math.isfinite(value)
    assert result["model_sha256"] == MODEL_SHA256 and result["source_revision"] == SOURCE_REVISION
    assert result["runtime"]["device"] == "cuda"
    # A second call reads the same identity-bound prediction artifact.
    from aso_agents.oligoai import released_model_predict
    directory = Path(result["artifact_json"]).parent
    inputs = json.loads((directory / "model-inputs.json").read_text())["inputs"]
    before = (directory / "model-results.json").stat().st_mtime_ns
    assert released_model_predict(inputs, directory)["predictions"][0]["predicted_inhibition_percent"] == value
    assert (directory / "model-results.json").stat().st_mtime_ns == before


@pytest.mark.parametrize("species", [
    "human",
    pytest.param("mouse", marks=pytest.mark.skipif(
        os.environ.get("ASO_RUN_MOUSE_SMOKE") != "1",
        reason="Set ASO_RUN_MOUSE_SMOKE=1 after installing the mouse checkpoint")),
])
def test_released_splice_model_inference(tmp_path, species):
    from aso_agents import splicing_agent as exported
    import importlib
    module = importlib.import_module("aso_agents.splicing_agent")
    assert exported is module.splicing_agent
    # The reference is synthetic; this checks model loading and coordinate plumbing.
    randomizer = random.Random(511)
    sequence = "".join(randomizer.choices("ACGT", k=24000))
    fasta = tmp_path / "neutral.fa"
    fasta.write_text(">chr1\n" + sequence + "\n")
    tiles = tmp_path / "tiles.json"
    tiles.write_text(json.dumps({"rows": [{"chrom": "chr1", "start": 11980, "end": 12000,
                                          "strand": "+", "sequence": sequence[11980:12000], "name": "neutral"}]}))
    goal = {"intent": "splice_modulation", "species": species, "exon_label": 2,
            "isoform_transcripts": {"fixture": {"transcript_id": "SNAPGENE::fixture", "strand": 1,
                "chrom": "chr1", "seq_region_name": "1", "exons_tx": [
                    {"index": 1, "start": 10000, "end": 11000},
                    {"index": 2, "start": 12001, "end": 12200},
                    {"index": 3, "start": 13000, "end": 14000}]}}}
    sites = {"sites": [{"label": "neutral_exon", "exon_index": 2,
                        "acceptor": {"chrom": "chr1", "position": 12001, "strand": "+"},
                        "donor": {"chrom": "chr1", "position": 12200, "strand": "+"}}]}
    result = module.evaluate_splice_shifts_on_tiles(str(tiles), "SNAPGENE::fixture", json.dumps(goal),
                                                  ref_fasta_path=str(fasta), splice_sites_json=json.dumps(sites))
    assert not result.get("error"), result
    assert result["count"] == 1
    assert math.isfinite(result["results"][0]["delta_inclusion_deletion"])
    assert math.isfinite(result["results"][0]["delta_inclusion_replacement"])
