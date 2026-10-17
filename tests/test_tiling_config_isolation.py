from pathlib import Path
from types import SimpleNamespace

from aso_agents import runtime_binaries, tiling_agent


def test_emitted_tiling_commands_keep_group_scoped_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "references"
    species_config = SimpleNamespace(
        genome_fasta=reference_root / "genome.fa",
        phastcons_path=reference_root / "phastcons.bw",
        phylop_path=reference_root / "phylop.bw",
        chrom_sizes_path=reference_root / "chrom.sizes",
        bowtie_prefix=reference_root / "bowtie" / "genome",
        gtf_path=reference_root / "genes.gtf",
        ucsc_build="test-build",
        conservation_mode="disabled",
    )
    monkeypatch.setattr(tiling_agent, "get_species_config", lambda _species: species_config)
    monkeypatch.setattr(runtime_binaries, "ensure_snakemake", lambda: {"status": "ok"})
    monkeypatch.setattr(runtime_binaries, "ensure_tiling_cli_tools", lambda: {"status": "ok"})
    monkeypatch.setattr(runtime_binaries, "refresh_runtime_path", lambda: "/usr/bin")
    monkeypatch.setattr(runtime_binaries, "snakemake_argv", lambda: ["snakemake"])

    goal = {"gene_symbol": "GENE", "species": "human"}
    first_output = tmp_path / "tiling" / "chemistry_20"
    second_output = tmp_path / "tiling" / "chemistry_18"
    first = tiling_agent.emit_pipeline_config(
        structured_goal_json=tiling_agent.json.dumps(goal),
        bed_json=tiling_agent.json.dumps(
            {"rows": [["chr1", 100, 140, "first", 0, "+"]]}
        ),
        chemistry_json=tiling_agent.json.dumps(
            {"recommendation": {"CHEMISTRY": "steric-hindrance", "length": 20}}
        ),
        output_dir=str(first_output),
    )
    second = tiling_agent.emit_pipeline_config(
        structured_goal_json=tiling_agent.json.dumps(goal),
        bed_json=tiling_agent.json.dumps(
            {"rows": [["chr1", 200, 240, "second", 0, "+"]]}
        ),
        chemistry_json=tiling_agent.json.dumps(
            {"recommendation": {"CHEMISTRY": "steric-hindrance", "length": 18}}
        ),
        output_dir=str(second_output),
    )

    first_config = Path(first["config_path"])
    second_config = Path(second["config_path"])
    first_bed = Path(first["bed_path"])
    second_bed = Path(second["bed_path"])

    assert first_config != second_config
    assert first_bed != second_bed
    assert first_config.parent == Path(f"{first_output}.runtime")
    assert second_config.parent == Path(f"{second_output}.runtime")
    assert f"OUTPUT_FOLDER: {first_output}" in first_config.read_text(encoding="utf-8")
    assert "TILE_LEN: 20" in first_config.read_text(encoding="utf-8")
    assert f"OUTPUT_FOLDER: {second_output}" in second_config.read_text(encoding="utf-8")
    assert "TILE_LEN: 18" in second_config.read_text(encoding="utf-8")
    assert "\tfirst\t" in first_bed.read_text(encoding="utf-8")
    assert "\tsecond\t" in second_bed.read_text(encoding="utf-8")
    assert str(first_config) in first["snakemake_cmd"]
    assert str(second_config) in second["snakemake_cmd"]
