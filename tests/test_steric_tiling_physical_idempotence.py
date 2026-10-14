"""A stale tiling marker cannot hide missing Snakemake deliverables."""

from aso_agents.pipeline_tools import _steric_tiling_physical_results_present


def test_all_groups_need_bed_and_idt_outputs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    groups = [{"output_dir": str(first)}, {"output_dir": str(second)}]
    assert _steric_tiling_physical_results_present(groups) is False
    (first / "A.scored.bed").write_text("chr1\t0\t20\n")
    (first / "A.scored.idt.tsv").write_text("ASO\n")
    assert _steric_tiling_physical_results_present(groups) is False
    (second / "B.scored.bed").write_text("chr2\t0\t20\n")
    assert _steric_tiling_physical_results_present(groups) is False
    (second / "B.scored.idt.tsv").write_text("ASO\n")
    assert _steric_tiling_physical_results_present(groups) is True
