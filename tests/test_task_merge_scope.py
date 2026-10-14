"""Combining related splice objectives must not erase a task's own contract."""
from copy import deepcopy

import pytest

from aso_agents import pipeline_tools as pipeline


def tasks():
    return [{"task_id": name, "task_type": "design_new_asos", "depends_on": [], "context": {
        "gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
        "structured_goal": {"gene_symbol": "NEUTRAL", "species": "human", "target_context": "cultured cells",
            "design_programs": [{"program_id": name, "modality": "steric", "intent": "splice_modulation"}],
            "splice_isoform_objectives": [{"transcript_id": "TX_" + name, "desired_abundance": delta}]}}}
        for name, delta in [("a", "decrease"), ("b", "increase")]]


@pytest.mark.parametrize("field,first,second", [
    ("chemistry", "fully modified steric", "gapmer with a DNA gap"),
    ("constraints", {"gc_content": {"min": 0.3, "max": 0.7}}, {"gc_content": {"min": 0.4, "max": 0.6}}),
    ("region_constraints", [{"chrom": "chr1", "start": 10, "end": 30}], [{"chrom": "chr1", "start": 40, "end": 60}]),
    ("reference_genome", "assembly_a", "assembly_b"),
])
def test_different_shared_settings_remain_separate_tasks(field, first, second):
    source = tasks()
    source[0]["context"]["structured_goal"][field] = first
    source[1]["context"]["structured_goal"][field] = second
    original = deepcopy(source)
    assert pipeline._merge_same_gene_splice_design_tasks(source) == source
    assert source == original


@pytest.mark.parametrize("field,value", [("candidate_minimum", 4), ("candidate_limit", 4), ("candidate_limit", 0)])
def test_even_equal_whole_task_counts_cannot_be_silently_shared(field, value):
    source = tasks()
    for task in source:
        task["context"]["structured_goal"][field] = value
    assert pipeline._merge_same_gene_splice_design_tasks(source) == source


def test_identical_unbounded_scope_still_combines_complementary_splice_objectives():
    source = tasks()
    for task in source:
        task["context"]["structured_goal"].update(candidate_minimum=0, candidate_limit=None,
            chemistry="fully modified steric", constraints={"gc_content": {"min": 0.3, "max": 0.7}})
    merged = pipeline._merge_same_gene_splice_design_tasks(source)
    assert len(merged) == 1
    goal = merged[0]["context"]["structured_goal"]
    assert {p["program_id"] for p in goal["design_programs"]} == {"a", "b"}
    assert {p["transcript_id"] for p in goal["splice_isoform_objectives"]} == {"TX_a", "TX_b"}
    assert goal["chemistry"] == "fully modified steric"


def test_preserved_tasks_keep_their_dependent_comparison_bindings():
    source = tasks()
    source[0]["context"]["chemistry"] = "chemistry_a"
    source[1]["context"]["chemistry"] = "chemistry_b"
    source.append({"task_id": "compare", "task_type": "compare_aso_sets", "depends_on": ["a", "b"], "context": {}})
    result = pipeline._merge_same_gene_splice_design_tasks(source)
    assert [t["task_id"] for t in result] == ["a", "b", "compare"]
    assert result[-1]["depends_on"] == ["a", "b"]
