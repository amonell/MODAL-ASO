import json
from itertools import product

import numpy as np
import pytest
from Bio.Seq import Seq

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.mapping_evidence import compare_antisense_target
from aso_agents.region_overlap import interval_overlap_summary


def test_each_canonical_base_pair_has_explicit_antiparallel_semantics():
    pairs = {("A", "T"), ("C", "G"), ("G", "C"), ("T", "A")}
    for aso, target in product("ACGT", repeat=2):
        result = compare_antisense_target(aso, target)
        expected = 0 if (aso, target) in pairs else 1
        assert result["mismatch_count"] == expected
        assert result["match_count"] == 1 - expected
        assert result["mismatch_positions_in_supplied_sequence_1based"] == ([1] if expected else [])


@pytest.mark.parametrize("length", [1, 5, 11, 17, 20, 23, 31])
def test_all_single_substitutions_reverse_positions_and_preserve_base_identities(length):
    # An independent library constructs the molecule; all three substitutions
    # at every target position are checked against known physical identities.
    target = ("AGTCCTGACA" * length)[:length]
    aso = str(Seq(target).reverse_complement())
    exact = compare_antisense_target(aso, target)
    assert exact["match_count"] == length
    assert exact["mismatch_count"] == 0
    assert exact["mismatch_details_in_target_orientation"] == []
    for index, original in enumerate(target):
        for replacement in "ACGT":
            if replacement == original:
                continue
            changed = target[:index] + replacement + target[index + 1:]
            result = compare_antisense_target(aso, changed)
            assert result["mismatch_count"] == 1
            assert result["match_count"] == length - 1
            assert result["mismatch_positions_in_target_sequence_1based"] == [index + 1]
            assert result["mismatch_positions_in_supplied_sequence_1based"] == [length - index]
            detail, = result["mismatch_details_in_target_orientation"]
            assert detail == {
                "target_position_1based": index + 1,
                "reference_base": replacement,
                "aligned_query_base": original,
                "supplied_sequence_position_1based": length - index,
                "supplied_sequence_base": aso[length - index - 1],
                "expected_supplied_sequence_base": str(Seq(replacement).complement()),
            }


def test_multiple_mismatches_keep_two_distinct_position_orders():
    result = compare_antisense_target("TCGTAACC", "AGTTACCT")
    assert result["mismatch_count"] == 3
    assert result["mismatch_positions_in_target_sequence_1based"] == [1, 7, 8]
    assert result["mismatch_positions_in_supplied_sequence_1based"] == [1, 2, 8]
    assert result["aligned_query_5to3"] == "GGTTACGA"
    assert result["target_5to3"] == "AGTTACCT"
    assert result["antisense_5to3"] == "TCGTAACC"


def test_same_direction_strings_are_not_silently_treated_as_binding():
    assert compare_antisense_target("AAAAA", "AAAAA")["mismatch_count"] == 5
    assert compare_antisense_target("AAAAA", "TTTTT")["mismatch_count"] == 0


def test_rna_and_lowercase_normalize_to_explicit_dna_alphabet():
    result = compare_antisense_target("ugca", "ugca")
    assert result["mismatch_count"] == 0
    assert result["antisense_5to3"] == result["target_5to3"] == "TGCA"
    assert result["sequence_alphabet"] == "DNA; U normalized to T"


@pytest.mark.parametrize("aso,target", [
    ("", ""), ("AC", "ACG"), ("ACN", "ACG"), ("ACG", "ACN"),
    ("A-C", "AGC"), ("A C", "AGC"), ("R", "A"), ("A", "?"),
    (None, "A"), ("A", 1), (b"A", "T"), ("/5MOErA/", "T"),
])
def test_ambiguous_gapped_or_nonsequence_comparisons_fail_explicitly(aso, target):
    with pytest.raises((TypeError, ValueError)):
        compare_antisense_target(aso, target)


@pytest.mark.parametrize("convention", ["inclusive", "half_open"])
def test_interval_relations_against_independent_base_set_oracle(convention):
    spans = [(a, b) for a in range(-5, 6) for b in range(a, 6)]
    for (a, b), (c, d) in product(spans, repeat=2):
        candidate, feature = set(range(a, b + 1)), set(range(c, d + 1))
        end_adjustment = 1 if convention == "half_open" else 0
        result = interval_overlap_summary(a, b + end_adjustment, c, d + end_adjustment,
                                          coordinate_convention=convention)
        overlap = candidate & feature
        assert result["overlap_nt"] == len(overlap)
        assert result["any_overlap"] == bool(overlap)
        assert result["contains_feature"] == (feature <= candidate)
        assert result["within_feature"] == (candidate <= feature)
        assert result["overlap_interval"] == (
            [min(overlap), max(overlap) + end_adjustment] if overlap else None)
        assert result["coordinate_convention"] == convention


def test_single_base_boundary_and_partial_overlap_are_not_containment():
    inclusive = interval_overlap_summary(-6, 1, 1, 4, coordinate_convention="inclusive")
    assert inclusive["overlap_nt"] == 1
    assert inclusive["any_overlap"] is True
    assert inclusive["contains_feature"] is False
    assert inclusive["within_feature"] is False
    half_open = interval_overlap_summary(-6, 1, 1, 4, coordinate_convention="half_open")
    assert half_open["overlap_nt"] == 0
    assert half_open["any_overlap"] is False
    assert half_open["overlap_interval"] is None


def test_integer_array_coordinates_serialize_without_loss():
    result = interval_overlap_summary(np.int64(-2), np.int64(4), np.int32(1), np.int32(7),
                                      coordinate_convention="inclusive")
    assert json.loads(json.dumps(result))["overlap_interval"] == [1, 4]


@pytest.mark.parametrize("bounds,convention", [
    ((3, 2, 1, 4), "inclusive"), ((1, 4, 3, 2), "inclusive"),
    ((2, 2, 1, 4), "half_open"), ((1, 4, 2, 2), "half_open"),
    ((1.0, 4, 2, 3), "inclusive"), ((True, 4, 2, 3), "inclusive"),
    ((1, 4, None, 3), "inclusive"), ((1, 4, "2", 3), "inclusive"),
    ((1, 4, 2, 3), "unspecified"),
])
def test_interval_summary_rejects_unknown_or_invalid_geometry(bounds, convention):
    with pytest.raises((TypeError, ValueError)):
        interval_overlap_summary(*bounds, coordinate_convention=convention)


def test_actual_analysis_runtime_exports_mismatch_and_overlap_annotations(tmp_path):
    observations = [
        {"aso": "TCGTAACC", "target": "GGTTACGT", "bounds": [-6, 1, -2, 3]},
        {"aso": "TCGTAACC", "target": "GGTTACGA", "bounds": [-2, 5, -2, 3]},
    ]
    destination = tmp_path / "annotations.json"
    (tmp_path / "declared_inputs.json").write_text(json.dumps({"observations": observations}))

    def generate(payload):
        contract = payload["execution_contract"]
        code = "import json\nfrom pathlib import Path\n"
        code += "from aso_agents.mapping_evidence import compare_antisense_target\n"
        code += "from aso_agents.region_overlap import interval_overlap_summary\n"
        code += "inputs=json.loads(Path('declared_inputs.json').read_text())\n"
        code += "rows=[{'alignment':compare_antisense_target(x['aso'],x['target']),"
        code += "'geometry':interval_overlap_summary(*x['bounds'],coordinate_convention='inclusive')} "
        code += "for x in inputs['observations']]\n"
        code += f"Path({str(destination)!r}).write_text(json.dumps(rows))\n"
        code += f"Path({contract['result_json']!r}).write_text(json.dumps({{'rows':rows}}))\n"
        code += f"Path({contract['summary_markdown']!r}).write_text('Measured antiparallel alignment and interval intersection.')\n"
        return {"python_code": code}

    result = execute_analysis_program(directory=tmp_path, payload={"declared_inputs": observations},
                                      generate=generate, required_outputs=[destination], max_attempts=1)
    assert result["ok"], result
    partial, full = json.loads(destination.read_text())
    assert partial["alignment"]["mismatch_positions_in_supplied_sequence_1based"] == [1]
    assert partial["alignment"]["mismatch_positions_in_target_sequence_1based"] == [8]
    assert partial["geometry"]["overlap_interval"] == [-2, 1]
    assert partial["geometry"]["overlap_nt"] == 4
    assert partial["geometry"]["any_overlap"] is True
    assert partial["geometry"]["contains_feature"] is False
    assert full["alignment"]["mismatch_count"] == 0
    assert full["geometry"]["contains_feature"] is True
    assert full["geometry"]["overlap_nt"] == 6
