from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace

import pytest
from aso_agents.workbench_response_agent import evaluation_evidence_audit_agent

from aso_agents.response_context import (
    decode_response_context, encode_response_context, render_response_context,
)


def test_complete_alignment_panel_survives_sharing_without_cross_row_binding():
    geometry = {"transcript_id": "TX_NEUTRAL.2", "rna_context": "mature_spliced_cdna",
                "strand": "-", "genomic_segments": [
                    {"chrom": "chrNeutral", "start": 201, "end": 208},
                    {"chrom": "chrNeutral", "start": 103, "end": 114}],
                "coordinate_system": "1based_inclusive", "recorded_segment_count": 2}
    rows, evidence = [], {}
    for index in range(6):
        hypotheses = [{"geometry": deepcopy(geometry), "mismatch_count": mismatch,
                       "orientation": orientation,
                       "geometry_reference_token": f"{{{{R{index}_{orientation}_{mismatch}}}}}"}
                      for mismatch in range(100) for orientation in ("same_strand", "reverse_complement")]
        row = {"input_row_id": f"row_{index}", "name": "Repeated name", "sequence": "ATCG",
               "chemistry": "RNase H gapmer" if index % 2 else "steric non-RNase H",
               "hypotheses": hypotheses, "measured_zero": 0, "unknown": None,
               "activity_measured": False}
        rows.append(row)
        evidence[f"R{index}_inventory"] = deepcopy(row)
    payload = {"provided_asos": rows, "measurement_evidence": evidence}
    before = deepcopy(payload)
    wire = render_response_context(payload)
    decoded = decode_response_context(json.loads(wire))
    assert decoded == payload == before
    assert len(wire) < len(json.dumps(payload, indent=2)) / 3
    assert [row["input_row_id"] for row in decoded["provided_asos"]] == [f"row_{i}" for i in range(6)]
    assert all(len(row["hypotheses"]) == 200 for row in decoded["provided_asos"])
    assert decoded["provided_asos"][0]["hypotheses"][199]["geometry_reference_token"] == "{{R0_reverse_complement_99}}"


def test_reserved_marker_text_from_a_request_is_literal_not_an_encoding_reference():
    payload = {"request": {"$ref": "https://untrusted.invalid/example"},
               "attachment": {"$table": {"columns": ["fake"], "rows": [[1]]}},
               "literal": {"$literal": {"$ref": "#/shared_values/E00001"}},
               "repeated": [{"$ref": "literal value", "other": None} for _ in range(5)]}
    assert decode_response_context(encode_response_context(payload)) == payload


def test_missing_keys_and_distinct_numeric_types_remain_distinct():
    rows = [{"a": 0, "b": None}, {"a": False}, {"a": 0.0, "b": ""}, {"a": "0", "b": []}]
    decoded = decode_response_context(encode_response_context({"rows": rows}))
    assert decoded == {"rows": rows}
    assert [type(row["a"]) for row in decoded["rows"]] == [int, bool, float, str]
    assert "b" not in decoded["rows"][1]


def test_rectangular_tables_preserve_column_values_despite_dictionary_key_order():
    rows = [{"a": i, "b": -i} if i % 2 else {"b": -i, "a": i} for i in range(8)]
    assert decode_response_context(encode_response_context({"rows": rows})) == {"rows": rows}


def test_small_requests_keep_plain_json_and_citations_unchanged():
    payload = {"request_text": "Evaluate this molecule", "citation": "{{R001_alignment_1}}"}
    assert json.loads(render_response_context(payload)) == payload


def test_native_composer_sends_lossless_context_and_returns_unchanged_judgment(monkeypatch):
    from aso_agents import pipeline_tools
    row = {"input_row_id": "neutral_row", "name": "neutral_probe",
           "near_match_candidates": [{"transcript_id": "TX_NEUTRAL", "coordinate_system": "1based_inclusive",
                                      "start": i + 10, "end": i + 29,
                                      "orientation": "reverse_complement"} for i in range(250)]}
    payload = {"task_type": "evaluate_existing_aso_set",
               "evaluation_summary": {"provided_asos": [row], "evidence_copy": deepcopy(row)}}
    response = {"answer_summary": "Activity remains unresolved.", "markdown": "",
                "evaluation_assessments": [{"input_row_id": "neutral_row", "name": "neutral_probe",
                                             "mechanistic_plausibility": "Binding remains conditional."}]}
    sent = []

    audits = []
    async def run(agent, **kwargs):
        if agent.name == evaluation_evidence_audit_agent.name:
            audits.append(kwargs['input'])
            return SimpleNamespace(final_output=json.dumps({'findings': []}))
        sent.append(kwargs["input"])
        delivered = decode_response_context(json.loads(sent[-1]))
        schema = delivered.pop("output_schema")
        assert schema["properties"]["evaluation_assessments"]["items"]["properties"]["input_row_id"]["enum"] == ["neutral_row"]
        assert delivered == payload
        return SimpleNamespace(final_output=json.dumps(response))

    monkeypatch.setattr(pipeline_tools.Runner, "run", run)
    result = asyncio.run(pipeline_tools._run_workbench_response_agent_once(payload))
    assert result.pop('native_consistency_review')['attempts'] == 1
    assert result == response
    assert len(sent) == len(audits) == 1


@pytest.mark.parametrize("shared,context", [
    ({}, {"$ref": "https://untrusted.invalid"}),
    ({}, {"$ref": "#/shared_values/missing"}),
    ({"E1": {"$ref": "#/shared_values/E1"}}, {"$ref": "#/shared_values/E1"}),
    ({}, {"$table": {"columns": ["a", "a"], "rows": [[1, 2]]}}),
    ({}, {"$table": {"columns": ["a"], "rows": [[1, 2]]}}),
])
def test_diagnostic_decoder_rejects_invalid_reference_structure(shared, context):
    with pytest.raises(ValueError):
        decode_response_context({"context_encoding": "lossless_shared_json_v1",
                                 "context": context, "shared_values": shared})
