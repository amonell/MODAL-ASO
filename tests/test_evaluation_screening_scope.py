import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.goal_contract import canonical_task_context_schema


@pytest.mark.parametrize("policy", ["enabled", "disabled"])
def test_screening_policy_survives_task_handoff(policy):
    context = {"off_target_screening": policy, "num_mismatches": 0}
    jsonschema.validate(context, canonical_task_context_schema())
    task = {"task_type": "evaluate_existing_aso_set", "context": context}
    scoped = pipeline._task_context_from_request_envelope({}, task)
    goal = pipeline._overlay_request_context_on_goal({}, scoped, "")
    assert goal["off_target_screening"] == policy
    assert goal["num_mismatches"] == 0


@pytest.mark.parametrize("policy", [False, True, "none", "no", None, {}])
def test_screening_policy_is_not_an_untyped_boolean_or_alias(policy):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"off_target_screening": policy}, canonical_task_context_schema())


def test_task_policy_does_not_leak_from_a_previous_evaluation_goal():
    previous = {"off_target_screening": "disabled", "gene_symbol": "TEST"}
    inherited = pipeline._overlay_request_context_on_goal(previous, {}, "")
    assert "off_target_screening" not in inherited
    assert previous["off_target_screening"] == "disabled"
    envelope = {"default_context": {"off_target_screening": "disabled"}}
    local = pipeline._task_context_from_request_envelope(
        envelope, {"context": {"off_target_screening": "enabled"}})
    assert local["off_target_screening"] == "enabled"
    assert envelope["default_context"]["off_target_screening"] == "disabled"


@pytest.mark.parametrize("policy", [None, "enabled", "disabled"])
def test_scoring_honors_scope_without_losing_reference_evidence(tmp_path, monkeypatch, policy):
    calls = []

    class Runtime:
        @staticmethod
        def _annotate_tile_rows_with_rnafold(rows):
            for row in rows:
                row["RNAfold_MFE"] = -1.5
            return len(rows)

        @staticmethod
        def _annotate_gapmer_off_targets(path, goal, constraints):
            calls.append("screen")
            return {"rows_with_hits": 0}

    def enrich(path, scope):
        calls.append("count_bins")
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_enrich_exact_off_target_bins_for_tiles", enrich)
    monkeypatch.setattr(pipeline, "_maybe_resolve_eval_gene_id", lambda *_: calls.append("gene") or {})
    monkeypatch.setattr(pipeline, "exact_gene_count_scope", lambda *_: {"searched": True})
    monkeypatch.setattr(pipeline, "annotate_off_target_expression", lambda **_: calls.append("expression"))
    binding = {"target_sequence": "TACCGGTT", "mismatches": 0, "start": 3, "end": 10}
    rows = [{"name": "repeat", "sequence": "AACCGGTA", "mapping_status": "mapped",
             "alignments": [deepcopy(binding)], "Off_target_exact_count": 7,
             "off_target_gene_count": 7, "stage_results": {"mapping": {"status": "pass"}}},
            {"name": "repeat", "sequence": "AACCGGTA", "mapping_status": "unmapped",
             "alignments": []}]
    goal = {} if policy is None else {"off_target_screening": policy}
    result = pipeline._score_provided_asos_with_existing_pipeline(
        rows=rows, eval_goal=goal, cdna_map={}, iso_label=None, runtime=Runtime(),
        run_dir=str(tmp_path), task_id="evaluation")
    assert len(rows) == 2
    assert rows[0]["alignments"] == [binding]
    assert rows[0]["stage_results"]["mapping"] == {"status": "pass"}
    assert rows[1]["alignments"] == []
    assert all(row["RNAfold_MFE"] == -1.5 for row in rows)
    if policy == "disabled":
        assert calls == []
        assert result["off_target_mapping"] == {
            "status": "not_run", "reason": "excluded_by_task_scope"}
        assert result["off_target_expression"]["reason"] == "off_target_screening_not_run"
        assert "off_target_count_scope" not in result
        assert "exact_off_target_bins" not in result
        assert result["merge"]["off_target_scored_rows"] == 0
        for row in rows:
            assert row["Off_target_exact_count"] is None
            assert row["off_target_gene_count"] is None
            assert row["Off_target_1mm_count"] is None
            assert row["stage_results"]["off_target_mapping"]["status"] == "not_run"
        artifact = json.loads(Path(result["scored_tiles_path"]).read_text())
        assert all(row["off_target_count_status"] == "not_run" for row in artifact["rows"])
        assert all(row["Off_target_exact_count"] is None for row in artifact["rows"])
    else:
        assert calls == ["gene", "screen", "count_bins"]
        assert result["off_target_mapping"] == {"rows_with_hits": 0}
        assert result["off_target_count_scope"] == {"searched": True}


def test_disabled_screening_keeps_local_reference_uniqueness(tmp_path):
    calls = []

    class Runtime:
        @staticmethod
        def _build_gapmer_kmer_index_by_len(cdna, lengths):
            calls.append((cdna, lengths))
            return {8: {"AACCGGTA": {"reference"}}}

        @staticmethod
        def _annotate_gapmer_uniqueness(path, index, label, **kwargs):
            data = json.loads(Path(path).read_text())
            data["rows"][0]["isoform_uniqueness_count"] = 1
            Path(path).write_text(json.dumps(data))
            return path

        @staticmethod
        def _annotate_gapmer_off_targets(*args):
            pytest.fail("Disabled external screening must not execute")

    rows = [{"name": "molecule", "sequence": "AACCGGTA"}]
    result = pipeline._score_provided_asos_with_existing_pipeline(
        rows=rows, eval_goal={"off_target_screening": "disabled"},
        cdna_map={"reference": "TTAACCGGTA"}, iso_label="reference", runtime=Runtime(),
        run_dir=str(tmp_path), task_id="evaluation")
    assert calls == [({"reference": "TTAACCGGTA"}, [8])]
    assert result["isoform_uniqueness"]["status"] == "complete"
    assert rows[0]["isoform_uniqueness_count"] == 1
    assert rows[0]["Off_target_exact_count"] is None
