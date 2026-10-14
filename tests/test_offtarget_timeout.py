from __future__ import annotations

import time
import importlib
from pathlib import Path
from typing import Any

from aso_agents import pipeline_tools

off_target_expression_agent = importlib.import_module("aso_agents.off_target_expression_agent")


def test_offtarget_expression_default_timeout_is_interactive(monkeypatch) -> None:
    monkeypatch.delenv("ASO_OFFTARGET_TIMEOUT_SEC", raising=False)

    assert pipeline_tools._offtarget_timeout_seconds({"species": "human"}) == 240


def _sleeping_offtarget_worker(
    initial_filter_json_path: str,
    structured_goal_json: str,
    output_queue: Any,
) -> None:
    _ = (initial_filter_json_path, structured_goal_json, output_queue)
    time.sleep(10)


def _large_result_worker(_path, _goal, output_queue):
    output_queue.put({"result": {"large": "x" * 2_000_000}})


def _crashing_worker(_path, _goal, _queue):
    raise RuntimeError("neutral process failure")


def test_no_deadline_drains_large_result_without_join_deadlock(monkeypatch):
    monkeypatch.setenv("ASO_OFFTARGET_TIMEOUT_SEC", "none")
    monkeypatch.setattr(pipeline_tools, "_offtarget_expression_worker", _large_result_worker)
    assert pipeline_tools._offtarget_timeout_seconds({"off_target_expression": {"timeout_sec": 1}}) is None
    result = pipeline_tools._run_offtarget_expression_with_timeout(initial_filter_json_path="unused", structured_goal={})
    assert len(result["large"]) == 2_000_000


def test_no_deadline_detects_worker_crash(monkeypatch):
    monkeypatch.setenv("ASO_OFFTARGET_TIMEOUT_SEC", "none")
    monkeypatch.setattr(pipeline_tools, "_offtarget_expression_worker", _crashing_worker)
    result = pipeline_tools._run_offtarget_expression_with_timeout(initial_filter_json_path="unused", structured_goal={})
    assert "worker_exited_nonzero" in result["error"]


def test_offtarget_expression_timeout_returns_recoverable_error(monkeypatch, tmp_path: Path) -> None:
    input_json = tmp_path / "tiles.json"
    input_json.write_text('{"rows": []}', encoding="utf-8")
    monkeypatch.setenv("ASO_OFFTARGET_TIMEOUT_SEC", "1")
    monkeypatch.setattr(pipeline_tools, "_offtarget_expression_worker", _sleeping_offtarget_worker)

    started = time.monotonic()
    result = pipeline_tools._run_offtarget_expression_with_timeout(
        initial_filter_json_path=str(input_json),
        structured_goal={"species": "human"},
    )
    elapsed = time.monotonic() - started

    assert result["error"] == "off_target_expression_timeout: exceeded 1s"
    assert result["timeout_sec"] == 1
    assert elapsed < 5


def test_offtarget_expression_direct_call_does_not_skip_general_baseline(monkeypatch, tmp_path: Path) -> None:
    input_json = tmp_path / "tiles.json"
    input_json.write_text('{"rows": []}', encoding="utf-8")
    monkeypatch.setenv("ASO_OFFTARGET_TIMEOUT_SEC", "1")
    monkeypatch.setattr(pipeline_tools, "_offtarget_expression_worker", _sleeping_offtarget_worker)

    result = pipeline_tools._run_offtarget_expression_with_timeout(
        initial_filter_json_path=str(input_json),
        structured_goal={
            "species": "human",
            "target_context": "General human baseline context; no specific tissue, cell type, disease, or model system.",
            "off_target_expression": {
                "target_context": "General human baseline context; no specific tissue, cell type, disease, or model system."
            },
        },
    )

    assert result["error"] == "off_target_expression_timeout: exceeded 1s"


def test_offtarget_expression_specific_context_still_uses_worker(monkeypatch, tmp_path: Path) -> None:
    input_json = tmp_path / "tiles.json"
    input_json.write_text('{"rows": []}', encoding="utf-8")
    monkeypatch.setenv("ASO_OFFTARGET_TIMEOUT_SEC", "1")
    monkeypatch.setattr(pipeline_tools, "_offtarget_expression_worker", _sleeping_offtarget_worker)

    result = pipeline_tools._run_offtarget_expression_with_timeout(
        initial_filter_json_path=str(input_json),
        structured_goal={
            "species": "human",
            "target_context": "human hepatocytes in liver",
            "off_target_expression": {"target_context": "human hepatocytes in liver"},
        },
    )

    assert result["error"] == "off_target_expression_timeout: exceeded 1s"


def test_hek293t_context_infers_kidney_epithelial_cell() -> None:
    tissue, cell_type = off_target_expression_agent._infer_tissue_and_cell_type(
        "Human HEK293T cells in vitro"
    )

    assert tissue == "kidney"
    assert cell_type == "kidney epithelial cell"


def test_common_tissue_contexts_infer_cellxgene_labels() -> None:
    assert off_target_expression_agent._infer_tissue_and_cell_type(
        "human iPSC-derived cortical neurons"
    ) == ("brain", "cerebral cortex neuron")
    assert off_target_expression_agent._infer_tissue_and_cell_type(
        "human iPSC-derived dopaminergic neurons"
    ) == ("brain", "dopaminergic neuron")
    assert off_target_expression_agent._infer_tissue_and_cell_type(
        "HGPS patient dermal fibroblasts"
    ) == ("skin of body", "skin fibroblast")
    assert off_target_expression_agent._infer_tissue_and_cell_type(
        "human hepatocyte-like cells"
    ) == ("liver", "hepatocyte")


def test_preferred_tissue_is_used_without_llm(monkeypatch: Any) -> None:
    calls = {"count": 0}

    def fake_selector(**_kwargs: Any) -> None:
        calls["count"] += 1
        return None

    monkeypatch.setattr(off_target_expression_agent, "_select_census_context_with_llm", fake_selector)

    selected = off_target_expression_agent._select_census_tissue_with_llm(
        target_context="HGPS patient dermal fibroblasts",
        species_hint="human",
        available_species=[{"organism": "homo_sapiens"}],
        available_tissues=[
            {"tissue": "brain", "cell_count": 100},
            {"tissue": "skin of body", "cell_count": 50},
        ],
        preferred_tissue="skin of body",
    )

    assert selected == "skin of body"
    assert calls["count"] == 0


def test_cell_type_selection_falls_back_without_llm(monkeypatch: Any) -> None:
    calls = {"count": 0}

    def fake_selector(**_kwargs: Any) -> None:
        calls["count"] += 1
        return None

    monkeypatch.setattr(off_target_expression_agent, "_select_census_context_with_llm", fake_selector)

    selected = off_target_expression_agent._select_census_cell_types_with_llm(
        target_context="human iPSC-derived dopaminergic neurons",
        species_hint="human",
        available_species=[{"organism": "homo_sapiens"}],
        available_cell_types=[
            {"cell_type": "neuron", "cell_count": 100},
            {"cell_type": "dopaminergic neuron", "cell_count": 10},
            {"cell_type": "fibroblast", "cell_count": 1000},
        ],
        selected_tissue="brain",
        preferred_cell_type=None,
    )

    assert "dopaminergic neuron" in selected
    assert calls["count"] > 0


def test_general_baseline_context_uses_broad_baseline_mode() -> None:
    assert off_target_expression_agent._is_broad_baseline_context(
        "General human baseline context; no specific tissue, cell type, disease, or model system."
    )
    assert off_target_expression_agent._infer_tissue_and_cell_type(
        "General human baseline context; no specific tissue, cell type, disease, or model system."
    ) == (None, None)


def test_preferred_cell_type_filters_unrelated_llm_choices(monkeypatch: Any) -> None:
    calls = {"count": 0}

    def fake_selector(**_kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"cell_types": ["T cell", "kidney epithelial cell"]}

    monkeypatch.setattr(off_target_expression_agent, "_select_census_context_with_llm", fake_selector)

    selected = off_target_expression_agent._select_census_cell_types_with_llm(
        target_context="Human HEK293T cells in vitro",
        species_hint="human",
        available_species=[{"organism": "homo_sapiens"}],
        available_cell_types=[
            {"cell_type": "epithelial cell of proximal tubule", "cell_count": 10},
            {"cell_type": "kidney epithelial cell", "cell_count": 8},
            {"cell_type": "T cell", "cell_count": 4},
        ],
        selected_tissue="kidney",
        preferred_cell_type="kidney epithelial cell",
    )

    assert selected == ["kidney epithelial cell"]
    assert calls["count"] == 0
