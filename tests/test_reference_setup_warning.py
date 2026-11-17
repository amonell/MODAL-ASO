import contextlib
import io
from types import SimpleNamespace

import pytest

import run_goal_intake as runtime
from aso_agents import pipeline_tools, reference_setup, runtime_warnings
from aso_agents.transcript_chooser_agent import EnsemblServiceError


@pytest.fixture
def local_files(tmp_path, monkeypatch):
    config = SimpleNamespace(gtf_path=tmp_path / "annotation.gtf", genome_fasta=tmp_path / "genome.fa")
    monkeypatch.setattr(reference_setup, "get_species_config", lambda species: config)
    return config


@pytest.mark.parametrize("annotation,genome", [(False, False), (True, False), (False, True), (True, True)])
def test_warning_describes_only_the_missing_files(local_files, annotation, genome):
    if annotation:
        local_files.gtf_path.write_text("annotation")
    if genome:
        local_files.genome_fasta.write_text(">chr1\nACGT\n")
    result = reference_setup.inspect_resolution_references("human")
    assert ("transcript annotation" in result["missing"]) is (not annotation)
    assert ("genome sequence" in result["missing"]) is (not genome)
    assert ("Transcript mapping will use Ensembl" in result["message"]) is (not annotation)
    assert bool(result["message"]) is (not (annotation and genome))
    if result["message"]:
        assert "ASO_REFERENCE_DATA_ROOT" in result["message"]
        assert "modal-aso setup --species human --features references" in result["message"]


def test_compressed_annotation_counts_as_available(local_files):
    import gzip
    with gzip.open(local_files.gtf_path.with_suffix(".gtf.gz"), "wt") as handle:
        handle.write("annotation")
    local_files.genome_fasta.write_text(">chr1\nACGT\n")
    assert reference_setup.inspect_resolution_references("mouse")["message"] == ""


def test_warning_uses_requested_species(local_files):
    result = reference_setup.inspect_resolution_references("mouse")
    assert "Local mouse reference" in result["message"]
    assert "--species mouse" in result["message"]
    assert "human" not in result["message"]


def test_frozen_run_warning_requires_local_annotation(local_files, monkeypatch):
    monkeypatch.setenv("ASO_DEPENDENCY_MODE", "frozen")
    result = reference_setup.inspect_resolution_references("human")
    assert "frozen references" in result["message"]
    assert "requires the local annotation" in result["message"]
    assert "will use Ensembl" not in result["message"]


def test_pretty_cli_shows_warning_during_captured_tool_output():
    screen, captured = io.StringIO(), io.StringIO()
    narrator = runtime._CliNarrator(stream=screen, enable_stream=False, enable_spinner=False)
    with contextlib.redirect_stdout(captured), runtime_warnings.warning_handler(narrator.on_runner_event):
        print("ordinary tool log")
        runtime_warnings.emit_warning("Local human transcript annotation is unavailable.")
        assert "Local human" in screen.getvalue()
        assert "Local human" not in captured.getvalue()
    assert "ordinary tool log" not in screen.getvalue()
    # The handler is scoped to the CLI call and resets afterwards.
    with contextlib.redirect_stdout(captured):
        runtime_warnings.emit_warning("Another run")
    assert "Another run" in captured.getvalue()
    assert "Another run" not in screen.getvalue()


def test_warning_precedes_lookup_is_saved_and_is_not_repeated(local_files, monkeypatch, tmp_path, capsys):
    state = {"run_dir": str(tmp_path), "stage_attempts": {}, "retry_policy": {"max_stage_retries": 1}}
    goal = {"gene_symbol": "GENE", "species": "human"}
    saved = []
    calls = []

    class FakeRuntime:
        @staticmethod
        def _resolve_gene_id(*args, **kwargs):
            calls.append("lookup")
            assert state["reference_setup"]["human"]["status"] in {"missing", "available"}
            assert saved[0] == "reference_setup_checked"
            if len(calls) == 1:
                assert "Local human reference files are unavailable" in capsys.readouterr().out
            raise EnsemblServiceError("Temporary outage", retryable=True, status_code=503)

    monkeypatch.setattr(pipeline_tools, "_prepare_execution_stage", lambda **kw: {"status": "ok", "state": state, "run_dir": str(tmp_path)})
    monkeypatch.setattr(pipeline_tools, "_load_goal_context_from_run_dir", lambda _: {
        "payload": {}, "final_payload": {"structured_goal": goal}, "structured_goal": goal,
    })
    monkeypatch.setattr(pipeline_tools, "_restore_design_objective_preseed_to_payload", lambda *a, **k: False)
    monkeypatch.setattr(pipeline_tools, "_normalized_design_objective_preseed", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_tools, "_goal_runtime_module", lambda: FakeRuntime())
    monkeypatch.setattr(pipeline_tools, "_save_state", lambda *a, **k: saved.append(k.get("snapshot_reason")))

    for _ in range(2):
        pipeline_tools.tool_resolve_isoforms_and_sites(str(tmp_path / "state.json"))
    assert capsys.readouterr().out == ""
    assert len([e for e in state["events"] if e["phase"] == "reference_setup"]) == 1
    assert state["reference_setup"]["human"]["missing"] == ["transcript annotation", "genome sequence"]

    local_files.gtf_path.write_text("annotation")
    local_files.genome_fasta.write_text(">chr1\nACGT\n")
    pipeline_tools.tool_resolve_isoforms_and_sites(str(tmp_path / "state.json"))
    assert state["reference_setup"]["human"]["status"] == "available"
    assert capsys.readouterr().out == ""
