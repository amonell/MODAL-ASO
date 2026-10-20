from pathlib import Path
from types import SimpleNamespace

import pytest

from aso_agents import pipeline_tools, request_artifact_evidence, species_config


def configure_references(tmp_path, monkeypatch):
    fasta = tmp_path / "configured.fa"
    annotation = tmp_path / "configured.gtf"
    config = SimpleNamespace(
        genome_fasta=fasta, gtf_path=annotation, ensembl_assembly="configured-build",
    )
    monkeypatch.setattr(species_config, "list_supported_species", lambda: {"configured_species": config})

    def no_lookup(*args, **kwargs):
        raise AssertionError("Availability must not perform reference retrieval")

    monkeypatch.setattr(species_config, "resolve_ensembl_species", no_lookup)
    return fasta, annotation


@pytest.mark.parametrize("with_index", [False, True])
def test_present_configured_references_reach_planning_and_analysis(
    tmp_path, monkeypatch, with_index,
):
    fasta, annotation = configure_references(tmp_path, monkeypatch)
    fasta.write_text(">reference\nACGT\n")
    annotation.write_text("# configured annotation\n")
    if with_index:
        Path(str(fasta) + ".fai").write_text("reference\t4\t11\t4\t5\n")

    declared = pipeline_tools._build_sandbox_declared_inputs({}, {"tasks": []})
    resources = declared.get("reference_resources", [])
    by_role = {row["role"]: row for row in resources}

    assert by_role["genome_fasta"]["path"] == str(fasta)
    assert by_role["annotation_gtf"]["path"] == str(annotation)
    assert ("genome_fasta_index" in by_role) is with_index
    assert all(row["species"] == "configured_species" for row in resources)
    assert all(row["expected_assembly"] == "configured-build" for row in resources)
    assert all(row["identity_verified"] is False for row in resources)
    # Presence and configured identity are distinct from verified identity.
    prompt = request_artifact_evidence.append_request_artifact_evidence(
        pipeline_tools, "Immutable parser prompt", "",
    )
    assert str(fasta) in prompt and str(annotation) in prompt
    assert "not identity verification" in prompt


def test_missing_files_and_directories_are_not_declared_available(tmp_path, monkeypatch):
    fasta, annotation = configure_references(tmp_path, monkeypatch)
    fasta.mkdir()
    # A stale index cannot establish that its missing or non-file FASTA exists.
    Path(str(fasta) + ".fai").write_text("stale\n")

    declared = pipeline_tools._build_sandbox_declared_inputs({}, {"tasks": []})
    assert declared.get("reference_resources", []) == []
    prompt = request_artifact_evidence.append_request_artifact_evidence(
        pipeline_tools, "Immutable parser prompt", "",
    )
    assert prompt == "Immutable parser prompt"
