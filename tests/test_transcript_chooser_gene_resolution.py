from __future__ import annotations

import gzip
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import requests

chooser = importlib.import_module("aso_agents.transcript_chooser_agent")
pipeline_tools = importlib.import_module("aso_agents.pipeline_tools")


def test_gene_symbol_resolution_uses_xref_fallback_after_lookup_service_error(monkeypatch: Any) -> None:
    chooser._ENSEMBL_GENE_LOOKUP_CACHE.clear()
    chooser._ENSEMBL_ID_LOOKUP_CACHE.clear()

    def fail_primary_lookup(_symbol: str, *, species: str = "human", expand: int = 1) -> None:
        raise chooser.EnsemblServiceError("Ensembl REST lookup failed (500): simulated outage")

    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_get_ensembl_json(path: str, params: dict[str, str] | None = None) -> tuple[Any, Any]:
        calls.append((path, params))
        if path == "/xrefs/symbol/mus_musculus/Mapt":
            return (
                SimpleNamespace(status_code=200, url=f"https://rest.ensembl.org{path}", text=""),
                [{"id": "ENSMUSG00000018411", "type": "gene"}],
            )
        if path == "/lookup/id/ENSMUSG00000018411":
            return (
                SimpleNamespace(status_code=200, url=f"https://rest.ensembl.org{path}", text=""),
                {
                    "id": "ENSMUSG00000018411",
                    "display_name": "Mapt",
                    "description": "microtubule-associated protein tau",
                    "object_type": "Gene",
                    "biotype": "protein_coding",
                    "species": "mus_musculus",
                    "assembly_name": "GRCm39",
                },
            )
        raise AssertionError(f"unexpected Ensembl call: {path}")

    monkeypatch.setattr(chooser, "_lookup_gene", fail_primary_lookup)
    monkeypatch.setattr(chooser, "_get_ensembl_json", fake_get_ensembl_json)

    gene = chooser._lookup_gene_by_symbol("Mapt", "mouse", expand=1)

    assert gene is not None
    assert gene["id"] == "ENSMUSG00000018411"
    assert gene["_resolution_source"] == "ensembl_xrefs_symbol"
    assert calls == [
        ("/xrefs/symbol/mus_musculus/Mapt", {"object_type": "gene"}),
        ("/lookup/id/ENSMUSG00000018411", {"expand": "1"}),
    ]


def test_lookup_gene_id_reports_xref_resolution_source(monkeypatch: Any) -> None:
    def fake_lookup_gene_by_symbol(_symbol: str, _species_key: str, *, expand: int) -> dict[str, Any]:
        assert expand == 0
        return {
            "id": "ENSMUSG00000018411",
            "display_name": "Mapt",
            "description": "microtubule-associated protein tau",
            "assembly_name": "GRCm39",
            "_resolution_source": "ensembl_xrefs_symbol",
        }

    monkeypatch.setattr(chooser, "_lookup_gene_by_symbol", fake_lookup_gene_by_symbol)

    payload = chooser.lookup_gene_id("Mapt", species="mouse")

    assert payload["gene_id"] == "ENSMUSG00000018411"
    assert payload["resolution_source"] == "ensembl_xrefs_symbol"


def test_ensembl_rest_settings_read_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("ENSEMBL_REST_RETRIES", "5")
    monkeypatch.setenv("ENSEMBL_REST_BACKOFF", "2.0")
    monkeypatch.setenv("ENSEMBL_REST_TIMEOUT", "12.5")
    assert chooser._ensembl_rest_settings() == (5, 2.0, 12.5)


def test_ensembl_rest_settings_reject_invalid_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("ENSEMBL_REST_RETRIES", "not-a-number")
    monkeypatch.setenv("ENSEMBL_REST_BACKOFF", "-3")
    monkeypatch.setenv("ENSEMBL_REST_TIMEOUT", "0")
    attempts, backoff, timeout = chooser._ensembl_rest_settings()
    assert attempts == 3
    assert backoff == 0.0
    assert timeout == 1.0


def test_get_ensembl_json_retries_500_then_reports_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ENSEMBL_REST_RETRIES", "3")
    monkeypatch.setenv("ENSEMBL_REST_BACKOFF", "1.5")
    sleeps: list[float] = []
    calls = {"n": 0}

    def fake_get(url: str, **_kwargs: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(
            status_code=500,
            url=url,
            text="Internal Server Error",
            history=[],
        )

    monkeypatch.setattr(chooser.requests, "get", fake_get)
    monkeypatch.setattr(chooser.time, "sleep", sleeps.append)

    with pytest.raises(chooser.EnsemblServiceError) as exc_info:
        chooser._get_ensembl_json("/lookup/symbol/homo_sapiens/GENE")

    err = exc_info.value
    assert calls["n"] == 3
    assert sleeps == [1.5, 3.0]
    assert err.retryable is True
    assert err.status_code == 500
    assert "Ensembl is likely down" in str(err)
    assert "Ensembl REST is unavailable (HTTP 500)" in str(err)
    assert "after 3 retries" in str(err)


def test_get_ensembl_json_timeout_is_retryable_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ENSEMBL_REST_RETRIES", "2")
    monkeypatch.setenv("ENSEMBL_REST_BACKOFF", "1.0")
    sleeps: list[float] = []

    def fake_get(_url: str, **_kwargs: Any) -> Any:
        raise requests.Timeout("timed out")

    monkeypatch.setattr(chooser.requests, "get", fake_get)
    monkeypatch.setattr(chooser.time, "sleep", sleeps.append)

    with pytest.raises(chooser.EnsemblServiceError) as exc_info:
        chooser._get_ensembl_json("/info/ping")

    err = exc_info.value
    assert sleeps == [1.0]
    assert err.retryable is True
    assert "Ensembl is likely down" in str(err)
    assert "Ensembl REST is unavailable (timeout)" in str(err)


def test_lookup_gene_by_symbol_outage_keeps_unavailable_error(monkeypatch: Any) -> None:
    chooser._ENSEMBL_GENE_LOOKUP_CACHE.clear()
    chooser._ENSEMBL_ID_LOOKUP_CACHE.clear()

    def fail_lookup(*_args: Any, **_kwargs: Any) -> None:
        raise chooser.EnsemblServiceError(
            "Ensembl REST is unavailable (HTTP 500) after 3 retries: https://rest.ensembl.org/lookup",
            retryable=True,
            status_code=500,
        )

    def fail_xrefs(*_args: Any, **_kwargs: Any) -> None:
        raise chooser.EnsemblServiceError(
            "Ensembl REST is unavailable (HTTP 500) after 3 retries: https://rest.ensembl.org/xrefs",
            retryable=True,
            status_code=500,
        )

    monkeypatch.setattr(chooser, "_lookup_gene", fail_lookup)
    monkeypatch.setattr(chooser, "_lookup_gene_by_xrefs", fail_xrefs)

    with pytest.raises(chooser.EnsemblServiceError) as exc_info:
        chooser._lookup_gene_by_symbol("GENE", "human", expand=0)

    err = exc_info.value
    assert err.retryable is True
    assert "Ensembl REST is unavailable during gene symbol resolution" in str(err)


def test_lookup_gene_by_symbol_permanent_error_dominates_transient_error(
    monkeypatch: Any,
) -> None:
    def fail_lookup(*_args: Any, **_kwargs: Any) -> None:
        raise chooser.EnsemblServiceError(
            "Ensembl REST is unavailable (HTTP 503)",
            retryable=True,
            status_code=503,
        )

    def reject_xrefs(*_args: Any, **_kwargs: Any) -> None:
        raise chooser.EnsemblServiceError(
            "Ensembl REST request failed (400): invalid request scope",
            retryable=False,
            status_code=400,
        )

    monkeypatch.setattr(chooser, "_lookup_gene", fail_lookup)
    monkeypatch.setattr(chooser, "_lookup_gene_by_xrefs", reject_xrefs)
    monkeypatch.setattr(
        chooser,
        "_lookup_gene_by_local_gtf_symbol",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(chooser.EnsemblServiceError) as exc_info:
        chooser._lookup_gene_by_symbol("GENE", "invalid_scope", expand=0)

    err = exc_info.value
    assert err.retryable is False
    assert err.status_code == 400
    assert "gene symbol resolution was rejected" in str(err)


def test_unexpected_host_is_not_retryable() -> None:
    err = chooser.EnsemblServiceError(
        "Unexpected Ensembl host 'example.com', expected 'rest.ensembl.org'.",
        retryable=False,
    )
    assert pipeline_tools._is_retryable_ensembl_service_error(err) is False
    assert pipeline_tools._ensembl_resolution_failure_details(err, "/tmp/run") == {"run_dir": "/tmp/run"}


def test_resolution_outage_is_marked_retryable() -> None:
    err = chooser.EnsemblServiceError(
        "Ensembl REST is unavailable (HTTP 500) after 3 retries: https://rest.ensembl.org/lookup",
        retryable=True,
        status_code=500,
    )
    details = pipeline_tools._ensembl_resolution_failure_details(err, "/tmp/run")
    assert details == {
        "run_dir": "/tmp/run",
        "failure_class": "infrastructure",
        "failure_reason": "external_service_unavailable",
        "retryable_external_service": True,
        "external_service": "ensembl",
    }
    assert pipeline_tools._error_has_retryable_external_service({"kind": "tool_resolve_isoforms_failed", **details})


def test_frozen_dependency_failure_is_infrastructure_but_not_retryable() -> None:
    err = chooser.FrozenDependencyUnavailableError("missing local transcript")

    details = pipeline_tools._ensembl_resolution_failure_details(err, "/tmp/run")

    assert details == {
        "run_dir": "/tmp/run",
        "failure_class": "infrastructure",
        "failure_reason": "frozen_dependency_unavailable",
        "retryable_external_service": False,
        "external_service": "local_reference_bundle",
    }
    assert pipeline_tools._error_has_retryable_external_service(details) is False


def test_tool_resolve_isoforms_marks_ensembl_outage_retryable(monkeypatch: Any, tmp_path: Any) -> None:
    state = {
        "run_dir": str(tmp_path),
        "stage_attempts": {"resolution": 2},
        "retry_policy": {"max_stage_retries": 1, "max_external_service_retries": 8},
    }

    class FakeRuntime:
        def _resolve_gene_id(self, *_args: Any, **_kwargs: Any) -> None:
            raise chooser.EnsemblServiceError(
                "Ensembl REST is unavailable (HTTP 500) after 3 retries: https://rest.ensembl.org/lookup",
                retryable=True,
                status_code=500,
            )

    monkeypatch.setattr(
        pipeline_tools,
        "_prepare_execution_stage",
        lambda **_kwargs: {"status": "ok", "state": state, "run_dir": str(tmp_path)},
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_load_goal_context_from_run_dir",
        lambda _run_dir: {
            "payload": {"request": "Design a gapmer ASO."},
            "final_payload": {"structured_goal": {"gene_symbol": "GENE", "species": "human"}},
            "structured_goal": {"gene_symbol": "GENE", "species": "human"},
            "request_envelope": {},
        },
    )
    monkeypatch.setattr(pipeline_tools, "_restore_design_objective_preseed_to_payload", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(pipeline_tools, "_request_isoform_conversion_preseed_from_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_normalized_design_objective_preseed", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline_tools, "_goal_runtime_module", lambda: FakeRuntime())
    monkeypatch.setattr(pipeline_tools, "_save_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_mark_last_tool", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_next_stage_attempt", lambda *_args, **_kwargs: 2)

    result = pipeline_tools.tool_resolve_isoforms_and_sites(str(tmp_path / "state.json"))

    assert result["status"] == "retryable_error"
    assert result["failure_class"] == "infrastructure"
    assert result["failure_reason"] == "external_service_unavailable"
    assert result["retryable_external_service"] is True
    assert result["external_service"] == "ensembl"
    assert "gene_resolution_failed" in str(result["message"])
    assert "Ensembl REST is unavailable" in str(result["message"])
    assert state["last_error"]["retryable_external_service"] is True


def test_tool_resolve_isoforms_preserves_typed_transcript_service_outage(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    state = {
        "run_dir": str(tmp_path),
        "stage_attempts": {},
        "retry_policy": {"max_stage_retries": 1, "max_external_service_retries": 8},
    }

    class FakeRuntime:
        @staticmethod
        def _resolve_gene_id(*_args: Any, **_kwargs: Any) -> str:
            return "GENE-ID"

        @staticmethod
        def _resolve_isoform_transcripts(*_args: Any, **_kwargs: Any) -> None:
            raise chooser.EnsemblServiceError(
                "Reference transcript service is unavailable.",
                retryable=True,
                status_code=503,
            )

    monkeypatch.setattr(
        pipeline_tools,
        "_prepare_execution_stage",
        lambda **_kwargs: {"status": "ok", "state": state, "run_dir": str(tmp_path)},
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_load_goal_context_from_run_dir",
        lambda _run_dir: {
            "payload": {"request": "Design an oligonucleotide."},
            "final_payload": {
                "structured_goal": {
                    "gene_symbol": "GENE",
                    "species": "human",
                    "splice_isoform_objectives": [
                        {"isoform_label": "requested isoform", "direction": "decrease"}
                    ],
                }
            },
            "structured_goal": {
                "gene_symbol": "GENE",
                "species": "human",
                "splice_isoform_objectives": [
                    {"isoform_label": "requested isoform", "direction": "decrease"}
                ],
            },
            "request_envelope": {},
        },
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_restore_design_objective_preseed_to_payload",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_request_isoform_conversion_preseed_from_text",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_tools,
        "_normalized_design_objective_preseed",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(pipeline_tools, "_goal_runtime_module", lambda: FakeRuntime())
    monkeypatch.setattr(pipeline_tools, "_save_goal_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_save_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_tools, "_mark_last_tool", lambda *_args, **_kwargs: None)

    result = pipeline_tools.tool_resolve_isoforms_and_sites(str(tmp_path / "state.json"))

    assert result["status"] == "retryable_error"
    assert result["failure_class"] == "infrastructure"
    assert result["failure_reason"] == "external_service_unavailable"
    assert result["retryable_external_service"] is True
    assert state["last_error"]["failure_class"] == "infrastructure"
def test_symbol_resolution_uses_local_gtf_after_symbol_service_errors(monkeypatch: Any) -> None:
    chooser._ENSEMBL_GENE_LOOKUP_CACHE.clear()
    chooser._ENSEMBL_ID_LOOKUP_CACHE.clear()
    # Exercise remote hydration after a symbol-only local index hit; complete
    # installed local gene payloads legitimately require no REST hydration.
    monkeypatch.setattr(chooser, "_local_gtf_gene_payload", lambda *args: None)

    def fail_primary_lookup(_symbol: str, *, species: str = "human", expand: int = 1) -> None:
        raise chooser.EnsemblServiceError("Ensembl REST lookup failed (500): simulated outage")

    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_get_ensembl_json(path: str, params: dict[str, str] | None = None) -> tuple[Any, Any]:
        calls.append((path, params))
        if path.startswith("/xrefs/symbol/homo_sapiens/"):
            raise chooser.EnsemblServiceError("Ensembl REST gene xref lookup failed (500): simulated outage")
        if path == "/lookup/id/ENSG00000000001":
            return (
                SimpleNamespace(status_code=200, url=f"https://rest.ensembl.org{path}", text=""),
                {
                    "id": "ENSG00000000001",
                    "display_name": "GENE1",
                    "description": "test gene",
                    "object_type": "Gene",
                    "biotype": "protein_coding",
                    "species": "homo_sapiens",
                    "assembly_name": "GRCh38",
                },
            )
        raise AssertionError(f"unexpected Ensembl call: {path}")

    monkeypatch.setattr(chooser, "_lookup_gene", fail_primary_lookup)
    monkeypatch.setattr(chooser, "_get_ensembl_json", fake_get_ensembl_json)
    monkeypatch.setattr(
        chooser,
        "_load_local_gtf_gene_symbol_index",
        lambda _species_key: {
            "gene1": [
                {
                    "gene_id": "ENSG00000000001",
                    "gene_name": "GENE1",
                    "gene_biotype": "protein_coding",
                }
            ]
        },
    )

    payload = chooser.lookup_gene_id("GENE1", species="human")

    assert payload["gene_id"] == "ENSG00000000001"
    assert payload["resolution_source"] == "local_gtf_gene_symbol"
    assert calls[-1] == ("/lookup/id/ENSG00000000001", {"expand": "0"})


def test_local_gtf_payload_supports_transcript_listing_and_structure(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    gtf_path = tmp_path / "annotation.gtf"
    gtf_path.write_text(
        """##gtf-version 2.2
chr1\tsource\tgene\t100\t900\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; gene_type \"protein_coding\"; gene_name \"GENE1\";
chr1\tsource\ttranscript\t100\t900\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_type \"protein_coding\"; gene_name \"GENE1\"; transcript_type \"protein_coding\"; transcript_name \"GENE1-201\"; tag \"MANE_Select\";
chr1\tsource\texon\t100\t199\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_name \"GENE1\"; exon_id \"ENSE00000000001.1\";
chr1\tsource\tCDS\t120\t199\t.\t+\t0\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_name \"GENE1\"; protein_id \"ENSP00000000001.1\";
chr1\tsource\texon\t500\t599\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_name \"GENE1\"; exon_id \"ENSE00000000002.1\";
chr1\tsource\tCDS\t500\t560\t.\t+\t2\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_name \"GENE1\"; protein_id \"ENSP00000000001.1\";
""",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        gtf_path=gtf_path,
        ensembl_species="homo_sapiens",
        ensembl_assembly="GRCh38",
        ucsc_build="hg38",
    )
    monkeypatch.setattr(chooser, "get_species_config", lambda _species: config)
    monkeypatch.setattr(chooser, "list_supported_species", lambda: {"human": config})
    chooser._ENSEMBL_ID_LOOKUP_CACHE.clear()
    chooser._LOCAL_GTF_GENE_SYMBOL_CACHE.clear()
    chooser._LOCAL_GTF_STABLE_ID_GENE_CACHE.clear()
    chooser._LOCAL_GTF_GENE_PAYLOAD_CACHE.clear()
    chooser._LOCAL_GTF_ID_PAYLOAD_CACHE.clear()

    gene = chooser._lookup_gene_by_local_gtf_symbol("GENE1", "human", expand=1)
    listing = chooser.list_transcripts("GENE1", "human", gene_id="ENSG00000000001")
    structure = chooser.get_transcript_structure("ENST00000000001")

    assert gene is not None
    assert gene["_resolution_source"] == "local_gtf_gene_symbol"
    assert gene["assembly_name"] == "GRCh38"
    assert listing["count"] == 1
    assert listing["transcripts"][0]["is_mane_select"] is True
    assert structure["id"] == "ENST00000000001"
    assert [exon["index"] for exon in structure["exons_tx"]] == [1, 2]
    assert all(exon["contains_coding"] for exon in structure["exons_tx"])


def test_frozen_transcript_lookup_uses_local_gtf_without_rest(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    gtf_path = tmp_path / "annotation.gtf"
    gtf_path.write_text(
        """##gtf-version 2.2
chr1\tsource\tgene\t100\t900\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; gene_type \"protein_coding\"; gene_name \"GENE1\";
chr1\tsource\ttranscript\t100\t900\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_type \"protein_coding\"; gene_name \"GENE1\"; transcript_type \"protein_coding\"; transcript_name \"GENE1-201\";
chr1\tsource\texon\t100\t199\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_name \"GENE1\"; exon_id \"ENSE00000000001.1\";
""",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        gtf_path=gtf_path,
        ensembl_species="homo_sapiens",
        ensembl_assembly="GRCh38",
        ucsc_build="hg38",
    )
    monkeypatch.setenv("ASO_DEPENDENCY_MODE", "frozen")
    monkeypatch.setattr(chooser, "get_species_config", lambda _species: config)
    monkeypatch.setattr(chooser, "list_supported_species", lambda: {"human": config})
    monkeypatch.setattr(
        chooser.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen lookup must not call REST")
        ),
    )
    chooser._ENSEMBL_ID_LOOKUP_CACHE.clear()
    chooser._LOCAL_GTF_GENE_SYMBOL_CACHE.clear()
    chooser._LOCAL_GTF_STABLE_ID_GENE_CACHE.clear()
    chooser._LOCAL_GTF_GENE_PAYLOAD_CACHE.clear()
    chooser._LOCAL_GTF_ID_PAYLOAD_CACHE.clear()

    structure = chooser.get_transcript_structure("ENST00000000001")
    gene = chooser.lookup_gene_id("GENE1", species="human")

    assert structure["id"] == "ENST00000000001"
    assert len(structure["exons_tx"]) == 1
    assert gene["gene_id"] == "ENSG00000000001"
    assert gene["resolution_source"] == "local_gtf_gene_symbol"


def test_frozen_missing_identifier_fails_without_rest(monkeypatch: Any) -> None:
    monkeypatch.setenv("ASO_DEPENDENCY_MODE", "frozen")
    monkeypatch.setattr(chooser, "list_supported_species", lambda: [])
    monkeypatch.setattr(
        chooser.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen lookup must not call REST")
        ),
    )

    with pytest.raises(chooser.FrozenDependencyUnavailableError):
        chooser._lookup_id("ENST00000000000")


def test_frozen_refseq_resolution_uses_version_paired_local_metadata(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    gtf_path = tmp_path / "annotation.gtf"
    gtf_path.write_text(
        """##gtf-version 2.2
chr1\tsource\tgene\t100\t900\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; gene_type \"protein_coding\"; gene_name \"GENE1\";
chr1\tsource\ttranscript\t100\t900\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_type \"protein_coding\"; gene_name \"GENE1\"; transcript_type \"protein_coding\"; transcript_name \"GENE1-201\";
chr1\tsource\texon\t100\t199\t.\t+\t.\tgene_id \"ENSG00000000001.1\"; transcript_id \"ENST00000000001.2\"; gene_name \"GENE1\"; exon_id \"ENSE00000000001.1\";
""",
        encoding="utf-8",
    )
    refseq_path = tmp_path / "metadata.RefSeq.gz"
    with gzip.open(refseq_path, "wt", encoding="utf-8") as handle:
        handle.write("ENST00000000001.2\tNM_000001.3\tNP_000001.1\n")
    config = SimpleNamespace(
        gtf_path=gtf_path,
        refseq_xref_path=refseq_path,
        ensembl_species="homo_sapiens",
        ensembl_assembly="GRCh38",
        ucsc_build="hg38",
    )
    monkeypatch.setenv("ASO_DEPENDENCY_MODE", "frozen")
    monkeypatch.setattr(chooser, "get_species_config", lambda _species: config)
    monkeypatch.setattr(chooser, "list_supported_species", lambda: {"human": config})
    monkeypatch.setattr(
        chooser.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen xref lookup must not call REST")
        ),
    )
    chooser._LOCAL_GTF_GENE_SYMBOL_CACHE.clear()
    chooser._LOCAL_GTF_STABLE_ID_GENE_CACHE.clear()
    chooser._LOCAL_GTF_GENE_PAYLOAD_CACHE.clear()
    chooser._LOCAL_GTF_ID_PAYLOAD_CACHE.clear()
    chooser._LOCAL_REFSEQ_TRANSCRIPT_CACHE.clear()

    exact = chooser.resolve_external_transcript_accession(
        "NM_000001.3", species="human"
    )
    versionless = chooser.resolve_external_transcript_accession(
        "NM_000001", species="human"
    )
    preflight = chooser.preflight_frozen_transcript_accessions(
        ["NM_000001.3", "ENST00000000001.2"]
    )

    assert exact["transcript_id"] == "ENST00000000001"
    assert exact["method"] == "local_gencode_refseq_xref"
    assert versionless["transcript_id"] == "ENST00000000001"
    assert preflight["ok"] is True
    assert preflight["checked_count"] == 2


def test_frozen_refseq_resolution_fails_closed_when_metadata_is_missing(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    config = SimpleNamespace(
        gtf_path=tmp_path / "missing.gtf",
        refseq_xref_path=tmp_path / "missing.RefSeq.gz",
        ensembl_species="homo_sapiens",
        ensembl_assembly="GRCh38",
        ucsc_build="hg38",
    )
    monkeypatch.setenv("ASO_DEPENDENCY_MODE", "frozen")
    monkeypatch.setattr(chooser, "get_species_config", lambda _species: config)
    monkeypatch.setattr(chooser, "list_supported_species", lambda: {"human": config})
    monkeypatch.setattr(
        chooser.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen xref lookup must not call REST")
        ),
    )
    chooser._LOCAL_REFSEQ_TRANSCRIPT_CACHE.clear()

    with pytest.raises(chooser.FrozenDependencyUnavailableError):
        chooser.resolve_external_transcript_accession(
            "NM_000001.3", species="human"
        )


def test_transcript_accession_extraction_is_nested_and_punctuation_neutral() -> None:
    accessions = chooser.extract_transcript_accessions(
        {
            "request": "Use NM_000001.3, then ENST00000000001.2; mutation c.123A>T.",
            "gold": {"transcript_ids": ["ENST00000000001"]},
        }
    )

    assert accessions == [
        "NM_000001.3",
        "ENST00000000001.2",
        "ENST00000000001",
    ]


def test_conserved_exon_helper_uses_resilient_symbol_resolution(monkeypatch: Any) -> None:
    gene_payload = {
        "id": "ENSG00000000001",
        "display_name": "GENE1",
        "object_type": "Gene",
        "species": "homo_sapiens",
        "assembly_name": "GRCh38",
        "Transcript": [
            {
                "id": "ENST00000334239",
                "is_canonical": 1,
                "Exon": [{"start": 100, "end": 199}],
            }
        ],
    }

    def fake_get_transcript_structure(_transcript_id: str) -> dict[str, Any]:
        return {
            "id": "ENST00000334239",
            "exons_tx": [{"index": 1, "start": 100, "end": 199}],
        }

    def fake_lookup_gene_by_symbol(symbol: str, species_key: str, expand: int) -> dict[str, Any]:
        assert symbol == "GENE1"
        assert species_key == "human"
        assert expand == 1
        return gene_payload

    monkeypatch.setattr(chooser, "get_transcript_structure", fake_get_transcript_structure)
    monkeypatch.setattr(chooser, "_lookup_gene_by_symbol", fake_lookup_gene_by_symbol)
    monkeypatch.setattr(chooser, "_gtex_list_tissues_internal", lambda: [])

    result = chooser.conserved_exons_for_transcript("GENE1", "ENST00000334239", species="human")

    assert not result.get("error")
    assert result["gene_symbol"] == "GENE1"
    assert result["top_exons"][0]["exon_index"] == 1
