from __future__ import annotations

import asyncio
from typing import Any

from aso_agents import exon_nomenclature


def _source_payload() -> dict[str, Any]:
    return {
        "status": "ready",
        "source_status": "authoritative",
        "ensembl_transcript_id": "ENST_SYNTHETIC",
        "refseq_accession": "NM_SYNTHETIC.1",  # gitleaks:allow
        "ensembl_exon_count": 4,
        "refseq_exon_count": 4,
        "refseq_cdna_length": 400,
        "source_comment": (
            "The four transcript exons are historically designated "
            "exon 1, 2a, 2, and 3."
        ),
        "structured_comment": {},
        "xref_record": {"dbname": "RefSeq_mRNA", "info_type": "DIRECT"},
        "xref_url": "https://example.invalid/xref",
        "source_url": "https://example.invalid/record",
        "xref_sha256": "x" * 64,
        "source_sha256": "y" * 64,
    }


def _transcript_exons() -> list[dict[str, int]]:
    return [
        {"index": index, "start": index * 100, "end": index * 100 + 20}
        for index in range(1, 5)
    ]


def test_source_backed_nomenclature_mapping_requires_independent_verification(
    monkeypatch: Any,
) -> None:
    source = _source_payload()
    mapper_output = {
        "status": "alternate_mapping",
        "requested_biological_label": "2",
        "resolved_transcript_ordinal": 3,
        "ordered_biological_labels": ["1", "2a", "2", "3"],
        "evidence_quote": source["source_comment"],
        "reason": "The source explicitly supplies the complete ordered labels.",
    }
    checker_output = {
        "status": "verified",
        "evidence_supports_complete_order": True,
        "requested_label_matches_resolved_ordinal": True,
        "transcript_counts_are_compatible": True,
        "reason": "The quoted order and both transcript counts agree.",
    }
    calls: list[str] = []

    monkeypatch.setattr(
        exon_nomenclature,
        "fetch_authoritative_exon_nomenclature_source",
        lambda *_args, **_kwargs: source,
    )

    async def fake_schema_agent(
        agent: Any, _payload: Any, _schema: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        calls.append(agent.name)
        return mapper_output if len(calls) == 1 else checker_output

    monkeypatch.setattr(exon_nomenclature, "_run_schema_agent", fake_schema_agent)

    resolved = asyncio.run(
        exon_nomenclature.resolve_transcript_exon_nomenclature(
            transcript_id="ENST_SYNTHETIC",
            requested_exon_label=2,
            transcript_exons=_transcript_exons(),
        )
    )

    assert resolved["status"] == "verified_alternate_mapping"
    assert resolved["requested_exon_label"] == "2"
    assert resolved["resolved_transcript_ordinal"] == 3
    assert resolved["evidence_refs"] == [source["xref_url"], source["source_url"]]
    assert calls == [
        "Authoritative Exon Nomenclature Mapper",
        "Authoritative Exon Nomenclature Mapping Checker",
    ]


def test_nomenclature_mapping_rejects_nonverbatim_source_evidence(
    monkeypatch: Any,
) -> None:
    source = _source_payload()
    monkeypatch.setattr(
        exon_nomenclature,
        "fetch_authoritative_exon_nomenclature_source",
        lambda *_args, **_kwargs: source,
    )
    calls = 0

    async def fake_schema_agent(
        _agent: Any, _payload: Any, _schema: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "status": "alternate_mapping",
            "requested_biological_label": "2",
            "resolved_transcript_ordinal": 3,
            "ordered_biological_labels": ["1", "2a", "2", "3"],
            "evidence_quote": "A passage that is absent from the source.",
            "reason": "Proposed from source evidence.",
        }

    monkeypatch.setattr(exon_nomenclature, "_run_schema_agent", fake_schema_agent)

    resolved = asyncio.run(
        exon_nomenclature.resolve_transcript_exon_nomenclature(
            transcript_id="ENST_SYNTHETIC",
            requested_exon_label=2,
            transcript_exons=_transcript_exons(),
        )
    )

    assert resolved["status"] == "unresolved"
    assert "verbatim" in resolved["reason"]
    assert calls == 1
