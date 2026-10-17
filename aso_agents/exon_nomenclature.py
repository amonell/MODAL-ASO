"""Source-backed mapping from biological exon labels to transcript ordinals.

Biological exon names are not always the same as a transcript's one-based exon
array index.  This module keeps those identities separate: request-facing
nomenclature is preserved, while an independently checked mapping may provide
the structural ordinal needed for coordinate materialization.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import re
from copy import deepcopy
from collections.abc import Callable
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from pathlib import Path
from functools import lru_cache

import requests
from Bio import SeqIO
from agents import Agent, ModelSettings, Runner
from jsonschema import Draft202012Validator

from aso_agents.async_utils import run_async_from_sync
from aso_agents.structured_outputs import strict_json_text_format


_MAPPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["alternate_mapping", "ordinal_only", "unresolved"],
        },
        "requested_biological_label": {"type": "string", "minLength": 1},
        "resolved_transcript_ordinal": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
        },
        "ordered_biological_labels": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "evidence_quote": {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 1200},
                {"type": "null"},
            ]
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [
        "status",
        "requested_biological_label",
        "resolved_transcript_ordinal",
        "ordered_biological_labels",
        "evidence_quote",
        "reason",
    ],
    "additionalProperties": False,
}

_CHECKER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["verified", "rejected"]},
        "evidence_supports_complete_order": {"type": "boolean"},
        "requested_label_matches_resolved_ordinal": {"type": "boolean"},
        "transcript_counts_are_compatible": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [
        "status",
        "evidence_supports_complete_order",
        "requested_label_matches_resolved_ordinal",
        "transcript_counts_are_compatible",
        "reason",
    ],
    "additionalProperties": False,
}


def _optional_positive_int(name: str) -> int | None:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


_MODEL = (
    str(
        os.environ.get("ASO_EXON_NOMENCLATURE_MODEL")
        or os.environ.get("ASO_GOAL_INTAKE_MODEL")
        or "gpt-5.2-2025-12-11"
    ).strip()
    or "gpt-5.2-2025-12-11"
)
_REASONING_EFFORT = (
    str(
        os.environ.get("ASO_EXON_NOMENCLATURE_REASONING_EFFORT")
        or os.environ.get("ASO_GOAL_INTAKE_REASONING_EFFORT")
        or "high"
    )
    .strip()
    .lower()
    or "high"
)
_MAX_OUTPUT_TOKENS = (
    _optional_positive_int("ASO_EXON_NOMENCLATURE_MAX_OUTPUT_TOKENS")
    or _optional_positive_int("ASO_GOAL_INTAKE_MAX_OUTPUT_TOKENS")
    or 500_000
)


exon_nomenclature_mapper_agent = Agent(
    name="Authoritative Exon Nomenclature Mapper",
    instructions=(
        "Determine whether the supplied authoritative transcript source explicitly "
        "defines a biological exon-label order that differs from one-based transcript "
        "ordinal order. Work only from the supplied source payload. Do not use memory, "
        "gene lore, or unstated conventions. Preserve the requested biological label. "
        "If the source explicitly states a complete ordered nomenclature whose item "
        "count agrees with both transcript representations, expand that order into "
        "ordered_biological_labels. Return alternate_mapping only when the requested "
        "label occurs exactly once and its one-based position differs from the numeric "
        "label. An inclusive numeric range in an explicit ordered nomenclature is "
        "an ordered sequence of its integer labels; expand it only when its endpoints "
        "and direction are unambiguous and the resulting count agrees with the "
        "transcript. Quote the smallest exact passage that establishes the order. Return "
        "ordinal_only when the source supports ordinary ordinal numbering or contains "
        "no explicit alternate order; return unresolved when the evidence conflicts or "
        "is insufficient. Never invent a label or coordinate. Return only the schema."
    ),
    tools=[],
    model=_MODEL,
    model_settings=ModelSettings(
        reasoning={"effort": _REASONING_EFFORT},
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        extra_body=strict_json_text_format(
            "aso_exon_nomenclature_mapping", _MAPPER_SCHEMA
        ),
    ),
)

exon_nomenclature_checker_agent = Agent(
    name="Authoritative Exon Nomenclature Mapping Checker",
    instructions=(
        "Independently verify the proposed biological-label to transcript-ordinal "
        "mapping against only the supplied authoritative source payload. Verify that "
        "the evidence explicitly establishes the complete ordered label sequence, the "
        "source's inclusive numeric ranges, if any, expand unambiguously to the "
        "proposed individual labels in order, the "
        "requested label occurs exactly once at the proposed one-based ordinal, the "
        "source and Ensembl transcript exon counts agree, and the evidence quote is "
        "verbatim. Reject any inference that depends on outside knowledge or an "
        "incomplete label list or on an unstated label within an ambiguous range. "
        "An unambiguous inclusive range is not an incomplete label list. Return only "
        "the schema."
    ),
    tools=[],
    model=_MODEL,
    model_settings=ModelSettings(
        reasoning={"effort": _REASONING_EFFORT},
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        extra_body=strict_json_text_format(
            "aso_exon_nomenclature_mapping_check", _CHECKER_SCHEMA
        ),
    ),
)


def _normalized_label(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("exon "):
        text = text[len("exon ") :].strip()
    return text


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def _stable_identifier(value: Any) -> str:
    return str(value or "").strip().split(".", 1)[0]


def _source_timeout_seconds() -> float:
    raw = str(
        os.environ.get("ASO_EXON_NOMENCLATURE_SOURCE_TIMEOUT_SECONDS") or "30"
    ).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 30.0
    return max(1.0, value)


def _source_error(reason: str, **details: Any) -> dict[str, Any]:
    payload = {
        "status": "unresolved",
        "source_status": "unavailable",
        "reason": str(reason or "authoritative source unavailable"),
    }
    payload.update(details)
    return payload


@lru_cache(maxsize=8)
def _local_xref_rows(path: str, modified_ns: int, size: int):
    """Read the installed annotation's paired, versioned accessions once."""
    raw = Path(path).read_bytes()
    content = gzip.decompress(raw) if path.endswith(".gz") else raw
    rows: dict[str, set[str]] = {}
    for line in content.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        transcript = fields[0].strip()
        for accession in fields[1:]:
            if re.fullmatch(r"(?:NM|NR|XM|XR)_\d+\.\d+", accession.strip()):
                rows.setdefault(transcript, set()).add(accession.strip())
    return rows, hashlib.sha256(raw).hexdigest()


def _local_annotation_xrefs(transcript_id: str, reference_annotation: str | None):
    if not reference_annotation or not re.search(r"\.\d+$", transcript_id):
        return None
    annotation = Path(reference_annotation)
    match = re.fullmatch(r"(.+?)(?:\.primary_assembly)?\.annotation\.gtf(?:\.gz)?", annotation.name)
    if not match or not annotation.is_file():
        return None
    for suffix in (".metadata.RefSeq.gz", ".metadata.RefSeq"):
        path = annotation.with_name(match.group(1) + suffix)
        if not path.is_file():
            continue
        stat = path.stat()
        rows, sha = _local_xref_rows(str(path), stat.st_mtime_ns, stat.st_size)
        accessions = rows.get(transcript_id) or set()
        if accessions:
            return ([{"dbname": "RefSeq_mRNA", "display_id": accession,
                      "info_type": "DIRECT", "ensembl_transcript_id": transcript_id,
                      "source": "local_annotation_version_pair"}
                     for accession in sorted(accessions)], str(path), sha)
    return None


def fetch_authoritative_exon_nomenclature_source(
    transcript_id: str,
    *,
    expected_exon_count: int,
    reference_annotation: str | None = None,
    expected_exon_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Retrieve an exact Ensembl-to-RefSeq transcript record and its exon evidence."""

    stable_transcript_id = _stable_identifier(transcript_id)
    if not stable_transcript_id:
        return _source_error("transcript identifier is missing")
    timeout = _source_timeout_seconds()
    headers = {
        "Accept": "application/json",
        "User-Agent": "MODAL-ASO exon-nomenclature resolver",
    }
    xref_url = (
        "https://rest.ensembl.org/xrefs/id/"
        + quote(stable_transcript_id, safe="")
    )
    local = _local_annotation_xrefs(transcript_id, reference_annotation)
    if local is not None:
        xrefs, xref_url, xref_sha256 = local
    else:
        try:
            xref_response = requests.get(
                xref_url,
                params={"external_db": "RefSeq_mRNA"},
                headers=headers,
                timeout=timeout,
            )
            xref_response.raise_for_status()
            xrefs = xref_response.json()
            xref_url = str(xref_response.url)
            xref_sha256 = _sha256_text(xref_response.text)
        except Exception as exc:
            return _source_error(
                "Ensembl transcript cross-reference retrieval failed",
                source_url=xref_url,
                exception_type=type(exc).__name__,
            )
    if not isinstance(xrefs, list):
        return _source_error(
            "Ensembl transcript cross-reference response was not a list",
            source_url=xref_url,
        )

    candidates = [
        item
        for item in xrefs
        if isinstance(item, Mapping)
        and str(item.get("dbname") or "").strip() == "RefSeq_mRNA"
        and str(item.get("display_id") or item.get("primary_id") or "").strip()
    ]
    candidates.sort(
        key=lambda item: (
            0 if str(item.get("info_type") or "").strip().upper() == "DIRECT" else 1,
            -float(item.get("ensembl_identity") or 0),
            -float(item.get("xref_identity") or 0),
            str(item.get("display_id") or item.get("primary_id") or ""),
        )
    )
    if not candidates:
        return _source_error(
            "Ensembl returned no RefSeq mRNA cross-reference",
            source_url=xref_url,
            xref_sha256=xref_sha256,
        )

    fetch_failures: list[dict[str, str]] = []
    for xref in candidates:
        accession = str(
            xref.get("display_id") or xref.get("primary_id") or ""
        ).strip()
        ncbi_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        try:
            ncbi_response = requests.get(
                ncbi_url,
                params={
                    "db": "nuccore",
                    "id": accession,
                    "rettype": "gb",
                    "retmode": "text",
                },
                headers={"User-Agent": headers["User-Agent"]},
                timeout=timeout,
            )
            ncbi_response.raise_for_status()
            record = SeqIO.read(io.StringIO(ncbi_response.text), "genbank")
        except Exception as exc:
            fetch_failures.append(
                {"accession": accession, "exception_type": type(exc).__name__}
            )
            continue

        refseq_exons = [feature for feature in record.features if feature.type == "exon"]
        if "." in accession and record.id != accession:
            fetch_failures.append({"accession": accession, "exception_type": "refseq_version_mismatch"})
            continue
        if len(refseq_exons) != int(expected_exon_count):
            fetch_failures.append(
                {
                    "accession": accession,
                    "exception_type": "transcript_exon_count_mismatch",
                }
            )
            continue
        exon_lengths = [len(exon.location) for exon in refseq_exons]
        if expected_exon_lengths and exon_lengths != list(expected_exon_lengths):
            fetch_failures.append({"accession": accession, "exception_type": "ordered_exon_lengths_mismatch"})
            continue
        comment = _compact_whitespace(record.annotations.get("comment"))
        if not comment:
            fetch_failures.append(
                {"accession": accession, "exception_type": "source_comment_missing"}
            )
            continue
        source_payload = {
            "status": "ready",
            "source_status": "authoritative",
            "ensembl_transcript_id": stable_transcript_id,
            "refseq_accession": str(record.id or accession),
            "ensembl_exon_count": int(expected_exon_count),
            "refseq_exon_count": len(refseq_exons),
            "refseq_cdna_length": len(record.seq),
            "refseq_exon_lengths": exon_lengths,
            "ensembl_exon_lengths": list(expected_exon_lengths or []),
            "source_ensembl_transcript_id": transcript_id,
            "source_comment": comment,
            "structured_comment": deepcopy(
                dict(record.annotations.get("structured_comment") or {})
            ),
            "xref_record": deepcopy(dict(xref)),
            "xref_url": xref_url,
            "source_url": str(ncbi_response.url),
            "xref_sha256": xref_sha256,
            "source_sha256": _sha256_text(ncbi_response.text),
        }
        return source_payload

    return _source_error(
        "No exact RefSeq transcript record with a compatible exon count was available",
        source_url=xref_url,
        xref_sha256=xref_sha256,
        fetch_failures=fetch_failures,
    )


def _parse_mapping_output(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value or ""))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _schema_errors(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    return [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(dict(payload)),
            key=lambda item: list(item.absolute_path),
        )
    ]


async def _run_schema_agent(
    agent: Agent,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    contract_validator: Callable[[Mapping[str, Any]], list[str]] | None = None,
) -> dict[str, Any]:
    active_payload = deepcopy(dict(payload))
    last_errors: list[str] = []
    for attempt in range(1, 4):
        if last_errors:
            active_payload["output_contract_feedback"] = {
                "attempt": attempt,
                "instruction": (
                    "Regenerate the complete response using exactly the required "
                    "schema. Do not omit, rename, or add fields."
                ),
                "validation_errors": last_errors,
            }
        result = await Runner.run(
            agent,
            input=json.dumps(active_payload, ensure_ascii=False, indent=2),
            max_steps=4,
        )
        parsed = _parse_mapping_output(result.final_output)
        last_errors = _schema_errors(parsed, schema)
        if not last_errors and contract_validator is not None:
            last_errors = contract_validator(parsed)
        if not last_errors:
            return parsed
    return {
        "status": "contract_failure",
        "reason": "model output did not satisfy the declared JSON contract",
        "validation_errors": last_errors,
    }


def _mapping_contract_errors(
    mapping: Mapping[str, Any],
    *,
    requested_label: Any,
    transcript_exons: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> list[str]:
    errors = _schema_errors(mapping, _MAPPER_SCHEMA)
    if errors or str(mapping.get("status") or "") != "alternate_mapping":
        return errors
    expected_count = len(transcript_exons)
    ordered = mapping.get("ordered_biological_labels")
    if not isinstance(ordered, list) or len(ordered) != expected_count:
        errors.append("ordered biological labels must match transcript exon count")
        return errors
    requested = _normalized_label(requested_label)
    if _normalized_label(mapping.get("requested_biological_label")) != requested:
        errors.append("mapper changed the requested biological label")
    matching_ordinals = [
        index
        for index, label in enumerate(ordered, start=1)
        if _normalized_label(label) == requested
    ]
    resolved = mapping.get("resolved_transcript_ordinal")
    if matching_ordinals != [resolved]:
        errors.append("requested label does not occur exactly at proposed ordinal")
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        errors.append("resolved transcript ordinal is not an integer")
    elif resolved < 1 or resolved > expected_count:
        errors.append("resolved transcript ordinal is outside transcript")
    else:
        ordinal_values = [
            int(exon.get("index"))
            for exon in transcript_exons
            if isinstance(exon, Mapping)
            and isinstance(exon.get("index"), int)
            and not isinstance(exon.get("index"), bool)
        ]
        if resolved not in ordinal_values:
            errors.append("resolved transcript ordinal is absent from Ensembl exons")
    try:
        numeric_requested = int(requested)
    except (TypeError, ValueError):
        numeric_requested = None
    if numeric_requested is not None and resolved == numeric_requested:
        errors.append("proposed mapping is not an alternate ordinal")
    evidence_quote = str(mapping.get("evidence_quote") or "")
    source_comment = str(source.get("source_comment") or "")
    if not evidence_quote or evidence_quote not in source_comment:
        errors.append("evidence quote is not verbatim in the authoritative source")
    if source.get("ensembl_exon_count") != source.get("refseq_exon_count"):
        errors.append("authoritative transcript exon counts disagree")
    return errors


def _accepted_mapping(
    mapping: Mapping[str, Any],
    checker: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "verified_alternate_mapping",
        "source": "authoritative_exon_nomenclature_mapping",
        "requested_exon_label": mapping["requested_biological_label"],
        "resolved_transcript_ordinal": mapping["resolved_transcript_ordinal"],
        "ordered_biological_labels": deepcopy(
            mapping["ordered_biological_labels"]
        ),
        "evidence_quote": mapping["evidence_quote"],
        "reason": mapping["reason"],
        "checker_reason": checker["reason"],
        "evidence_refs": [source["xref_url"], source["source_url"]],
        "ensembl_transcript_id": source["ensembl_transcript_id"],
        "refseq_accession": source["refseq_accession"],
        "xref_sha256": source["xref_sha256"],
        "source_sha256": source["source_sha256"],
        "ensembl_exon_count": source["ensembl_exon_count"],
        "refseq_exon_count": source["refseq_exon_count"],
        "source_ensembl_transcript_id": source.get("source_ensembl_transcript_id"),
        "refseq_exon_lengths": source.get("refseq_exon_lengths"),
        "ensembl_exon_lengths": source.get("ensembl_exon_lengths"),
    }


def _checker_contract_errors(checker: Mapping[str, Any]) -> list[str]:
    errors = _schema_errors(checker, _CHECKER_SCHEMA)
    if errors:
        return errors
    all_checks_pass = all(
        checker.get(field) is True
        for field in (
            "evidence_supports_complete_order",
            "requested_label_matches_resolved_ordinal",
            "transcript_counts_are_compatible",
        )
    )
    expected_status = "verified" if all_checks_pass else "rejected"
    if str(checker.get("status") or "") != expected_status:
        errors.append(
            "status must be verified exactly when every verification field is true; "
            "otherwise status must be rejected"
        )
    return errors


async def resolve_transcript_exon_nomenclature(
    *,
    transcript_id: str,
    requested_exon_label: Any,
    transcript_exons: Sequence[Mapping[str, Any]],
    reference_annotation: str | None = None,
) -> dict[str, Any]:
    """Return a verified structural mapping without changing the request label."""

    requested = _normalized_label(requested_exon_label)
    if not requested or not transcript_exons:
        return {
            "status": "unresolved",
            "source": "authoritative_exon_nomenclature_mapping",
            "requested_exon_label": str(requested_exon_label or ""),
            "reason": "requested exon label or transcript exon structure is missing",
        }
    source = fetch_authoritative_exon_nomenclature_source(
        transcript_id,
        expected_exon_count=len(transcript_exons),
        reference_annotation=reference_annotation,
        expected_exon_lengths=[int(exon["end"]) - int(exon["start"]) + 1
                               for exon in sorted(transcript_exons, key=lambda exon: int(exon["index"]))],
    )
    if source.get("status") != "ready":
        return {
            **deepcopy(source),
            "source": "authoritative_exon_nomenclature_mapping",
            "requested_exon_label": str(requested_exon_label),
        }
    prompt = {
        "requested_biological_exon_label": requested,
        "ensembl_transcript": {
            "transcript_id": _stable_identifier(transcript_id),
            "exons_in_transcript_order": [
                {
                    "ordinal": exon.get("index"),
                    "start": exon.get("start"),
                    "end": exon.get("end"),
                }
                for exon in transcript_exons
                if isinstance(exon, Mapping)
            ],
        },
        "authoritative_source": {
            key: deepcopy(source.get(key))
            for key in (
                "ensembl_transcript_id",
                "refseq_accession",
                "ensembl_exon_count",
                "refseq_exon_count",
                "refseq_cdna_length",
                "source_comment",
                "structured_comment",
                "xref_record",
                "xref_url",
                "source_url",
            )
        },
    }
    mapping = await _run_schema_agent(
        exon_nomenclature_mapper_agent,
        prompt,
        _MAPPER_SCHEMA,
        contract_validator=lambda candidate: _mapping_contract_errors(
            candidate,
            requested_label=requested_exon_label,
            transcript_exons=transcript_exons,
            source=source,
        ),
    )
    if str(mapping.get("status") or "") == "ordinal_only":
        return {
            "status": "ordinal_default",
            "source": "transcript_exon_index",
            "requested_exon_label": str(requested_exon_label),
            "reason": str(
                mapping.get("reason")
                or "authoritative source did not establish alternate nomenclature"
            ),
            "evidence_refs": [source["xref_url"], source["source_url"]],
            "xref_sha256": source["xref_sha256"],
            "source_sha256": source["source_sha256"],
        }
    if str(mapping.get("status") or "") != "alternate_mapping":
        return {
            "status": "unresolved", "source": "authoritative_exon_nomenclature_mapping",
            "requested_exon_label": str(requested_exon_label),
            "reason": str(mapping.get("reason") or "The source mapping remains unresolved."),
            "evidence_refs": [source["xref_url"], source["source_url"]],
            "xref_sha256": source["xref_sha256"], "source_sha256": source["source_sha256"],
        }
    contract_errors = _mapping_contract_errors(
        mapping,
        requested_label=requested_exon_label,
        transcript_exons=transcript_exons,
        source=source,
    )
    if contract_errors:
        return {
            "status": "unresolved",
            "source": "authoritative_exon_nomenclature_mapping",
            "requested_exon_label": str(requested_exon_label),
            "reason": "; ".join(contract_errors),
            "evidence_refs": [source["xref_url"], source["source_url"]],
            "xref_sha256": source["xref_sha256"],
            "source_sha256": source["source_sha256"],
        }
    checker = await _run_schema_agent(
        exon_nomenclature_checker_agent,
        {**prompt, "proposed_mapping": mapping},
        _CHECKER_SCHEMA,
        contract_validator=_checker_contract_errors,
    )
    checker_verified = bool(
        str(checker.get("status") or "") == "verified"
        and checker.get("evidence_supports_complete_order") is True
        and checker.get("requested_label_matches_resolved_ordinal") is True
        and checker.get("transcript_counts_are_compatible") is True
    )
    if not checker_verified:
        return {
            "status": "unresolved",
            "source": "authoritative_exon_nomenclature_mapping",
            "requested_exon_label": str(requested_exon_label),
            "reason": str(checker.get("reason") or "independent check rejected mapping"),
            "evidence_refs": [source["xref_url"], source["source_url"]],
            "xref_sha256": source["xref_sha256"],
            "source_sha256": source["source_sha256"],
        }
    return _accepted_mapping(mapping, checker, source)


def resolve_transcript_exon_nomenclature_sync(
    *,
    transcript_id: str,
    requested_exon_label: Any,
    transcript_exons: Sequence[Mapping[str, Any]],
    reference_annotation: str | None = None,
) -> dict[str, Any]:
    return run_async_from_sync(
        lambda: resolve_transcript_exon_nomenclature(
            transcript_id=transcript_id,
            requested_exon_label=requested_exon_label,
            transcript_exons=transcript_exons,
            reference_annotation=reference_annotation,
        )
    )


__all__ = [
    "fetch_authoritative_exon_nomenclature_source",
    "resolve_transcript_exon_nomenclature",
    "resolve_transcript_exon_nomenclature_sync",
]
