"""Binding-evidence precedence for user-supplied ASO evaluation panels.

Generic gapmer proxies can only break ties after physical antisense evidence.
In particular, an exact same-sequence (sense) hit is not an exact antisense
target for the supplied molecule.
"""

from __future__ import annotations

import json
from typing import Any


def evaluation_ranking_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require one recorded chemistry before comparing a panel's generic proxies.

    Exact supplied descriptions are compared conservatively. Different descriptions
    are not assumed chemically equivalent, and no mechanism is inferred from prose.
    The underlying sequence measurements remain useful without a shared ranking.
    """
    identities = []
    for row in rows:
        value = row.get("chemistry") or row.get("Chemistry")
        if isinstance(value, str):
            value = value.strip()
            identity = value if value and value.lower() not in {"unknown", "unspecified", "not recorded"} else None
        elif isinstance(value, dict) and value:
            identity = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            identity = None
        identities.append(identity)
    comparable = len(rows) < 2 or (all(identities) and len(set(identities)) == 1)
    return {
        "allow_shared_ranking": bool(comparable),
        "status": "available_within_recorded_chemistry" if comparable else "not_comparable",
        "reason": "single_recorded_chemistry_or_single_row" if comparable else "mixed_or_unresolved_recorded_chemistries",
        "scope": (
            "Generic sequence/binding proxies within one recorded chemistry; no potency or splice-efficacy claim."
            if comparable else
            "No shared numeric rank: recorded chemistries differ or are unresolved. Compare per-molecule binding and sequence evidence; effects across mechanisms are not calibrated."
        ),
    }


def suppress_incomparable_ranks(rows: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    """Remove ordinal comparisons while retaining the measured proxy components."""
    if policy["allow_shared_ranking"]:
        return
    for row in rows:
        row.pop("selection_rank", None)
        row.pop("knockdown_proxy_rank", None)
        for key in ("selection_score_components", "knockdown_proxy_score_components"):
            components = row.get(key)
            if isinstance(components, dict):
                components.pop("generic_gapmer_proxy_rank", None)
        row["ranking_status"] = policy["status"]
        row["ranking_scope"] = policy["scope"]
        stages = row.get("stage_results")
        if isinstance(stages, dict):
            stages["ranking"] = {"status": "not_run", "reason_code": policy["reason"]}


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _same_requested_transcript(record: dict[str, Any], requested: str) -> bool:
    if not requested:
        return False
    identity = record.get("reference_identity")
    versioned = str(identity.get("transcript_id") or "") if isinstance(identity, dict) else ""
    observed = versioned or str(record.get("transcript_id") or record.get("isoform_label") or "")
    if not observed or observed.split(".", 1)[0] != requested.split(".", 1)[0]:
        return False
    if "." in requested and "." in observed and requested != observed:
        return False
    return True


def requested_antisense_mismatches(entry: dict[str, Any], transcript_id: str) -> int | None:
    """Return supported mature-cDNA mismatch count; ignore sense and locus-only hits."""
    if not transcript_id:
        return None
    for record in _records(entry.get("mapping_hits")):
        if (record.get("match_orientation") == "reverse_complement"
                and record.get("mapping_scope") == "spliced_cdna"
                and _same_requested_transcript(record, transcript_id)):
            return 0
    near = []
    for record in _records(entry.get("near_match_candidates")):
        if (record.get("match_orientation") != "reverse_complement"
                or record.get("mapping_scope") != "spliced_cdna"
                or not _same_requested_transcript(record, transcript_id)):
            continue
        value = record.get("mismatch_count")
        if type(value) is int and value >= 0:
            near.append(value)
    return min(near) if near else None


def rank_scored_provided_rows(
    scored_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    transcript_id: str,
) -> dict[str, Any]:
    """Mutate scored rows only when requested-transcript antisense evidence exists."""
    entries = []
    for position, scored in enumerate(scored_rows):
        index = scored.get("provided_row_index")
        source = (original_rows[index] if type(index) is int and 0 <= index < len(original_rows)
                  and isinstance(original_rows[index], dict) else {})
        mismatches = requested_antisense_mismatches(source, transcript_id)
        old_rank = scored.get("selection_rank")
        prior = old_rank if type(old_rank) is int and old_rank > 0 else position + 1
        entries.append((scored, mismatches, prior, position))
    if not entries or all(mismatches is None for _, mismatches, _, _ in entries):
        return {"status": "not_run", "reason": "requested_transcript_antisense_evidence_unavailable"}
    ordered = sorted(entries, key=lambda item: (
        item[1] is None, item[1] if item[1] is not None else float("inf"), item[2], item[3]))
    for rank, (scored, mismatches, prior, _) in enumerate(ordered, 1):
        scored["selection_rank"] = rank
        components = scored.get("selection_score_components")
        if not isinstance(components, dict):
            components = {}
        components["requested_transcript_antisense_min_mismatches"] = mismatches
        components["generic_gapmer_proxy_rank"] = prior
        components["binding_evidence_scope"] = "exact or best ungapped reverse-complement match in requested mature transcript"
        scored["selection_score_components"] = components
    scored_rows[:] = [item[0] for item in ordered]
    return {"status": "complete", "method": "requested_transcript_antisense_evidence_then_generic_gapmer_proxy",
            "transcript_id": transcript_id, "row_count": len(scored_rows),
            "ordering": [
                {"criterion": "requested_transcript_antisense_min_mismatches",
                 "explanation": "Exact reverse-complement mature-transcript sites precede near matches with increasing mismatches; sense-only matches do not count as exact antisense sites."},
                {"criterion": "generic_gapmer_proxy_rank",
                 "explanation": "The existing generic gapmer proxy order breaks remaining ties; neither rank establishes measured activity."},
            ]}
