"""Lossless serialization of repeated report evidence for an LLM context.

This changes representation only. The scientific payload, response contract and
delivered artifacts keep their original data and identifiers.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json


ENCODING = "lossless_shared_json_v1"
_MARKERS = {"$ref", "$table", "$literal"}


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def encode_response_context(payload, *, min_shared_bytes=192):
    """Store identical values once and encode rectangular object lists as tables.

    References are local JSON pointers, never filesystem or network resources.
    Rows retain their order; only identical column sets are eligible for a table.
    Literal marker objects are escaped, including ones supplied by a requester.
    """
    counts = Counter()
    fingerprints = {}

    def fingerprint(value):
        if isinstance(value, (dict, list)) and id(value) in fingerprints:
            return fingerprints[id(value)]
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
        result = (hashlib.sha256(raw).hexdigest(), len(raw))
        if isinstance(value, (dict, list)):
            fingerprints[id(value)] = result
        return result

    def count(value):
        if isinstance(value, dict):
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)
        elif not isinstance(value, str):
            return
        key, size = fingerprint(value)
        if size >= min_shared_bytes:
            counts[key] += 1

    count(payload)
    shared, identifiers = {}, {}

    def encode(value):
        if not isinstance(value, (dict, list, str)):
            return value
        key, size = fingerprint(value)
        if size >= min_shared_bytes and counts[key] > 1:
            if key not in identifiers:
                identifier = f"E{len(identifiers) + 1:05d}"
                identifiers[key] = identifier
                shared[identifier] = encode_value(value)
            return {"$ref": "#/shared_values/" + identifiers[key]}
        return encode_value(value)

    def encode_value(value):
        if isinstance(value, dict):
            encoded = {key: encode(child) for key, child in value.items()}
            if _MARKERS.intersection(value):
                return {"$literal": encoded}
            return encoded
        if isinstance(value, list):
            if (len(value) >= 4 and isinstance(value[0], dict) and value[0]
                    and all(isinstance(row, dict) and row.keys() == value[0].keys()
                            for row in value)):
                columns = list(value[0])
                return {"$table": {"columns": columns,
                                   "rows": [[encode(row[key]) for key in columns]
                                            for row in value]}}
            return [encode(child) for child in value]
        return value

    context = encode(payload)
    return {"context_encoding": ENCODING, "context": context,
            "shared_values": shared}


def decode_response_context(document):
    """Expand our local representation for diagnostics and round-trip checks."""
    if document.get("context_encoding") != ENCODING:
        raise ValueError("Unknown response context encoding")
    shared = document["shared_values"]

    def decode(value, stack=()):
        if isinstance(value, list):
            return [decode(child, stack) for child in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"$literal"}:
            return {key: decode(child, stack) for key, child in value["$literal"].items()}
        if set(value) == {"$ref"}:
            pointer = value["$ref"]
            if not isinstance(pointer, str) or not pointer.startswith("#/shared_values/"):
                raise ValueError("Only local evidence references are valid")
            identifier = pointer[len("#/shared_values/"):]
            if identifier in stack or identifier not in shared:
                raise ValueError("Cyclic or missing evidence reference")
            return decode(shared[identifier], (*stack, identifier))
        if set(value) == {"$table"}:
            table = value["$table"]
            columns = table["columns"]
            if len(set(columns)) != len(columns):
                raise ValueError("Duplicate evidence table columns")
            rows = []
            for row in table["rows"]:
                if len(row) != len(columns):
                    raise ValueError("Evidence table width mismatch")
                rows.append({key: decode(child, stack) for key, child in zip(columns, row)})
            return rows
        return {key: decode(child, stack) for key, child in value.items()}

    return decode(document["context"])


def render_response_context(payload):
    """Keep ordinary inputs plain and use sharing only when it saves context."""
    plain = _dump(payload)
    if len(plain.encode("utf-8")) < 24_000:
        return plain
    encoded = _dump(encode_response_context(payload))
    return encoded if len(encoded.encode("utf-8")) < len(plain.encode("utf-8")) else plain


def panel_reference_alignment_summary(row):
    """Preserve each reference/context/orientation when thinning panel evidence.

    A primary-reference mapping label cannot represent a multi-reference result.
    These are summaries of retained records, not an additional alignment search
    or an inference about activity. Missing identity metadata stays missing.
    """
    groups = {}
    for field in ("mapping_hits", "near_match_candidates"):
        for hit in row.get(field) or []:
            identity = hit.get("reference_identity") or {}
            context = {
                "transcript_id": identity.get("transcript_id") or hit.get("transcript_id"),
                "species": identity.get("species"), "assembly": identity.get("assembly"),
                "version_attested": identity.get("version_attested"),
                "reference_source": identity.get("reference_source"),
                "rna_context": hit.get("rna_context"), "mapping_scope": hit.get("mapping_scope"),
                "strand": hit.get("strand"), "match_orientation": hit.get("match_orientation"),
            }
            key = _dump(context)
            if key not in groups:
                groups[key] = {**context, "exact_mapping_record_count": 0,
                               "retained_near_match_record_count": 0,
                               "minimum_retained_mismatches": None,
                               "recorded_requested_site_overlaps": [],
                               "has_recorded_reference_junction": False}
            group = groups[key]
            exact = field == "mapping_hits"
            group["exact_mapping_record_count" if exact else "retained_near_match_record_count"] += 1
            mismatch = 0 if exact else hit.get("mismatch_count")
            if isinstance(mismatch, int) and not isinstance(mismatch, bool) and mismatch >= 0:
                previous = group["minimum_retained_mismatches"]
                group["minimum_retained_mismatches"] = mismatch if previous is None else min(previous, mismatch)
            for overlap in hit.get("requested_site_overlaps") or []:
                if overlap not in group["recorded_requested_site_overlaps"]:
                    group["recorded_requested_site_overlaps"].append(deepcopy(overlap))
            group["has_recorded_reference_junction"] |= bool(hit.get("reference_junctions_spanned"))
    return {"groups": list(groups.values()),
            "scope": "All retained source records grouped by recorded reference identity, RNA context, strand and query orientation. Record counts may include repeated windows. Missing groups or identity fields are not evidence of a negative search or biological inactivity."}


def evaluation_response_batches(payload, *, target_context_bytes=240_000):
    """Partition a large panel at observation boundaries without sampling rows.

    Each assessment receives its complete original row and citation dictionary.
    A panel overview retains every other scientific/measurement field while
    avoiding repeated exhaustive alignment lists. No rank or biological verdict
    is computed here. The original complete payload remains the delivery source.
    """
    summary = payload.get("evaluation_summary") or {}
    rows = summary.get("provided_asos") or []
    if (payload.get("task_type") != "evaluate_existing_aso_set" or len(rows) < 2
            or len(render_response_context(payload).encode("utf-8")) <= target_context_bytes):
        return [payload]
    ids = [row.get("input_row_id") for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Large evaluation panels require distinct source input_row_id values")
    repeated_detail = {"mapping_hits", "near_match_candidates", "evidence_inventory",
                       "reference_tokens", "measurement_tokens"}
    overview = []
    for row in rows:
        item = {key: deepcopy(value) for key, value in row.items() if key not in repeated_detail}
        item["alignment_inventory_counts"] = {
            "exact_mapping_count": len(row.get("mapping_hits") or []),
            "retained_near_match_count": len(row.get("near_match_candidates") or []),
            "scope": "Counts of recorded retained hypotheses; not activity or an exhaustive binding count.",
        }
        item["reference_alignment_summary"] = panel_reference_alignment_summary(row)
        overview.append(item)
    shared_summary = {key: value for key, value in summary.items()
                      if key not in {"provided_asos", "measurement_evidence", "provided_aso_count"}}
    evidence = summary.get("measurement_evidence") or {}
    batches = []
    for row in rows:
        part = {key: deepcopy(value) for key, value in payload.items() if key != "evaluation_summary"}
        part_summary = deepcopy(shared_summary)
        part_summary.update(
            provided_asos=[deepcopy(row)], provided_aso_count=1,
            full_panel_observation_count=len(rows), panel_comparison_overview=deepcopy(overview),
            measurement_evidence={key: deepcopy(value) for key, value in evidence.items()
                                  if value.get("input_row_id") == row["input_row_id"]},
        )
        part_summary["assessment_scope"] = (
            "Return an assessment only for the observation in provided_asos. Its complete retained "
            "alignment evidence and original citation tokens are supplied. The panel_comparison_overview "
            "contains the other observations' recorded measurements, identities, chemistry and reference "
            "context for evidence-limited comparative ranking; its exhaustive alignment lists are assessed "
            "in their own observation requests. Do not supply assessments or borrow alignment citations "
            "for those other rows. Use recorded panel ranks only in their stated scope; if the available "
            "comparison evidence is insufficient, leave the ranking uncertain. No observation is discarded."
        )
        part["evaluation_summary"] = part_summary
        batches.append(part)
    return batches


def merge_evaluation_responses(responses, payload):
    """Assemble original per-observation judgments, without another science draw."""
    from .input_rows import assessment_coverage_errors, assessment_row_lookup
    assessments = [deepcopy(item) for response in responses
                   for item in response.get("evaluation_assessments") or []]
    rows = (payload.get("evaluation_summary") or {}).get("provided_asos") or []
    errors = assessment_coverage_errors(assessments, rows)
    if errors:
        raise ValueError("Evaluation response assembly failed: " + "; ".join(errors))
    resolve = assessment_row_lookup(rows)
    by_id = {resolve(item)["input_row_id"]: item for item in assessments}
    merged = {"evaluation_assessments": [by_id[row["input_row_id"]] for row in rows]}
    for key in ("answer_summary", "markdown"):
        values = [str(response.get(key) or "").strip() for response in responses]
        merged[key] = "\n\n".join(dict.fromkeys(value for value in values if value))
    for key in ("assumptions", "limitations", "evidence_refs", "suggested_followups"):
        values, seen = [], set()
        for response in responses:
            supplied = response.get(key) or []
            for value in supplied if isinstance(supplied, list) else [supplied]:
                identity = _dump(value)
                if identity not in seen:
                    values.append(deepcopy(value))
                    seen.add(identity)
        merged[key] = values
    # Preserve every original component, including unexpected model commentary.
    merged["original_component_responses"] = deepcopy(responses)
    return merged
