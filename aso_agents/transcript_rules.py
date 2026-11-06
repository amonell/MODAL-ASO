"""Resolve a typed transcript selection rule against annotated candidates."""
from copy import deepcopy
import json

from agents import Agent, Runner


_rule_agent = Agent(
    name="Transcript rule interpreter",
    instructions=(
        "Translate the user's transcript selection rule into annotation predicates. "
        "Use only the supplied rule and user evidence; do not add preferences. "
        "Return JSON {selection:{filters:{},order_by:null,unresolved_reason:null},objective_labels:[],rule_quote:string|null}. "
        "rule_quote must be one verbatim excerpt from user_evidence containing the selection rule. "
        "When rule is empty, check user_evidence for an explicitly requested annotation rule. "
        "If none is requested, or the request only names an isoform alias or accession, "
        "return selection=null, rule_quote=null and objective_labels=[]; another resolver handles those identities. "
        "selection has filters (an object), order_by (null or cdna_length, "
        "coding_bp_length, protein_length_aa), and unresolved_reason (null or a string). "
        "Filters may use is_canonical, is_mane_select, is_mane_plus_clinical (booleans), "
        "biotype, display_name (exact strings), or protein_length_aa (integer). "
        "Canonical means is_canonical=true; MANE Select means is_mane_select=true. "
        "Protein-coding means biotype=protein_coding. Longest transcript means "
        "order_by=cdna_length; longest coding sequence means coding_bp_length; "
        "longest protein means protein_length_aa. Ordering selects the greatest value. "
        "Do not use genomic span as transcript length. Never invent an accession. "
        "If the rule requires evidence outside these fields, has alternatives or "
        "cannot be represented exactly, set unresolved_reason and leave filters empty. "
        "For a scope that does not request one reference (for example all isoforms), "
        "return selection=null, rule_quote=null and objective_labels=[]. "
        "objective_labels contains only supplied objective labels whose reference is governed "
        "by this rule in the user's request. A gene-level mRNA objective uses the requested "
        "reference rule; a distinct isoform, preservation target or explicit accession retains "
        "its own identity. Do not bind a named alternative isoform to the primary rule. "
        "The caller will apply these predicates to all candidates and require a unique match."
    ),
    model="gpt-5.2-2025-12-11",
)

_BOOL_FIELDS = {"is_canonical", "is_mane_select", "is_mane_plus_clinical"}
_STRING_FIELDS = {"biotype", "display_name"}
_ORDER_FIELDS = {"cdna_length", "coding_bp_length", "protein_length_aa"}
_REFERENCE_FIELDS = {"transcript_id", "reference_transcript_id", "primary_transcript_id",
                     "reference_accession"}


def select_rule_match(plan, transcripts):
    """Apply only supported typed predicates; missing facts cannot win a ranking."""
    if not isinstance(plan, dict) or set(plan) != {"filters", "order_by", "unresolved_reason"}:
        raise ValueError("transcript_rule_response_invalid")
    filters, order = plan["filters"], plan["order_by"]
    if not isinstance(filters, dict) or order not in _ORDER_FIELDS | {None}:
        raise ValueError("transcript_rule_predicates_invalid")
    if plan["unresolved_reason"] is not None and not isinstance(plan["unresolved_reason"], str):
        raise ValueError("transcript_rule_reason_invalid")
    if plan["unresolved_reason"]:
        return []
    if not filters and order is None:
        raise ValueError("transcript_rule_has_no_predicates")
    for key, value in filters.items():
        valid = ((key in _BOOL_FIELDS and type(value) is bool)
                 or (key in _STRING_FIELDS and isinstance(value, str) and bool(value))
                 or (key == "protein_length_aa" and type(value) is int and value > 0))
        if not valid:
            raise ValueError("transcript_rule_predicate_invalid: " + key)
    matches = [row for row in transcripts if isinstance(row, dict) and row.get("id")
               and all(row.get(key) == value and row.get(key) is not None for key, value in filters.items())]
    if order and matches:
        if any(type(row.get(order)) is not int or row[order] <= 0 for row in matches):
            return []
        best = max(row[order] for row in matches)
        matches = [row for row in matches if row[order] == best]
    return matches


def _bind_references(value, rule, transcript_id):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _REFERENCE_FIELDS and item == rule:
                value[key] = transcript_id
            elif key == "transcript_ids" and isinstance(item, list):
                value[key] = [transcript_id if ref == rule else ref for ref in item]
            elif key not in {"transcript_lock", "isoform_transcripts", "cross_species_targets"}:
                _bind_references(item, rule, transcript_id)
    elif isinstance(value, list):
        for item in value:
            _bind_references(item, rule, transcript_id)


def resolve_transcript_rule(runtime, goal, final_payload, user_evidence):
    lock = goal.get("transcript_lock")
    explicit_rule = isinstance(lock, dict) and lock.get("kind") == "verbal_rule"
    scope_rule = (not lock and goal.get("isoform_scope_source") == "user_specified"
                  and isinstance(goal.get("isoform_scope"), str) and bool(goal["isoform_scope"].strip()))
    # Some intake responses retain the requested rule only in an objective's
    # label. Check the original evidence before sending that label to web alias
    # matching. Named alternatives and gene-wide scopes keep their own resolver.
    unbound_objectives = (not lock and not goal.get("transcript_id") and not goal.get("isoform_transcripts")
                         and bool(goal.get("splice_isoform_objectives")))
    if not explicit_rule and not scope_rule and not unbound_objectives:
        return {"status": "not_required"}
    rule = str((lock.get("value") if explicit_rule else goal.get("isoform_scope")) or "").strip()
    if explicit_rule and not rule:
        raise ValueError("transcript_rule_missing")
    objectives = [item for item in goal.get("splice_isoform_objectives") or [] if isinstance(item, dict)]
    response = Runner._run_sync(_rule_agent, input=json.dumps({
        "rule": rule, "user_evidence": user_evidence,
        "objectives": objectives,
        "available_annotation_fields": sorted(_BOOL_FIELDS | _STRING_FIELDS | _ORDER_FIELDS),
    }), max_steps=1)
    parsed = json.loads(response.final_output) if isinstance(response.final_output, str) else response.final_output
    if not isinstance(parsed, dict) or set(parsed) != {"selection", "objective_labels", "rule_quote"}:
        raise ValueError("transcript_rule_response_invalid")
    plan, labels = parsed["selection"], parsed["objective_labels"]
    if not isinstance(labels, list) or any(not isinstance(label, str) or label not in
            {item.get("isoform_label") for item in objectives} for label in labels):
        raise ValueError("transcript_rule_objective_binding_invalid")
    if plan is None and not explicit_rule and not labels and parsed["rule_quote"] is None:
        return {"status": "not_required"}
    quote = parsed["rule_quote"]
    if not isinstance(quote, str) or not quote.strip() or quote not in user_evidence:
        raise ValueError("transcript_rule_quote_unverified")
    rule = rule or quote
    listing = runtime.list_transcripts(symbol=goal.get("gene_symbol"),
        species=goal.get("species") or "human", gene_id=goal.get("gene_id"))
    if not isinstance(listing, dict) or listing.get("error"):
        raise ValueError("transcript_rule_annotation_unavailable")
    candidates = listing.get("transcripts") or []
    matches = select_rule_match(plan, candidates)
    if len(matches) != 1:
        return {"status": "needs_clarification", "rule": rule, "plan": plan,
                "matching_transcript_ids": [row["id"] for row in matches],
                "questions": [f'Which transcript accession should I use for “{rule}”? '
                              'The available annotations do not identify one unique match.']}
    selected = matches[0]
    transcript_id = str(selected["id"])
    from .declared_reference_catalog import _check_context
    structure = runtime._fetch_transcript_structure(transcript_id)
    _check_context(runtime, structure, goal, transcript_id)
    if not runtime._source_identifier_matches(structure, transcript_id) or not structure.get("exons_tx"):
        raise ValueError("transcript_rule_structure_unverified")
    current = goal.get("transcript_id")
    if current and current not in {rule, transcript_id, selected.get("versioned_id")}:
        raise ValueError("transcript_rule_conflicts_with_declared_reference")
    receipt = {"status": "resolved", "rule": rule, "rule_quote": quote, "plan": plan,
               "objective_labels": labels,
               "selected_transcript_id": transcript_id,
               "annotation": deepcopy(selected), "original_lock": deepcopy(lock)}
    bound = deepcopy(goal)
    _bind_references(bound, rule, transcript_id)
    for objective in bound.get("splice_isoform_objectives") or []:
        if isinstance(objective, dict) and objective.get("isoform_label") in labels:
            explicit = {objective.get(key) for key in _REFERENCE_FIELDS if objective.get(key)}
            if explicit - {transcript_id, selected.get("versioned_id")}:
                raise ValueError("transcript_rule_conflicts_with_objective_reference")
            objective["transcript_id"] = transcript_id
    bound["transcript_id"] = transcript_id
    bound["transcript_policy"] = "user_locked"
    bound["transcript_lock"] = {"kind": "explicit_id", "value": transcript_id,
                                "source": "resolved_user_rule", "original_rule": rule}
    bound["transcript_rule_resolution"] = receipt
    catalog = bound.setdefault("isoform_transcripts", {})
    from .gapmer_selectivity import resolve_catalog_label
    if resolve_catalog_label(transcript_id, catalog, catalog) is None:
        catalog[transcript_id] = {"transcript_id": transcript_id,
            "metadata": {**runtime.resolved_transcript_metadata(structure, selected),
                         "transcript_rule_resolution": deepcopy(receipt)},
            "sites": runtime._extract_splice_sites(structure),
            "exons_tx": deepcopy(structure["exons_tx"]),
            "chrom": structure.get("seq_region_name"),
            "strand": "-" if str(structure.get("strand")) in {"-", "-1"} else "+"}
    goal.clear()
    goal.update(bound)
    _bind_references(final_payload, rule, transcript_id)
    for key in ("transcript_id", "transcript_policy", "transcript_lock", "transcript_rule_resolution", "isoform_transcripts", "splice_isoform_objectives"):
        if key in goal:
            final_payload[key] = deepcopy(goal[key])
    return receipt
