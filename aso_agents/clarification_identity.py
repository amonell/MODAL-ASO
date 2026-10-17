"""Conservative explicit choices for the coordinate-preservation fast path.

Unrecognized prose remains with semantic intake. Mere mentions, including
comparison/exclusion references, are not permission to replace a transcript.
"""
import re
from copy import deepcopy

_ID = re.compile(r"\b(?:ENS[A-Z]*T\d+(?:\.\d+)?|[NX][MR]_\d+(?:\.\d+)?)\b", re.I)
_CHOICE = re.compile(
    r"\b(?:use|choose|select|switch\s+to|target)\s+(?:(?:the|transcript|isoform)\s+)*$"
    r"|\b(?:transcript|isoform)(?:\s+id)?\s*(?:is|=|:)\s*$", re.I
)
_NON_CHOICE = re.compile(
    r"\b(?:absent\s+from|missing\s+from|excluded\s+from|rather\s+than|instead\s+of"
    r"|not\s+(?:present|included|found)\s+in|exclude|avoid|preserve|protect)\s*"
    r"(?:(?:the|transcript|isoform)\s+)*$"
    r"|\b(?:not|never|do\s+not|don't)\s+(?:use|choose|select|target)\s*"
    r"(?:(?:the|transcript|isoform)\s+)*$"
    r"|\bnot\s+(?:(?:the|transcript|isoform)\s+)*$", re.I
)


def explicit_transcript_choices(answers):
    """Return positively designated IDs and other unclassified references."""
    choices = []
    unclear = []
    for item in answers:
        text = str(item.get("answer") or "").strip()
        for match in _ID.finditer(text):
            prefix = re.split(r"[;\n]|[.!?]\s+", text[:match.start()])[-1]
            if _NON_CHOICE.search(prefix):
                continue
            if _ID.fullmatch(text.rstrip(".")) or _CHOICE.search(prefix):
                if match.group() not in choices:
                    choices.append(match.group())
            else:
                if match.group() not in unclear:
                    unclear.append(match.group())
    return choices, unclear


def apply_confirmed_transcript_identities(goal, confirmations):
    """Bind explicit choices to every corresponding objective and retire stale references."""
    selected = {item.get("isoform_label"): item.get("transcript_id")
                for item in confirmations if item.get("decision") in {"accept", "user_selected"}
                and item.get("isoform_label") and item.get("transcript_id")}
    changes = []
    gene = str(goal.get("gene_symbol") or "").casefold()
    containers = [goal, *(goal.get("design_programs") or [])]
    for container in containers:
        if not isinstance(container, dict):
            continue
        targets = [container]
        for field in ("splice_isoform_objectives", "preservation_targets", "evaluation_targets"):
            targets.extend(item for item in container.get(field) or [] if isinstance(item, dict))
        for target in targets:
            target_gene = str(target.get("gene_symbol") or "").casefold()
            if gene and target_gene and gene != target_gene:
                continue
            tid = selected.get(target.get("isoform_label"))
            if tid and target.get("transcript_id") != tid:
                changes.append({"isoform_label": target["isoform_label"], "previous_transcript_id": target.get("transcript_id"),
                                "transcript_id": tid, "scope": "requested_identity"})
                target["transcript_id"] = tid
                if "ensembl_transcript_id" in target:
                    target["ensembl_transcript_id"] = tid
    for field in ("isoform_transcripts", "preseeded_isoform_transcripts"):
        catalog = goal.get(field)
        if not isinstance(catalog, dict):
            continue
        for label, tid in selected.items():
            entry = catalog.get(label)
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata") or {}
            aliases = {str(value).casefold() for value in (
                entry.get("transcript_id"), entry.get("id"), entry.get("versioned_id"),
                metadata.get("id"), metadata.get("versioned_id")) if value}
            if str(tid).casefold() not in aliases:
                goal.setdefault("superseded_isoform_transcript_selections", []).append({
                    "isoform_label": label, "catalog_field": field, "previous_reference": deepcopy(entry),
                    "requested_transcript_id": tid, "reason": "explicit_user_selection"})
                del catalog[label]
                changes.append({"isoform_label": label, "transcript_id": tid, "scope": field})
    if changes:
        goal.pop("isoform_resolution_locked", None)
    return changes
