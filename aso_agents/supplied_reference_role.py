"""Keep supplied annotation identity separate from the prediction background."""
from pathlib import Path
import hashlib
import re


def _path(value):
    return str(Path(str(value)).expanduser().resolve()) if value else ""


def artifact_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_roles(goal, annotations, user_evidence=""):
    """Only quoted user choices or an explicit positive background request resolve a role."""
    records = goal.get("supplied_sequence_roles") or []
    roles = {}
    for annotation in annotations:
        path = _path(annotation.get("snapgene_path"))
        choices = set()
        for record in records:
            if not isinstance(record, dict) or _path(record.get("artifact_ref_path")) != path:
                continue
            quote = str(record.get("source_quote") or "").strip()
            role = record.get("role")
            if record.get("sequence_sha256") and record["sequence_sha256"] != annotation.get("sequence_sha256"):
                continue
            if quote and quote in user_evidence and role in {"design_background", "annotation_only"}:
                choices.add(role)
        if len(choices) == 1:
            roles[path] = choices.pop()
        elif not choices and not records and goal.get("use_supplied_sequence_variants") is True:
            # Existing explicit patient-background requests remain compatible.
            roles[path] = "design_background"
        else:
            roles[path] = None
    return roles


def role_question(goal, annotations, user_evidence=""):
    roles = resolved_roles(goal, annotations, user_evidence)
    pending = []
    for annotation in annotations:
        path = _path(annotation.get("snapgene_path"))
        variants = annotation.get("patient_variants") or []
        if not variants or roles.get(path):
            continue
        question = (f"The supplied file '{path}' has {len(variants)} detected sequence difference(s) "
                    "from the reference. Should those differences define the design background, "
                    "or should this file supply isoform annotations only? Reply 'design background' "
                    "or 'annotation only' for this file.")
        pending.append({"artifact_ref_path": path, "question": question,
                        "sequence_sha256": annotation.get("sequence_sha256"),
                        "variant_count": len(variants)})
    goal["pending_supplied_reference_roles"] = pending
    if not pending:
        return None
    return {"kind": "supplied_reference_background", "restart_stage": "resolution",
            "reason": "The supplied sequence differs from the reference, and its genetic-background role is unresolved.",
            "questions": [item["question"] for item in pending],
            "why_this_matters": "The background changes the sequence used for binding and splice predictions.",
            "context": "The file's requested isoform annotations are retained with either choice. Sequence differences have been detected but have not been applied.",
            "suggested_answers": ["design background", "annotation only"]}


def apply_role_answers(goal, answers):
    pending = goal.get("pending_supplied_reference_roles") or []
    by_question = {str(item.get("question") or ""): str(item.get("answer") or "").strip()
                   for item in answers if isinstance(item, dict)}
    if not pending or len(by_question) != len(pending) or set(by_question) != {p["question"] for p in pending}:
        raise ValueError("Please answer the background-role question for each supplied file.")
    updates = []
    for item in pending:
        answer = by_question[item["question"]]
        normalized = re.sub(r"\s+", " ", answer.lower()).rstrip(".! ")
        if normalized in {"design background", "use as design background", "use the supplied sequence as the design background", "patient background", "apply the differences"}:
            role = "design_background"
        elif normalized in {"annotation only", "annotation-only", "annotations only", "use for annotation only", "reference annotations only"}:
            role = "annotation_only"
        else:
            raise ValueError("Please choose 'design background' or 'annotation only' for each file; an ambiguous answer cannot change the prediction background.")
        updates.append({"artifact_ref_path": item["artifact_ref_path"], "role": role,
                        "source_quote": answer, "sequence_sha256": item.get("sequence_sha256")})
    paths = {_path(item["artifact_ref_path"]) for item in updates}
    retained = [record for record in goal.get("supplied_sequence_roles") or []
                if isinstance(record, dict) and _path(record.get("artifact_ref_path")) not in paths]
    goal["supplied_sequence_roles"] = retained + updates
    goal.pop("pending_supplied_reference_roles", None)
    return updates
