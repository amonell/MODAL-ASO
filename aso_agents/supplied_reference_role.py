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


def _explicit_role(text):
    """Conservative intent recognition; unclear free text gets the role question."""
    text = re.sub(r"\s+", " ", str(text).casefold()).strip().rstrip(".! ")
    if "?" in text or re.search(r"\b(?:not|never|without|whether|either|or|if|should|could|can|maybe|might)\b|n['’]t\b", text):
        return None
    background = (
        text in {"design background", "patient background", "apply the differences"}
        or bool(re.search(r"\b(?:use|treat|apply|design|model|predict)\b.*\b(?:design|patient|genetic)(?:[- ]specific)? background\b", text))
        or bool(re.search(r"\b(?:apply|incorporate)\b.*\b(?:variants|mutations|differences)\b", text))
    )
    annotation = bool(re.search(r"\bannotations?[- ]only\b|\bannotations? alone\b", text))
    if background == annotation:
        return None
    return "design_background" if background else "annotation_only"


def _quoted_role_is_explicit(record, role, evidence, path, multiple_files):
    quote = str(record.get("source_quote") or "").strip()
    if not quote or quote not in evidence or _explicit_role(quote) != role:
        return False
    # A quotation must not remove negation or an unresolved choice from its
    # surrounding user sentence. Accepted clarification answers are separate
    # evidence lines, so they can resolve an earlier ambiguous request.
    contexts = [line for line in evidence.splitlines() if quote in line]
    if not any(_explicit_role(line) == role for line in contexts):
        return False
    # Short answers acquire this sequence binding only after the per-file
    # clarification is applied. Free-form intake choices for several files must
    # identify their own file in the quote rather than borrowing another's role.
    if multiple_files and not record.get("sequence_sha256"):
        return path in quote or Path(path).name in quote
    return True


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
            role = record.get("role")
            if record.get("sequence_sha256") and record["sequence_sha256"] != annotation.get("sequence_sha256"):
                continue
            if role in {"design_background", "annotation_only"} and _quoted_role_is_explicit(
                record, role, user_evidence, path, len(annotations) > 1
            ):
                choices.add(role)
        if len(choices) == 1:
            roles[path] = choices.pop()
        elif not choices and not records and not user_evidence.strip() and goal.get("use_supplied_sequence_variants") is True:
            # A direct structured caller may explicitly select this flag. In
            # free-text intake the same flag can be inferred by the model and
            # cannot substitute for evidence of the user's background choice.
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
