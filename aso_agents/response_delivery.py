"""Lossless representation repair and projection onto a declared response scope.

No assessment is invented or merged. Extra *known* panel observations belong to
their own components; the raw response remains in the native response archive.
Unknown identities and duplicate/missing in-scope observations still fail normal
coverage and evidence validation.
"""
from copy import deepcopy

from .input_rows import assessment_row_lookup
from .response_artifacts import digest
from .citation_tokens import citation_placeholders


def evaluation_delivery_schema(payload):
    if payload.get("task_type") != "evaluate_existing_aso_set":
        return None
    rows = (payload.get("evaluation_summary") or {}).get("provided_asos") or []
    if not rows or any(not row.get("input_row_id") for row in rows):
        return None
    text = {"type": "string"}
    texts = {"type": "array", "items": text}
    fields = {"input_row_id": {"type": "string", "enum": [r["input_row_id"] for r in rows]},
              "name": {"type": "string", "enum": list(dict.fromkeys(r["name"] for r in rows))},
              **{key: text for key in ("mechanistic_plausibility", "selectivity_limitations",
                                      "uncertainty", "ranking_rationale")}, "evidence_refs": texts}
    properties = {key: text for key in ("answer_summary", "markdown")}
    properties.update({key: texts for key in ("assumptions", "limitations", "evidence_refs", "suggested_followups")})
    properties["evaluation_assessments"] = {
        "type": "array", "minItems": len(rows), "maxItems": len(rows),
        "items": {"type": "object", "properties": fields, "required": list(fields),
                  "additionalProperties": False}}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def normalize_evaluation_delivery(response, payload):
    if payload.get("task_type") != "evaluate_existing_aso_set" or not isinstance(response, dict):
        return response
    result = deepcopy(response)
    summary = payload.get("evaluation_summary") or {}
    rows = summary.get("provided_asos") or []
    overview = summary.get("panel_comparison_overview") or []
    # The same observation occurs in both provided_asos and the panel overview.
    identities = {r["input_row_id"]: r for r in overview + rows if r.get("input_row_id")}
    resolve = assessment_row_lookup(list(identities.values()))
    changes = []
    assessments = result.get("evaluation_assessments")
    if isinstance(assessments, dict):
        converted = []
        for row_id, value in assessments.items():
            if not isinstance(value, dict) or row_id not in identities:
                return result  # Leave an unknown representation for validation.
            item = deepcopy(value)
            if item.get("input_row_id") not in (None, row_id):
                raise ValueError("Assessment map key conflicts with its input_row_id.")
            item["input_row_id"] = row_id
            if resolve(item) is None:
                raise ValueError("Assessment map identity does not match a supplied observation.")
            converted.append(item)
        assessments = result["evaluation_assessments"] = converted
        changes.append("input_row_id_map_to_array")
    excluded = []
    if overview and isinstance(assessments, list):
        in_scope = {row.get("input_row_id") for row in rows}
        # Never discard a foreign or ambiguous row just to satisfy the contract.
        resolved = [resolve(item) for item in assessments]
        if all(row is not None for row in resolved):
            excluded = [row["input_row_id"] for row in resolved if row["input_row_id"] not in in_scope]
            if excluded:
                result["evaluation_assessments"] = [item for item, row in zip(assessments, resolved)
                                                    if row["input_row_id"] in in_scope]
                changes.append("project_known_panel_rows_to_declared_component")
    if changes:
        result["native_delivery_normalization"] = {
            "operations": changes, "original_response_sha256": digest(response),
            "out_of_scope_input_row_ids": excluded,
            "assessment_text_changed": False,
        }
    return result


def normalize_panel_ranking_rationales(response, payload):
    """Bind panel ranking prose to recorded facts before scientific review.

    A component composer may see one molecule at a time. Free-form relative
    explanations from that component can therefore contradict its siblings.
    Keep the original model answer in the immutable draft archive and render a
    modest, source-backed per-molecule rationale for the delivered assessment.
    """
    if payload.get('task_type') != 'evaluate_existing_aso_set' or not isinstance(response, dict):
        return response
    panel = (payload.get('evaluation_summary') or {}).get('panel_comparison_overview') or []
    assessments = response.get('evaluation_assessments') or []
    if len(panel) < 2 or not isinstance(assessments, list) or not assessments:
        return response
    by_id = {row.get('input_row_id'): row for row in panel if isinstance(row, dict)
             and row.get('input_row_id')}
    if len(by_id) != len(panel):
        return response
    result = deepcopy(response)
    changed = []
    for item in result['evaluation_assessments']:
        if not isinstance(item, dict) or item.get('input_row_id') not in by_id:
            continue
        row = by_id[item['input_row_id']]
        facts = []
        rank = row.get('selection_rank')
        if type(rank) is int and 1 <= rank <= len(panel):
            facts.append(f'Recorded computational selection rank: {rank} of {len(panel)}.')
        else:
            facts.append('No comparable computational selection rank is recorded for this molecule.')
        orientation = row.get('orientation_interpretation') or {}
        exact = orientation.get('original_molecule_exact_binding')
        if exact is True and orientation.get('match_orientation') == 'reverse_complement':
            facts.append('An exact antisense sequence-alignment hypothesis is recorded for the submitted molecule.')
        elif exact is False:
            facts.append('No exact antisense alignment for the submitted molecule is recorded in the declared search; this does not establish inactivity.')
        else:
            facts.append('The exact antisense relationship of the submitted molecule is unresolved in this summary.')
        gc = row.get('gc_fraction')
        if isinstance(gc, (int, float)) and not isinstance(gc, bool) and 0 <= gc <= 1:
            facts.append(f'Recorded GC fraction: {gc:g}.')
        facts.append('These are exploratory computational descriptors, not measured efficacy or a validated potency comparison.')
        replacement = ' '.join(facts)
        if item.get('ranking_rationale') != replacement:
            item['ranking_rationale'] = replacement
            changed.append(item['input_row_id'])
    if changed:
        result['native_delivery_normalization'] = {
            **(result.get('native_delivery_normalization') or {}),
            'source_bound_panel_ranking': True,
            'panel_ranking_input_row_ids': changed,
            'original_response_sha256': digest(response),
            'assessment_text_changed': True,
        }
    return result


def normalize_structured_assessment_markdown(response):
    """Render a missing Markdown view from complete, identified model assessments.

    This changes representation only. It never fills absent assessment content or
    changes the model's original response, which remains in the response archive.
    """
    if not isinstance(response, dict) or str(response.get("markdown") or "").strip():
        return response
    # Never promote unresolved model citation syntax into a publication-facing
    # report. Only the evaluation-specific path has a measurement binder.
    def has_unresolved_ref(value):
        if isinstance(value, dict):
            return any(has_unresolved_ref(item) for item in value.values())
        if isinstance(value, list):
            return any(has_unresolved_ref(item) for item in value)
        return bool(citation_placeholders(value))
    if has_unresolved_ref(response):
        return response
    summary = response.get("answer_summary")
    assessments = response.get("evaluation_assessments")
    if not isinstance(summary, str) or not summary.strip():
        return response
    if isinstance(assessments, dict):
        if not assessments or any(not isinstance(item, dict) or item.get("input_row_id") != row_id
                                  for row_id, item in assessments.items()):
            return response
        rows = list(assessments.values())
    elif isinstance(assessments, list):
        rows = assessments
    else:
        return response
    if not rows:
        return response
    required = ("input_row_id", "mechanistic_plausibility", "selectivity_limitations",
                "uncertainty", "ranking_rationale")
    if any(not isinstance(item, dict) or any(not isinstance(item.get(key), str)
           or not item[key].strip() for key in required) for item in rows):
        return response
    identities = [item["input_row_id"] for item in rows]
    if len(identities) != len(set(identities)):
        return response
    lines = [summary.strip(), "", "## Per-observation assessments", ""]
    labels = (("mechanistic_plausibility", "Mechanistic plausibility"),
              ("selectivity_limitations", "Selectivity limitations"),
              ("uncertainty", "Uncertainty"),
              ("ranking_rationale", "Ranking rationale"))
    for item in rows:
        name = str(item.get("name") or "").strip()
        lines.append("### " + item["input_row_id"] + (" — " + name if name else ""))
        lines.append("")
        for key, label in labels:
            lines.append("- **" + label + ":** " + item[key].strip())
        refs = item.get("evidence_refs")
        if isinstance(refs, list) and refs:
            lines.append("- **Evidence references:** " + "; ".join(str(ref).strip() for ref in refs if str(ref).strip()))
        lines.append("")
    for key, title in (("limitations", "Limitations"), ("assumptions", "Assumptions")):
        items = response.get(key)
        if isinstance(items, list) and items:
            lines.extend(["## " + title, ""])
            lines.extend("- " + str(item).strip() for item in items if str(item).strip())
            lines.append("")
    result = deepcopy(response)
    result["markdown"] = "\n".join(lines).strip() + "\n"
    result["native_delivery_normalization"] = {
        **(result.get("native_delivery_normalization") or {}),
        "structured_assessments_to_markdown": True,
        "markdown_source_sha256": digest(response),
        "assessment_text_changed": False,
    }
    return result
