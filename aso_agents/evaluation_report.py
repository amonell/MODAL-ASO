"""Human evaluation reports with lossless, separately linked evidence."""
from html import escape
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import uuid

from .input_rows import assessment_row_lookup
from .sequence_orientation import preferred_binding_hits


EVIDENCE_FILE = "evaluation_evidence.html"


def package_evaluation_evidence(markdown_text, destination, evidence_outputs):
    """Relocate generated asset links and retain an immutable, portable bundle.

    Only the renderer-owned link destinations are changed. Scientific prose,
    anchors, source observations and the canonical report remain untouched.
    Content-addressed directories keep independent exports from sharing or
    overwriting different evidence under the same generic filename.
    """
    destination = Path(destination)
    assets = {}
    for key in ("evaluation_evidence_json", "evaluation_evidence_html"):
        if evidence_outputs.get(key):
            source = Path(evidence_outputs[key])
            assets[source.name] = source.read_bytes()
    if not assets:
        return markdown_text, {}
    identity = {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    relative = Path(destination.name + ".assets") / digest
    bundle = destination.parent / relative
    bundle.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, data in assets.items():
        target = bundle / name
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError("export_evidence_integrity_error: " + str(target))
        else:
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        link = quote((relative / name).as_posix(), safe="/")
        # Exact renderer-owned URI tokens, never a prose/intent heuristic.
        markdown_text = markdown_text.replace("](" + name + ")", "](" + link + ")")
        markdown_text = markdown_text.replace("](" + name + "#", "](" + link + "#")
        exported[name] = str(target)
    return markdown_text, exported


def evidence_link(label, anchor):
    text = str(label).replace("[", "\\[").replace("]", "\\]")
    return f"[{text}]({EVIDENCE_FILE}#{quote(str(anchor), safe='')})"


def citation_display(reference, item):
    value = item["value"]
    if not isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    label = str(item.get("name") or "ASO") + " " + str(item.get("field") or "evidence").replace("_", " ")
    if isinstance(value, dict) and all(value.get(k) is not None for k in ("chrom", "start", "end", "coordinate_system")):
        label += f" {value['chrom']}:{value['start']}–{value['end']} ({value['coordinate_system']})"
    return evidence_link(label, reference)


def _cell(value):
    return str(value if value is not None else "unavailable").replace("|", "\\|").replace("\n", " ")


def evaluation_markdown(response, summary):
    rows = summary.get("provided_asos") or []
    resolve = assessment_row_lookup(rows)
    lines = ["# Supplied ASO evaluation", "", str(response.get("answer_summary") or ""), "",
             "## Molecule overview", "",
             "Exact and near-match counts below describe retained antisense alignment records within the searched references. They do not establish binding or activity.", "",
             "| ASO | Evaluated molecule (5′→3′) | Exact antisense records | Minimum retained antisense mismatches | Other-gene screening count | Screening status |",
             "|---|---|---:|---:|---:|---|"]
    for row in rows:
        orientation = row.get("orientation_interpretation") or {}
        scoped = {**row, "sequence_role_confirmation": row.get("sequence_role_confirmation") or orientation.get("sequence_role_confirmation") or {}}
        exact = preferred_binding_hits(scoped, row.get("mapping_hits") or [])
        near = preferred_binding_hits(scoped, row.get("near_match_candidates") or [])
        mismatches = [hit["mismatch_count"] for hit in near if isinstance(hit.get("mismatch_count"), int)]
        values = [row.get("name") or row.get("input_row_id"),
                  orientation.get("evaluated_molecule_sequence") or row.get("sequence"),
                  len(exact), min(mismatches) if mismatches else None,
                  row.get("off_target_gene_count"), row.get("off_target_count_status") or "not attested"]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines.extend(["", "## Reference context", ""])
    for key, value in (summary.get("context_summary") or {}).items():
        rendered = evidence_link(key.replace("_", " "), "context_summary") if isinstance(value, (dict, list)) else str(value)
        lines.append(f"- {key}: {rendered}")
    if summary.get("reference_catalog"):
        lines.extend(["", evidence_link("Complete reference transcript catalog", "reference_catalog")])
    scope = (summary.get("provided_aso_scoring") or {}).get("off_target_count_scope")
    if scope:
        lines.extend(["", "## Exact gene-screening count definition", ""])
        for key, value in scope.items():
            rendered = evidence_link(key.replace("_", " "), "off_target_count_scope") if isinstance(value, (list, dict)) else str(value)
            lines.append(f"- {key}: {rendered}")
    fields = ("mechanistic_plausibility", "selectivity_limitations", "uncertainty", "ranking_rationale", "evidence_refs")
    for assessment in response.get("evaluation_assessments") or []:
        lines.extend(["", "## " + str(assessment.get("name") or "ASO"), ""])
        if assessment.get("input_row_id"):
            lines.append("- Input row: " + assessment["input_row_id"])
        all_fields = [*fields, *(field for field in assessment if field not in {*fields, "name", "input_row_id"})]
        for field in all_fields:
            value = assessment.get(field)
            if value in (None, "", []):
                continue
            if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value)):
                rendered = evidence_link(field.replace("_", " "), "assessment-" + str(assessment.get("input_row_id") or assessment.get("name")))
            else:
                rendered = "; ".join(str(item) for item in value) if isinstance(value, list) else str(value)
            lines.append(f"- {field}: {rendered}")
        row = resolve(assessment) or {}
        anchor = "row-" + str(row.get("input_row_id") or row.get("name"))
        identity = row.get("resolved_sequence_identity") or {}
        if identity:
            lines.append(f"- Confirmed sequence identity: {identity.get('status')}; {identity.get('role')}; {identity.get('evaluated_molecule_sequence')}.")
        orientation = row.get("orientation_interpretation") or {}
        if orientation.get("likely_reversed_entry"):
            lines.append(
                "- Likely reversed entry: the supplied molecule has no exact antisense "
                "site in the searched context, while its reverse-complement interpretation "
                "matches the transcript-oriented site. The original molecule was not replaced; "
                "this sequence comparison does not establish cellular binding or activity."
            )
        lines.append("- " + evidence_link("Complete mapping, geometry, orientation and measured evidence for this observation", anchor))
    for field, title in (("assumptions", "Assumptions"), ("limitations", "Limitations"), ("suggested_followups", "Suggested follow-up")):
        if response.get(field):
            lines.extend(["", "## " + title, "", *["- " + str(value) for value in response[field]]])
    if str(response.get("markdown") or "").strip():
        lines.extend(["", "## Additional model commentary", "", response["markdown"]])
    lines.extend(["", "Full evidence is retained in [the evidence file](evaluation_evidence.json) and "
                  "[the browsable evidence record](evaluation_evidence.html). Every scientific assessment above retains the model's original judgment."])
    return "\n".join(lines)


def write_evaluation_evidence(directory, response):
    """Persist one cited evidence record, with stable browser anchors."""
    directory = Path(directory)
    summary = response.get("response_evaluation_summary") or response.get("evaluation_summary") or {}
    record = {"context_summary": summary.get("context_summary"),
              "reference_catalog": summary.get("reference_catalog"),
              "off_target_count_scope": (summary.get("provided_aso_scoring") or {}).get("off_target_count_scope"),
              "measurement_citations": response.get("measurement_citations") or {},
              "evaluation_assessments": response.get("evaluation_assessments") or [],
              "provided_asos": summary.get("provided_asos") or []}
    json_path = directory / "evaluation_evidence.json"
    html_path = directory / EVIDENCE_FILE
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    sections = [(key, key.replace("_", " "), record[key]) for key in ("context_summary", "reference_catalog", "off_target_count_scope")]
    sections.extend((key, f"{item.get('name')} — {item.get('field')}", item)
                    for key, item in record["measurement_citations"].items())
    sections.extend(("row-" + str(row.get("input_row_id") or row.get("name")),
                     str(row.get("name") or row.get("input_row_id")), row) for row in record["provided_asos"])
    sections.extend(("assessment-" + str(row.get("input_row_id") or row.get("name")),
                     "Assessment: " + str(row.get("name") or row.get("input_row_id")), row) for row in record["evaluation_assessments"])
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Evaluation evidence</title>',
            '<style>body{font-family:Arial,sans-serif;max-width:1000px;margin:2em auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1em 0}</style></head><body>',
            '<h1>Evaluation evidence</h1><p>Original observations and cited reference values. These are computational records, not experimental activity.</p>']
    for anchor, title, value in sections:
        html.append(f'<section id="{escape(str(anchor), quote=True)}"><h2>{escape(title)}</h2><details><summary>Show full record</summary><pre>{escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details></section>')
    html.append('</body></html>')
    html_path.write_text("\n".join(html), encoding="utf-8")
    return {"evaluation_evidence_json": str(json_path), "evaluation_evidence_html": str(html_path)}
