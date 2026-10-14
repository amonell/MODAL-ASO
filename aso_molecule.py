"""Pure physical molecule records and independent vendor-notation attestation.

The parser never calls the encoder and never interprets natural-language requests.
"""
from copy import deepcopy
import hashlib
import json

from aso_chemistry import chemistry_components, chemistry_label, require_chemistry_length, require_linkage_specification

MOLECULE_SCHEMA = "modal-aso-molecule/1.0"
ASSUMPTION_STATUSES = {
    "explicit_request", "bundled_disclosure_assent", "assumed_default",
    "observed_export_intent_unknown", "model_selected", "configured_intent_unknown",
}


def parse_ordering_notation(notation):
    """Read exact supported IDT syntax, including every base modification/bond."""
    if not isinstance(notation, str) or not notation:
        raise ValueError("Original ordering notation is required.")
    atoms, links, offset = [], [], 0
    while offset < len(notation):
        terminal = None
        if notation[offset] == "/":
            end = notation.find("/", offset + 1)
            if end < 0:
                raise ValueError("Unclosed ordering token; no substitution performed.")
            token = notation[offset + 1:end]
            if token == "iMe-dC":
                base, sugar, modification = "C", "DNA", "5mC"
            elif (len(token) == 7 and token[0] in {"5", "i", "3"}
                    and token[1:6] == "2MOEr" and token[6] in "ACGT"):
                terminal, base, sugar = token[0], token[6], "2MOE"
                modification = "5mC" if base == "C" else "none"
            else:
                raise ValueError("Unsupported or malformed ordering token; no substitution performed.")
            offset = end + 1
        elif notation[offset] == "m":
            if offset + 1 >= len(notation) or notation[offset + 1] not in "ACGU":
                raise ValueError("Unsupported OMe nucleotide token; no substitution performed.")
            base, sugar, modification = notation[offset + 1], "2OME", "none"
            offset += 2
        elif notation[offset] in "ACGT":
            base, sugar, modification = notation[offset], "DNA", "none"
            offset += 1
        else:
            raise ValueError("Unsupported or malformed ordering token; no substitution performed.")
        atoms.append({"reference_base": "T" if base == "U" else base,
                      "nucleobase": base, "sugar": sugar,
                      "base_modification": modification, "terminal_token": terminal})
        if offset < len(notation):
            star = notation[offset] == "*"
            links.append("PS" if star else "PO")
            offset += int(star)
            if offset == len(notation):
                raise ValueError("Dangling linkage marker has no following nucleotide.")
    for index, atom in enumerate(atoms):
        if atom["sugar"] == "2MOE":
            terminal = "5" if index == 0 else "3" if index == len(atoms) - 1 else "i"
            if atom["terminal_token"] != terminal:
                raise ValueError("Original MOE terminal notation contradicts its physical position.")
    chemistry = [a["sugar"] + ("_methylC" if a["base_modification"] == "5mC" else "")
                 for a in atoms]
    return {"sequence": "".join(a["reference_base"] for a in atoms),
            "sugars_5to3": [a["sugar"] for a in atoms],
            "base_modifications_5to3": [a["base_modification"] for a in atoms],
            "nucleobases_5to3": [a["nucleobase"] for a in atoms],
            "nucleotide_chemistry_5to3": chemistry, "linkages_5to3": links}


def validate_parsed_chemistry(parsed, sequence, architecture, linkages):
    gapmer = require_chemistry_length(architecture, len(sequence))
    sugar = chemistry_components(architecture).get("wing_chemistry")
    if sugar not in {"2MOE", "2OME"}:
        raise ValueError("Recipient notation validation supports MOE/OMe/DNA profiles only.")
    expected = ["DNA" if gapmer and gapmer.in_gap(i + 1) else sugar for i in range(len(sequence))]
    if (parsed["sequence"] != sequence or parsed["sugars_5to3"] != expected
            or parsed["linkages_5to3"] != linkages):
        raise ValueError("Original ordering notation contradicts the declared sequence, sugars or linkages; no substitution performed.")


def molecule_record(sequence, architecture, notation, *, linkages=None,
                    assumption_status="observed_export_intent_unknown"):
    """Attest an existing molecule; never generate/repair a missing order string."""
    if assumption_status not in ASSUMPTION_STATUSES:
        raise ValueError("Declare chemistry assumption status independently of encoding.")
    if (not isinstance(sequence, str) or not sequence or sequence != sequence.upper()
            or set(sequence) - set("ACGT")):
        raise ValueError("An uppercase physical A/C/G/T sequence is required.")
    architecture = chemistry_label(architecture)
    require_chemistry_length(architecture, len(sequence))
    components = chemistry_components(architecture)
    if architecture in {"steric-pmo", "unspecified"}:
        if notation or linkages is not None:
            raise ValueError("PMO/unspecified chemistry cannot carry PS/PO vendor notation.")
        status = "vendor_notation_unsupported" if architecture == "steric-pmo" else "chemistry_unspecified"
        sugar = components.get("wing_chemistry", "unspecified")
        parsed = {"sequence": sequence, "sugars_5to3": [sugar] * len(sequence),
                  "base_modifications_5to3": ["unspecified"] * len(sequence),
                  "nucleobases_5to3": list(sequence),
                  "nucleotide_chemistry_5to3": [sugar] * len(sequence),
                  "linkages_5to3": [components.get("backbone", "unspecified")] * (len(sequence) - 1)}
    else:
        links = require_linkage_specification(len(sequence), linkages)
        parsed = parse_ordering_notation(notation)
        validate_parsed_chemistry(parsed, sequence, architecture, links)
        status = "attested"
    record = {"schema_version": MOLECULE_SCHEMA, "antisense_5to3": sequence,
              "architecture": architecture, "platform": components["platform"],
              "nucleotide_sugars_5to3": parsed["sugars_5to3"],
              "nucleobases_5to3": parsed["nucleobases_5to3"],
              "base_modifications_5to3": parsed["base_modifications_5to3"],
              "nucleotide_chemistry_5to3": parsed["nucleotide_chemistry_5to3"],
              "linkages_5to3": parsed["linkages_5to3"],
              "PS_stereochemistry": "unspecified" if architecture == "unspecified" else "not_stereodefined" if "PS" in parsed["linkages_5to3"] else "not_applicable",
              "terminal_modifications": [] if status == "attested" else None,
              "conjugates": [] if status == "attested" else None,
              "ordering_notation": notation or "", "ordering_status": status,
              "chemistry_assumption_status": assumption_status}
    physical = {k: v for k, v in record.items() if k not in {"chemistry_assumption_status", "ordering_status"}}
    record["physical_molecule_sha256"] = hashlib.sha256(
        json.dumps(physical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def require_profile_base_modifications(record):
    """Check a newly generated native molecule against its profile's base policy.

    Historical recipient attestation preserves observed bases; native generation
    must additionally fulfill the configured base modifications.
    """
    if record["ordering_status"] != "attested":
        return
    expected = ["5mC" if base == "C" and record["architecture"] != "steric-2ome-ps"
                and not record["architecture"].startswith("2ome:") else "none"
                for base in record["antisense_5to3"]]
    if record["base_modifications_5to3"] != expected:
        raise ValueError("Ordering notation contradicts the configured base modifications; do not substitute.")


def validate_molecule_record(record):
    """Re-attest the complete record, including its digest and position arrays."""
    if not isinstance(record, dict):
        raise ValueError("Physical molecule record must be an object.")
    links = record.get("linkages_5to3") if record.get("ordering_status") == "attested" else None
    expected = molecule_record(record.get("antisense_5to3"), record.get("architecture"),
                               record.get("ordering_notation"), linkages=links,
                               assumption_status=record.get("chemistry_assumption_status"))
    if record != expected:
        raise ValueError("Physical molecule record contradicts independently parsed chemistry.")
    return deepcopy(expected)


def validate_candidate_molecule(row):
    """Require rendered row fields to agree with the attested molecule."""
    annotations = row.get("source_annotations") or {}
    record = row.get("molecule_specification") or annotations.get("molecule_specification")
    if isinstance(record, str):
        record = json.loads(record)
    record = validate_molecule_record(record)
    for source in (row, annotations):
        for field in ("synthesized_antisense_sequence", "Synthesis", "chemistry_synthesis"):
            if field in source and source[field] != record["antisense_5to3"]:
                raise ValueError("Candidate sequence contradicts its attested physical molecule.")
        for field in ("IDT", "chemistry_idt"):
            if field in source and source[field] != record["ordering_notation"]:
                raise ValueError("Candidate ordering string contradicts its attested physical molecule.")
    return record
