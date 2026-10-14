"""Bind generated molecules to their producing program and attest native exports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from aso_chemistry import canonical_chemistry_configuration, chemistry_label
from aso_molecule import (molecule_record, parse_ordering_notation,
                         require_profile_base_modifications, validate_molecule_record)
from aso_recipient_report import attest_record, validate_recipient_report, write_recipient_report


def candidate_configuration(row, goal):
    """Resolve only a bound producer; never borrow a sibling program's chemistry."""
    program_id = row.get("program_id")
    profiles = (goal.get("chemistry_profiles") or {}).get("by_program") or {}
    if program_id in profiles:
        return canonical_chemistry_configuration(profiles[program_id])
    programs = [p for p in goal.get("design_programs") or [] if isinstance(p, dict)]
    matches = [p for p in programs if program_id and p.get("program_id") == program_id]
    if len(matches) == 1 and matches[0].get("chemistry"):
        return canonical_chemistry_configuration(matches[0]["chemistry"])
    if len(programs) == 1 and programs[0].get("chemistry"):
        return canonical_chemistry_configuration(programs[0]["chemistry"])
    if len(programs) > 1:
        return {}
    return canonical_chemistry_configuration(goal.get("chemistry"))


def _observed_profile(notation):
    """Derive a representable layout from exact tokens, never from prose."""
    sugars = parse_ordering_notation(notation)["sugars_5to3"]
    if all(sugar == "2MOE" for sugar in sugars):
        return "steric-hindrance"
    if all(sugar == "2OME" for sugar in sugars):
        return "steric-2ome-ps"
    if "DNA" not in sugars or sugars[0] not in {"2MOE", "2OME"} or sugars[-1] != sugars[0]:
        raise ValueError("Unbound ordering notation has no supported native architecture.")
    first, last = sugars.index("DNA"), len(sugars) - 1 - sugars[::-1].index("DNA")
    if any(sugar != "DNA" for sugar in sugars[first:last + 1]):
        raise ValueError("Unbound ordering notation contains a discontinuous DNA gap.")
    if any(sugar != sugars[0] for sugar in sugars[:first] + sugars[last + 1:]):
        raise ValueError("Unbound ordering notation contains conflicting wing sugars.")
    return ("2ome:" if sugars[0] == "2OME" else "") + f"{first}-{last-first+1}-{len(sugars)-last-1}"


def attest_candidate_rows(rows, goal):
    """Attest before publishing; do not change bases, order strings, ranks or sites."""
    from .candidate_presentation import bind_candidate_program_groups
    bound = bind_candidate_program_groups(rows, goal)
    result = []
    for row in bound:
        row = deepcopy(row)
        annotations = row.get("source_annotations") or {}
        sequence = row.get("synthesized_antisense_sequence") or row.get("Synthesis") or annotations.get("Synthesis")
        notation = (row.get("IDT") or row.get("chemistry_idt") or annotations.get("IDT")
                    or annotations.get("chemistry_idt") or "")
        configuration = candidate_configuration(row, goal)
        local = row.get("chemistry_label") or annotations.get("chemistry_label")
        profile = configuration.get("CHEMISTRY") or chemistry_label(local)
        if not profile and notation:
            profile = _observed_profile(notation)
        if not sequence:
            if profile or notation:
                raise ValueError("Chemical export lacks its physical antisense sequence.")
            row["chemistry_export_status"] = "physical_sequence_unresolved"
            result.append(row)
            continue
        profile = profile or "unspecified"
        status = configuration.get("chemistry_assumption_status") or "configured_intent_unknown" if configuration else "observed_export_intent_unknown"
        molecule = molecule_record(sequence, profile, notation,
                                   linkages=configuration.get("linkage_specification", configuration.get("linkages")),
                                   assumption_status=status)
        if configuration:
            require_profile_base_modifications(molecule)
        existing = row.get("molecule_specification") or annotations.get("molecule_specification")
        if isinstance(existing, str):
            existing = json.loads(existing)
        if existing and validate_molecule_record(existing)["physical_molecule_sha256"] != molecule["physical_molecule_sha256"]:
            raise ValueError("Generated molecule changed after physical chemistry attestation.")
        row["molecule_specification"] = molecule
        row["chemistry_export_status"] = molecule["ordering_status"]
        result.append(row)
    return result


def write_native_chemistry_exports(rows, directory):
    """Content-addressed, idempotent recipient views of the selected native inventory."""
    records = []
    for index, row in enumerate(rows, 1):
        molecule = row.get("molecule_specification")
        if not molecule:
            continue
        validate_molecule_record(molecule)
        records.append(attest_record(f"candidate_{index}", molecule["antisense_5to3"],
            molecule["architecture"], molecule["ordering_notation"],
            linkages=molecule["linkages_5to3"] if molecule["ordering_status"] == "attested" else None,
            assumption_status=molecule["chemistry_assumption_status"], native_rank=row.get("selection_rank")))
    if not records:
        return {}
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = Path(directory) / "chemistry" / digest
    if path.exists():
        validate_recipient_report(path)
        if json.loads((path / "candidates.json").read_text())["records"] != records:
            raise ValueError("Native chemistry export identity collision.")
    else:
        write_recipient_report(path, records, title="ASO chemistry",
            context="Selected native candidates. Physical sequence and chemistry are preserved across all views. PMO and unspecified chemistry have no supported vendor ordering notation.",
            provenance={"source": "selected_native_generated_aso_set", "inventory_sha256": digest})
    return {"chemical_identity_json": str(path / "candidates.json"),
            "chemical_identity_csv": str(path / "candidates.csv"),
            "chemical_identity_xlsx": str(path / "candidates.xlsx"),
            "chemical_identity_md": str(path / "report.md"),
            "chemical_identity_html": str(path / "report.html")}
