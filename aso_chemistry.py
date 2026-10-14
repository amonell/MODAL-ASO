"""Physical MOE/OMe/DNA full-PS and morpholino oligonucleotide architectures.

This module has no model or scientific-runtime dependencies so both native
design and the standalone tiler use the same molecular encoding. It parses a
machine-readable numeric descriptor, never interprets a user's prose.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GapmerArchitecture:
    wing_5prime: int
    dna_gap: int
    wing_3prime: int
    wing_chemistry: str = "2MOE"

    @property
    def length(self):
        return self.wing_5prime + self.dna_gap + self.wing_3prime

    @property
    def label(self):
        prefix = "2ome:" if self.wing_chemistry == "2OME" else ""
        return prefix + f"{self.wing_5prime}-{self.dna_gap}-{self.wing_3prime}"

    def in_gap(self, position_1based):
        return (
            isinstance(position_1based, int)
            and not isinstance(position_1based, bool)
            and self.wing_5prime < position_1based <= self.wing_5prime + self.dna_gap
        )


def gapmer_architecture(label):
    """Return an architecture for an exact positive wing-gap-wing descriptor."""
    if not isinstance(label, str):
        return None
    label = label.strip()
    sugar = "2MOE"
    if label.startswith("2ome:"):
        sugar, label = "2OME", label[len("2ome:"):]
    parts = label.split("-")
    if len(parts) != 3 or any(
        not part or not part.isascii() or not part.isdecimal() for part in parts
    ):
        return None
    counts = tuple(int(part) for part in parts)
    if any(count <= 0 for count in counts):
        return None
    return GapmerArchitecture(*counts, wing_chemistry=sugar)


STERIC_SUGARS = {"steric-hindrance": "2MOE", "steric-2ome-ps": "2OME",
                  "steric-pmo": "MORPHOLINO"}


def chemistry_components(label):
    architecture = gapmer_architecture(label)
    if architecture:
        return {"platform": "gapmer", "wing_chemistry": architecture.wing_chemistry,
                "gap_chemistry": "DNA", "backbone": "PS"}
    if label in STERIC_SUGARS:
        return {"platform": "steric_blocker", "wing_chemistry": STERIC_SUGARS[label],
                "gap_chemistry": "none",
                "backbone": "PHOSPHORODIAMIDATE" if label == "steric-pmo" else "PS"}
    if label == "unspecified":
        return {"platform": ""}
    raise ValueError("Unresolved chemical architecture: " + str(label))


def require_modality_chemistry(modality, label):
    """Reject a known architecture that contradicts its typed design mechanism."""
    expected = {'gapmer': 'gapmer', 'steric': 'steric_blocker'}.get(modality)
    if expected is None or label == 'unspecified':
        return
    actual = chemistry_components(label)['platform']
    if actual != expected:
        raise ValueError(f'Design modality {modality} requires {expected} architecture; '
                         f'selected {label} encodes {actual}. Resolve the chemistry/mechanism conflict before design.')
    architecture = gapmer_architecture(label)
    if modality == 'gapmer' and architecture and architecture.dna_gap < 6:
        raise ValueError('Native RNase H gapmer design requires a supported DNA gap of at least 6 nt; '
                         'a shorter physical architecture cannot establish that mechanism. Do not substitute another gap.')


def architecture_label(architecture):
    """Compile typed chemical components; never parse a user's prose."""
    if not isinstance(architecture, dict):
        raise ValueError("Chemical architecture must be an object.")
    sugar, backbone = architecture.get("sugar"), architecture.get("backbone")
    if (sugar, backbone) == ("MORPHOLINO", "PHOSPHORODIAMIDATE"):
        counts = tuple(architecture.get(key) for key in ("wing_5prime", "dna_gap", "wing_3prime"))
        if architecture.get("kind") == "steric_blocker" and counts == (0, 0, 0):
            if architecture.get("linkages") is not None:
                raise ValueError("PMO cannot use PS/PO linkages.")
            return "steric-pmo"
        raise ValueError("PMO is a fully morpholino steric blocker without a DNA gap.")
    if sugar not in {"2MOE", "2OME"} or backbone != "PS":
        raise ValueError("Unsupported chemical building blocks: " + str((sugar, backbone)))
    counts = tuple(architecture.get(key) for key in ("wing_5prime", "dna_gap", "wing_3prime"))
    if any(type(n) is not int for n in counts):
        raise ValueError("Chemical segment lengths must be integer nucleotide counts.")
    if architecture.get("kind") == "steric_blocker" and counts == (0, 0, 0):
        if architecture.get("linkages") is not None:
            links = architecture["linkages"]
            if not isinstance(links, (list, tuple)):
                raise ValueError("Linkages must be a PS/PO array.")
            require_linkage_specification(len(links) + 1, links, allow_mixed=False)
        return next(label for label, value in STERIC_SUGARS.items() if value == sugar)
    if architecture.get("kind") == "gapmer" and all(n > 0 for n in counts):
        if "linkages" in architecture:
            require_linkage_specification(sum(counts), architecture["linkages"], allow_mixed=False)
        return GapmerArchitecture(*counts, wing_chemistry=sugar).label
    raise ValueError("Steric blockers have no DNA gap/wing partition; gapmer segments must be positive.")


def chemistry_label(label):
    architecture = gapmer_architecture(label)
    if architecture:
        return architecture.label
    if isinstance(label, str) and label.strip() in {*STERIC_SUGARS, "unspecified"}:
        return label.strip()
    return None


def require_chemistry_length(label, length):
    """Reject substitutions; an exact requested length is never an override."""
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("Oligonucleotide length must be a positive integer.")
    architecture = gapmer_architecture(label)
    if architecture:
        if architecture.length != length:
            raise ValueError(
                f"{architecture.label} gapmer requires {architecture.length}-nt ASOs; "
                f"got length {length}. Resolve the chemistry/length conflict before tiling."
            )
    elif chemistry_label(label) not in {*STERIC_SUGARS, "unspecified"}:
        raise ValueError(
            "Unresolved chemistry: provide a canonical wing-gap-wing MOE/DNA full-PS "
            "descriptor, 2ome:W-G-V, a fully MOE/OMe full-PS steric profile, or steric-pmo."
        )
    return architecture


def require_linkage_specification(length, linkages=None, *, allow_mixed=True):
    """Validate one PS/PO value for each 5prime-to-3prime internucleotide link.

    None retains the existing full-PS default; callers must separately record
    whether that default was specified, assumed, or inferred from an export.
    """
    if type(length) is not int or length < 1:
        raise ValueError("Linkage specification requires a positive nucleotide length.")
    if linkages is None:
        return ["PS"] * (length - 1)
    if not isinstance(linkages, (list, tuple)) or len(linkages) != length - 1:
        raise ValueError("Specify exactly length-1 internucleotide linkages.")
    if any(value not in ("PS", "PO") for value in linkages):
        raise ValueError("Unsupported linkage: only PS and PO are represented.")
    if not allow_mixed and any(value != "PS" for value in linkages):
        raise ValueError("Mixed PS/PO native design is unsupported; preserve the request, do not substitute full PS.")
    return list(linkages)


def encode_idt_oligo(sequence, label, *, linkages=None):
    """Encode explicit PS/PO linkages, retaining full PS only as the default."""
    sequence = str(sequence).upper()
    architecture = require_chemistry_length(label, len(sequence))
    if any(base not in "ACGT" for base in sequence):
        raise ValueError("Chemical encoding requires an unambiguous A/C/G/T sequence.")
    if chemistry_label(label) in {"unspecified", "steric-pmo"} and linkages is not None:
        raise ValueError("PS/PO linkages cannot be assigned to unspecified or PMO chemistry.")
    if chemistry_label(label) == "unspecified":
        # A physical base sequence is not an orderable chemical architecture.
        # Keep the sequence in its own output field without inventing chemistry.
        return ""
    if chemistry_label(label) == "steric-pmo":
        # PMOs have a phosphorodiamidate morpholino backbone. The IDT
        # MOE/OMe/PS notation is inapplicable; retain the physical base
        # sequence and typed chemistry in separate export fields.
        return ""
    links = require_linkage_specification(len(sequence), linkages)
    units = []
    sugar = chemistry_components(chemistry_label(label))["wing_chemistry"]
    for index, base in enumerate(sequence):
        if architecture and architecture.in_gap(index + 1):
            units.append("/iMe-dC/" if base == "C" and sugar == "2MOE" else base)
        elif sugar == "2OME":
            # IDT RNA notation: mA/mC/mG/mU and * between PS-linked bases.
            # Reference/mapping sequences retain the DNA alphabet; this is the
            # physical RNA building block at synthesis, not a sequence redesign.
            units.append("m" + ("U" if base == "T" else base))
        else:
            terminus = "5" if index == 0 else "3" if index == len(sequence) - 1 else "i"
            units.append(f"/{terminus}2MOEr{base}/")
    notation = "".join(unit + ("*" if index < len(links) and links[index] == "PS" else "")
                       for index, unit in enumerate(units))
    # A separate parser checks the emitted physical identity, not just its label.
    from aso_molecule import molecule_record, require_profile_base_modifications
    record = molecule_record(sequence, chemistry_label(label), notation, linkages=links)
    require_profile_base_modifications(record)
    return notation


CHEMICAL_CONFIGURATION_FIELDS = {
    "CHEMISTRY", "profile", "chemistry", "name", "length", "aso_length_nt",
    "platform", "wing_chemistry", "gap_chemistry", "backbone", "sugar",
    "kind", "wing_5prime", "dna_gap", "wing_3prime", "architecture",
    "linkages", "linkage_specification", "rationale", "notes", "source",
    "chemistry_assumption_status", "cytosine_methylation", "stereochemistry",
    "terminal_modifications", "conjugates",
}
CHEMISTRY_PAYLOAD_FIELDS = {"recommendation", "unsupported_reason", "source", "mechanism_review",
                            "requested_length_override", "chemical_requirements"}


def _validate_chemical_object(value, length=None):
    """Validate every field in a chemical object before projecting its label."""
    unknown = set(value) - CHEMICAL_CONFIGURATION_FIELDS
    if unknown:
        raise ValueError("Unsupported chemistry field(s): " + ", ".join(sorted(unknown))
                         + "; preserve the requirement, do not substitute a default profile.")
    labels = [chemistry_label(value[k]) for k in ("CHEMISTRY", "profile", "name", "chemistry")
              if isinstance(value.get(k), str) and chemistry_label(value[k])]
    if len(set(labels)) > 1:
        raise ValueError("Conflicting chemistry identities; do not substitute a profile.")
    label = labels[0] if labels else None
    if label and any(isinstance(value.get(k), str) and not chemistry_label(value[k])
                     for k in ("CHEMISTRY", "profile", "chemistry")):
        raise ValueError("Conflicting resolved and unresolved chemical identities; do not substitute a profile.")
    if isinstance(value.get("architecture"), dict):
        nested = value["architecture"]
        _validate_chemical_object(nested, value.get("length", length))
        try:
            nested_label = architecture_label(nested)
        except ValueError as exc:
            raise ValueError(str(exc) + '; do not substitute a partial architecture.') from exc
        if label and label != nested_label:
            raise ValueError("Declared profile contradicts its typed architecture.")
        label = nested_label
    if value.get("kind") in {"gapmer", "steric_blocker"}:
        try:
            label_from_parts = architecture_label(value)
        except ValueError as exc:
            raise ValueError(str(exc) + '; do not substitute an unsupported architecture.') from exc
        if label and label != label_from_parts:
            raise ValueError("Declared profile contradicts its typed architecture.")
        label = label_from_parts
    components = chemistry_components(label) if label else {}
    for field, expected in (("platform", components.get("platform")),
                            ("wing_chemistry", components.get("wing_chemistry")),
                            ("sugar", components.get("wing_chemistry")),
                            ("gap_chemistry", components.get("gap_chemistry")),
                            ("backbone", components.get("backbone"))):
        observed = {"steric blocker": "steric_blocker"}.get(value[field], value[field]) if field == "platform" and field in value else value.get(field)
        if field in value and label and observed != expected:
            raise ValueError(f"Unsupported {field}: {value[field]}; {label} represents {expected}; do not substitute.")
    if value.get("stereochemistry") is not None and not (
            value["stereochemistry"] == "not_stereodefined" and components.get("backbone") == "PS"):
        raise ValueError("Explicit PS stereochemistry is unsupported by native design; do not substitute an unspecified pattern.")
    for field in ("terminal_modifications", "conjugates"):
        if value.get(field) not in (None, []):
            raise ValueError(f"Explicit {field} are unsupported by native design; do not discard them.")
    if "cytosine_methylation" in value:
        requested = value["cytosine_methylation"]
        default = "all" if components.get("wing_chemistry") == "2MOE" else "none"
        if requested not in (None, "profile_default", default) or label in {"steric-pmo", "unspecified"} and requested not in (None, "profile_default"):
            raise ValueError("Requested cytosine methylation differs from the represented profile; do not substitute or discard it.")
    if "chemistry_assumption_status" in value:
        from aso_molecule import ASSUMPTION_STATUSES
        if value["chemistry_assumption_status"] not in ASSUMPTION_STATUSES:
            raise ValueError("Invalid chemical assumption status.")
    declared_length = value.get("length", value.get("aso_length_nt", length))
    if declared_length is not None and (type(declared_length) is not int or declared_length < 1):
        raise ValueError("Chemical length must be a positive integer; do not substitute a default length.")
    if value.get("length") is not None and value.get("aso_length_nt") is not None and value["length"] != value["aso_length_nt"]:
        raise ValueError("Conflicting chemical lengths; do not substitute a default length.")
    architecture = gapmer_architecture(label)
    if architecture and declared_length is not None:
        require_chemistry_length(label, declared_length)
    local_length = architecture.length if architecture else declared_length
    present = [value[k] for k in ("linkages", "linkage_specification")
               if k in value and value[k] is not None]
    if len(present) == 2 and present[0] != present[1]:
        raise ValueError("Conflicting internucleotide linkage arrays.")
    for links in present:
        if label in {"steric-pmo", "unspecified"} or value.get("sugar") == "MORPHOLINO":
            raise ValueError("PS/PO linkages are unsupported for PMO/unspecified chemistry.")
        if not isinstance(links, (list, tuple)):
            raise ValueError("Explicit linkages require a PS/PO array; do not substitute full PS.")
        checked_length = local_length if type(local_length) is int else len(links) + 1
        require_linkage_specification(checked_length, links, allow_mixed=False)


def require_native_linkage_identity(value, length=None, *, _chemical_object=False):
    """Compatibility entry point for complete native chemical-identity checks.

    Only explicitly identified chemical objects are checked against the chemical
    field registry. Goal metadata is not treated as a chemical specification.
    Unresolved prose is preserved for semantic selection; it is never parsed here.
    """
    if isinstance(value, list):
        for entry in value:
            require_native_linkage_identity(entry, length, _chemical_object=_chemical_object)
        return
    if not isinstance(value, dict):
        return
    if "recommendation" in value and (_chemical_object or set(value) <= CHEMISTRY_PAYLOAD_FIELDS):
        unexpected = set(value) - CHEMISTRY_PAYLOAD_FIELDS
        if unexpected:
            raise ValueError("Unsupported chemistry payload field(s): " + ", ".join(sorted(map(str, unexpected))) + "; do not discard them.")
    if "recommendation" in value and value.get("unsupported_reason"):
        raise ValueError("Unsupported chemistry: " + str(value["unsupported_reason"]) + "; do not substitute a recommendation.")
    if "recommendation" in value and value.get("chemical_requirements"):
        from aso_agents.chemical_requirements import require_native_requirement_capabilities, requirement_configuration_errors
        requirements = value["chemical_requirements"]
        require_native_requirement_capabilities(requirements)
        recommendation = value["recommendation"]
        if isinstance(recommendation, dict) and chemistry_label(recommendation.get("CHEMISTRY")):
            represented = canonical_chemistry_configuration(recommendation, length)
            errors = requirement_configuration_errors(requirements, represented)
            if errors:
                raise ValueError("; ".join(errors))
    if ((_chemical_object and "recommendation" not in value) or any(k in value for k in ("CHEMISTRY", "profile"))
            or value.get("kind") in {"gapmer", "steric_blocker"}):
        _validate_chemical_object(value, length)
    local_length = value.get("length", value.get("aso_length_nt", length))
    label = chemistry_label(value.get("CHEMISTRY") or value.get("profile")
                            or (value.get("chemistry") if isinstance(value.get("chemistry"), str) else None))
    architecture = gapmer_architecture(label)
    if "backbone" in value:
        declared = value["backbone"]
        if declared not in {"PS", "PHOSPHORODIAMIDATE"}:
            raise ValueError("Unsupported native backbone; do not substitute full PS.")
        expected_typed = {"2MOE": "PS", "2OME": "PS", "MORPHOLINO": "PHOSPHORODIAMIDATE"}.get(value.get("sugar"))
        if (expected_typed and declared != expected_typed) or (value.get("kind") == "gapmer" and declared != "PS"):
            raise ValueError("Typed native backbone conflicts with chemical architecture; do not substitute full PS.")
        if chemistry_label(label) and declared != chemistry_components(label).get("backbone"):
            raise ValueError("Explicit native backbone conflicts with selected chemistry; do not substitute full PS.")
    if architecture:
        local_length = architecture.length
    if value.get("kind") == "gapmer":
        counts = [value.get(k) for k in ("wing_5prime", "dna_gap", "wing_3prime")]
        if all(type(n) is int and n > 0 for n in counts):
            local_length = sum(counts)
    for key in ("linkages", "linkage_specification"):
        if key in value and value[key] is not None:
            links = value[key]
            if not isinstance(links, (list, tuple)):
                raise ValueError("Explicit linkages require a PS/PO array; do not substitute full PS.")
            checked_length = local_length if type(local_length) is int else len(links) + 1
            require_linkage_specification(checked_length, links, allow_mixed=False)
            if label in {"steric-pmo", "unspecified"} or value.get("sugar") == "MORPHOLINO":
                raise ValueError("PS/PO linkages are unsupported for PMO/unspecified chemistry.")
    for key in ("recommendation", "architecture", "chemistry", "constraints", "design_programs", "design_jobs", "programs"):
        if key in value:
            require_native_linkage_identity(value[key], local_length,
                                           _chemical_object=key in {"recommendation", "architecture", "chemistry"})


def canonical_chemistry_configuration(value, length=None):
    """Preserve every validated field; aliases do not erase chemical choices."""
    from copy import deepcopy
    require_native_linkage_identity(value, length, _chemical_object=isinstance(value, dict))
    raw = value.get("recommendation", value) if isinstance(value, dict) else {"CHEMISTRY": value}
    if not isinstance(raw, dict):
        return {}
    label = next((chemistry_label(raw[k]) for k in ("CHEMISTRY", "chemistry", "profile", "name")
                  if isinstance(raw.get(k), str) and chemistry_label(raw[k])), None)
    if not label:
        return {}
    _validate_chemical_object(raw, length)
    result = deepcopy(raw)
    for alias in ("profile", "name", "chemistry"):
        result.pop(alias, None)
    result["CHEMISTRY"] = label
    if result.get("platform") == "steric blocker":
        result["platform"] = "steric_blocker"
    result.setdefault("platform", chemistry_components(label)["platform"])
    architecture = gapmer_architecture(label)
    result.setdefault("length", architecture.length if architecture else length or 20)
    if isinstance(value, dict) and value.get("chemical_requirements"):
        from aso_agents.chemical_requirements import require_native_requirement_capabilities, requirement_configuration_errors
        require_native_requirement_capabilities(value["chemical_requirements"])
        errors = requirement_configuration_errors(value["chemical_requirements"], result)
        if errors:
            raise ValueError("; ".join(errors))
    return result
