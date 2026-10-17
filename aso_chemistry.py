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


def architecture_label(architecture):
    """Compile typed chemical components; never parse a user's prose."""
    if not isinstance(architecture, dict):
        raise ValueError("Chemical architecture must be an object.")
    sugar, backbone = architecture.get("sugar"), architecture.get("backbone")
    if (sugar, backbone) == ("MORPHOLINO", "PHOSPHORODIAMIDATE"):
        counts = tuple(architecture.get(key) for key in ("wing_5prime", "dna_gap", "wing_3prime"))
        if architecture.get("kind") == "steric_blocker" and counts == (0, 0, 0):
            return "steric-pmo"
        raise ValueError("PMO is a fully morpholino steric blocker without a DNA gap.")
    if sugar not in {"2MOE", "2OME"} or backbone != "PS":
        raise ValueError("Unsupported chemical building blocks: " + str((sugar, backbone)))
    counts = tuple(architecture.get(key) for key in ("wing_5prime", "dna_gap", "wing_3prime"))
    if any(type(n) is not int for n in counts):
        raise ValueError("Chemical segment lengths must be integer nucleotide counts.")
    if architecture.get("kind") == "steric_blocker" and counts == (0, 0, 0):
        return next(label for label, value in STERIC_SUGARS.items() if value == sugar)
    if architecture.get("kind") == "gapmer" and all(n > 0 for n in counts):
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


def encode_idt_oligo(sequence, label):
    """Encode the physical antisense molecule, with all n-1 PS linkages."""
    sequence = str(sequence).upper()
    architecture = require_chemistry_length(label, len(sequence))
    if any(base not in "ACGT" for base in sequence):
        raise ValueError("Chemical encoding requires an unambiguous A/C/G/T sequence.")
    if chemistry_label(label) == "unspecified":
        # A physical base sequence is not an orderable chemical architecture.
        # Keep the sequence in its own output field without inventing chemistry.
        return ""
    if chemistry_label(label) == "steric-pmo":
        # PMOs have a phosphorodiamidate morpholino backbone. The IDT
        # MOE/OMe/PS notation is inapplicable; retain the physical base
        # sequence and typed chemistry in separate export fields.
        return ""
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
    return "*".join(units)
