"""Check a selected chemistry against the native molecule architecture."""
import re
from aso_chemistry import gapmer_architecture, chemistry_label, chemistry_components, architecture_label
from copy import deepcopy


class ChemistryResolutionError(ValueError):
    """An explicit capability or selection failure, never a default substitution."""


def chemistry_selection_schema(requested_length=None):
    architecture = {"kind": {"type": "string", "enum": ["gapmer", "steric_blocker"]},
                    "sugar": {"type": "string", "enum": ["2MOE", "2OME", "MORPHOLINO"]},
                    "backbone": {"type": "string", "enum": ["PS", "PHOSPHORODIAMIDATE"]},
                    **{key: {"type": "integer", "minimum": 0} for key in
                       ("wing_5prime", "dna_gap", "wing_3prime")}}
    length = {"type": "integer", "minimum": 1}
    if requested_length is not None:
        length["const"] = requested_length
    recommendation = {"architecture": {"type": "object", "properties": architecture,
                        "required": list(architecture), "additionalProperties": False},
                      "length": length, "rationale": {"type": "string"}}
    return {"type": "object", "properties": {
        "recommendation": {"anyOf": [{"type": "null"}, {"type": "object", "properties": recommendation,
                            "required": list(recommendation), "additionalProperties": False}]},
        "unsupported_reason": {"type": ["string", "null"]}},
        "required": ["recommendation", "unsupported_reason"], "additionalProperties": False}


def compile_chemistry_proposal(payload):
    """Compile typed components to the canonical tiler representation."""
    if not isinstance(payload, dict):
        return payload
    if payload.get("recommendation") is None and payload.get("unsupported_reason"):
        raise ChemistryResolutionError("unsupported_chemistry: " + str(payload["unsupported_reason"]))
    recommendation = payload.get("recommendation")
    if not isinstance(recommendation, dict) or "architecture" not in recommendation:
        return payload  # Backward-compatible canonical proposals stay validated.
    label = architecture_label(recommendation["architecture"])
    result = deepcopy(payload)
    result["recommendation"] = {"CHEMISTRY": label, **chemistry_components(label),
                                "length": recommendation.get("length"),
                                "rationale": recommendation.get("rationale")}
    return result


PROFILE_LENGTHS = {"5-10-5": 20, "3-10-3": 16}
PROFILE_PLATFORMS = {**{key: "gapmer" for key in PROFILE_LENGTHS},
                     "steric-hindrance": "steric_blocker", "steric-2ome-ps": "steric_blocker",
                     "steric-pmo": "steric_blocker"}


def chemistry_contract_errors(payload, requested_length=None):
    recommendation = payload.get("recommendation") if isinstance(payload, dict) else None
    if not isinstance(recommendation, dict):
        return ["Return a recommendation object."]
    label = recommendation.get("CHEMISTRY")
    architecture = gapmer_architecture(label)
    if not chemistry_label(label):
        return ["CHEMISTRY must identify a supported MOE/OMe full-PS or morpholino/phosphorodiamidate architecture. Descriptive requests require semantic resolution, not a default substitution."]
    errors = []
    length = recommendation.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        errors.append("length must be a positive integer nucleotide count.")
    if architecture and length != architecture.length:
        errors.append(f"The native {label} architecture requires exactly {architecture.length} nt.")
    if requested_length is not None and length != requested_length:
        errors.append(f"The requested exact oligo length is {requested_length} nt, not a minimum or preference.")
    components = chemistry_components(label)
    platform = components["platform"]
    if recommendation.get("platform") != platform:
        errors.append(f"The selected profile has platform {platform}.")
    for field in ("wing_chemistry", "gap_chemistry", "backbone"):
        if label == "unspecified" and field in recommendation:
            errors.append(f"Unspecified chemistry cannot select {field}.")
        supported = components.get(field)
        if field in recommendation and recommendation[field] != supported:
            errors.append(f"Unsupported {field}: the physical encoder supports {supported}; do not substitute it for another requested chemistry.")
    rationale = str(recommendation.get("rationale") or "").strip()
    if not rationale:
        errors.append("Explain the selected chemistry's mechanism and constraint compatibility.")
    # A branch rationale describes this configuration. Other architectures need
    # their own independently checked configuration before being recommended.
    mentioned = set(re.findall(r"(?<!\d)\d+\s*[-–]\s*\d+\s*[-–]\s*\d+(?!\d)", rationale))
    for profile in mentioned:
        normalized = re.sub(r"\s+", "", profile).replace("–", "-")
        if normalized != (label.split(":", 1)[-1] if architecture else label):
            errors.append(f"rationale discusses unselected architecture {profile}; scope it to this selected profile only.")
    if requested_length is not None and re.search(
        rf"(?:\b(?:at least|minimum(?:\s+of)?|floor(?:\s+of)?)\s+{requested_length}\s*(?:nt|bp|nucleotides?)\b|"
        rf"\b{requested_length}\s*[- ]?(?:nt|bp|nucleotides?)\s+(?:floor|minimum)\b)",
        rationale, re.I,
    ):
        errors.append("rationale changes the exact requested length into a lower bound.")
    return errors
