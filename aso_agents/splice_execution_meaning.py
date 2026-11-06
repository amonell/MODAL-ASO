"""Expose the execution meaning of typed splice fields to semantic review.

This module never reads request wording to choose a biological direction.
The reviewer compares these descriptions with the original user evidence.
"""

SPLICE_EFFECT_SEMANTICS = (
    "desired_effect names the exon outcome the ASOs should promote. "
    "For an exon-level intervention, reducing skipping means promoting inclusion "
    "(desired_effect=inclusion, desired_delta=increase); increasing skipping means "
    "promoting skipping (desired_effect=skipping, desired_delta=decrease). "
    "The delta in these programs describes exon usage. A decrease attached to an "
    "outcome name in user prose must be interpreted semantically before selecting "
    "these fields. Compare the actual execution_meaning below with the original "
    "request, including its negation and direction. A program label or objective "
    "that repeats the request cannot override opposite executable fields. "
    "An identified isoform's abundance is a separate quantity: an exon-skipped "
    "isoform can be requested to increase or decrease. A default_exon_context "
    "objective expresses exon inclusion and follows the program effect pairing."
)


def splice_execution_meaning(goal):
    effects = {"skipping": "Promote exon skipping; decrease exon inclusion.",
               "inclusion": "Promote exon inclusion; decrease exon skipping."}
    rows = []
    for collection in ("design_programs", "splice_isoform_objectives"):
        for index, item in enumerate(goal.get(collection) or []):
            if not isinstance(item, dict):
                continue
            is_program = collection == "design_programs"
            if is_program and item.get("intent") != "splice_modulation":
                continue
            effect = item.get("desired_effect")
            exon_scope = is_program or item.get("objective_scope") == "default_exon_context"
            if exon_scope and effect in effects:
                meaning = effects[effect]
            elif not is_program and not exon_scope and item.get("desired_abundance") in {"increase", "decrease"}:
                meaning = item["desired_abundance"].capitalize() + " the abundance of the identified isoform."
            else:
                continue
            rows.append({"pointer": f"/{collection}/{index}", "execution_meaning": meaning,
                         **{key: item[key] for key in ("program_id", "isoform_label", "transcript_id", "exon_label",
                            "desired_effect", "desired_delta", "desired_abundance", "objective_scope") if key in item}})
    return rows
