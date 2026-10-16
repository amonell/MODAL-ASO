"""Canonical model-facing contract for executable ASO design goals."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .coordinate_contract import COORDINATE_ORIGIN_INSTRUCTIONS


CANONICAL_TARGET_TYPES = frozenset({
    "gene", "transcript", "transcript_cohort", "isoform", "isoform_cohort",
    "exon", "splice_junction", "splice_acceptor", "splice_donor", "splice_site",
    "splice_site_window", "binding_site", "region", "regulatory_region", "sequence",
    "mutant_variant", "wildtype_variant", "protein", "other",
})


CANONICAL_TASK_TYPES = frozenset(
    {
        "analyze_artifact",
        "answer_question",
        "build_composite_report",
        "compare_aso_sets",
        "design_new_asos",
        "evaluate_existing_aso_set",
        "sandbox_analysis",
        "trace_filter_decisions",
    }
)


CANONICAL_TASK_CONTEXT_FIELDS = frozenset(
    {
        "num_mismatches",
        "artifact_refs",
        "chemistry",
        "chemistry_policy",
        "oligoai_evaluation",
        "comparison_contract",
        "evaluation_targets",
        "gene_id",
        "gene_symbol",
        "platform",
        "provided_aso_files",
        "provided_asos",
        "provided_sequence_role",
        "species",
        "species_targets",
        "structured_goal",
        "target_context",
        "transcript_id",
        "web_retrieval",
    }
)

CANONICAL_TASK_CONTEXT_FIELD_SEMANTICS = {
    "num_mismatches": "Explicit requested maximum substitution count for off-target sequence searching in this task, including zero. This is a search budget, not a maximum off-target hit count or permission to mismatch the intended target.",
    "oligoai_evaluation": "Explicit OligoAI prediction conditions: enabled, dosage_nm and transfection_method; omit unstated values so runtime assumptions remain labelled.",
    "artifact_refs": "User-supplied run, file, or artifact references consumed by this task.",
    "chemistry": "User-supplied ASO chemistry or mechanism context for this task.",
    "chemistry_policy": "Explicit chemical-architecture selection policy: select or leave_unspecified, independent of biological mechanism.",
    "comparison_contract": "User-specified physical eligibility and deterministic pooled sequence-metric ranking for a comparison task.",
    "evaluation_targets": "Structured biological targets evaluated by this task.",
    "gene_id": "Explicitly supplied stable gene identifier for this task.",
    "gene_symbol": "Explicitly supplied gene symbol for this task.",
    "platform": "User-supplied experimental or computational platform context.",
    "provided_aso_files": "User-supplied files containing existing ASOs.",
    "provided_asos": "Existing ASO values supplied by the user for evaluation or comparison.",
    "provided_sequence_role": "Explicit user-declared role of the supplied 5-prime to 3-prime sequence strings in this task: antisense molecules, target sites, or unknown. Per-row declarations take precedence; binding outcomes do not change this input fact.",
    "species": "Requested organism for this task.",
    "species_targets": "Structured organisms included in this task.",
    "structured_goal": "Authoritative executable goal for a design task.",
    "target_context": "Tissue, cell type, disease model, or other biological context for this task.",
    "transcript_id": "Explicitly supplied transcript identifier for this task.",
    "web_retrieval": "Source-retrieval requirements for externally resolvable task facts.",
}

CANONICAL_STRUCTURED_GOAL_FIELDS = frozenset(
    {
        "num_mismatches",
        "artifact_refs",
        "aso_overlap_requirement",
        "candidate_limit",
        "candidate_limit_per_site",
        "candidate_minimum",
        "candidate_selection_policy",
        "conditional_exon_skip_events",
        "target_accessibility",
        "target_accessibility_request_quote",
        "oligoai_evaluation",
        "chemistry",
        "chemistry_policy",
        "constraints",
        "cross_species_requirement",
        "design_programs",
        "evaluation_targets",
        "explicit_splice_targets",
        "flanks",
        "focus_exons_only",
        "gene_id",
        "gene_symbol",
        "intent",
        "isoform_scope",
        "isoform_scope_source",
        "junction_gapmers",
        "min_asos_per_site",
        "minimum_overlap_bp",
        "mirna_binding",
        "motif_tiling",
        "off_target_expression",
        "off_target_species",
        "patient_variant",
        "preservation_targets",
        "rbp_binding",
        "region_constraints",
        "run_splice_eval",
        "run_splice_router",
        "site_direction_hints",
        "source_attribution",
        "species",
        "species_targets",
        "splice_isoform_objectives",
        "target_context",
        "target_region",
        "target_scope",
        "transcript_lock",
        "transcript_policy",
    }
)

CANONICAL_GOAL_COLLECTION_FIELDS = {
    "design_programs": frozenset(
        {
            "coordinate_space",
            "coordinate_convention",
            "transcript_ids",
            "chemistry",
            "chemistry_policy",
            "desired_delta",
            "desired_effect",
            "end",
            "execution_policy",
            "exon_label",
            "exon_labels",
            "flanks",
            "focus_exons_only",
            "intent",
            "isoform_label",
            "label",
            "length",
            "modality",
            "notes",
            "oligoai_evaluation",
            "objective",
            "position",
            "program_id",
            "region_constraints",
            "run_splice_eval",
            "run_splice_router",
            "site_context",
            "site_kind",
            "start",
            "target_context",
            "target_region",
            "target_type",
            "targeting_policy",
            "transcript_id",
        }
    ),
    "evaluation_targets": frozenset(
        {
            "chrom",
            "desired_delta",
            "desired_effect",
            "end",
            "exon_label",
            "exon_index",
            "gene_symbol",
            "isoform_label",
            "label",
            "objective_type",
            "position",
            "reference_transcript_id",
            "site_kind",
            "splice_relative_window",
            "start",
            "strand",
            "target_type",
            "transcript_id",
        }
    ),
    "preservation_targets": frozenset(
        {
            "action",
            "gene_symbol",
            "isoform_label",
            "label",
            "notes",
            "preservation_target_type",
            "rationale",
            "transcript_id",
        }
    ),
    "splice_isoform_objectives": frozenset(
        {
            "aggregate_class",
            "desired_abundance",
            "desired_effect",
            "exon_label",
            "exon_labels",
            "isoform_label",
            "notes",
            "objective_scope",
            "rationale",
            "reference_accession",
            "role",
            "transcript_id",
        }
    ),
}

CANONICAL_GOAL_COLLECTION_REQUIRED_FIELDS = {
    "design_programs": frozenset(
        {
            "program_id",
            "label",
            "modality",
            "intent",
            "objective",
        }
    ),
    # Every evaluation target declares its semantic role. Direction is then
    # conditionally required by the semantic validator for outcome/route
    # targets and forbidden for identity anchors and monitoring safeguards.
    "evaluation_targets": frozenset({"objective_type"}),
    "preservation_targets": frozenset({"action", "preservation_target_type"}),
    "splice_isoform_objectives": frozenset({"desired_abundance"}),
}

_REGION_INTERVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "chrom": {"type": "string", "description": "Coordinate reference: chromosome for coordinate_space=genomic; exact transcript identifier for coordinate_space=cdna. The field name does not force genomic coordinates."},
        "genome_build": {"type": "string"},
        "end": {"type": "integer", "minimum": 0},
        "label": {"type": "string"},
        "score": {"type": "number"},
        "start": {"type": "integer", "minimum": 0},
        "strand": {"type": "string", "enum": ["+", "-", "."]},
    },
    "required": ["chrom", "start", "end"],
    "additionalProperties": False,
}

_SPLICE_RELATIVE_WINDOW_SCHEMA = {
    "type": "object",
    "properties": {
        "anchor_site": {"type": "string", "enum": ["acceptor", "donor"]},
        "exon_label": {"type": ["string", "integer"]},
        "feature_name": {"type": "string"},
        "reference_transcript_id": {"type": "string"},
        "relative_end_nt": {"type": "integer"},
        "relative_start_nt": {"type": "integer"},
    },
    "additionalProperties": False,
}

_EVALUATION_SPLICE_RELATIVE_WINDOW_SCHEMA = deepcopy(_SPLICE_RELATIVE_WINDOW_SCHEMA)
_EVALUATION_SPLICE_RELATIVE_WINDOW_SCHEMA["properties"]["exon_index"] = {
    "type": "integer", "minimum": 1,
    "description": "Explicit one-based transcript annotation exon ordinal, distinct from the biological exon_label.",
}
_EVALUATION_SPLICE_RELATIVE_WINDOW_SCHEMA["description"] = (
    "Evaluation feature interval relative to a transcript exon splice boundary. "
    "Identify the anchor with reference_transcript_id and exon_index or exon_label. "
    "Signed endpoints are inclusive and follow transcript orientation on either strand; "
    "offset zero is the donor's last exonic nucleotide or the acceptor's first exonic nucleotide. "
    "These offsets are not genomic start/end coordinates and impose no ASO design filter."
)


def _closed_object(properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


_OLIGOAI_CONDITIONS_SCHEMA = _closed_object({
    "enabled": {"type": "boolean"},
    "dosage_nm": {"type": "number", "exclusiveMinimum": 0},
    "transfection_method": {"type": "string", "enum": ["Electroporation", "Gymnosis", "Other", "Lipofection"]},
})


_STRING_OR_INTEGER_SCHEMA = {
    "anyOf": [{"type": "string"}, {"type": "integer"}],
}

_JSON_SCALAR_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
    ],
}

_ARTIFACT_REF_SCHEMA = _closed_object(
    {
        "path": {"type": "string"},
        "artifact_type": {"type": "string"},
        "label": {"type": "string"},
    }
)

_PROVIDED_ASO_FILE_SCHEMA = _closed_object(
    {
        "path": {"type": "string"},
        "file_format": {"type": "string"},
        "label": {"type": "string"},
    }
)

_PROVIDED_ASO_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        _closed_object(
            {
                "name": {"type": "string"},
                "sequence": {"type": "string"},
                "sequence_role": {"type": "string", "enum": ["antisense_oligo_sequence", "target_site_sequence", "unknown"]},
                "chemistry": {"type": "string"},
                "platform": {"type": "string"},
                "notes": {"type": "string"},
            }
        ),
    ]
}

_FLANKS_SCHEMA = _closed_object(
    {
        "up": {"type": "integer", "minimum": 0},
        "down": {"type": "integer", "minimum": 0},
    }
)

_CROSS_SPECIES_REQUIREMENT_SCHEMA = _closed_object(
    {
        "required": {"type": "boolean"},
        "species": {"type": "array", "items": {"type": "string"}},
        "match_type": {"type": "string"},
        "mode": {"type": "string"},
        "return_nonconserved_candidates": {"type": "boolean"},
    }
)

_RBP_BINDING_SCHEMA = _closed_object(
    {
        "rbp": {"type": "string"},
        "assay": {"type": "string"},
        "biosample_hint": {"type": "string"},
        "max_peaks": {"type": "integer", "minimum": 1},
        "aso_overlap_requirement": {
            "type": "string",
            "enum": ["any_overlap", "full_containment", "contains_feature"],
        },
    }
)

_MIRNA_BINDING_SCHEMA = _closed_object(
    {
        "mirnas": {"type": "array", "items": {"type": "string"}},
        "source": {"type": "string"},
        "target_level": {"type": "string"},
        "operator": {"type": "string", "enum": ["union", "intersect"]},
        "max_sites": {"type": "integer", "minimum": 1},
        "aso_overlap_requirement": {
            "type": "string",
            "enum": ["any_overlap", "full_containment", "contains_feature"],
        },
    }
)

_MOTIF_TILING_SCHEMA = _closed_object(
    {
        "motifs": {"type": "array", "items": {"type": "string"}},
        "target_level": {
            "type": "string",
            "enum": ["pre_mrna", "mature_mrna"],
        },
        "operator": {"type": "string", "enum": ["union", "intersect"]},
        "max_hits": {"type": "integer", "minimum": 1},
        "strand_mode": {"type": "string", "enum": ["transcribed", "both"]},
        "aso_overlap_requirement": {
            "type": "string",
            "enum": ["any_overlap", "full_containment", "contains_feature"],
        },
    }
)

_OFF_TARGET_EXPRESSION_SCHEMA = _closed_object(
    {
        "target_context": {"type": "string"},
        "max_cells": {"type": "integer", "minimum": 1},
        "sample_seed": {"type": "integer", "minimum": 0},
        "max_h5ad_bytes": {"type": "integer", "minimum": 1},
        "max_h5ad_gb": {"type": "number", "exclusiveMinimum": 0},
        "use_census": {"type": "boolean"},
        "allow_h5ad_fallback": {"type": "boolean"},
    }
)

_SOURCE_ATTRIBUTION_SCHEMA = _closed_object(
    {
        "required": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "string"}},
    }
)

_TRANSCRIPT_LOCK_SCHEMA = _closed_object(
    {
        "kind": {
            "type": "string",
            "enum": ["explicit_id", "reference_accession", "verbal_rule"],
        },
        "value": {"type": "string"},
        "version": {"type": "integer", "minimum": 1},
        "note": {"type": "string"},
        "source": {"type": "string"},
    }
)

_JUNCTION_GAPMERS_SCHEMA = _closed_object(
    {
        "enabled": {"type": "boolean"},
        "scope": {
            "type": "string",
            "enum": ["focused_exons", "splice_plan", "all_exons"],
        },
        "span_min_each_side": {"type": "integer", "minimum": 0},
    }
)

_WEB_RETRIEVAL_SCHEMA = _closed_object(
    {
        "needed": {"type": "boolean"},
        "required": {"type": "boolean"},
        "reason": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}},
        "source_policy": {
            "type": "string",
            "enum": ["official_first", "primary_sources_only", "general_web"],
        },
        "max_searches": {"type": "integer", "minimum": 1, "maximum": 3},
    }
)

_PROGRAM_EXECUTION_POLICY_SCHEMA = _closed_object(
    {
        "role": {"type": "string", "enum": ["primary", "fallback"]},
        "activation_condition": {
            "type": "string",
            "enum": ["always", "no_viable_primary_routes"],
        },
    }
)

_AGGREGATE_CLASS_SCHEMA = _closed_object(
    {
        "basis": {
            "type": "string",
            "enum": ["exon_presence", "role", "residual_other_isoforms"],
        },
        "exon_label": deepcopy(_STRING_OR_INTEGER_SCHEMA),
        "presence": {"type": "string", "enum": ["included", "excluded"]},
        "role": {
            "type": "string",
            "enum": ["beneficial", "pathogenic", "context"],
        },
    }
)

_EXPLICIT_SPLICE_TARGET_SCHEMA = _closed_object(
    {
        "chrom": {"type": "string"},
        "start": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "One-based inclusive genomic start for the biological splice target."
            ),
        },
        "end": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "One-based inclusive genomic end for the biological splice target."
            ),
        },
        "strand": {"type": "string", "enum": ["+", "-", "."]},
        "label": {"type": "string"},
        "isoform_label": {"type": "string"},
        "exon_label": deepcopy(_STRING_OR_INTEGER_SCHEMA),
        "requested_exon_label": deepcopy(_STRING_OR_INTEGER_SCHEMA),
        "desired_delta": {"type": "string", "enum": ["increase", "decrease"]},
        "notes": {"type": "string"},
    }
)
_EXPLICIT_SPLICE_TARGET_SCHEMA["required"] = ["chrom", "start", "end"]

_SITE_DIRECTION_HINT_SCHEMA = _closed_object(
    {
        "isoform_label": {"type": "string"},
        "site_hint": {"type": "string"},
        "desired_delta": {"type": "string", "enum": ["increase", "decrease"]},
        "exon_label": deepcopy(_STRING_OR_INTEGER_SCHEMA),
        "notes": {"type": "string"},
    }
)

_SPECIES_TARGET_SCHEMA = _closed_object(
    {
        "species": {"type": "string"},
        "label": {"type": "string"},
        "gene_symbol": {"type": "string"},
        "gene_id": {"type": "string"},
        "transcript_id": {"type": "string"},
    }
)

CANONICAL_REGION_CONSTRAINT_FIELDS = frozenset(
    {
        "allow_coordinate_conversion",
        "aso_overlap_requirement",
        "constraint_id",
        "coordinate_convention",
        "coordinate_space",
        "exon_label",
        "exon_numbering_transcript_id",
        "feature_name",
        "gene_symbol",
        "genome_build",
        "intervals_bed0_half_open",
        "isoform_label",
        "label",
        "minimum_overlap_bp",
        "mirna_binding",
        "motif_tiling",
        "notes",
        "operator",
        "rbp_binding",
        "reference_transcript_id",
        "rows",
        "score",
        "source",
        "source_type",
        "splice_relative_window",
        "strand",
        "transcript_id",
    }
)

_REGION_CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "allow_coordinate_conversion": {"type": "boolean"},
        "aso_overlap_requirement": {
            "type": "string",
            "enum": ["any_overlap", "full_containment", "contains_feature"],
        },
        "constraint_id": {"type": "string"},
        "coordinate_convention": {"type": "string"},
        "coordinate_space": {"type": "string", "enum": ["genomic", "cdna"], "description": "Origin of the supplied interval numbers. Mature-RNA/transcript positions are cdna even when a genome assembly is named. Preserve this origin independently of zero/one-based indexing."},
        "exon_label": {"type": ["string", "integer"]},
        "exon_numbering_transcript_id": {"type": "string"},
        "feature_name": {"type": "string", "enum": ["start_codon", "cds"]},
        "gene_symbol": {"type": "string"},
        "genome_build": {"type": "string"},
        "intervals_bed0_half_open": {
            "type": "array",
            "items": _REGION_INTERVAL_SCHEMA,
        },
        "isoform_label": {"type": "string"},
        "label": {"type": "string"},
        "minimum_overlap_bp": {"type": "integer", "minimum": 1},
        "mirna_binding": _MIRNA_BINDING_SCHEMA,
        "motif_tiling": _MOTIF_TILING_SCHEMA,
        "notes": {"type": "string"},
        "operator": {
            "type": "string",
            "enum": ["union", "intersect", "include", "exclude"],
            "description": "Ordered Boolean combination of candidate-match results, not geometric merging of feature intervals. union/include accepts either the preceding result or this constraint; intersect requires both; exclude requires the preceding result and failure of this constraint. The first constraint seeds the result (negated for exclude). Rows within one constraint are alternative locations for that constraint.",
        },
        "rbp_binding": _RBP_BINDING_SCHEMA,
        "reference_transcript_id": {"type": "string"},
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "items": deepcopy(_JSON_SCALAR_SCHEMA),
            },
        },
        "score": {"type": "number"},
        "source": {"type": "string"},
        "source_type": {
            "type": "string",
            "enum": [
                "rbp_binding",
                "mirna_binding",
                "motif_tiling",
                "exon",
                "explicit_rows",
                "splice_relative_window",
                "transcript_feature",
            ],
        },
        "splice_relative_window": _SPLICE_RELATIVE_WINDOW_SCHEMA,
        "strand": {"type": "string", "enum": ["+", "-", "."]},
        "transcript_id": {"type": "string"},
    },
    "required": ["aso_overlap_requirement"],
    "allOf": [{
        "if": {"properties": {"source_type": {"const": "explicit_rows"}}, "required": ["source_type"]},
        "then": {"anyOf": [
            {"properties": {"intervals_bed0_half_open": {"minItems": 1}}, "required": ["intervals_bed0_half_open"]},
            {"properties": {"rows": {"minItems": 1}}, "required": ["rows"]},
        ]},
    }, {
        "if": {"properties": {"source_type": {"const": "transcript_feature"}}, "required": ["source_type"]},
        "then": {
            "required": ["feature_name"],
            "not": {"anyOf": [{"required": ["rows"]}, {"required": ["intervals_bed0_half_open"]}]},
        },
    }],
    "additionalProperties": False,
}

CANONICAL_SEQUENCE_CONSTRAINT_FIELDS = frozenset(
    {"allele_specific", "gc_content", "homopolymers", "max_binding_sites"}
)

_ALLELE_SPECIFIC_SCHEMA = {
    "type": "object",
    "properties": {
        "non_target_allele": {"type": "string"},
        "require_variant_overlap": {"type": "boolean"},
        "rsid": {"type": "string"},
        "target_allele": {"type": "string"},
    },
    "required": ["target_allele", "non_target_allele", "require_variant_overlap"],
    "additionalProperties": False,
}

_PATIENT_VARIANT_SCHEMA = {
    "type": "object",
    "properties": {
        "alt": {"type": "string"},
        "chr": {"type": "string"},
        "gene_symbol": {"type": "string"},
        "genome_build": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": ["genomic", "cdna", "protein", "hgvs", "unknown"],
        },
        "non_target_allele": {"type": "string"},
        "notation": {"type": "string"},
        "pos": {"type": "integer", "minimum": 1},
        "ref": {"type": "string"},
        "reference_accession": {"type": "string"},
        "rsid": {"type": "string"},
        "target_allele": {"type": "string"},
        "transcript_id": {"type": "string"},
    },
    "required": ["notation", "kind"],
    "additionalProperties": False,
}

_GC_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "max": {"type": "number", "minimum": 0, "maximum": 1},
        "max_inclusive": {"type": "boolean"},
        "max_percent": {"type": "number", "minimum": 0, "maximum": 100},
        "min": {"type": "number", "minimum": 0, "maximum": 1},
        "min_inclusive": {"type": "boolean"},
        "min_percent": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "additionalProperties": False,
}

_HOMOPOLYMER_SCHEMA = {
    "type": "object",
    "properties": {
        "avoid_sequences": {"type": "array", "items": {"type": "string"}},
        "max_run": {"type": "integer", "minimum": 0},
        "per_base_max": {
            "type": "object",
            "properties": {
                base: {"type": "integer", "minimum": 0} for base in ("A", "C", "G", "T")
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_SEQUENCE_CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "allele_specific": _ALLELE_SPECIFIC_SCHEMA,
        "gc_content": _GC_CONTENT_SCHEMA,
        "homopolymers": _HOMOPOLYMER_SCHEMA,
        "max_binding_sites": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

CANONICAL_GOAL_FIELD_SCHEMAS = {
    "num_mismatches": {"type": "integer", "minimum": 0, "description": "Explicit off-target substitution search budget. Preserve requested values, including zero; runtime supports 0 through 3 and must report larger requests as unsupported rather than silently reduce them."},
    "transcript_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True,
                       "description": "Required transcript cohort for this program. For a shared-sequence gapmer panel every selected target must match every member. Independent transcript panels use separate programs; alternative isoform directions use splice_isoform_objectives."},
    "coordinate_space": {"type": "string", "enum": ["genomic", "cdna"]},
    "coordinate_convention": {"type": "string", "enum": ["bed0_half_open", "one_based_inclusive"]},
    "action": {"type": "string", "enum": ["preserve"]},
    "aggregate_class": _AGGREGATE_CLASS_SCHEMA,
    "artifact_refs": {"type": "array", "items": _ARTIFACT_REF_SCHEMA},
    "aso_overlap_requirement": {
        "type": "string",
        "enum": ["any_overlap", "full_containment", "contains_feature"],
    },
    "candidate_limit": {"type": ["integer", "null"], "minimum": 1},
    "candidate_limit_per_site": {"type": "integer", "minimum": 1},
    "candidate_minimum": {"type": "integer", "minimum": 0},
    "candidate_selection_policy": {"type": "string", "enum": ["ranked", "diverse_target_sites", "nonoverlapping_target_sites"]},
    "conditional_exon_skip_events": {"type": "array", "items": {"type": "object", "properties": {
        "exon_label": {"type": "integer", "minimum": 1},
        "transcript_id": {"type": ["string", "null"]},
        "request_quote": {"type": "string"}}, "required": ["exon_label", "request_quote"]}},
    "target_accessibility": {"type": "boolean"},
    "target_accessibility_request_quote": {"type": "string", "minLength": 1},
    "oligoai_evaluation": deepcopy(_OLIGOAI_CONDITIONS_SCHEMA),
    "chemistry": {"type": "string"},
    "chemistry_policy": {"type": "string", "enum": ["select", "leave_unspecified"],
                         "description": "Permission to select chemistry: select allows delegated/default platform choice; leave_unspecified prohibits chemical specification by explicit user instruction. Missing chemistry alone permits selection."},
    "chrom": {"type": "string"},
    "constraints": _SEQUENCE_CONSTRAINT_SCHEMA,
    "cross_species_requirement": _CROSS_SPECIES_REQUIREMENT_SCHEMA,
    "end": {"type": "integer", "minimum": 0},
    "evaluation_targets": {"type": "array", "items": {}},
    "reference_transcript_id": {"type": "string"},
    "splice_relative_window": _EVALUATION_SPLICE_RELATIVE_WINDOW_SCHEMA,
    "execution_policy": _PROGRAM_EXECUTION_POLICY_SCHEMA,
    "explicit_splice_targets": {
        "type": "array",
        "items": _EXPLICIT_SPLICE_TARGET_SCHEMA,
    },
    "exon_label": deepcopy(_STRING_OR_INTEGER_SCHEMA),
    "exon_index": {
        "type": "integer", "minimum": 1,
        "description": "Explicit one-based transcript annotation exon ordinal in 5-prime to 3-prime order. Preserve a separately supplied biological name in exon_label.",
    },
    "exon_labels": {
        "type": "array",
        "items": deepcopy(_STRING_OR_INTEGER_SCHEMA),
    },
    "flanks": _FLANKS_SCHEMA,
    "focus_exons_only": {"type": "boolean"},
    "gene_id": {"type": "string"},
    "gene_symbol": {"type": "string"},
    "desired_delta": {"type": "string", "enum": ["increase", "decrease"]},
    "desired_abundance": {
        "type": "string",
        "enum": ["increase", "decrease"],
    },
    "desired_effect": {"type": "string", "enum": ["inclusion", "skipping"]},
    "modality": {"type": "string", "enum": ["steric", "gapmer"]},
    "intent": {
        "type": "string",
        "enum": ["knockdown", "splice_modulation", "binding"],
    },
    "isoform_label": {"type": "string"},
    "isoform_scope": {
        "anyOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
        ]
    },
    "isoform_scope_source": {
        "type": "string",
        "enum": [
            "user_specified",
            "resolver_inferred",
            "default_exon_splice_context",
        ],
    },
    "junction_gapmers": _JUNCTION_GAPMERS_SCHEMA,
    "label": {"type": "string"},
    "length": {"type": "integer", "minimum": 1},
    "minimum_overlap_bp": {"type": "integer", "minimum": 1},
    "min_asos_per_site": {"type": "integer", "minimum": 1},
    "mirna_binding": _MIRNA_BINDING_SCHEMA,
    "motif_tiling": _MOTIF_TILING_SCHEMA,
    "notes": {"type": "string"},
    "objective": {"type": "string"},
    "objective_scope": {
        "type": "string",
        "enum": [
            "specific_transcript",
            "specific_isoform",
            "aggregate_class",
            "default_exon_context",
        ],
    },
    "objective_type": {
        "type": "string",
        "enum": [
            "identity_anchor",
            "primary_outcome",
            "route_target",
            "monitoring_safeguard",
        ],
    },
    "off_target_expression": _OFF_TARGET_EXPRESSION_SCHEMA,
    "off_target_species": {"type": "array", "items": {"type": "string"}},
    "patient_variant": _PATIENT_VARIANT_SCHEMA,
    "position": {"type": "integer", "minimum": 0},
    "preservation_target_type": {
        "type": "string",
        "enum": ["gene", "isoform", "transcript"],
    },
    "preservation_targets": {"type": "array", "items": {}},
    "program_id": {"type": "string"},
    "rbp_binding": _RBP_BINDING_SCHEMA,
    "rationale": {"type": "string"},
    "reference_accession": {"type": "string"},
    "region_constraints": {
        "type": "array",
        "items": _REGION_CONSTRAINT_SCHEMA,
    },
    "role": {
        "type": "string",
        "enum": ["beneficial", "pathogenic", "context"],
    },
    "run_splice_eval": {"type": "boolean"},
    "run_splice_router": {"type": "boolean"},
    "site_context": {
        "type": "string",
        "enum": ["site_window", "upstream_intron", "downstream_intron", "exon_body"],
    },
    "site_direction_hints": {
        "type": "array",
        "items": _SITE_DIRECTION_HINT_SCHEMA,
    },
    "source_attribution": _SOURCE_ATTRIBUTION_SCHEMA,
    "species": {"type": "string"},
    "species_targets": {"type": "array", "items": _SPECIES_TARGET_SCHEMA},
    "splice_isoform_objectives": {"type": "array", "items": {}},
    "start": {"type": "integer", "minimum": 0},
    "strand": {"type": "string", "enum": ["+", "-", "."]},
    "target_context": {"type": "string"},
    "target_region": {"type": "string"},
    "target_scope": {"type": "string"},
    "target_type": {"type": "string", "enum": sorted(CANONICAL_TARGET_TYPES)},
    "targeting_policy": {
        "type": "string",
        "enum": ["explicit_user_region", "default_exploratory"],
    },
    "transcript_id": {"type": "string"},
    "transcript_lock": _TRANSCRIPT_LOCK_SCHEMA,
    "transcript_policy": {
        "type": "string",
        "enum": ["user_locked", "agent_choice"],
    },
    "site_kind": {"type": "string", "enum": ["acceptor", "donor", "both"]},
}


CANONICAL_TASK_CONTEXT_FIELD_SCHEMAS = {
    "num_mismatches": {"type": "integer", "minimum": 0, "description": "Requested off-target substitution search budget for this task, independent of hit-count filters."},
    "oligoai_evaluation": deepcopy(_OLIGOAI_CONDITIONS_SCHEMA),
    "artifact_refs": {"type": "array", "items": _ARTIFACT_REF_SCHEMA},
    "chemistry": {"type": "string"},
    "chemistry_policy": {"type": "string", "enum": ["select", "leave_unspecified"],
                         "description": "Permission to select chemistry: select allows delegated/default platform choice; leave_unspecified prohibits chemical specification by explicit user instruction. Missing chemistry alone permits selection."},
    "comparison_contract": {
        "type": "object",
        "required": ["transcript_id", "exon_ordinal", "binding_requirement", "ranking_metric", "tie_breaker",
                     "length_nt", "max_homopolymer_nt", "gc_min", "gc_max", "gc_target"],
        "properties": {
            "transcript_id": {"type": "string"},
            "exon_ordinal": {"type": "integer"},
            "binding_requirement": {"type": "string", "enum": ["exact_antisense_in_exon"]},
            "ranking_metric": {"type": "string", "enum": ["absolute_gc_distance"]},
            "ranking_order": {"type": "string", "enum": ["ascending"],
                              "description": "Absolute-distance ranking minimizes distance; ascending is the implemented default."},
            "observation_policy": {"type": "string", "enum": ["preserve_all_origins"],
                                   "description": "Retain every submitted/generated observation, including repeated sequences across origins; this is the implemented default."},
            "tie_breaker": {"type": "string", "enum": ["ascending_observation_id"]},
            "length_nt": {"type": "integer"},
            "max_homopolymer_nt": {"type": "integer"},
            "gc_min": {"type": "number"},
            "gc_max": {"type": "number"},
            "gc_target": {"type": "number"},
        },
        "additionalProperties": False,
    },
    "evaluation_targets": {"type": "array", "items": {}},
    "gene_id": {"type": "string"},
    "gene_symbol": {"type": "string"},
    "platform": {"type": "string"},
    "provided_aso_files": {
        "type": "array",
        "items": _PROVIDED_ASO_FILE_SCHEMA,
    },
    "provided_asos": {"type": "array", "items": _PROVIDED_ASO_SCHEMA},
    "provided_sequence_role": {"type": "string", "enum": ["antisense_oligo_sequence", "target_site_sequence", "unknown"]},
    "species": {"type": "string"},
    "species_targets": {"type": "array", "items": _SPECIES_TARGET_SCHEMA},
    "target_context": {"type": "string"},
    "transcript_id": {"type": "string"},
    "web_retrieval": _WEB_RETRIEVAL_SCHEMA,
}


# These descriptions define meaning, not spellings that should be accepted as
# aliases.  They are supplied to the semantic checker so it can move a
# supported fact to an existing canonical destination without teaching the
# runtime every malformed key a model might invent.
CANONICAL_GOAL_FIELD_SEMANTICS = {
    "num_mismatches": "Explicit off-target substitution search budget; separate from maximum allowed binding-site count, and from mismatches permitted at an intended target.",
    "action": "Canonical preservation action requested for a protected biological entity.",
    "artifact_refs": "User-supplied local or external artifacts that are inputs to the design.",
    "aso_overlap_requirement": "any_overlap means the ASO intersects the feature; full_containment means the ASO lies inside the feature; contains_feature means the ASO covers the entire feature and may extend beyond it.",
    "candidate_limit": "Maximum candidate count for the complete goal or one design program. Use null when no maximum applies to this scope; independent per-site limits retain their own field.",
    "candidate_limit_per_site": "Maximum candidate count independently retained for each target site.",
    "candidate_minimum": "Minimum number of distinct oligonucleotide sequences required in the complete panel; paired with candidate_limit for an acceptable count range. Omission requires at least one. Set zero only when the user permits a documented empty result after the requested search; unavailable evidence must remain a limitation, never a claim of biological absence.",
    "candidate_selection_policy": "ranked retains the best individual candidates; diverse_target_sites is a soft preference for separated targets and may add overlapping alternatives to meet a count. nonoverlapping_target_sites is required when the user makes mutual target nonoverlap a hard condition: choose distinct physical molecules jointly, preserve all count constraints, never fill with overlapping targets, and report unmet counts only for the verified eligible pool. Keep exact requested counts in candidate_minimum and candidate_limit; separate program labels do not guarantee different molecules.",
    "target_accessibility": "Whether candidate selection requires sequence-based target RNA accessibility predictions. This measures full binding-window unpaired probability on the target transcript, separately from oligo self-folding. It provides relative ranking, not an experimentally validated efficacy threshold.",
    "target_accessibility_request_quote": "An exact passage from the user's request explicitly requiring sequence-based target RNA accessibility prediction. Required when target_accessibility is true; generic evidence or accessibility of a genomic region is insufficient.",
    "oligoai_evaluation": "Explicit OligoAI efficacy prediction conditions: enabled, positive dosage_nm and supported transfection_method. Omit any unstated field; runtime labels missing conditions as an assumed reference scenario. This does not measure efficacy or alter physical chemistry, candidate selection or safeguards.",
    "conditional_exon_skip_events": "User-requested hypothetical complete exon removals to assess from the named reference sequence, independent of the desired ASO intervention direction.",
    "chemistry": "Requested oligonucleotide chemical architecture, distinct from mechanism and selection policy.",
    "chemistry_policy": "select permits ordinary chemistry selection. leave_unspecified records an explicit instruction to defer chemical architecture/platform and deliver base sequences without synthesis encoding. Omitted chemistry alone does not imply leave_unspecified; scope this policy per affected design program.",
    "constraints": "User-requested executable sequence and mapping filters only.",
    "cross_species_requirement": "Required cross-species target matching or conservation policy.",
    "design_programs": "Distinct requested ASO design branches or routes.",
    "evaluation_targets": "Structured biological sites, isoforms, exons, or intervals against which the result is evaluated.",
    "explicit_splice_targets": "Explicitly supplied splice targets used by downstream targeting.",
    "flanks": "Requested upstream and downstream sequence-window sizes.",
    "focus_exons_only": "Whether targeting is restricted to the specified exons.",
    "gene_id": "Explicitly supplied stable gene identifier.",
    "gene_symbol": "Explicitly supplied gene symbol.",
    "intent": "Canonical biological design intent: knockdown, splice modulation, or steric binding.",
    "isoform_scope": "Requested scope over transcripts or isoforms.",
    "isoform_scope_source": "Evidence provenance for the chosen isoform scope.",
    "junction_gapmers": "Requested gapmer targeting policy for transcript junctions.",
    "min_asos_per_site": "Minimum number of candidates required for every target site.",
    "minimum_overlap_bp": "Minimum number of bases by which a candidate must overlap an allowed interval.",
    "mirna_binding": "Required miRNA-site overlap restricting candidate eligibility. Omit for annotation-only overlap analysis, which belongs in a downstream analysis task and must preserve the requested library.",
    "motif_tiling": "Required sequence-motif targeting that restricts candidate eligibility. Omit when motifs are only annotations on an otherwise complete library.",
    "off_target_expression": "Off-target expression analysis settings, including biological context.",
    "off_target_species": "Organisms explicitly requiring genome/transcriptome-wide off-target screening. Target conservation alone does not request this extra analysis. The primary design organism is screened by default; optional secondary screens report their actual coverage.",
    "patient_variant": "Typed user-supplied variant identity used to resolve a patient, disease, or allele-specific sequence context.",
    "preservation_targets": "User-named genes, transcripts, or isoforms whose abundance or function must be preserved by selectivity.",
    "rbp_binding": "Required RNA-binding-protein-site targeting that restricts candidate eligibility. Omit for annotation-only overlap analysis.",
    "region_constraints": "Executable genomic, cDNA, exon, or splice-relative targeting regions.",
    "run_splice_eval": "Whether the design branch requires splice-effect evaluation.",
    "run_splice_router": "Whether the design branch requires splice-route selection.",
    "site_direction_hints": "User-supported directional hints for splice sites.",
    "source_attribution": "Sources that the user requires the design or answer to cite.",
    "species": "Requested organism for the design.",
    "species_targets": "Structured organisms included in a multi-species request.",
    "splice_isoform_objectives": "Desired abundance or exon effect for each requested isoform objective.",
    "target_context": "Tissue, cell type, disease model, or explicitly broad biological context.",
    "target_region": "User-described biological region to target.",
    "target_scope": "Whether targeting is isoform-specific or gene-wide.",
    "transcript_lock": "An explicitly user-locked transcript and its provenance.",
    "transcript_policy": "Whether transcript selection is user-locked or delegated to the platform.",
    "aggregate_class": "Named aggregate isoform class addressed by an isoform objective.",
    "chrom": "Chromosome or contig for a genomic target.",
    "coordinate_convention": "Coordinate indexing and interval-boundary convention supplied by the user or source.",
    "coordinate_space": "Whether target coordinates are genomic or transcript cDNA coordinates.",
    "desired_abundance": "Requested increase or decrease in an isoform's abundance.",
    "desired_delta": "Requested increase or decrease produced by a design program or evaluated target.",
    "desired_effect": "Requested exon inclusion or exon skipping effect.",
    "end": "End coordinate or relative endpoint for a target.",
    "execution_policy": "Whether a design program is primary or fallback and the canonical condition that activates it.",
    "exon_label": "One requested exon label or number.",
    "exon_index": "Explicit one-based transcript annotation exon ordinal, distinct from biological exon nomenclature.",
    "exon_labels": "Multiple requested exon labels or numbers.",
    "exon_numbering_transcript_id": "Transcript whose exon numbering defines the requested exon label.",
    "gc_content": "Canonical lower and upper GC-fraction bounds.",
    "genome_build": "Reference genome assembly for genomic coordinates.",
    "homopolymers": "Canonical limits on repeated identical bases or forbidden repeated sequences.",
    "intervals_bed0_half_open": "Normalized zero-based, half-open genomic intervals.",
    "isoform_label": "User-supplied isoform name when no stable transcript identifier was supplied.",
    "label": "Human-readable label for a program, target, interval, or constraint.",
    "length": "Explicit requested ASO length in nucleotides.",
    "max_binding_sites": "Maximum allowed number of genomic binding or mapping sites.",
    "modality": "Canonical ASO modality: steric or gapmer.",
    "notes": "Non-executable user-supported context that does not replace structured fields.",
    "objective": "Concise biological objective for one design program.",
    "objective_scope": "Whether an isoform objective applies to a transcript, named isoform, aggregate class, or default exon context.",
    "objective_type": "Canonical role of an evaluation target: identity_anchor, primary_outcome, route_target, or monitoring_safeguard.",
    "operator": "Ordered combination of per-candidate constraint matches: union/include is OR, intersect is AND, exclude is AND NOT. This does not merge source intervals before testing overlap. Rows within a constraint are alternative locations.",
    "position": "Single site position supplied for a target.",
    "preservation_target_type": "Whether a protected biological entity is a gene, isoform, or transcript.",
    "program_id": "Stable platform-generated identifier for one design branch within the request; it is not a user-supplied biological fact.",
    "rationale": "User-supported rationale for an isoform objective.",
    "reference_accession": "Explicitly supplied reference accession for an isoform objective.",
    "reference_transcript_id": "Transcript used to interpret a relative or exon-numbered region.",
    "role": "Whether an isoform is beneficial, pathogenic, or contextual.",
    "rows": "Explicit tabular region rows preserved from a supplied artifact or request.",
    "score": "User-supplied score attached to a region or interval.",
    "site_context": "User-described local context around a requested site.",
    "site_kind": "Requested splice-site kind: acceptor, donor, or both.",
    "source": "Name or identifier of the source defining a region constraint.",
    "source_type": "Canonical kind of evidence or coordinates defining a region constraint.",
    "splice_relative_window": "Window expressed relative to a splice acceptor or donor.",
    "start": "Start coordinate or relative endpoint for a target.",
    "strand": "Reference strand for a genomic target.",
    "target_type": "Canonical biological entity whose direction is specified. Splice acceptor/donor/junction/site targets use transcript_id to locate a feature; transcript/isoform targets refer to abundance of that RNA outcome. Resolve descriptive categories semantically to the enum and keep the description in label. Use other only for an entity outside the supported categories, not for an unresolved choice between them.",
    "targeting_policy": "Requested targeting policy for one design program.",
    "transcript_id": "Explicitly supplied stable transcript identifier at a program, target, or constraint.",
}


def _canonical_object_schema(
    fields: Any,
    field_schemas: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    schemas = CANONICAL_GOAL_FIELD_SCHEMAS if field_schemas is None else field_schemas
    missing = sorted(str(field) for field in fields if str(field) not in schemas)
    if missing:
        raise ValueError(f"canonical fields lack schemas: {', '.join(missing)}")
    return {
        "type": "object",
        "properties": {
            str(field): deepcopy(schemas[str(field)]) for field in sorted(fields)
        },
        "additionalProperties": False,
    }


def canonical_structured_goal_schema() -> Dict[str, Any]:
    """Return the strict schema shared by model output and runtime validation."""

    scalar_fields = CANONICAL_STRUCTURED_GOAL_FIELDS.difference(
        CANONICAL_GOAL_COLLECTION_FIELDS
    )
    properties = _canonical_object_schema(scalar_fields)["properties"]
    for collection, fields in CANONICAL_GOAL_COLLECTION_FIELDS.items():
        item_schema = _canonical_object_schema(fields)
        required_fields = CANONICAL_GOAL_COLLECTION_REQUIRED_FIELDS.get(
            collection,
            frozenset(),
        )
        if required_fields:
            item_schema["required"] = sorted(required_fields)
        properties[collection] = {
            "type": "array",
            "items": item_schema,
        }
        if collection == "design_programs":
            properties[collection]["minItems"] = 1
    return {
        "type": "object",
        "properties": properties,
        "required": ["design_programs"],
        "additionalProperties": False,
    }


def canonical_task_context_schema() -> Dict[str, Any]:
    """Return the strict schema for task-local request context."""

    context_schemas = deepcopy(CANONICAL_TASK_CONTEXT_FIELD_SCHEMAS)
    context_schemas["evaluation_targets"] = {
        "type": "array",
        "items": _canonical_object_schema(
            CANONICAL_GOAL_COLLECTION_FIELDS["evaluation_targets"]
        ),
    }
    context_schemas["structured_goal"] = canonical_structured_goal_schema()
    schema = _canonical_object_schema(
        CANONICAL_TASK_CONTEXT_FIELDS,
        context_schemas,
    )
    schema["properties"]["structured_goal"] = canonical_structured_goal_schema()
    return schema


def _unique_patch_value_schema(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = [deepcopy(candidate) for candidate in candidates]
    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate:
            continue
        digest = repr(candidate)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(candidate)
    return {"anyOf": unique}


def canonical_goal_patch_value_schema() -> Dict[str, Any]:
    """Return the exact value union accepted by structured-goal patches."""

    candidates: List[Dict[str, Any]] = [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
    ]
    candidates.extend(
        deepcopy(schema)
        for field, schema in CANONICAL_GOAL_FIELD_SCHEMAS.items()
        if field not in CANONICAL_GOAL_COLLECTION_FIELDS
    )
    for fields in CANONICAL_GOAL_COLLECTION_FIELDS.values():
        item_schema = _canonical_object_schema(fields)
        candidates.extend(
            [item_schema, {"type": "array", "items": deepcopy(item_schema)}]
        )
    return _unique_patch_value_schema(candidates)


def canonical_patch_value_schema() -> Dict[str, Any]:
    """Return the union accepted across goal and non-design context patches."""

    candidates = list(canonical_goal_patch_value_schema()["anyOf"])
    candidates.extend(
        deepcopy(schema)
        for field, schema in CANONICAL_TASK_CONTEXT_FIELD_SCHEMAS.items()
        if field != "evaluation_targets"
    )
    return _unique_patch_value_schema(candidates)


def _registry_schema_summary(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(schema[key])
        for key in ("type", "enum", "const", "minimum", "maximum")
        if key in schema
    }


def canonical_goal_field_registry() -> List[Dict[str, Any]]:
    """Return semantic, machine-readable canonical destination metadata."""

    entries: List[Dict[str, Any]] = []

    def _walk(schema: Dict[str, Any], path: str, scope: str) -> None:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return
        for field in sorted(properties):
            field_schema = properties[field]
            if not isinstance(field_schema, dict):
                continue
            field_path = f"{path}.{field}"
            entries.append(
                {
                    "path_pattern": field_path,
                    "semantic_role": CANONICAL_GOAL_FIELD_SEMANTICS.get(
                        field,
                        f"Canonical {field.replace('_', ' ')} value in this scope.",
                    ),
                    "applies_to": scope,
                    "schema": _registry_schema_summary(field_schema),
                    "incompatible_with": [],
                }
            )
            if field_schema.get("type") == "object":
                _walk(field_schema, field_path, scope)
            elif field_schema.get("type") == "array" and isinstance(
                field_schema.get("items"), dict
            ):
                _walk(field_schema["items"], f"{field_path}[*]", field)

    _walk(
        canonical_structured_goal_schema(),
        "context.structured_goal",
        "structured_goal",
    )
    return entries


def canonical_task_context_field_registry() -> List[Dict[str, Any]]:
    """Return semantic metadata for exact task-context destinations."""

    schema = canonical_task_context_schema()
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    return [
        {
            "path_pattern": f"tasks[*].context.{field}",
            "semantic_role": CANONICAL_TASK_CONTEXT_FIELD_SEMANTICS.get(
                field,
                f"Canonical {field.replace('_', ' ')} value for one task.",
            ),
            "applies_to": "task_context",
            "schema": _registry_schema_summary(properties.get(field) or {}),
            "incompatible_with": [],
        }
        for field in sorted(CANONICAL_TASK_CONTEXT_FIELDS)
    ]


def canonical_goal_destination_catalog() -> Dict[str, List[str]]:
    """Return compact, schema-derived legal object fields by canonical path."""

    catalog: Dict[str, List[str]] = {}

    def _walk(node: Dict[str, Any], path: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") != "object":
            return
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return
        catalog[path] = sorted(str(field) for field in properties)
        for field, nested in properties.items():
            if not isinstance(nested, dict):
                continue
            nested_path = f"{path}.{field}"
            if nested.get("type") == "object":
                _walk(nested, nested_path)
                continue
            if nested.get("type") == "array" and isinstance(
                nested.get("items"),
                dict,
            ):
                _walk(nested["items"], f"{nested_path}[*]")

    _walk(canonical_structured_goal_schema(), "context.structured_goal")
    return catalog


def canonical_goal_requirement_catalog() -> Dict[str, List[str]]:
    """Return schema-derived required child fields for every goal object path."""

    catalog: Dict[str, List[str]] = {}

    def _walk(node: Dict[str, Any], path: str) -> None:
        if not isinstance(node, dict) or node.get("type") != "object":
            return
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return
        catalog[path] = sorted(str(field) for field in node.get("required", []))
        for field, nested in properties.items():
            if not isinstance(nested, dict):
                continue
            nested_path = f"{path}.{field}"
            if nested.get("type") == "object":
                _walk(nested, nested_path)
            elif nested.get("type") == "array" and isinstance(
                nested.get("items"), dict
            ):
                _walk(nested["items"], f"{nested_path}[*]")

    _walk(canonical_structured_goal_schema(), "context.structured_goal")
    return catalog


CANONICAL_TARGET_CONTEXT_INSTRUCTIONS = (
    "- Every design_new_asos task requires target_context for execution: a supplied tissue, cell type, disease/model context, or a user-delegated computational baseline. The sparse construction schema may omit it while clarification is pending. A missing biological context is not a public-reference fact and must not be silently invented.\n"
    "- Interpret context permission in the scope of the user's request and the question they answered. An explicit instruction to choose reasonable defaults, use your judgment, or proceed without additional context preferences in reply to a tissue/model question delegates a broad computational baseline. The user need not repeat a particular phrase such as 'broad baseline'. Record general baseline for the supported organism, identify it as a user-delegated design assumption, and do not ask the same context question again. This records permission, not knowledge of an experimental tissue or patient.\n"
    "- Preserve every more specific tissue/model requirement or restriction. Default permission for another field does not authorize changing context; inability to supply information is not permission. Never use context-default delegation to invent a missing patient variant, custom reference, assay measurement, target identity, or biological result. If neither a context nor applicable delegation exists, ask once for context or permission to use a broad baseline.\n"
)


CANONICAL_CLARIFICATION_BOUNDARY_INSTRUCTIONS = (
    "Clarification boundary contract:\n"
    "- Public gene, transcript, sequence, and standard annotation facts belong to authoritative source resolution and must not be shifted to the user.\n"
    + CANONICAL_TARGET_CONTEXT_INSTRUCTIONS
    +
    "- A source cited to substantiate a clarification answer is authoritative only within that paired question-and-answer scope. It is provenance for the answer, not an additional user request. Never import unrelated objectives, identifiers, directions, constraints, or deliverables from clarification source evidence.\n"
    "- A named public isoform or transcript product remains a public-reference resolution target even when it is disease-associated or produced by alternative, cryptic, or noncanonical splicing. Its label is sufficient to begin source resolution unless the request says the target is private, custom, unannotated, allele-specific, or rejects the standard public mapping. Do not require the user to supply its mutation, transcript ID, sequence, or coordinates merely because the mechanism is noncanonical.\n"
    "- A user-defined, private, unannotated, cryptic, or otherwise nonstandard target feature is not safely resolvable from a partial label. When execution depends on its exact identity and the request supplies neither an exact genomic interval with strand nor an explicit mapping to a standard annotated feature, request that user-specific identity once. Never infer the mapping from a numeric fragment or a similarly named public feature.\n"
    "- A transcript exon ordinal is standard only when the user supplies an unqualified integer ordinal. A qualified exon label is not made resolvable merely by also supplying a transcript identifier; require its exact interval and strand or an explicit mapping to a standard annotated exon.\n"
    "- When a request explicitly rejects the authoritative feature or coordinate that its otherwise standard label resolves to, that rejection removes the standard mapping. If the request does not identify the intended replacement by an exact interval with strand or an explicit standard-feature mapping, ask once for target_feature_mapping_missing; do not guess which alternative feature the user intended.\n"
    "- Genomic intervals from different genome assemblies cannot be combined in one coordinate-space operation without an explicit target genome build or user-approved normalization policy. Ask once for that user-specific choice rather than silently merging or lifting coordinates.\n"
    "- For an RNA target whose gene or transcript identity is supplied, transcribed strand is an authoritative public-annotation fact. Do not ask the user for strand unless the requested target is genuinely independent of that resolvable transcription context or the supplied evidence explicitly conflicts.\n"
    "- Do not infer a candidate-count impossibility from target-interval length alone. any_overlap permits an oligo to extend beyond the constrained feature, whereas full_containment does not; sequence materialization and ranking determine the available unique candidates. Ask about count feasibility only when the explicit overlap semantics and executable search space establish a real user choice that no declared platform policy can handle.\n"
    "- Collect every independent user-specific execution blocker visible in the current evidence in one clarification round. Do not ask for optional metadata or platform-defaultable settings.\n"
)


def canonical_resolution_capability_catalog() -> List[Dict[str, str]]:
    """Declare source/default capabilities shared by clarification reviewers."""

    return [
        {
            "capability": "public_gene_and_transcript_resolution",
            "scope": "Canonical public gene, transcript, sequence, annotation, and coordinate lookup.",
        },
        {
            "capability": "public_named_isoform_resolution",
            "scope": (
                "Named public isoform and transcript-product labels can be mapped "
                "through configured transcript annotation and source-backed lookup, "
                "including disease-associated alternative or cryptic splice products."
            ),
        },
        {
            "capability": "public_binding_source_resolution",
            "scope": "Canonical RBP/CLIP and miRNA target-source requests represented in the structured goal.",
        },
        {
            "capability": "pipeline_defaults",
            "scope": "Unspecified execution settings for which the platform has an explicit default policy.",
        },
        {
            "capability": "splice_relative_window_overlap_default",
            "scope": (
                "An exact splice-relative ASO tiling window defaults to full "
                "containment when the window itself is targeted, or any overlap "
                "when the window represents a separately named embedded feature; "
                "an explicit supported user policy overrides it."
            ),
        },
    ]


CANONICAL_DESIGN_GOAL_INSTRUCTIONS = (
    CANONICAL_CLARIFICATION_BOUNDARY_INSTRUCTIONS
    + COORDINATE_ORIGIN_INSTRUCTIONS
    + "Canonical executable design-goal contract:\n"
    "- When an ASO must cover an annotated start codon or satisfy a coding-sequence boundary, use a region constraint with source_type='transcript_feature', feature_name='start_codon' or 'cds', and reference_transcript_id when supplied. The platform resolves annotation coordinates; do not invent absolute rows or assume the start codon occupies transcript bases 1-3. Preserve the requested overlap relation. Constraints that must all hold use intersect, including features on different explicitly requested transcripts. A shared panel stores every required transcript in its program transcript_ids.\n"
    "- chemistry_policy is a permission decision, not a record of whether the user named a chemical formula. select authorizes platform selection and is the meaning of delegated/default chemistry. leave_unspecified prohibits selecting modifications, backbone and synthesis platform and is used only when the user deliberately defers that choice. If the user simply omits chemistry, omit the policy so ordinary platform selection remains available. Preserve an explicit chemical formula separately in chemistry. Scope different policies to their own programs.\n"
    "- Treat the user request and recorded clarification answers as evidence. Preserve every explicitly supplied design fact. Do not use biological background knowledge to invent a gene, species, transcript, isoform, exon, coordinate, route, direction, length, chemistry, or target context. The species field contains only an organism supported by request or clarification evidence, or an organism established from an explicit identifier by an authoritative source; never store a lookup instruction, uncertainty note, or resolution plan in species. Omit unresolved species during intake and let authoritative resolution supply it instead of encoding how to find it as though it were an organism.\n"
    "- Every design task must store its authoritative goal at context.structured_goal. Do not leave executable facts only in the task label, request summary, assumptions, or prose.\n"
    "- Explicit mature-transcript windows use coordinate_space='cdna' and an exact reference transcript. Executable windows belong in region_constraints with BED0 half-open bounds; use operator='intersect' when a second required feature must also be satisfied. A required covered feature uses contains_feature; a minimum partial overlap uses any_overlap plus integer minimum_overlap_bp, never a newly invented overlap enum. Program-local start/end windows must carry coordinate_space and coordinate_convention ('bed0_half_open' or 'one_based_inclusive'); absolute endpoints are not flanks. Keep an outer allowed window and a required covered feature as separate constraints.\n"
    "- An available source may supply values without making every value a requested execution constraint. Create an optional canonical object or collection only when request or clarification evidence requires its semantics. If that optional structure has required children whose values are unsupported, omit the structure instead of asking the user to complete model-authored state.\n"
    "- context.structured_goal.design_programs is required and contains one object per requested design branch or route. Each object uses program_id, label, modality, intent, and objective. Add desired_delta when the evidence defines a biological direction.\n"
    "- design_programs[*].program_id is a platform-generated structural identifier. Generate a concise, non-empty identifier that is unique within the goal and stable for the branch semantics. It does not require a literal in user evidence, and the user must never be asked to supply or approve it.\n"
    "- A design program represents a distinct intervention branch, modality, objective, control-flow role, targeting policy, or executable variant required to preserve a finite user-specified parameter domain; it does not represent each site or interval. Multiple sites served by the same branch remain one design program and are represented as distinct evaluation_targets and region-constraint rows.\n"
    "- When the request makes one design program conditional on another program failing, encode that control flow at design_programs[*].execution_policy. Use {role:'primary', activation_condition:'always'} for the primary program and {role:'fallback', activation_condition:'no_viable_primary_routes'} for the dormant fallback. Do not leave program priority or activation only in a label, objective, note, or invented field. Omit execution_policy when the user did not define primary/fallback control flow.\n"
    "- design_programs[*].modality must be exactly 'steric' or 'gapmer'. design_programs[*].intent must be exactly 'knockdown', 'splice_modulation', or 'binding'. Use 'knockdown' when the requested end result is lower total gene or RNA output, including a splice-mediated route; use 'splice_modulation' when altered exon inclusion or isoform balance is itself the requested end result; use 'binding' for a non-splicing steric blockade. Translating explicit request language into these schema enums is normalization, not invention.\n"
    "- Use design_programs[*].desired_delta, with exactly 'increase' or 'decrease', for the requested biological direction of that program. It is required for knockdown and splice-modulation programs, whose requested mechanism determines a direction, and for a binding program only when the user states a downstream biological direction. A non-splicing binding-site blockade remains executable without an abundance-direction claim; omit desired_delta rather than inventing one or asking the user. When desired_effect='skipping', a design program's desired_delta must be 'decrease'; when desired_effect='inclusion', it must be 'increase'. Apply that pairing to an evaluation target only when its identity is the exon and desired_delta therefore describes exon usage. A transcript, isoform, transcript-cohort, or isoform-cohort evaluation target uses desired_delta for the abundance of that identified outcome, independently of the structural splice event that produces it. For splice_isoform_objectives, desired_abundance likewise describes the abundance of the identified isoform or transcript and is independent of the structural splice event: an exon-skipped isoform may be requested to increase, and an exon-included isoform may be requested to decrease. Only a default_exon_context objective, which has no identified isoform abundance of its own, pairs desired_effect='skipping' with desired_abundance='decrease' and desired_effect='inclusion' with desired_abundance='increase'. These outcome fields are distinct from the downstream router's site-action direction. When the request defines separate directional routes, every route must have its own desired_delta. Do not encode a supported direction only in objective, route_goal, route_label, notes, safeguards, or a key named direction.\n"
    "- When the user explicitly gives one oligo length, store it as the positive integer design_programs[*].length on every affected program. When the user gives a finite set or bounded inclusive integer range of allowed oligo lengths, exhaustively create otherwise-equivalent program variants so every allowed integer appears exactly once at design_programs[*].length for each affected branch. Give every variant a unique program_id. Do not choose a preferred or default member, omit an allowed member, or encode the domain only inside chemistry, a label, objective, note, or assumption. Omit length when the user did not specify it.\n"
    "- Preserve requested candidate counts without aggregation. For an inclusive global count range, set candidate_minimum to its lower bound and top-level candidate_limit to its upper bound; a finite panel range cannot have a null candidate_limit. For an exact global count, set both fields to that count. Use top-level candidate_limit only for a maximum that applies to the complete design task. When the request says a count applies independently to each site, window, or route, store that exact number at top-level candidate_limit_per_site. Never multiply a per-unit count by the number of branches, convert it into an overall total, or leave it only on program-local candidate_limit fields. For an exact per-unit count, also store that number at min_asos_per_site. For a minimum-only count, set min_asos_per_site without inventing an upper bound. Distinguish the count of molecules from oligo length and tiling/search-row counts.\n"
    "- Put each target fact in the object whose scope the evidence supports. Global biological outcomes belong in evaluation_targets and splice_isoform_objectives. Put a target identity on a design program only when the evidence explicitly assigns that identity to that intervention branch; never distribute global outcome identities across parallel programs merely because their counts match. Use exon_label for a numeric exon, site_kind for acceptor/donor/both, transcript_id for an explicitly supplied transcript identifier in that object's scope, isoform_label for a user-supplied isoform description without an identifier, and desired_delta for direction. Do not leave target identity only in prose.\n"
    "- transcript_id holds one accession. A program targeting a transcript cohort stores its members in transcript_ids, never a comma-joined singular ID. Distinct requested isoform directions remain in splice_isoform_objectives; shared knockdown and splice switching are separate programs with their own mechanisms. Exact per-site or per-panel counts set both candidate_limit_per_site and min_asos_per_site.\n"
    "- When the request assigns a named isoform or transcript to a particular intervention route, store that identity on the route_target evaluation target, the corresponding design program, and a corresponding splice_isoform_objectives item. Use design_programs[*].isoform_label or design_programs[*].transcript_id; a name present only in the program label or objective is not executable route scope. The splice objective must use the same typed identity, objective_scope='specific_isoform' or 'specific_transcript' as appropriate, and the evidence-supported desired_abundance. Never downgrade a named isoform-abundance route to default_exon_context. A transcript supplied only to locate an exon or splice junction is a reference identity, not an isoform-abundance outcome: preserve it on transcript_lock and the program, use an exon evaluation target and default_exon_context for the structural splice objective, and do not alternate that objective between default_exon_context and specific_transcript during repair. Do not add an identity to a program when the evidence states only a global outcome.\n"
    "- Every evaluation_targets item must classify objective_type as exactly identity_anchor, primary_outcome, route_target, or monitoring_safeguard. Use identity_anchor when the item preserves a site, coordinate, exon endpoint, or other target identity needed to execute a directional program but the request does not define a change in that item's own abundance or usage; identity anchors must omit desired_delta. A primary_outcome or route_target must carry the evidence-supported desired_delta for that identified outcome. A monitoring_safeguard records a requested observation whose outcome direction was not specified and must also omit desired_delta. Monitoring or identity anchoring alone never creates a directional claim or a clarification need.\n"
    "- When the user asks to spare, preserve, protect, or avoid changing a named gene, transcript, or isoform, add one preservation_targets item per protected entity. Set action='preserve', set preservation_target_type to exactly 'gene', 'transcript', or 'isoform', and store its identity in gene_symbol, transcript_id, or isoform_label. Do not represent a preservation constraint as increase/decrease, and do not leave it only in objective, label, notes, assumptions, or request prose.\n"
    "- Use patient_variant for a user-supplied variant that defines the patient, disease, or target sequence context, whether or not the requested ASO must discriminate between alleles. Preserve the supplied notation without inventing an accession; downstream authoritative resolution qualifies a bare or gene-qualified cDNA coordinate against an explicit reference accession when available or a source-backed canonical transcript otherwise. The presence of patient_variant alone does not imply allele-specific targeting. Add constraints.allele_specific only when request or clarification evidence explicitly requires selecting a target allele while avoiding a non-target allele; in that case, preserve target_allele, non_target_allele, and require_variant_overlap there.\n"
    "- Treat exon_label, exon_labels, and requested_exon_label as request-nomenclature fields. Preserve the exact biological label or ordered labels stated by the user; they are not storage for a resolver's structural transcript-exon index. Public annotation, declared artifacts, and resolved coordinates may map a requested label to a different internal exon index for execution, but they must not renumber the request-facing fields. Keep the requested label in the canonical goal and let explicit target coordinates or downstream resolution metadata carry the structural mapping. Change request nomenclature only when the user request or a later user clarification explicitly supplies the replacement label.\n"
    "- An exon-to-exon junction named by its two endpoint exons must preserve both ordered exon labels at design_programs[*].exon_labels and as distinct evaluation_targets items with numeric exon_label values. Never collapse a junction pair into one endpoint or encode the pair as a compound exon_label string.\n"
    "- Copy explicitly supplied gene, species, and target context into the structured goal. Every design_new_asos task requires target_context for runtime execution even though the sparse construction schema permits it to be absent while clarification is pending. Apply the shared target-context permission contract: copy supplied context or materialize an explicitly delegated computational baseline in every affected structured goal. Do not leave that resolved permission only in assumptions, and do not request it again. If neither context nor applicable delegation is supplied, preserve one context clarification. When the user explicitly locks a transcript as the design reference, store it as transcript_policy='user_locked' and transcript_lock={value: the supplied identifier, source: 'user_request'}, and also put transcript_id on each program and evaluation target to which the evidence assigns that lock. If the supplied transcript identifier includes an explicit numeric version, also store that integer at transcript_lock.version; never leave an explicitly supplied version only in prose. Naming transcripts as biological outcomes does not by itself lock or assign a design-reference transcript. Do not create a top-level transcript_id key.\n"
    "- For a coupled multi-species design, species is one anchor organism used for primary reference resolution, while species_targets and cross_species_requirement.species preserve the complete requested organism set. The anchor must be one member of that set. Never concatenate, join, or invent a composite species name. Do not split a coupled conservation objective into separate tasks merely to represent its organisms.\n"
    "- Distinguish missing user intent from source-resolvable execution facts. When the request supplies a biological target and organism plus a reference-selection policy or resolvable scope, preserve that scope and let downstream authoritative resolution obtain the accession, annotation, sequence, and coordinates. Use transcript_policy='agent_choice' when the evidence delegates reference selection. Do not ask the user to provide public-reference identifiers, sequences, or coordinates merely because they are absent from request evidence. Clarify only when a material user-specific choice remains ambiguous or no supplied identity can be resolved.\n"
    "- Use isoform_scope_source='user_specified' when the isoform scope is explicitly supplied by the user, 'resolver_inferred' when it comes from transcript resolution, and 'default_exon_splice_context' for an exon splice objective without a separately stated isoform-abundance outcome, including when the exon is located using a user-specified reference transcript. Use no other value and omit this optional field when its provenance is unsupported.\n"
    "- Canonical biological genomic coordinates in evaluation_targets, explicit_splice_targets, and design_program target fields are one-based and inclusive, matching genomic coordinates presented in ordinary user requests and transcript annotations. Preserve the supplied biological start and end there. BED conversion is scoped only to region_constraints[*].intervals_bed0_half_open: for a one-based inclusive interval [s,e], that row is [s-1,e). Never copy a BED0 start back into a biological target object. The synthesis pass and every semantic audit pass must treat a one-base start discrepancy between these representations as a contract error and repair the biological target from request evidence.\n"
    "- For explicitly supplied genomic intervals, preserve one labeled entry per interval in evaluation_targets. Coordinates in a source describe target identity or context; they do not by themselves prove that the user requested an executable overlap filter. Create a region_constraints entry only when request or clarification evidence explicitly makes the interval an execution constraint and supports the entry's required overlap semantics. For an exact splice_relative_window requested for ASO tiling, the declared platform default is full_containment when the window itself is the target and any_overlap when the window represents a separately named embedded feature; an explicit supported user policy always overrides that default. When an explicit genomic-row constraint is supported, use source_type='explicit_rows', coordinate_space='genomic', and intervals_bed0_half_open containing the corresponding converted chrom/start/end/strand rows; preserve an explicitly supplied genome build, coordinate convention, aso_overlap_requirement, and minimum_overlap_bp. Otherwise omit the optional region_constraints entry while retaining the interval as target identity/context. Do not put chrom, strand, genome-build, overlap, or GC aliases inside design_programs.\n"
    "- For source coordinates on different assemblies, preserve each original genome_build and coordinate convention on its region constraint/interval. Clarify conversion permission AND ambiguous coordinate conventions together. If user evidence authorizes validated conversion, set allow_coordinate_conversion=true on the relevant region_constraints; keep original-build coordinates there and let the native materializer perform evidenced conversion to the installed reference. Never invent converted coordinates, keep asking an already answered conversion question, or remove the original build labels. If source intervals were explicitly BED0, retain their original starts in intervals_bed0_half_open and express their one-based biological identities as start+1/end in evaluation_targets. Conversion admission is not proof that mapping succeeded; report each conversion result and original interval.\n"
    "- Preserve the direction of containment: full_containment means the complete ASO lies INSIDE a target interval. contains_feature means the complete target feature lies INSIDE the ASO. To require junction crossing with at least U bases before and D bases after an acceptor boundary, use a splice_relative_window from -U to D-1 with contains_feature; for a donor use 1-U to D because offset zero is the last exonic base. Both offset conventions follow transcript orientation on either genomic strand. Create separate acceptor and donor constraints with operator=union when either junction is allowed; a whole-exon any_overlap constraint does not encode junction crossing. Never encode the required covered core as full_containment.\n"
    "- region_constraints, aso_overlap_requirement, and minimum_overlap_bp are direct children of context.structured_goal, never children of constraints. constraints contains only gc_content, homopolymers, and max_binding_sites. Represent GC bounds at constraints.gc_content with min, max, min_inclusive, and max_inclusive; use fractional values from 0 through 1. Keep an explicitly supplied oligo length on each affected design program.\n"
    "- Preserve an explicit off-target substitution search budget at context.num_mismatches for evaluation/comparison tasks and context.structured_goal.num_mismatches for design tasks. A request for exact plus one-mismatch evidence has budget 1, including separate exact evidence. Zero means exact-only. This is independent of max_binding_sites, sequence identity at the intended target, and any filtering threshold. Do not invent a budget when unspecified or silently lower an unsupported request.\n"
    "- evaluation_targets contain target identity and coordinates only. A transcript-level primary outcome does not acquire a splice-site kind merely because the request also defines splice-site intervention routes; site_kind is present only when evidence explicitly scopes it to that target. Store genome_build, coordinate_convention, source_type, operator, and intervals_bed0_half_open on region_constraints items, not on evaluation_targets.\n"
    "- Store program target fields directly at design_programs[*].transcript_id, isoform_label, exon_label, exon_labels, site_kind, position, start, or end as applicable. Do not wrap them in a target or evaluation_target object. For an exon-level splice program that does not restrict targeting to one junction, use site_kind='both' to declare that the execution plan must consider both acceptor and donor sites.\n"
    "- For a splice_modulation program with splice_relative_window constraints, site_kind is the execution anchor, not a claim that the named feature is itself a splice junction. When all of that program's windows use one anchor_site, set site_kind to that exact acceptor or donor value; when its windows use both anchor types, set site_kind='both'. Do not remove the anchor merely because the window describes a regulatory feature.\n"
    "- Represent count-style splice-window language as inclusive relative offsets without changing its nucleotide count. For an acceptor whose offset 0 is the first exonic nucleotide, N nucleotides upstream through M nucleotides into the exon is relative_start_nt=-N and relative_end_nt=M-1. For a donor whose offset 0 is the last exonic nucleotide, N nucleotides upstream through M nucleotides downstream is relative_start_nt=-(N-1) and relative_end_nt=M. Preserve explicitly signed endpoint offsets verbatim instead of applying this count conversion.\n"
    "- When request evidence names a specific biological feature represented by a splice-relative window, store that exact name at region_constraints[*].splice_relative_window.feature_name. Do not leave the feature identity only in the constraint label, program label, objective, or notes.\n"
    "- If distinct routes refer to distinct transcripts, isoforms, sites, or windows, emit distinct evaluation_targets whose structured fields distinguish those routes. Never collapse distinct routes into identical generic targets.\n"
    "- For every program whose requested route changes splicing, including splice-mediated knockdown, set design_programs[*].run_splice_router=true and design_programs[*].run_splice_eval=true. Store these as JSON booleans on the affected program; do not leave the route implied by intent, desired_effect, label, or objective prose.\n"
    "- Use splice_isoform_objectives[*].desired_abundance for isoform abundance direction and design_programs[*].desired_delta for program direction. These are the only canonical direction keys for those objects. Both fields accept exactly 'increase' or 'decrease'.\n"
    "- Do not infer pathogenic or beneficial disease roles from an increase/decrease instruction, exon inclusion/skipping, or a reference transcript lock. A structural reference is contextual. Retain an explicitly supported disease role, but otherwise omit the optional role or use context.\n"
    "- When an exon-level effect is supported, store it at design_programs[*].desired_effect and/or splice_isoform_objectives[*].desired_effect using exactly 'inclusion' or 'skipping'. Do not emit variants such as 'exon_inclusion' or 'exon_skipping'. splice_isoform_objectives[*].role is optional; when supported, it must be exactly 'beneficial', 'pathogenic', or 'context'. Omit an unsupported optional role instead of turning an intervention type or action into a role.\n"
    "- A splice effect is scoped to the exact object supported by the evidence. Never copy evaluation_targets[*].desired_effect or splice_isoform_objectives[*].desired_effect into a design program merely because only one target exists. A program receives desired_effect, site_kind, run_splice_router, and run_splice_eval only when the evidence assigns a splice-changing mechanism to that program.\n"
    "- desired_effect and site_kind describe splice effects and splice sites. Omit them for non-splicing binding programs and binding-site evaluation targets unless the request explicitly defines a splice effect or splice site.\n"
    "- Targeting sequence across an exon-to-exon junction does not by itself request splice modulation. Do not infer desired_effect, site_kind, run_splice_router, or run_splice_eval solely from junction targeting when the requested mechanism is cleavage or binding.\n"
    "- When a gapmer request explicitly targets sequence spanning an exon-to-exon junction in spliced RNA, set junction_gapmers.enabled=true. Treat it as gapmer cDNA tiling with run_splice_router=false and run_splice_eval=false unless the evidence separately requests splice modulation. Do not leave junction-spanning execution only in target prose or exon labels.\n"
    "- When the request names evidence or annotation sources, or requires source attribution, store the exact user-supplied source names at context.structured_goal.source_attribution as {required: true, sources: [...]}. Do not substitute similar source names or leave provenance only in assumptions. Omit source_attribution when the request does not require or name sources.\n"
    "- If an executable field is required but is not supported by the request or clarification history, return need_clarification and identify that canonical field in blocked_tasks[*].missing_fields. Do not fill it with a plausible default or unsupported identifier.\n"
)
