"""Evidence delivery for supplied-molecule evaluation; no efficacy decisions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from .input_rows import assessment_row_lookup
from .alignment_reporting import alignment_geometry
from .alignment_identity import annotate_window_groups
from .chemical_position_evidence import chemical_position_evidence


def alternative_transcripts_without_exact_binding(catalog, binding_hits):
    """An exact hit in one transcript does not settle alternative-isoform binding."""
    matched = {str(hit.get("transcript_id") or "").split(".")[0]
               for hit in binding_hits if isinstance(hit, dict)}
    return sorted({str(entry.get("transcript_id") or entry.get("id") or label).split(".")[0]
                   for label, entry in (catalog or {}).items() if isinstance(entry, dict)} - matched - {""})


def row_evidence_inventory(row):
    """Expose absent evidence explicitly, without borrowing another row's hits."""
    hypotheses = []
    for kind, key in (("exact_mapping", "mapping_hits"), ("near_match", "near_match_candidates")):
        for index, hit in enumerate(row.get(key) or []):
            hypotheses.append({
                "kind": kind, "transcript_id": hit.get("transcript_id"),
                "rna_context": hit.get("rna_context") or hit.get("mapping_scope"),
                "orientation": hit.get("match_orientation"),
                "requested_site_overlaps": deepcopy(hit.get("requested_site_overlaps") or []),
                "reference_junctions_spanned": deepcopy(hit.get("reference_junctions_spanned") or []),
                "source_field": f"{key}[{index}]",
                "geometry": alignment_geometry(hit),
            })
    window_identity = annotate_window_groups(hypotheses)
    groups = {}
    for hypothesis in hypotheses:
        fields = ("kind", "transcript_id", "rna_context", "orientation")
        key = tuple(hypothesis.get(field) for field in fields)
        group = groups.setdefault(key, {
            **{field: hypothesis.get(field) for field in fields},
            "retained_count": 0, "source_fields": [],
        })
        group["retained_count"] += 1
        group["source_fields"].append(hypothesis["source_field"])
    register_sources = [item["source_field"] for item in hypotheses if item["requested_site_overlaps"]]
    junction_sources = [item["source_field"] for item in hypotheses if item["reference_junctions_spanned"]]
    availability = {
        "requested_feature_membership": {
            "status": "recorded" if row.get("requested_feature_membership") else "not_recorded",
            "source_field": "requested_feature_membership",
            "features": [{
                "source_field": f"requested_feature_membership[{index}]",
                **{key: deepcopy(feature[key]) for key in (
                    "requested_feature_identity_resolved", "identity_resolution", "feature",
                    "transcript_id", "evaluated_transcript_id", "chrom", "start", "end",
                    "coordinate_system", "exon_index") if key in feature},
                "recorded_membership_observation_count": len(feature.get("observations") or []),
            } for index, feature in enumerate(row.get("requested_feature_membership") or [])],
            "interpretation": "Recorded reference-feature geometry and membership observations are separate from the requested splice-site register and junction annotations. Inspect the referenced observations with their RNA context and orientation. Resolved annotation ordinals do not attest an alternate biological name or measured activity.",
        },
        "requested_site_register": {
            "status": "recorded" if register_sources else "not_recorded",
            "source_record_count": len(register_sources), "source_fields": register_sources,
        },
        "independent_reference_junction_geometry": {
            "status": "recorded" if junction_sources else "not_recorded",
            "source_record_count": len(junction_sources), "source_fields": junction_sources,
        },
        "count_scope": "Source records with the indicated evidence, not distinct windows or biological events.",
        "interpretation": "Requested-site registration, independently annotated junction geometry, biological feature nomenclature and measured activity are separate. A missing register or unresolved biological label does not remove recorded reference geometry. Recorded geometry does not by itself resolve biological nomenclature, establish binding, or demonstrate selectivity or activity. Not recorded means absent from this inventory, not biologically absent.",
    }
    return {
        "input_row_id": row.get("input_row_id"),
        "exact_mapping_count": len(row.get("mapping_hits") or []),
        "retained_near_match_count": len(row.get("near_match_candidates") or []),
        "near_match_search": deepcopy(row.get("near_match_search") or
                                     (row.get("binding_evidence") or {}).get("near_match_search_scope") or {}),
        "retained_hypotheses": hypotheses,
        "retained_hypothesis_groups": list(groups.values()),
        "recorded_window_identity": window_identity,
        "feature_evidence_availability": availability,
        "scope": "Only this input row. Empty retained evidence is not evidence of absent binding or activity. Other panel rows cannot supply this molecule's site registers or alignments. An empty requested-site register is not evidence of being outside a requested feature. Reconcile actual reference junctions and block geometry separately from request-facing exon nomenclature.",
    }


def exact_gene_count_scope(config, target_gene_id, mapping_summary):
    """Describe the actual two-component native screening algorithm, not efficacy."""
    annotation = Path(config.gtf_path)
    genome = Path(config.genome_fasta)
    return {
        "metric": "distinct annotated non-target gene screening count; not a genome-wide binding or safety count",
        "count_unit": "unique version-stripped GTF gene IDs; union of genomic-exon and adjacent-junction hits",
        "assembly": config.ensembl_assembly,
        "genome_fasta": str(genome.resolve()),
        "annotation_gtf": str(annotation.resolve()),
        "annotation_bytes": annotation.stat().st_size if annotation.is_file() else None,
        "bowtie_index_prefix": str(config.bowtie_prefix),
        "genomic_search": "Bowtie 1: -a -f -v 0 -y -m 2000 --max <overmapped.fasta> --sam --sam-nohead --no-unal --seed 1; seed length equals shortest query length",
        "counted_genomic_rna_universe": "Only ungapped genomic hits wholly contained within an annotated GTF exon contribute genes. Intronic-only and intergenic hits do not contribute to this count.",
        "genomic_strand_policy": "Bowtie searches both genomic strands; exon-to-gene assignment does not require the SAM alignment strand to match the GTF transcript strand. This is a screening proxy, not a strand-specific antisense binding count.",
        "junction_search": "Enumerate windows spanning each pair of adjacent annotated exons in transcript order, using reference genome bases and GTF strand; require at least one query base on each side. Windows spanning more than one junction are not enumerated.",
        "junction_strand_policy": "Transcript-oriented junction sequence must exactly equal the recorded query sequence; the junction search does not additionally reverse-complement the query.",
        "mismatch_policy": "0 substitutions only; 1/2/3-mismatch searches not performed",
        "indel_policy": "ungapped exact matching; insertion/deletion alignments not searched",
        "intended_target_exclusion": "Exclude the entire target gene, including all its transcripts and loci, by version-stripped gene ID; not just the requested site or transcript.",
        "excluded_gene_id": str(target_gene_id or "").split('.')[0] or None,
        "exclusion_applied": bool(target_gene_id),
        "genomic_alignment_cap_per_query": mapping_summary.get("mapper_alignment_cap", 2000),
        "overmapped_query_count": mapping_summary.get("overmapped_queries"),
        "junction_search_summary": deepcopy(mapping_summary.get("junction_mapping") or {}),
        "limitations": "Does not establish mismatched, intronic/pre-mRNA, strand-specific or experimental selectivity. Same-gene isoform selectivity is outside this count. Inspect the per-row query sequence/role and completeness status before interpreting a zero.",
    }


def reference_context(config, genome, transcript_id, transcript_entry, requested_id, patient_fasta=None,
                      searched_cdna=False, searched_premrna=False):
    """Identify the loaded reference separately from a user's requested version."""
    filename = getattr(genome, "filename", None)
    loaded = Path(filename).resolve() if filename else None
    configured = Path(config.genome_fasta).resolve()
    metadata = (transcript_entry or {}).get("metadata") or {}
    versioned = metadata.get("versioned_id")
    if not versioned and metadata.get("version") is not None and transcript_id:
        versioned = f"{transcript_id.split('.')[0]}.{metadata['version']}"
    requested_id = requested_id or None
    version_status = "not_requested"
    if requested_id:
        version_status = ("verified" if requested_id == versioned else "mismatch" if versioned
                          else "unverified") if "." in requested_id else "stable_id_only_requested"
    index = Path(str(loaded) + ".fai") if loaded else None
    return {
        "reference_build": config.ensembl_assembly if loaded == configured else None,
        "reference_build_basis": "loaded_configured_genome" if loaded == configured else "unverified",
        "reference_fasta": str(loaded) if loaded else None,
        "reference_fasta_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest() if index and index.is_file() else None,
        "requested_transcript_id": requested_id,
        "resolved_transcript_id": versioned or transcript_id,
        "transcript_version_status": version_status,
        "transcript_reference_source": metadata.get("reference_source"),
        "transcript_reference_annotation": metadata.get("reference_annotation"),
        "coordinate_system": "1based_inclusive; genomic blocks in transcript order",
        "rna_contexts_searched": (["mature_spliced_cdna"] if searched_cdna else []) +
                                 (["unspliced_transcript_locus_pre_mrna"] if searched_premrna else []),
        "patient_cdna_reference_fasta": patient_fasta,
        "pre_mrna_reference": "configured_genome; patient substitution is not applied to locus mapping",
        "isoform_score_rna_context": "mature_spliced_cdna; does not test intronic pre_mrna binding",
    }


def annotate_binding_evidence(row, sites, preferred_transcript):
    """Report alignment and geometry without converting them into activity labels."""
    near = row.get("near_match_candidates") or []
    hits = row.get("mapping_hits") or []
    for match in [*hits, *near]:
        scope = match.get("mapping_scope") or "pre_mrna_locus"
        match["rna_context"] = "mature_spliced_cdna" if scope == "spliced_cdna" else "unspliced_transcript_locus_pre_mrna"
        overlaps = []
        # Do not label other isoforms as the requested transcript's site.
        if str(match.get("transcript_id") or "").split('.')[0] == str(preferred_transcript or "").split('.')[0]:
            segments = match.get("genomic_segments") or [match.get("genomic_span") or {}]
            for site in sites:
                kind = site.get("site_kind")
                anchor = site.get(kind) or {}
                position = anchor.get("position")
                if not isinstance(position, int):
                    continue
                if match.get("strand") and anchor.get("strand") and match["strand"] != anchor["strand"]:
                    continue
                if any(str(s.get("chrom", "")).removeprefix("chr") == str(anchor.get("chrom", "")).removeprefix("chr")
                       and isinstance(s.get("start"), int) and isinstance(s.get("end"), int)
                       and s["start"] <= position <= s["end"] for s in segments):
                    overlaps.append({"site_kind": kind, "position": position, "chrom": anchor.get("chrom"),
                                     "strand": anchor.get("strand"), "exon_index": site.get("exon_index"),
                                     "transcript_id": preferred_transcript, "coordinate_system": "1based_inclusive"})
        match["requested_site_overlaps"] = overlaps
    antisense = [m for m in near if m.get("match_orientation") == "reverse_complement"]
    row["binding_evidence"] = {
        "exact_mapping_status": row.get("mapping_status"),
        "antisense_near_match_hypotheses": len(antisense),
        "antisense_near_matches_overlapping_requested_site": [deepcopy(m) for m in antisense if m["requested_site_overlaps"]],
        "interpretation": "No exact match is an exact-search outcome, not evidence of no binding. Near-match overlap is geometric evidence for a conditional binding hypothesis, not measured affinity or activity. Missing perturbation scores do not test a mismatched molecule's inactivity.",
        "near_match_search_scope": deepcopy(row.get("near_match_search") or {
            "status": "not_attested", "reason": "No execution-scope record supplied; interpret only the listed hypotheses."}),
    }
    row["rna_context"] = "mature_spliced_cdna" if row.get("mapping_scope") == "spliced_cdna" else "unspliced_transcript_locus_pre_mrna" if row.get("mapping_scope") == "pre_mrna_locus" else "no_exact_mapping; see per-hypothesis RNA context"


def attach_measurement_references(summary):
    """Give the composer stable tokens; rendering copies the measured values."""
    evidence = {}
    for index, row in enumerate(summary.get("provided_asos") or []):
        position_evidence = chemical_position_evidence(row)
        if position_evidence is not None:
            row['chemical_position_evidence'] = position_evidence
        refs = {}
        for field, value in (row.get("splice_eval_summary") or {}).items():
            if value is None or value == "":
                continue
            ref = f"M{index + 1:03d}_{field}"
            evidence[ref] = {"name": row.get("name"), "input_row_id": row.get("input_row_id"), "field": "splice_eval_summary." + field, "value": deepcopy(value)}
            refs[field] = "{{" + ref + "}}"
        row["measurement_tokens"] = refs
        from .splice_prediction_records import splice_prediction_records
        prediction_records = splice_prediction_records(row)
        for record_index, record in enumerate(prediction_records):
            tokens = {}
            for field, value in record['numeric_measurements'].items():
                if record['source_field'] == 'splice_eval_summary':
                    tokens[field] = refs[field]
                    continue
                ref = f'M{index + 1:03d}_splice_record_{record_index:03d}_{field}'
                evidence[ref] = {
                    'name': row.get('name'), 'input_row_id': row.get('input_row_id'),
                    'field': record['source_field'] + '.' + field, 'value': value,
                    'recorded_identity': deepcopy(record['recorded_identity']),
                }
                tokens[field] = '{{' + ref + '}}'
            record['measurement_tokens'] = tokens
        row['recorded_splice_predictions'] = prediction_records
        reference_tokens = {}
        # Genomic coordinates are reference facts, not generated narrative.
        # Bind a whole interval so endpoints and the coordinate convention travel together.
        features = row.get("requested_feature_membership") or []
        for number, feature in enumerate(features, 1):
            if not all(feature.get(k) is not None for k in ("chrom", "start", "end", "coordinate_system")):
                continue
            ref = f"R{index + 1:03d}_feature_{number}"
            evidence[ref] = {"name": row.get("name"), "input_row_id": row.get("input_row_id"), "field": f"requested_feature_membership[{number - 1}]",
                             "value": {k: deepcopy(feature[k]) for k in ("feature", "label", "transcript_id", "evaluated_transcript_id", "requested_exon_label", "exon_index", "requested_feature_identity_resolved", "identity_resolution", "chrom", "start", "end", "coordinate_system") if k in feature}}
            reference_tokens[f"feature_{number}"] = "{{" + ref + "}}"
        row["reference_tokens"] = reference_tokens
        row["evidence_inventory"] = row_evidence_inventory(row)
        for number, hypothesis in enumerate(row["evidence_inventory"]["retained_hypotheses"], 1):
            ref = f"R{index + 1:03d}_alignment_{number}"
            evidence[ref] = {
                "name": row.get("name"), "input_row_id": row.get("input_row_id"),
                "field": hypothesis["source_field"], "value": deepcopy(hypothesis["geometry"]),
                "window_identity_status": hypothesis["window_identity_status"],
                "row_local_window_group_id": hypothesis.get("window_group_id"),
            }
            token = "{{" + ref + "}}"
            reference_tokens[f"alignment_{number}"] = token
            hypothesis["geometry_reference_token"] = token
        ref = f"R{index + 1:03d}_evidence_inventory"
        evidence[ref] = {"name": row.get("name"), "input_row_id": row.get("input_row_id"),
                         "field": "evidence_inventory", "value": deepcopy(row["evidence_inventory"])}
        reference_tokens["evidence_inventory"] = "{{" + ref + "}}"
        groups = {}
        for match in row.get('near_match_candidates') or []:
            context = match.get('rna_context') or ('mature_spliced_cdna' if match.get('mapping_scope') == 'spliced_cdna' else 'unspliced_transcript_locus_pre_mrna')
            orientation = match.get('match_orientation') or 'unknown'
            count = match.get('mismatch_count')
            if isinstance(count, int) and not isinstance(count, bool):
                groups.setdefault((context, orientation), []).append(count)
        measurements = []
        for (context, orientation), counts in sorted(groups.items()):
            value = {'rna_context':context, 'match_orientation':orientation,
                     'minimum_recorded_mismatches':min(counts), 'maximum_recorded_mismatches':max(counts),
                     'scope':'Recorded retained hypotheses only; not a genome-wide bound or an activity threshold.'}
            measurements.append(value)
            key = f'near_match_{context}_{orientation}'
            ref = f'R{index + 1:03d}_{key}'
            evidence[ref] = {'name':row.get('name'), 'input_row_id':row.get('input_row_id'), 'field':key, 'value':value}
            reference_tokens[key] = '{{' + ref + '}}'
        row['near_match_measurements'] = measurements
        orientation = row.get("orientation_interpretation") or {}
        confirmation = orientation.get("sequence_role_confirmation") or {}
        if confirmation.get("role") in {"antisense_oligo_sequence", "target_site_sequence"}:
            row["resolved_sequence_identity"] = {
                "status": "user_confirmed", "role": confirmation["role"],
                "evaluated_molecule_sequence": orientation.get("evaluated_molecule_sequence"),
                "interpretation": "The requester has answered the identity question. Assess this confirmed molecule; sequence-role clarification is no longer pending. Biological binding or activity may remain uncertain."}
    summary["measurement_evidence"] = evidence
    summary["assessment_delivery"] = "Return exactly one assessment per supplied input_row_id, preserving its name. Names and sequences may repeat across distinct observations; never merge them. input_reconciliation records redundant parser aliases, not additional user observations. Use measurement_tokens for numeric perturbation results and reference_tokens for feature coordinates and mismatch summaries; do not transcribe these values. Use a retained hypothesis's geometry_reference_token for its genomic segment count and coordinates; never independently write a block count such as single genomic block or two blocks. The token binds that exact alignment's transcript, RNA context, orientation and segment list together. Missing segments leave their count unknown; a bounding span does not prove continuity. Keep mature-RNA and pre-mRNA hypotheses separate. Honor resolved_sequence_identity independently of uncertainty about biological activity. Per-row assessments are the single source of narrative and ranking. Leave markdown empty; the application renders the same assessments in the CSV and report."
    return summary


def assessment_measurement_errors(response, summary):
    """Validate every citation before delivery, including report-level fields.

    The composer can repair serialization inside its existing bounded contract
    loop. Neither the validator nor renderer guesses a missing evidence token,
    borrows another observation's measurement, or changes a scientific judgment.
    """
    evidence = summary.get("measurement_evidence") or {}
    resolve = assessment_row_lookup(summary.get("provided_asos") or [])
    errors = []

    def inspect_assessment(assessment, path):
        if not isinstance(assessment, dict):
            errors.append(path + ": Expected an evaluation assessment object.")
            return
        row = resolve(assessment)
        if row is None:
            errors.append(path + ": Unknown or ambiguous evaluation input row; supply its exact input_row_id and name.")
            return
        for ref in re.findall(r"\{\{([^{}]+)\}\}", json.dumps(assessment, ensure_ascii=False)):
            if ref in evidence:
                item = evidence[ref]
                if item["name"] != row.get("name") or item.get("input_row_id") != row.get("input_row_id"):
                    errors.append(path + ": Measurement reference belongs to a different molecule/input row: " + ref)
        for field in ("mechanistic_plausibility", "ranking_rationale", "uncertainty"):
            text = str(assessment.get(field) or "")
            if re.search(r"(?:delta_[a-z_]+|perturbation(?: score)?)\s*[=:]\s*[−+\-]?\d", text, re.I):
                errors.append(path + "." + field + ": Use a measurement reference instead of a numerical transcription: " + str(assessment.get("name")))

    def inspect(value, path="$"):
        if isinstance(value, dict):
            for key, item in value.items():
                child = path + "." + key
                if key == "evaluation_assessments" and isinstance(item, list):
                    for index, assessment in enumerate(item):
                        inspect_assessment(assessment, child + "[" + str(index) + "]")
                inspect(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, path + "[" + str(index) + "]")
        elif isinstance(value, str):
            for ref in re.findall(r"\{\{([^{}]+)\}\}", value):
                if ref not in evidence:
                    errors.append(path + ": Unknown evaluation measurement reference: " + ref +
                                  ". Cite an exact supplied token, or express non-citation prose without template delimiters; never invent a reference.")
    inspect(response)
    return list(dict.fromkeys(errors))


def bind_assessment_measurements(response, summary):
    """Resolve references once in every field; invalid references fail visibly."""
    errors = assessment_measurement_errors(response, summary)
    if errors:
        raise ValueError("; ".join(errors))
    evidence = summary.get("measurement_evidence") or {}
    used = set()
    resolve = assessment_row_lookup(summary.get("provided_asos") or [])

    def bind(value):
        if isinstance(value, dict):
            return {k: bind(v) for k, v in value.items()}
        if isinstance(value, list):
            return [bind(v) for v in value]
        if not isinstance(value, str):
            return value
        def replace(match):
            ref = match.group(1)
            if ref not in evidence:
                raise ValueError("Unknown evaluation measurement reference: " + ref)
            used.add(ref)
            from .evaluation_report import citation_display
            return citation_display(ref, evidence[ref])
        return re.sub(r"\{\{([^{}]+)\}\}", replace, value)

    result = bind(deepcopy(response))
    result["measurement_citations"] = {key: evidence[key] for key in sorted(used)}
    result["original_unbound_response"] = deepcopy(response)
    from .evaluation_report import evaluation_markdown
    result["markdown"] = evaluation_markdown(result, summary)
    return result


def unsupported_measurement_reference_findings(response, source):
    """Locate unsupported citation syntax in assessment prose for scoped repair.

    Each finding is bound to the exact draft substring and the source row. This
    does not rewrite a citation, infer a value, or accept an unsupported claim.
    """
    from .evaluation_response_audit import narrative_fields
    from .citation_tokens import citation_placeholders
    summary = source.get('evaluation_summary') if isinstance(source, dict) else None
    if not isinstance(summary, dict):
        return []
    evidence = summary.get('measurement_evidence') or {}
    source_rows = summary.get('provided_asos') or []
    assessments = response.get('evaluation_assessments') if isinstance(response, dict) else None
    if not isinstance(assessments, list) or not isinstance(source_rows, list):
        return []
    by_id = {row.get('input_row_id'): index for index, row in enumerate(source_rows)
             if isinstance(row, dict) and row.get('input_row_id')}
    findings = []
    for path, prose in narrative_fields(response).items():
        parts = path.split('/')
        if len(parts) < 4 or parts[1] != 'evaluation_assessments' or not parts[2].isdigit():
            continue
        index = int(parts[2])
        if index >= len(assessments) or not isinstance(assessments[index], dict):
            continue
        row_id = assessments[index].get('input_row_id')
        if row_id not in by_id:
            continue
        for reference in citation_placeholders(prose):
            if reference in evidence:
                continue
            findings.append({
                'response_path': path,
                'quote': '{{' + reference + '}}',
                'kind': 'unsupported_factual_claim',
                'evidence_paths': ['/evaluation_summary/provided_asos/' + str(by_id[row_id])],
                'reason': 'This is not an available evidence token. Correct only this field using an exact supplied token or evidence-scoped prose; preserve the other assessments.',
            })
    return findings
