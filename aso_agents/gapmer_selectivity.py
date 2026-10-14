"""Reference-bound junction scope and protected-transcript gapmer evidence.

No request prose or biological names are interpreted here. Inputs are typed
objectives, resolved transcript structures and the actual synthesized molecule.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from aso_chemistry import encode_idt_oligo, gapmer_architecture
from .preservation_context import preserved_catalog_labels, preservation_resolution_objectives


def _aliases(label, entry):
    meta = entry.get('metadata') or {}
    return {str(value).strip().casefold() for value in
            (label, entry.get('transcript_id'), entry.get('id'), entry.get('versioned_id'),
             meta.get('id'), meta.get('versioned_id')) if value}


def _structure(entry):
    return (str(entry.get('chrom') or (entry.get('metadata') or {}).get('seq_region_name') or '').removeprefix('chr'),
            '-' if str(entry.get('strand')) in {'-', '-1'} else '+',
            tuple((e['start'], e['end']) for e in entry.get('exons_tx') or []))


def resolve_catalog_label(label, catalog, resolved):
    """Join a declared alias through its attested structure; never strip versions."""
    identity = str(label or '').strip().casefold()
    if not identity:
        return None
    direct = [key for key, entry in catalog.items() if identity in _aliases(key, entry)]
    if len(direct) == 1:
        return direct[0]
    sources = [(key, entry) for key, entry in resolved.items()
               if identity in _aliases(key, entry)]
    matches = set()
    for key, entry in sources:
        source_ids = _aliases('', entry)
        for target, other in catalog.items():
            if source_ids & _aliases('', other) and _structure(entry) == _structure(other):
                # A shared stable accession does not overrule conflicting versions.
                versions = [{a for a in _aliases('', x) if a.rpartition('.')[2].isdecimal()
                             and a.rpartition('.')[0]} for x in (entry, other)]
                if all(versions) and not versions[0] & versions[1]:
                    continue
                matches.add(target)
    return next(iter(matches)) if len(matches) == 1 else None


def bind_uniqueness_context(context, catalog, resolved):
    bound = deepcopy(context)
    for field in ('target_labels', 'beneficial_labels', 'pathogenic_labels'):
        labels = list(context.get(field) or [])
        if field == 'target_labels' and context.get('target_label'):
            labels.append(context['target_label'])
        matches = []
        for label in dict.fromkeys(labels):
            match = resolve_catalog_label(label, catalog, resolved)
            if match is None:
                raise ValueError('isoform_sequence_catalog_identity_unresolved: ' + str(label))
            matches.append(match)
        bound[field] = list(dict.fromkeys(matches))
    bound['target_label'] = next(iter(bound['target_labels']), None)
    return bound


def _ordinal(entry, requested):
    mappings = entry.get('exon_nomenclature_resolutions') or (entry.get('metadata') or {}).get('exon_nomenclature_resolutions') or {}
    mapping = mappings.get(str(requested)) or {}
    if mapping.get('status') == 'verified_alternate_mapping':
        return mapping['resolved_transcript_ordinal']
    return requested


def explicit_junction_region(goal, program, resolved, primary):
    """Materialize two typed endpoints as one adjacent mature-RNA junction.

    When target ordinals cannot form the requested junction, a resolved protected
    isoform can anchor the exon labels: both exon intervals must project exactly
    to consecutive target exons, with a unique result. Record this comparative
    interpretation explicitly; ambiguous or unavailable mappings fail closed.
    """
    pair = program.get('exon_labels')
    is_junction = program.get('target_type') == 'splice_junction'
    is_junction |= bool((goal.get('junction_gapmers') or {}).get('enabled') and pair)
    if not is_junction or not pair:
        return None
    if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int or x < 1 for x in pair):
        return {'error': 'explicit_junction_requires_two_ordered_exon_labels'}
    requests = [program.get(k) for k in ('isoform_label', 'transcript_id') if program.get(k)]
    matches = {resolve_catalog_label(value, resolved, resolved) for value in requests}
    if not requests:
        matches = {primary}
    if len(matches) != 1 or None in matches or next(iter(matches)) not in resolved:
        return {'error': 'explicit_junction_reference_unresolved'}
    label = next(iter(matches)); entry = resolved[label]
    exons = entry.get('exons_tx') or []
    indices = [e.get('index') for e in exons]
    candidates = {}

    def record(left, right, basis, reference):
        if left not in indices or right not in indices:
            return
        position = indices.index(left)
        if position + 1 >= len(indices) or indices[position + 1] != right:
            return
        boundary = sum(e['end'] - e['start'] + 1 for e in exons[:position + 1])
        candidates.setdefault(boundary, []).append({'basis': basis, 'reference_label': reference,
            'requested_exon_labels': pair, 'target_exon_ordinals': [left, right]})

    left, right = [_ordinal(entry, x) for x in pair]
    record(left, right, 'target_transcript_exon_order', label)
    if not candidates:
        for ref_label in preserved_catalog_labels(goal, resolved):
            reference = resolved[ref_label]
            if _structure(entry)[:2] != _structure(reference)[:2]:
                continue
            endpoints = []
            for requested in pair:
                refs = [e for e in reference.get('exons_tx') or [] if e.get('index') == _ordinal(reference, requested)]
                projected = [e.get('index') for e in exons if len(refs) == 1
                             and (e['start'], e['end']) == (refs[0]['start'], refs[0]['end'])]
                endpoints.append(projected[0] if len(projected) == 1 else None)
            record(*endpoints, 'protected_reference_exon_intervals_projected_to_target', ref_label)
    if len(candidates) != 1:
        return {'error': 'explicit_junction_mapping_unresolved_or_ambiguous',
                'requested_exon_labels': pair, 'candidate_boundaries': sorted(candidates)}
    boundary, provenance = next(iter(candidates.items()))
    span = (goal.get('junction_gapmers') or {}).get('span_min_each_side') or 1
    if type(span) is not int or span < 1:
        return {'error': 'junction_span_invalid'}
    transcript = entry.get('transcript_id') or label
    strand = _structure(entry)[1]
    return {'rows': [[transcript, boundary - span, boundary + span,
                      program.get('label') or program.get('program_id') or 'junction', 0, strand]],
            'sources': ['explicit_junction'], 'tile_overlap_requirement': 'contains_feature',
            'match_mode': 'junction_crossing', 'isoform_label': label,
            'junction_boundary_cdna': boundary, 'junction_mapping': provenance}


def _closest_matches(target, transcript, architecture):
    if len(transcript) < len(target) or set(transcript) - set('ACGT'):
        raise ValueError('protected_transcript_sequence_incomplete')
    bases = np.frombuffer(transcript.encode('ascii'), dtype=np.uint8)
    windows = np.lib.stride_tricks.sliding_window_view(bases, len(target))
    mismatches = windows != np.frombuffer(target.encode('ascii'), dtype=np.uint8)
    counts = mismatches.sum(axis=1)
    minimum = int(counts.min())
    sites = []
    for offset in np.flatnonzero(counts == minimum):
        positions = sorted(len(target) - int(p) for p in np.flatnonzero(mismatches[offset]))
        sites.append({'transcript_cdna_start_1based': int(offset) + 1,
                      'transcript_cdna_end_1based': int(offset) + len(target),
                      'target_5to3': transcript[int(offset):int(offset) + len(target)],
                      'mismatch_positions_aso_1based': positions,
                      'dna_gap_mismatch_positions_aso_1based': [p for p in positions if architecture.in_gap(p)],
                      'mismatch_count': minimum})
    return minimum, sites


def annotate_preservation(path, goal, context, cdna_map, chemistry, *, catalog=None, resolved=None):
    """Separate sequence evidence from a conservative design-selection policy."""
    required = bool(preservation_resolution_objectives(goal))
    if not required:
        return {'status': 'not_required'}
    if catalog is not None and resolved is not None:
        protected = []
        for objective in preservation_resolution_objectives(goal):
            label = resolve_catalog_label(objective['isoform_label'], catalog, resolved)
            if label is None:
                raise ValueError('gapmer_protected_transcript_identity_unresolved')
            protected.append(label)
        protected = list(dict.fromkeys(protected))
    else:
        # Without structures, accept exact declared identities only. A beneficial
        # splice objective is not automatically a preservation constraint.
        requested = {objective['isoform_label'] for objective in preservation_resolution_objectives(goal)}
        protected = sorted(requested.intersection(context.get('beneficial_labels') or []))
    if not protected:
        raise ValueError('gapmer_protected_transcript_identity_unresolved')
    payload = json.loads(Path(path).read_text())
    for row in payload.get('rows') or []:
        target = str(row.get('Sequence') or row.get('sequence') or '').upper()
        antisense = target.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        architecture = gapmer_architecture(chemistry)
        if (not architecture or architecture.length != len(target)
                or set(target) - set('ACGT') or not target
                or str(row.get('Synthesis') or '').upper() != antisense
                or row.get('IDT') != encode_idt_oligo(antisense, chemistry)):
            raise ValueError('gapmer_preservation_physical_chemistry_unverified')
        references = []
        for label in protected:
            transcript = str(cdna_map.get(label) or '').upper()
            minimum, sites = _closest_matches(target, transcript, architecture)
            references.append({'transcript_label': label,
                'transcript_sequence_sha256': hashlib.sha256(transcript.encode()).hexdigest(),
                'minimum_ungapped_mismatches': minimum, 'closest_sites': sites})
        exact = any(ref['minimum_ungapped_mismatches'] == 0 for ref in references)
        intact_gap = any(not site['dna_gap_mismatch_positions_aso_1based']
                         for ref in references for site in ref['closest_sites'])
        status = 'exact_preserved_transcript_match' if exact else (
            'closest_preserved_match_has_intact_dna_gap' if intact_gap else 'sequence_discrimination_supported')
        row['gapmer_preservation_assessment'] = {
            'status': status, 'recommended_for_selective_panel': not (exact or intact_gap),
            'chemistry': chemistry, 'dna_gap_positions_aso_1based':
                [architecture.wing_5prime + 1, architecture.wing_5prime + architecture.dna_gap],
            'protected_references': references,
            'method': 'Exhaustive ungapped full-length comparison of the target-oriented complement to each protected mature transcript; all equally closest sites retained. Mismatch positions use the actual synthesized ASO 5prime-to-3prime orientation.',
            'interpretation': 'Conservative computational shortlist: exclude perfect protected-transcript complements and candidates whose closest protected alignment leaves the entire DNA gap paired. Gap mismatches support sequence discrimination but do not establish RNase H discrimination, cellular selectivity, efficacy or safety. Bulged alignments, kinetics and experimental activity are not evaluated.'}
    payload.setdefault('schema', {}).setdefault('columns', {})['gapmer_preservation_assessment'] = 'object'
    Path(path).write_text(json.dumps(payload, indent=2) + '\n')
    return {'status': 'complete', 'protected_labels': protected, 'rows_assessed': len(payload.get('rows') or [])}


def filter_preservation(path):
    payload = json.loads(Path(path).read_text())
    rows = payload.get('rows') or []
    assessed = [row for row in rows if 'gapmer_preservation_assessment' in row]
    if not assessed:
        return {'status': 'not_required', 'output_path': str(path)}
    if len(assessed) != len(rows):
        raise ValueError('gapmer_preservation_assessment_incomplete')
    kept, excluded = [], []
    for row in rows:
        assessment = row['gapmer_preservation_assessment']
        decision = assessment.get('recommended_for_selective_panel')
        if type(decision) is not bool:
            raise ValueError('gapmer_preservation_decision_missing')
        (kept if decision else excluded).append(row)
    source = Path(path)
    output = source.with_name(source.stem + '.preservation_selected.json')
    rejected = source.with_name(source.stem + '.preservation_excluded.json')
    for destination, selected in ((output, kept), (rejected, excluded)):
        destination.write_text(json.dumps({**payload, 'rows': selected}, indent=2) + '\n')
    return {'status': 'complete', 'input_path': str(source), 'output_path': str(output),
            'excluded_path': str(rejected), 'checked_count': len(rows),
            'passing_count': len(kept), 'failed_count': len(excluded),
            'policy': 'Preserve all evidence; only sequence-discriminating gapmers enter the recommended panel.'}
