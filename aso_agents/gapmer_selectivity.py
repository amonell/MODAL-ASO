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
    aliases = {str(value).strip().casefold() for value in
            (label, entry.get('transcript_id'), entry.get('id'), entry.get('versioned_id'),
             meta.get('id'), meta.get('versioned_id')) if value}
    for record in (entry, meta):
        version = record.get('version')
        if isinstance(version, bool) or not str(version or '').isdecimal():
            continue
        for field in ('id', 'transcript_id'):
            identity = str(record.get(field) or '').strip().casefold()
            if identity and not _version(identity):
                aliases.add(f'{identity}.{version}')
    return aliases


def _version(identity):
    base, dot, version = identity.rpartition('.')
    return (base, version) if dot and base and version.isdecimal() else None


def _versions(aliases):
    versions = {}
    for alias in aliases:
        pair = _version(alias)
        if pair:
            versions.setdefault(pair[0], set()).add(pair[1])
    return versions


def _compatible_identity(left, right):
    """A shared stable accession cannot erase an explicit version conflict."""
    versions = [_versions(aliases) for aliases in (left, right)]
    if any(len(values) != 1 for group in versions for values in group.values()):
        return False
    return all(versions[0][key] == versions[1][key]
               for key in versions[0].keys() & versions[1].keys())


def _reference_context_compatible(left, right):
    from .species_config import normalize_species
    for field in ('species', 'assembly_name'):
        groups = []
        for entry in (left, right):
            values = {str(value) for value in
                      (entry.get(field), (entry.get('metadata') or {}).get(field)) if value}
            if field == 'species':
                values = {normalize_species(value) for value in values}
            groups.append(values)
        if any(len(values) > 1 for values in groups) or (all(groups) and groups[0] != groups[1]):
            return False
    return True


def _complete_structure(entry):
    exons = entry.get('exons_tx') or []
    chrom = entry.get('chrom') or (entry.get('metadata') or {}).get('seq_region_name')
    strand = entry.get('strand')
    if not chrom or str(strand) not in {'+', '-', '1', '-1'} or not exons:
        return None
    if any(not isinstance(e, dict) or type(e.get('start')) is not int
           or type(e.get('end')) is not int or e['start'] < 1 or e['end'] < e['start'] for e in exons):
        return None
    return _structure(entry)


def _structure(entry):
    return (str(entry.get('chrom') or (entry.get('metadata') or {}).get('seq_region_name') or '').removeprefix('chr'),
            '-' if str(entry.get('strand')) in {'-', '-1'} else '+',
            tuple((e['start'], e['end']) for e in entry.get('exons_tx') or []))


def resolve_catalog_label(label, catalog, resolved):
    """Join a declared alias through its attested structure; never strip versions."""
    identity = str(label or '').strip().casefold()
    if not identity:
        return None
    sources = [(key, entry) for key, entry in resolved.items()
               if identity in _aliases(key, entry)]
    if len(sources) > 1 or any(not _compatible_identity(_aliases(key, entry), set())
                               for key, entry in sources):
        return None
    direct = [(key, entry) for key, entry in catalog.items() if identity in _aliases(key, entry)]
    if direct:
        if len(direct) != 1:
            return None
        key, entry = direct[0]
        aliases = _aliases(key, entry)
        if not _compatible_identity(aliases, set()):
            return None
        for source_key, source in sources:
            if (not _compatible_identity(aliases, _aliases(source_key, source))
                    or not _reference_context_compatible(entry, source)):
                return None
            structures = [_complete_structure(record) for record in (entry, source)]
            if all(structures) and structures[0] != structures[1]:
                return None
        return key
    matches = set()
    for key, entry in sources:
        source_ids = _aliases('', entry)
        for target, other in catalog.items():
            other_ids = _aliases('', other)
            structure = _complete_structure(entry)
            if (source_ids & other_ids and structure and structure == _complete_structure(other)
                    and _compatible_identity(source_ids, _aliases(target, other))
                    and _reference_context_compatible(entry, other)
                    and all(base in _versions(other_ids) for base in _versions(source_ids))):
                matches.add(target)
    return next(iter(matches)) if len(matches) == 1 else None


def resolve_uniqueness_roles(target, beneficial, pathogenic, catalog, resolved):
    """Keep sequence hits usable even when an auxiliary role cannot be assigned."""
    result = {}
    for role, labels in (('target', [target] if target else []),
                         ('beneficial', beneficial or []), ('pathogenic', pathogenic or [])):
        bindings = []
        for label in dict.fromkeys(labels):
            match = resolve_catalog_label(label, catalog, resolved)
            bindings.append({'requested_label': label, 'catalog_label': match,
                             'status': 'resolved' if match is not None else 'unresolved'})
        result[role] = {'status': 'unresolved' if any(b['status'] == 'unresolved' for b in bindings) else 'resolved',
                        'bindings': bindings,
                        'catalog_labels': sorted({b['catalog_label'] for b in bindings if b['catalog_label'] is not None})}
    result['status'] = 'unresolved' if any(v['status'] == 'unresolved' for v in result.values()) else 'resolved'
    result['basis'] = 'Unique catalog identity or attested reference aliases and matching transcript structure; versions are never stripped.'
    return result


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


def _ungapped_match_profile(target, transcript, architecture):
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
    histogram = np.bincount(counts, minlength=len(target) + 1)
    return {'minimum_ungapped_mismatches': minimum, 'closest_sites': sites,
            'ungapped_mismatch_site_counts': {str(n): int(count) for n, count in enumerate(histogram)},
            'ungapped_windows_searched': int(counts.size),
            'target_length_nt': len(target), 'transcript_length_nt': len(transcript)}


def _closest_matches(target, transcript, architecture):
    """Retain the closest-site interface for callers that do not need counts."""
    profile = _ungapped_match_profile(target, transcript, architecture)
    return profile['minimum_ungapped_mismatches'], profile['closest_sites']


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
            profile = _ungapped_match_profile(target, transcript, architecture)
            references.append({'transcript_label': label,
                'transcript_sequence_sha256': hashlib.sha256(transcript.encode()).hexdigest(),
                **profile})
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
            'method': 'Exhaustive ungapped full-length comparison of the target-oriented complement to each protected mature transcript; all equally closest sites retained. The mismatch histogram counts every full-length window at a distinct transcript start position, including overlapping sites, separately for each exact substitution count from zero through the oligo length. These are transcript-site counts, not gene counts or cumulative mismatch thresholds. Mismatch positions use the actual synthesized ASO 5prime-to-3prime orientation.',
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
