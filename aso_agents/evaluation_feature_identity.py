"""Resolve requested feature labels independently of supplied ASO alignments."""
from copy import deepcopy
import json


def transcript_identifier(entry, fallback=''):
    metadata = entry.get('metadata') or {}
    own_id = str(entry.get('transcript_id') or entry.get('id') or fallback)
    if (entry.get('explicit_target_modified') or metadata.get('explicit_target_modified')
            or '::' in own_id):
        return own_id or str(metadata.get('id') or '')
    value = str(metadata.get('versioned_id') or entry.get('versioned_id')
                or entry.get('transcript_id') or entry.get('id') or fallback)
    if '.' not in value and metadata.get('version') is not None:
        value += '.' + str(metadata['version'])
    return value


def transcript_matches(requested, actual):
    if not requested or not actual:
        return False
    requested_base, separator, requested_model = requested.partition('::')
    actual_base, actual_separator, actual_model = actual.partition('::')
    if (separator, requested_model) != (actual_separator, actual_model):
        return False
    head, dot, version = requested_base.rpartition('.')
    if dot and version.isdigit():
        return requested_base == actual_base
    actual_head, actual_dot, actual_version = actual_base.rpartition('.')
    return requested_base == (actual_head if actual_dot and actual_version.isdigit() else actual_base)


def _positive_ordinal(value):
    if isinstance(value, bool):
        return None
    text = str(value or '').strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def is_resolved_feature_identity(identity):
    """Reference ordinals and alternate biological names retain distinct provenance."""
    return isinstance(identity, dict) and identity.get('status') in {
        'explicit_transcript_ordinal', 'verified_biological_label', 'resolved_reference_ordinal',
    }


def _source_backed_ordinal_default(resolution, label):
    if not isinstance(resolution, dict):
        return False
    if (resolution.get('status') != 'ordinal_default'
            or resolution.get('source') != 'transcript_exon_index'
            or _positive_ordinal(resolution.get('requested_exon_label')) != label):
        return False
    refs = resolution.get('evidence_refs')
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) and ref for ref in refs):
        return False
    # Native source bindings follow a compatible transcript lookup. A failed
    # lookup or a cached annotation guess must not impersonate that result.
    return all(isinstance(resolution.get(key), str)
               and len(resolution[key]) == 64
               and all(char in '0123456789abcdef' for char in resolution[key].lower())
               for key in ('source_sha256', 'xref_sha256'))


def resolve_evaluation_target_identities(targets, catalog, default_transcript_id, resolver=None):
    """Attach explicit identity provenance; never choose a feature from an ASO hit.

    Biological aliases need an attested nomenclature relation. A source-backed
    ordinary numbering resolution identifies the named reference ordinal; failed
    or unattempted lookups remain provisional.
    An explicit exon_index denotes a transcript ordinal. Versioned reference
    requests are never satisfied by a different version.
    """
    result = []
    for raw in targets:
        target = deepcopy(raw)
        if isinstance(target.get('splice_relative_window'), dict):
            result.append(_resolve_relative_window(target, catalog, default_transcript_id, resolver))
            continue
        label = target.get('exon_label')
        ordinal = _positive_ordinal(target.get('exon_index'))
        if label is None and ordinal is None:
            result.append(target)
            continue
        reference = str(target.get('exon_numbering_reference_transcript_id')
                        or target.get('exon_numbering_transcript_id')
                        or target.get('reference_transcript_id')
                        or target.get('transcript_id') or default_transcript_id or '')
        matches = {}
        for key, entry in catalog.items():
            if not isinstance(entry, dict):
                continue
            actual = transcript_identifier(entry, key)
            if (reference == key and '.' not in reference) or transcript_matches(reference, actual):
                signature = json.dumps([actual, entry.get('chrom'), entry.get('strand'),
                    entry.get('exons_tx')], sort_keys=True)
                matches[signature] = (actual, entry)
        identity = {'requested_exon_label': label, 'requested_reference_transcript_id': reference,
                    'status': 'unresolved', 'biological_label_verified': False}
        if len(matches) != 1:
            identity['reason'] = 'The requested numbering reference is unavailable or ambiguous.'
        else:
            actual, entry = next(iter(matches.values()))
            identity['reference_transcript_id'] = actual
            resolution = None
            if ordinal is not None:
                identity.update(status='explicit_transcript_ordinal', resolved_transcript_ordinal=ordinal,
                    reason='The request specifies an exon_index; this is a transcript annotation ordinal.')
            elif _positive_ordinal(label) is not None:
                metadata = entry.get('metadata') or {}
                resolution = deepcopy((entry.get('exon_nomenclature_resolutions')
                    or metadata.get('exon_nomenclature_resolutions') or {}).get(str(label)))
                if resolution is None and callable(resolver):
                    resolution = resolver(entry, _positive_ordinal(label))
                identity['nomenclature_evidence'] = deepcopy(resolution)
                resolved = _positive_ordinal((resolution or {}).get('resolved_transcript_ordinal'))
                if (resolution or {}).get('status') == 'verified_alternate_mapping' and resolved:
                    identity.update(status='verified_biological_label', biological_label_verified=True,
                                    resolved_transcript_ordinal=resolved)
                    ordinal = resolved
                elif _source_backed_ordinal_default(resolution, _positive_ordinal(label)):
                    ordinal = _positive_ordinal(label)
                    identity.update(status='resolved_reference_ordinal',
                        resolved_transcript_ordinal=ordinal,
                        numbering_convention='one_based_transcript_ordinal',
                        reason='The source-backed numbering resolution uses ordinary one-based exon order in the requested reference transcript. This identifies the reference feature without claiming an alternate biological nomenclature mapping or activity.')
                else:
                    # Preserve an explicit annotation-order hypothesis without
                    # presenting it as the requested biological feature.
                    ordinal = _positive_ordinal(label)
                    identity.update(status='annotation_ordinal_hypothesis', resolved_transcript_ordinal=ordinal,
                        reason='Biological exon-label identity is unresolved. This candidate uses the transcript annotation ordinal and must not establish requested-feature membership.')
            if ordinal is not None:
                exons = [e for e in entry.get('exons_tx') or []
                         if isinstance(e, dict) and _positive_ordinal(e.get('index')) == ordinal]
                if len(exons) == 1:
                    exon = exons[0]
                    ordinals = {_positive_ordinal(e.get('index')) for e in entry.get('exons_tx') or []
                                if isinstance(e, dict)} - {None}
                    identity.update(chrom=entry.get('chrom') or (entry.get('metadata') or {}).get('seq_region_name'),
                                    start=exon.get('start'), end=exon.get('end'),
                                    strand='-' if str(entry.get('strand')) in {'-', '-1'} else '+',
                                    coordinate_system='1based_inclusive')
                    identity['splice_boundaries'] = {
                        'acceptor': ordinal != min(ordinals), 'donor': ordinal != max(ordinals),
                    }
                else:
                    identity.update(status='unresolved', biological_label_verified=False,
                                    reason='The resolved ordinal is unavailable or ambiguous in the selected reference.')
        target['exon_identity_resolution'] = identity
        result.append(target)
    return result


def _resolve_relative_window(target, catalog, default_transcript_id, resolver):
    """Materialize a typed splice-relative feature without inferring prose or hits."""
    window = target['splice_relative_window']
    resolution = {'status': 'unresolved', 'requested_window': deepcopy(window)}
    target['relative_window_resolution'] = resolution
    site = window.get('anchor_site')
    offsets = [window.get('relative_start_nt'), window.get('relative_end_nt')]
    if (site not in {'acceptor', 'donor'} or
            not all(type(value) is int for value in offsets) or offsets[0] > offsets[1]):
        resolution['reason'] = 'A relative window requires one splice anchor and ordered signed integer endpoints.'
        return target
    # Nested identity is explicit; no sibling target or ASO mapping can supply it.
    identity_source = window if any(window.get(key) is not None for key in ('exon_index', 'exon_label')) else target
    anchor = {key: deepcopy(identity_source.get(key)) for key in ('exon_index', 'exon_label')}
    anchor['reference_transcript_id'] = window.get('reference_transcript_id') or target.get('reference_transcript_id')
    anchor['transcript_id'] = target.get('transcript_id') or default_transcript_id
    resolved_anchor = resolve_evaluation_target_identities([anchor], catalog, default_transcript_id, resolver)[0]
    identity = resolved_anchor.get('exon_identity_resolution') or {}
    resolution['anchor_identity_resolution'] = identity
    if not is_resolved_feature_identity(identity):
        resolution['reason'] = 'The relative-window exon anchor is unavailable or unresolved.'
        return target
    if not (identity.get('splice_boundaries') or {}).get(site):
        resolution['reason'] = 'The requested exon end is a transcript terminus, not an annotated splice boundary.'
        return target
    if (not identity.get('chrom') or identity.get('strand') not in {'+', '-'} or
            not all(type(identity.get(key)) is int for key in ('start', 'end'))):
        resolution['reason'] = 'The resolved exon anchor lacks complete genomic geometry.'
        return target
    negative = identity['strand'] == '-'
    position = ((identity['start'] if negative else identity['end']) if site == 'donor'
                else (identity['end'] if negative else identity['start']))
    endpoints = [position + (-1 if negative else 1) * value for value in offsets]
    geometry = {'chrom': identity['chrom'], 'strand': identity['strand'],
                'start': min(endpoints), 'end': max(endpoints), 'coordinate_system': '1based_inclusive'}
    if geometry['start'] < 1:
        resolution['reason'] = 'The requested window extends below genomic coordinate one; it was not clipped.'
        return target
    for key in ('chrom', 'strand', 'start', 'end', 'coordinate_system'):
        value = target.get(key)
        expected = geometry[key]
        if key == 'chrom' and value is not None:
            value, expected = str(value).removeprefix('chr'), str(expected).removeprefix('chr')
        if value is not None and value != expected:
            resolution['reason'] = 'Explicit genomic coordinates conflict with the resolved relative window.'
            return target
    resolution.update(status='resolved', anchor_site=site, anchor_position_1based=position,
                      reference_transcript_id=identity['reference_transcript_id'], **geometry)
    target.update(geometry)
    return target
