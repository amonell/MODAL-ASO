"""Bind splice predictions to explicit objectives and order computational designs.

These are recommendation annotations, never activity estimates or eligibility
filters. Missing predictions and disagreement remain visible.
"""
from copy import deepcopy
import hashlib
import json
import math

from .evaluation_feature_identity import (
    is_resolved_feature_identity, resolve_evaluation_target_identities,
    transcript_identifier, transcript_matches,
)


def finite_number(value):
    if isinstance(value, bool) or value in (None, ''):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def prediction_evidence(values, direction):
    """Describe signed observations; agreement affects priority, not eligibility."""
    parsed = [finite_number(value) for value in values]
    observed = [value for value in parsed if value is not None]
    result = {'direction': direction, 'values': parsed, 'observed_count': len(observed),
              'expected_count': len(parsed), 'mean_delta': None,
              'directional_mean': None, 'state': 'unavailable', 'rank_key': [1, 0.0]}
    if not observed:
        return result
    mean = sum(observed) / len(observed)
    result['mean_delta'] = mean
    if not isinstance(direction, str) or direction not in {'increase', 'decrease', 'neutral', 'preserve'}:
        result['state'] = 'direction_unresolved'
        return result
    if direction in {'neutral', 'preserve'}:
        # Opposite predictions must not cancel into an apparent neutrality claim.
        magnitude = max(abs(value) for value in observed)
        result.update(state='zero_observed_change' if magnitude == 0 else 'nonzero_predicted_change',
                      directional_mean=-magnitude, rank_key=[2, -magnitude])
        return result
    sign = 1 if direction == 'increase' else -1
    signed = [sign * value for value in observed]
    directional_mean = sign * mean
    complete = len(observed) == len(parsed)
    if min(signed) < 0 < max(signed):
        state, tier = 'discordant', 3 if directional_mean >= 0 else 2
    elif max(signed) > 0:
        state, tier = ('direction_supported', 4) if complete else ('partial_direction_support', 3)
    elif min(signed) < 0:
        state, tier = ('opposing', 0) if complete else ('partial_opposing', 1)
    else:
        state, tier = 'zero_observed_change', 1
    result.update(state=state, directional_mean=directional_mean,
                  rank_key=[tier, directional_mean])
    return result


def resolve_routing_exon_objectives(goal):
    """Resolve only an explicit exon outcome in the current job's reference.

    Scoring caches contain raw measurements. Job-specific objectives are bound
    here, after cache reuse, so another job's direction cannot leak into a row.
    """
    job = goal.get('routing_job')
    if isinstance(job, dict):
        sources = [job]
    else:
        sources = [target for target in goal.get('evaluation_targets') or []
                   if isinstance(target, dict) and target.get('target_type') == 'exon']
        if not sources:
            sources = [goal]
    targets = []
    expanded = []
    for source in sources:
        labels = source.get('exon_labels')
        if isinstance(labels, list) and labels:
            expanded.extend({**source, 'exon_label': label} for label in labels)
        else:
            expanded.append(source)
    for source in expanded:
        effect = source.get('desired_effect')
        if not isinstance(effect, str) or effect not in {'inclusion', 'skipping'}:
            continue
        if source.get('exon_label') is None and source.get('exon_index') is None:
            continue
        target = {key: source[key] for key in (
            'exon_label', 'exon_index', 'transcript_id', 'reference_transcript_id',
            'exon_numbering_reference_transcript_id', 'exon_numbering_transcript_id',
        ) if key in source}
        target['desired_effect'] = effect
        targets.append(target)
    catalog = goal.get('isoform_transcripts') or {}
    if not isinstance(catalog, dict):
        catalog = {}
    resolved = resolve_evaluation_target_identities(targets, catalog, goal.get('transcript_id'))
    objectives = []
    for target in resolved:
        identity = target.get('exon_identity_resolution') or {}
        definition = {'desired_effect': target['desired_effect'],
            'direction': 'increase' if target['desired_effect'] == 'inclusion' else 'decrease',
            'identity': identity, 'status': 'unresolved', 'requested_feature': target}
        if is_resolved_feature_identity(identity):
            reference = identity.get('reference_transcript_id')
            entries = [entry for label, entry in catalog.items() if isinstance(entry, dict)
                       and transcript_matches(reference, transcript_identifier(entry, label))]
            strands = {'-' if str(entry.get('strand')) in {'-', '-1'} else '+'
                       if str(entry.get('strand')) in {'+', '1'} else 'unknown'
                       for entry in entries}
            start, end = finite_number(identity.get('start')), finite_number(identity.get('end'))
            if (entries and strands <= {'+', '-'} and len(strands) == 1
                    and start is not None and end is not None and start.is_integer()
                    and end.is_integer() and 0 < start <= end and identity.get('chrom')):
                strand = next(iter(strands))
                definition.update(status='resolved', reference_transcript_id=reference,
                    chrom=str(identity['chrom']), strand=strand,
                    exon_index=identity['resolved_transcript_ordinal'],
                    acceptor_position=int(end if strand == '-' else start),
                    donor_position=int(start if strand == '-' else end))
        signature = {key: value for key, value in definition.items()
                     if key not in {'identity', 'requested_feature'}}
        signature['requested_reference'] = identity.get('requested_reference_transcript_id')
        signature['requested_exon_label'] = target.get('exon_label')
        signature['requested_exon_index'] = target.get('exon_index')
        definition['objective_id'] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if not any(old['objective_id'] == definition['objective_id'] for old in objectives):
            objectives.append(definition)
    return objectives


def _chrom(value):
    text = str(value or '').strip()
    return text[3:] if text.lower().startswith('chr') else text


def _matches_pair(site, objective):
    if site.get('reference_binding_status') in {'conflicting', 'unresolved'}:
        return False
    if not transcript_matches(objective['reference_transcript_id'],
                              str(site.get('reference_transcript_id') or '')):
        return False
    for kind in ('acceptor', 'donor'):
        point = site.get(kind)
        if not isinstance(point, dict):
            return False
        if _chrom(point.get('chrom')) != _chrom(objective['chrom']):
            return False
        if finite_number(point.get('position')) != objective[kind + '_position']:
            return False
        strand = str(point.get('strand') or site.get('strand') or '')
        if strand not in ({'-', '-1'} if objective['strand'] == '-' else {'+', '1'}):
            return False
    return True


def preserve_splice_site_evidence(result, catalog):
    """Copy measurements with their resolved reference; never add an outcome."""
    sites = deepcopy(result.get('splice_site_deltas') or [])
    if not isinstance(sites, list):
        return []
    for site in sites:
        if not isinstance(site, dict):
            continue
        label = str(site.get('isoform_label') or '')
        references = set()
        matched_entries = []
        for key, entry in catalog.items() if isinstance(catalog, dict) else []:
            if not isinstance(entry, dict):
                continue
            reference = transcript_identifier(entry, key)
            if label and (label == key or transcript_matches(label, reference)):
                references.add(reference)
                matched_entries.append(entry)
        if len(references) == 1:
            reference = next(iter(references))
            from .explicit_target_identity import prediction_geometry_status
            geometry = {prediction_geometry_status(site, entry) for entry in matched_entries}
            if site.get('reference_transcript_id') not in (None, '', reference) or 'conflicting' in geometry:
                site['reference_binding_status'] = 'conflicting'
            elif geometry != {'consistent'}:
                site['reference_binding_status'] = 'unresolved'
            else:
                site['reference_transcript_id'] = reference
                site['reference_binding_status'] = 'resolved_catalog_identity'
        else:
            site['reference_binding_status'] = 'unresolved'
    return sites


def objective_prediction_evidence(row, objective):
    """Use explicitly identified paired measurements, never a global scalar."""
    result = {'objective_id': objective['objective_id'], 'desired_effect': objective['desired_effect'],
              'reference_transcript_id': objective.get('reference_transcript_id'),
              'exon_index': objective.get('exon_index'), 'chrom': objective.get('chrom'),
              'strand': objective.get('strand'), 'acceptor_position': objective.get('acceptor_position'),
              'donor_position': objective.get('donor_position'), 'measurements': [],
              'status': 'unresolved_identity', 'rank_key': [1, 0.0]}
    if objective.get('status') != 'resolved':
        return result
    sites = row.get('splice_site_deltas') or []
    if isinstance(sites, str):
        try:
            sites = json.loads(sites)
        except (TypeError, ValueError):
            sites = []
    by_window = {}
    for index, site in enumerate(sites if isinstance(sites, list) else []):
        if not isinstance(site, dict) or not _matches_pair(site, objective):
            continue
        if site.get('evaluation_status') != 'evaluated':
            continue
        values = [finite_number(site.get('delta_inclusion_deletion')),
                  finite_number(site.get('delta_inclusion_replacement'))]
        window = site.get('window_bp')
        key = json.dumps(window, sort_keys=True)
        entry = by_window.setdefault(key, {'window_bp': window, 'observations': {}})
        signature = json.dumps(values)
        observation = entry['observations'].setdefault(signature,
            {'values': values, 'source_site_indices': []})
        observation['source_site_indices'].append(index)
    if not by_window:
        result['status'] = 'paired_measurement_unavailable'
        return result
    if any(len(entry['observations']) != 1 for entry in by_window.values()):
        result.update(status='conflicting_pair_measurements', raw_windows=list(by_window.values()))
        return result
    for entry in by_window.values():
        observation = next(iter(entry['observations'].values()))
        result['measurements'].append({'window_bp': entry['window_bp'], **observation,
            'prediction': prediction_evidence(observation['values'], objective['direction'])})
    result['status'] = 'recorded_pair_predictions'
    # Several windows are separate, correlated views, not independent replicates.
    result['rank_key'] = min(item['prediction']['rank_key'] for item in result['measurements'])
    result['interpretation'] = 'Paired splice-probability proxy; not measured exon inclusion or ASO activity.'
    return result


def annotate_route_recommendation(row, primary_reports, objectives):
    """Add reproducible priority and scope without excluding any candidate."""
    selected = []
    for report in primary_reports:
        if not isinstance(report, dict):
            continue
        selected.append({'label': report.get('label'),
            'prediction': prediction_evidence(
                [report.get('delta_deletion'), report.get('delta_replacement')],
                report.get('direction'))})
    paired = [objective_prediction_evidence(row, objective) for objective in objectives]
    site_key = min((item['prediction']['rank_key'] for item in selected), default=[1, 0.0])
    if paired:
        primary_key = min(item['rank_key'] for item in paired)
        states = [measurement['prediction']['state'] for item in paired
                  for measurement in item['measurements']]
        incomplete = any(item['status'] != 'recorded_pair_predictions' for item in paired)
        basis = 'explicit_exon_objective'
    else:
        primary_key = site_key
        states = [item['prediction']['state'] for item in selected]
        incomplete = not states
        basis = 'selected_splice_site'
    if incomplete or not states or any(state in {'unavailable', 'direction_unresolved'} for state in states):
        status = 'incomplete_prediction_evidence'
        note = 'The requested effect lacks complete, unambiguous scored evidence. Retained as exploratory.'
    elif all(state == 'direction_supported' for state in states):
        status = 'direction_supported'
        note = 'Recorded perturbation scores support the requested direction; cellular ASO activity remains unmeasured.'
    elif all(state == 'opposing' for state in states):
        status = 'opposing_predictions'
        note = 'Recorded perturbation scores oppose the requested direction. Retained as an exploratory candidate.'
    elif all(state == 'zero_observed_change' for state in states):
        status = 'no_directional_support'
        note = 'The recorded predictions show no directional change. Retained as exploratory.'
    else:
        status = 'mixed_or_partial_predictions'
        note = 'Recorded predictions disagree or provide partial support. Retained as an exploratory candidate.'
    row.update(selected_effect_evidence=selected, exon_objective_evidence=paired,
        recommendation_basis=basis, recommendation_status=status, recommendation_note=note,
        recommendation_rank_key=[*primary_key, *site_key])
    return row


def recommendation_sort_key(candidate, direction=None):
    """Higher tuples lead; absent direction never turns magnitude into benefit."""
    saved = candidate.get('recommendation_rank_key')
    if isinstance(saved, (list, tuple)) and saved and all(finite_number(value) is not None for value in saved):
        key = tuple(float(value) for value in saved)
    else:
        values = [candidate.get('selected_delta_deletion'), candidate.get('selected_delta_replacement')]
        if all(finite_number(value) is None for value in values):
            values = [candidate.get('mean_splice_delta')]
        key = tuple(prediction_evidence(values, direction)['rank_key'])
    return (*key, finite_number(candidate.get('score', candidate.get('Score'))) or 0.0)


RECOMMENDATION_FIELDS = (
    'selected_effect_evidence', 'exon_objective_evidence', 'recommendation_basis',
    'recommendation_status', 'recommendation_note', 'recommendation_rank_key',
    'reference_scope', 'selected_transcript_context', 'shared_with_beneficial',
    'conflict_between_objectives', 'route_interpretation',
    'safeguard_report',
)


def report_recommendation_fields(row):
    """Keep row-level evidence intact through CSV and report normalization."""
    result = {}
    annotations = row.get('source_annotations')
    if not isinstance(annotations, dict):
        annotations = {}
    for key in RECOMMENDATION_FIELDS:
        value = row.get(key, annotations.get(key))
        if isinstance(value, str) and key in {
            'selected_effect_evidence', 'exon_objective_evidence', 'recommendation_rank_key',
            'reference_scope', 'selected_transcript_context',
            'safeguard_report',
        }:
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                continue
        if value is not None:
            result[key] = deepcopy(value)
    return result
