"""Reference-bound signed splice measurements for reading, filtering and export."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def junction_measurements(row, context, sites, name_site):
    source = {**(row.get('source_annotations') or {}), **row}
    entries = source.get('splice_site_deltas') or []
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except ValueError:
            entries = []
    grouped = defaultdict(list)
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        kind = entry.get('site_kind')
        if kind not in {'acceptor', 'donor'}:
            continue
        point = entry.get(kind) or {}
        label = entry.get('canonical_label') or entry.get('label') or entry.get('display_label')
        reference = entry.get('reference_transcript_id') or context.get('transcript')
        scope = {'reference': reference, 'job': context.get('job'), 'program': context.get('program'),
                 'assembly': entry.get('assembly') or entry.get('assembly_name') or context.get('assembly'),
                 'chrom': point.get('chrom'), 'strand': point.get('strand') or context.get('reference_strand'),
                 'position': point.get('position'), 'kind': kind,
                 'coordinate_system': '1based_inclusive',
                 'model': entry.get('model') or source.get('splice_model') or context.get('splice_model'),
                 'model_sha256': entry.get('model_sha256') or source.get('model_sha256')}
        if not reference or not scope['chrom'] or scope['position'] is None:
            # An unresolved identity stays local to its recorded response view.
            scope.update(unresolved_context=context['id'], recorded_label=label)
        key = hashlib.sha256(json.dumps(scope, sort_keys=True, default=str).encode()).hexdigest()[:24]
        values = {mode: number(entry.get(f'delta_{kind}_{mode}')) for mode in ('deletion', 'replacement')}
        status = entry.get('evaluation_status')
        evaluated = status in {None, 'evaluated', 'completed', 'scored', 'ok'}
        mean = sum(values.values()) / 2 if evaluated and all(v is not None for v in values.values()) else None
        bound = sites.get(label) or {}
        compatible = not entry.get('reference_transcript_id') or bound.get('reference_transcript_id') == entry['reference_transcript_id']
        naming = {label: {**bound, **entry, **point,
            'verified_exon_label': bound.get('verified_exon_label'),
            'requested_exon_label': bound.get('requested_exon_label'),
            'annotation_exon_index': bound.get('annotation_exon_index')}} if compatible else {label: {**entry, **point,
            'verified_exon_label': None, 'annotation_exon_index': None, 'requested_exon_label': None}}
        grouped[key].append({'id': 'junction_' + key, 'label': name_site(label, naming),
                             'scope': scope, 'predictions': values, 'mean': mean,
                             'status': 'available' if mean is not None else 'unavailable',
                             'recorded_status': status, 'source_label': label,
                             'reference_binding_status': entry.get('reference_binding_status')})
    result = []
    for items in grouped.values():
        unique = list({json.dumps(item, sort_keys=True, default=str): item for item in items}.values())
        result.append(unique[0] if len(unique) == 1 else {
            **unique[0], 'mean': None, 'status': 'conflicting_records', 'variants': unique})
    return result


def junction_attributes(records):
    catalog = {}
    for index, row in enumerate(records):
        for measurement in row.get('junction_measurements') or []:
            key = measurement['id']
            if key not in catalog:
                catalog[key] = {'id': key, 'label': measurement['label'] + ' · mean splice effect',
                    'group': 'Splice junction predictions', 'kind': 'number', 'unit': '',
                    'direction': 'higher', 'low': -1, 'high': 1, 'quantity': 'splice_delta',
                    'description': 'Change in use of this junction. Negative values mean reduced use; positive values mean increased use. Both simulations must be available.',
                    'reference_scope': measurement['scope'], 'values': [None] * len(records)}
            catalog[key]['values'][index] = measurement['mean']
    labels = defaultdict(list)
    for attr in catalog.values():
        labels[attr['label']].append(attr)
    for attrs in labels.values():
        if len(attrs) < 2:
            continue
        positions = {(a['reference_scope'].get('chrom'), a['reference_scope'].get('position')) for a in attrs}
        references = {a['reference_scope'].get('reference') for a in attrs}
        for index, attr in enumerate(attrs, 1):
            scope = attr['reference_scope']
            if len(positions) == len(attrs):
                point = scope.get('position')
                qualifier = f"{scope.get('chrom') or '?'}:{point:,}" if type(point) is int else str(point)
            elif len(references) == len(attrs):
                qualifier = str(scope.get('reference') or 'Reference unavailable')
            else:
                qualifier = f'measurement {index}'
            attr['label'] += ' · ' + qualifier
    return list(catalog.values())
