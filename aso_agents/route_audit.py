"""Retain measured evidence for considered routes outside the recommended panel."""
import csv
import hashlib
import json
from pathlib import Path


def _vector(value):
    if isinstance(value, list):
        return value
    return value.split(',') if isinstance(value, str) and value else []


def write_route_audit(scored_path, decisions, output_directory):
    """Export original per-site measurements without recommending new molecules.

    The caller supplies typed route decisions and their reasons. This module
    never chooses biological targets, reverses signs, or changes rankings.
    Audit CSVs intentionally have no .route suffix and are not native design
    panel inputs. All measured signs are retained, including opposing effects.
    """
    if not decisions:
        return {'status':'not_needed', 'routes':[]}
    source = Path(scored_path)
    if not source.is_file():
        return {'status':'unavailable', 'reason':'scored_data_missing', 'routes':[]}
    raw = source.read_bytes()
    payload = json.loads(raw)
    rows = payload.get('rows') or []
    directory = Path(output_directory)/'route_audits'
    directory.mkdir(parents=True, exist_ok=True)
    fields = ['source_row', 'route_label', 'site_action', 'recommendation_status',
              'decision_reason', 'candidate_name', 'target_sequence',
              'antisense_sequence', 'chrom', 'start', 'end', 'strand',
              'delta_deletion', 'delta_replacement']
    routes = []
    seen = set()
    for decision in decisions:
        label, direction = decision['label'], decision['direction']
        if (label, direction) in seen:
            continue
        seen.add((label, direction))
        # Filenames are opaque identifiers, never gene- or coordinate-specific.
        name = hashlib.sha256(json.dumps([label,direction]).encode()).hexdigest()[:16]
        path = directory/(name+'.evidence.csv')
        exported = []
        for index, row in enumerate(rows):
            labels = [str(v).strip() for v in _vector(row.get('delta_names'))]
            if label not in labels:
                continue
            site_index = labels.index(label)
            deletion = _vector(row.get('delta_deletion'))
            replacement = _vector(row.get('delta_replacement'))
            exported.append({
                'source_row':index, 'route_label':label, 'site_action':direction,
                'recommendation_status':'not_recommended',
                'decision_reason':decision['reason'],
                'candidate_name':row.get('Name',row.get('name')),
                'target_sequence':row.get('target_binding_sequence',row.get('sequence',row.get('Sequence'))),
                'antisense_sequence':row.get('synthesized_antisense_sequence',row.get('Synthesis')),
                'chrom':row.get('chrom',row.get('Chrom')), 'start':row.get('start',row.get('Start')),
                'end':row.get('end',row.get('End')), 'strand':row.get('strand',row.get('Strand')),
                'delta_deletion':deletion[site_index] if site_index<len(deletion) else None,
                'delta_replacement':replacement[site_index] if site_index<len(replacement) else None,
            })
        with path.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerows(exported)
        routes.append({**decision,'recommendation_status':'not_recommended',
                       'path':str(path),'row_count':len(exported),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'status':'recorded','source':str(source),
            'source_sha256':hashlib.sha256(raw).hexdigest(),'routes':routes,
            'interpretation':'Measured route evidence only; these routes are outside the recommended candidate panel. Rows do not establish efficacy or isoform selectivity.'}
    (directory/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
