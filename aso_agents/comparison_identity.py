"""Separate source observations, sequence overlap, and recorded molecular identity."""
from __future__ import annotations

import hashlib
import json


def add_recorded_molecule_overlap(payload: dict, compare_sets: list[dict]) -> None:
    """Annotate exact recorded sequence/chemistry identity without inferring chemistry.

    Missing chemistry or physical sequence leaves that observation unresolved.
    Different chemistry labels are never assumed chemically equivalent.
    """
    inventory: dict[tuple[str, str], dict] = {}
    keys_by_set: dict[str, set] = {}
    unresolved_by_set: dict[str, int] = {}
    for panel in compare_sets:
        set_id = panel['set_id']
        keys_by_set[set_id] = set()
        unresolved_by_set[set_id] = 0
        for ordinal, row in enumerate(panel.get('rows') or [], 1):
            sequence = row.get('synthesized_antisense_sequence')
            if panel.get('source_type') == 'provided' and row.get('declared_sequence_role') != 'target_site_sequence':
                sequence = row.get('sequence') or row.get('raw_sequence')
            sequence = str(sequence or '').strip().upper()
            chemistry = row.get('chemistry')
            if isinstance(chemistry, dict):
                chemistry = json.dumps(chemistry, sort_keys=True, ensure_ascii=False)
            chemistry = ' '.join(str(chemistry or '').split())
            if not sequence or chemistry.casefold() in {'', 'unknown', 'not recorded', 'not specified', 'unspecified'}:
                unresolved_by_set[set_id] += 1
                continue
            key = (sequence, chemistry.casefold())
            keys_by_set[set_id].add(key)
            molecule = inventory.setdefault(key, {
                'molecule_id': 'molecule_' + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
                'sequence_5to3': sequence, 'recorded_chemistry': chemistry,
                'set_ids': [], 'observations': [],
            })
            if set_id not in molecule['set_ids']:
                molecule['set_ids'].append(set_id)
            metadata = row.get('input_metadata') if isinstance(row.get('input_metadata'), dict) else {}
            molecule['observations'].append({
                'set_id': set_id, 'set_label': panel.get('label'),
                'observation_id': row.get('input_row_id'),
                'candidate_id': metadata.get('candidate_id') or metadata.get('molecule_id') or row.get('name') or row.get('aso_id') or row.get('Name'),
                'source_metadata': metadata,
                'source_file': row.get('source_file') or panel.get('artifact_path'),
                'source_row': row.get('source_row'), 'set_row_1based': ordinal,
            })
    unresolved = sum(unresolved_by_set.values())
    payload.update({
        'molecule_identity_basis': 'Identical recorded physical 5-prime-to-3-prime sequence and chemistry label; '
            'chemistry labels are compared after whitespace/case normalization only. '
            'Chemical equivalence between different labels is not inferred.',
        'molecule_identity_status': 'complete_for_recorded_labels' if not unresolved else 'partially_unresolved',
        'observation_count': sum(len(panel.get('rows') or []) for panel in compare_sets),
        'distinct_sequence_count': len(set().union(*(panel.get('sequence_set') or set() for panel in compare_sets))),
        'known_molecule_count': len(inventory),
        'molecule_count': len(inventory) if not unresolved else None,
        'unresolved_molecule_observations': unresolved,
        'molecule_inventory': list(inventory.values()),
    })
    for summary in payload.get('sets') or []:
        set_id = summary['set_id']
        summary['molecule_count'] = len(keys_by_set[set_id]) if not unresolved_by_set[set_id] else None
        summary['unresolved_molecule_observations'] = unresolved_by_set[set_id]
    for pair in payload.get('pairwise_overlaps') or []:
        a, b = pair['set_a_id'], pair['set_b_id']
        shared = keys_by_set[a] & keys_by_set[b]
        pair['known_shared_molecule_count'] = len(shared)
        pair['shared_molecule_count'] = len(shared) if not (unresolved_by_set[a] or unresolved_by_set[b]) else None
        pair['shared_molecule_ids'] = [inventory[key]['molecule_id'] for key in sorted(shared)]
        pair['molecule_identity_status'] = ('complete_for_recorded_labels' if pair['shared_molecule_count'] is not None
                                            else 'partially_unresolved')
