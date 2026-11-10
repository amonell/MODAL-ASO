"""Compact views of persisted OligoAI evidence for the candidate explorer."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json


def evaluation_preview(evaluation, limit=40):
    """Bound legacy-tab metadata; full predictions live in the explorer and files."""
    context_fields = {'transcript_id', 'species', 'assembly', 'context_basis',
                      'reference_sequence_sha256', 'start1', 'end1',
                      'binding_interval_bed0_half_open', 'input_id', 'status',
                      'reason', 'predicted_inhibition_percent', 'flank_padding_nt', 'masked_flank_nt'}
    entries = evaluation.get('candidates') or []
    preview = deepcopy({key:value for key,value in evaluation.items() if key!='candidates'})
    preview['candidates'] = [{**deepcopy({k:v for k,v in entry.items() if k!='contexts'}),
                              'contexts': [{k:deepcopy(v) for k,v in context.items() if k in context_fields}
                                           for context in entry.get('contexts') or []]}
                             for entry in entries[:limit]]
    preview['candidate_preview_count'] = min(len(entries), limit)
    preview['candidate_preview_truncated'] = len(entries) > limit
    return preview


def activity_join_key(row):
    from .report_explorer_data import join_key, physical_identity, source_row
    source = source_row(row)
    return (source.get('program_id'), source.get('job_id'), source.get('input_row_id'),
            physical_identity(row), join_key(row))


def activity_index(rows, evaluation=None, *, candidate_numbers=None):
    """Keep predictions bound to their producing row, molecule and job."""
    from .report_explorer_data import physical_identity
    evaluation = evaluation or {}
    result = defaultdict(list)
    detached = {entry.get('candidate_number'): entry for entry in evaluation.get('candidates') or []}
    numbers = candidate_numbers if candidate_numbers is not None else range(1,len(rows)+1)
    for index, row in zip(numbers,rows):
        entry = row.get('oligoai_evaluation')
        if not entry:
            saved = detached.get(index) or {}
            if saved.get('physical_molecule_sha256') == physical_identity(row):
                entry = saved
        if entry:
            result[activity_join_key(row)].append((entry, evaluation))
    return result


def _reason(value):
    text = str(value or '')
    labels = {
        'explicitly_disabled': 'OligoAI was disabled for this request.',
        'no_exact_antisense_site_in_available_target_references': 'An exact binding site was unavailable in the supplied RNA references.',
        'physical_chemistry_not_attested': 'The recorded molecule needs a complete chemistry specification.',
    }
    for prefix, label in (
        ('unsupported_model_sugar', 'Sugar chemistry outside the model’s supported inputs'),
        ('unsupported_model_backbone', 'Backbone chemistry outside the model’s supported inputs'),
        ('unsupported_model_attachment', 'Attachment chemistry outside the model’s supported inputs'),
    ):
        if text.startswith(prefix):
            return label + text[len(prefix):]
    return labels.get(text, text.replace('_', ' '))


def project_activity(row, attached=()):
    """Retain each prediction and condition separately, without inference."""
    from .report_explorer_data import number, physical_identity, sequence
    entries = list(attached)
    own = row.get('oligoai_evaluation')
    if own and not any(entry == own for entry, _ in entries):
        entries.append((own, {}))
    projected = []
    for entry, evaluation in entries:
        identity = entry.get('physical_molecule_sha256')
        if identity and identity != physical_identity(row):
            continue
        if entry.get('antisense_5to3') and entry['antisense_5to3'] != sequence(row):
            continue
        conditions = entry.get('conditions') or evaluation.get('conditions') or {}
        for context in entry.get('contexts') or [{}]:
            scenario = {key: context.get(key) for key in (
                'transcript_id', 'species', 'assembly', 'context_basis', 'reference_sequence_sha256')}
            scenario.update({key: conditions.get(key) for key in (
                'dosage_nm', 'transfection_method', 'dose_basis', 'method_basis')})
            scenario.update({key: conditions[key] for key in ('unmodeled_conditions', 'unresolved_conditions')
                             if conditions.get(key)})
            scenario.update({key: evaluation.get(key) for key in ('model_sha256', 'source_revision')})
            scenario['dosage_nm'] = number(scenario['dosage_nm'])
            experiment = {key:scenario.get(key) for key in (
                'dosage_nm', 'transfection_method', 'dose_basis', 'method_basis',
                'unmodeled_conditions', 'unresolved_conditions', 'model_sha256', 'source_revision')}
            conditions_id = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()[:20]
            basis = {'genomic_pre_mrna': 'pre-mRNA',
                     'mature_transcript_extrapolation': 'mature RNA',
                     'mature_rna_extrapolation': 'mature RNA'}.get(
                         scenario['context_basis'], str(scenario['context_basis'] or '').replace('_', ' '))
            dose = f"{scenario['dosage_nm']:g} nM" if scenario['dosage_nm'] is not None else ''
            label = ' · '.join(str(v) for v in (scenario['species'], scenario['transcript_id'], basis,
                                               dose, scenario['transfection_method']) if v) or 'Recorded evaluation'
            if conditions.get('unresolved_conditions'):
                label = 'Prediction settings need clarification'
            conditions_label = ' · '.join(v for v in (dose, scenario['transfection_method']) if v) or 'Recorded conditions'
            scenario_id = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()[:20]
            value = number(context.get('predicted_inhibition_percent'))
            status = context.get('status') or entry.get('status') or 'unavailable'
            if status != 'predicted':
                value = None
            elif value is None:
                status = 'error'
            interval = context.get('binding_interval_bed0_half_open')
            location = ''
            if isinstance(interval, (list, tuple)) and len(interval) == 3:
                if number(interval[1]) is not None and number(interval[2]) is not None:
                    location = f'{interval[0]}:{int(interval[1])+1:,}–{int(interval[2]):,}'
            elif number(context.get('start1')) is not None and number(context.get('end1')) is not None:
                location = f"cDNA {int(context['start1']):,}–{int(context['end1']):,}"
            projected.append({**scenario, 'scenario_id': scenario_id, 'label': label,
                'conditions_id': conditions_id, 'conditions_label': conditions_label,
                'value': value, 'status': status,
                'reason': _reason(context.get('reason') or entry.get('reason')),
                'location': location, 'input_id': context.get('input_id'),
                'flank_padding_nt': context.get('flank_padding_nt'),
                'masked_flank_nt': context.get('masked_flank_nt')})
    return list({json.dumps(item, sort_keys=True): item for item in projected}.values())
