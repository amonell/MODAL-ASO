"""Resolve plural evaluation references from typed dependency identities."""
from copy import deepcopy
import csv
import json


def sequence_inventory_available(path):
    """An available empty table differs from an absent or unreadable table."""
    try:
        if not path.is_file():
            return False
        if path.suffix.lower() == '.json':
            payload = json.loads(path.read_text())
            return isinstance(payload, dict) and isinstance(payload.get('rows'), list)
        if path.suffix.lower() in ('.csv', '.tsv'):
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t' if path.suffix.lower() == '.tsv' else ',')
                return bool(set(reader.fieldnames or []) & {'sequence', 'Sequence', 'seq', 'ASO'})
    except (OSError, UnicodeError, ValueError, csv.Error):
        return False
    return False


def requested_dependency_contexts(candidate_dirs, envelope, context_reader, normalize_species):
    task = envelope.get('task_context') or {}
    targets = envelope.get('species_targets') or task.get('species_targets') or []
    targets = [target for target in targets if isinstance(target, dict)]
    if not targets:
        return [], []
    contexts, seen = [], set()
    for directory in candidate_dirs:
        context = context_reader([directory])
        goal = context.get('structured_goal') or {}
        if not goal:
            continue
        if context.get('run_dir') in seen:
            continue
        if any(target_matches_goal(target, goal, normalize_species) for target in targets):
            contexts.append(context)
            seen.add(context.get('run_dir'))
    coverage = []
    for target in targets:
        matched = [context.get('run_dir') for context in contexts
                   if target_matches_goal(target, context['structured_goal'], normalize_species)]
        coverage.append({'requested_identity': deepcopy(target),
                         'status': 'resolved' if matched else 'unavailable', 'source_run_dirs': matched})
    return contexts, coverage


def target_matches_goal(target, goal, normalize_species):
    identities = []
    if target.get('species'):
        identities.append(bool(goal.get('species')) and normalize_species(target['species']) == normalize_species(goal['species']))
    if target.get('gene_id'):
        identities.append(str(target['gene_id']).split('.')[0] == str(goal.get('gene_id') or '').split('.')[0])
    if target.get('gene_symbol'):
        identities.append(str(target['gene_symbol']).casefold() == str(goal.get('gene_symbol') or '').casefold())
    return bool(identities) and all(identities)


def merge_reference_catalog(destination, incoming, source_goal, normalize_species):
    """No organism inference from transcript names or matching genome sequence."""
    for label, original in incoming.items():
        entry = deepcopy(original)
        metadata = entry.setdefault('metadata', {})
        source_species = source_goal.get('species')
        if source_species:
            declared = entry.get('species') or metadata.get('species')
            if declared and normalize_species(declared) != normalize_species(source_species):
                raise ValueError('dependency_reference_species_conflict')
            if not declared:
                metadata['species'] = normalize_species(source_species)
        if label in destination:
            previous = destination[label]
            previous_metadata = previous.get('metadata') or {}
            for field in ('transcript_id', 'chrom', 'strand', 'exons_tx'):
                if previous.get(field) != entry.get(field):
                    raise ValueError('dependency_reference_identity_conflict:' + str(label))
            previous_species = previous.get('species') or previous_metadata.get('species')
            incoming_species = entry.get('species') or metadata.get('species')
            if previous_species and incoming_species and normalize_species(previous_species) != normalize_species(incoming_species):
                raise ValueError('dependency_reference_species_conflict:' + str(label))
            for field in ('assembly_name', 'versioned_id'):
                if previous_metadata.get(field) and metadata.get(field) and previous_metadata[field] != metadata[field]:
                    raise ValueError('dependency_reference_identity_conflict:' + str(label))
            continue
        destination[label] = entry
    return destination


def recorded_trace_markdown(payloads):
    lines = []
    for payload in payloads:
        contexts = payload.get('reference_traces') or [{
            'run_dir': payload.get('generated_reference_run_dir'),
            'reference_identity': {}, 'trace_rows': payload.get('trace_rows') or []}]
        for context in contexts:
            identity = context.get('reference_identity') or {}
            label = ' / '.join(str(identity[key]) for key in ('species', 'gene_symbol') if identity.get(key))
            label = label or str(context.get('run_dir') or 'Unresolved reference')
            lines.extend(['', '### ' + label, '',
                          '| Supplied molecule | Entered generated pool | Recorded outcome | First observed stage |',
                          '|---|---|---|---|'])
            for row in context.get('trace_rows') or []:
                entered = row.get('entered_pipeline')
                stage = row.get('first_drop_stage') if entered is True else 'Not an observed downstream rejection'
                values = [row.get('name') or row.get('input_row_id') or row.get('sequence'),
                          'yes' if entered is True else 'no' if entered is False else 'unknown',
                          row.get('terminal_classification') or row.get('terminal_status') or 'unknown',
                          stage or 'none recorded']
                lines.append('| ' + ' | '.join(str(value).replace('|', '\\|') for value in values) + ' |')
    if lines:
        lines.extend(['', 'These outcomes describe membership in the recorded generated pools. A molecule that was never generated has no observed downstream filter rejection. Absence from a generated pool does not establish lack of target binding or activity.'])
    return '\n'.join(lines)
