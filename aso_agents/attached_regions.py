"""Bind typed user-supplied transcript intervals to their declared design programs."""

import json
from copy import deepcopy
from pathlib import Path


TRANSCRIPT_CONVENTIONS = {
    'mature_transcript_1based_inclusive',
    'cdna_1based_inclusive',
}


def _mentioned(identifier, text):
    """Match a feature ID as a complete token without guessing a site from prose."""
    identifier = str(identifier or '').casefold()
    text = str(text or '').casefold()
    if not identifier:
        return False
    offset = 0
    while (index := text.find(identifier, offset)) >= 0:
        before = text[index - 1] if index else ''
        after_index = index + len(identifier)
        after = text[after_index] if after_index < len(text) else ''
        if not (before.isalnum() or before == '_') and not (after.isalnum() or after == '_'):
            return True
        offset = index + 1
    return False


def _features(goal, input_root):
    input_root = Path(input_root).resolve()
    found = []
    for reference in goal.get('artifact_refs') or []:
        if not isinstance(reference, dict):
            continue
        path = Path(str(reference.get('path') or ''))
        if path.suffix.lower() != '.json' or not path.is_absolute():
            continue
        path = path.resolve()
        if not path.is_relative_to(input_root) or not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError, UnicodeError):
            continue
        entries = payload.get('features') if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            convention = entry.get('coordinate_system') or (payload.get('coordinate_system') if isinstance(payload, dict) else None)
            if convention not in TRANSCRIPT_CONVENTIONS:
                continue
            feature_id = str(entry.get('feature_id') or entry.get('id') or '').strip()
            transcript = str(entry.get('reference_transcript') or entry.get('transcript_id') or '').strip()
            start, end = entry.get('start'), entry.get('end')
            if not feature_id or not transcript or isinstance(start, bool) or isinstance(end, bool):
                continue
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue
            if start < 1 or end < start:
                continue
            found.append({'feature_id': feature_id, 'transcript': transcript,
                          'start': start, 'end': end, 'coordinate_system': convention,
                          'source_path': str(path)})
    return found


def bind_attached_transcript_intervals(goal, input_root='/input'):
    """Return a goal with exact, uniquely named attachment intervals attached.

    Existing explicit constraints are authoritative. Ambiguous or mismatched
    feature IDs are left unresolved, never replaced with a broad gene region.
    """
    if not isinstance(goal, dict):
        return goal, {'bound_programs': [], 'unresolved_programs': []}
    features = _features(goal, input_root)
    updated = deepcopy(goal)
    bound, unresolved = [], []
    for program in updated.get('design_programs') or []:
        if not isinstance(program, dict) or program.get('region_constraints'):
            continue
        if program.get('targeting_policy') != 'explicit_user_region':
            continue
        text = ' '.join(str(program.get(key) or '') for key in ('program_id', 'label', 'objective'))
        matches = [feature for feature in features if _mentioned(feature['feature_id'], text)
                   and (not program.get('transcript_id') or
                        str(program['transcript_id']) == feature['transcript'])]
        if len(matches) != 1:
            unresolved.append(str(program.get('program_id') or program.get('label') or 'unnamed'))
            continue
        feature = matches[0]
        program['region_constraints'] = [{
            'constraint_id': feature['feature_id'], 'label': feature['feature_id'],
            'source_type': 'explicit_rows', 'operator': 'union',
            'coordinate_space': 'cdna',
            'coordinate_convention': feature['coordinate_system'],
            'aso_overlap_requirement': 'full_containment',
            'rows': [[feature['transcript'], feature['start'] - 1, feature['end'],
                      feature['feature_id'], 0, '.']],
            'source': 'user_attached_interval',
            'source_path': feature['source_path'],
        }]
        bound.append(str(program.get('program_id') or program.get('label') or 'unnamed'))
    return updated, {'bound_programs': bound, 'unresolved_programs': unresolved,
                     'attachment_features': len(features)}
