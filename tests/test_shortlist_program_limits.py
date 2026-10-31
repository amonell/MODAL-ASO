"""Program-local count contracts must reach the real final export."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents.quoted_program_obligations import apply_program_obligations
from aso_agents.shortlist_clarification import size_question


def goal_with_count(second=False):
    goal = {'species': 'human', 'design_programs': [
        {'program_id': 'bounded', 'modality': 'steric', 'length': 20}]}
    if second:
        goal['design_programs'].append({'program_id': 'open', 'modality': 'steric', 'length': 18})
    quote = 'At most three 20-nt ASOs'
    receipt = apply_program_obligations(goal, {'program_candidate_counts': [
        {'length_nt': 20, 'modality': 'steric', 'count': 3, 'kind': 'maximum', 'quote': quote}]}, quote)
    assert receipt['status'] == 'applied'
    return goal


def export(goal, tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    from aso_chemistry import encode_idt_oligo
    rows = []
    for program_index, program in enumerate(goal['design_programs']):
        for i in range(5):
            index = 5 * program_index + i
            length = program['length']
            sequence = ('ACGT' * 6)[:length - 4] + ''.join('ACGT'[(index // 4**n) % 4] for n in range(4))
            start = 100 + 30 * index
            rows.append({'Synthesis': sequence,
                'Sequence': sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1],
                'IDT': encode_idt_oligo(sequence, 'steric-hindrance'), 'selection_rank': index + 1,
                'Chrom': 'chrFixture', 'Start': start, 'End': start + length, 'Strand': '+',
                'binding_intervals_bed0_half_open': [['chrFixture', start, start + length]],
                'source_modality': 'steric', 'program_id': program['program_id'],
                'route_label': 'route_' + str(i % 2)})
    original = deepcopy(rows)
    monkeypatch.setattr(pipeline, '_collect_generated_candidate_rows', lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, '_collect_generated_stage_sets', lambda *a: {})
    monkeypatch.setattr(pipeline, '_goal_payload_context', lambda *a: {'structured_goal': goal})
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(artifacts['generated_aso_set_json']).read_text())
    pool = json.loads(Path(artifacts['candidate_assessments_json']).read_text())
    assert pool['pool_row_count'] == len(original)
    counts = Counter(row.get('program_id') or row['source_annotations']['program_id'] for row in panel['rows'])
    return panel, counts


def test_quoted_single_program_cap_reaches_export(tmp_path, monkeypatch):
    goal = goal_with_count()
    assert size_question(goal, True) is None
    panel, counts = export(goal, tmp_path, monkeypatch)
    assert counts == {'bounded': 3}
    assert len(panel['rows']) == 3


@pytest.mark.parametrize('preference', [None, 'all_eligible', 6])
def test_partly_sized_request_keeps_local_cap_and_late_global_choice(preference, tmp_path, monkeypatch):
    goal = goal_with_count(second=True)
    assert size_question(goal, False) is None
    assert size_question(goal, True)['restart_stage'] == 'report'
    if preference == 'all_eligible':
        goal['shortlist_size_preference'] = preference
    elif preference is not None:
        goal['candidate_limit'] = preference
    if preference is not None:
        assert size_question(goal, True) is None
    panel, counts = export(goal, tmp_path, monkeypatch)
    assert counts['bounded'] == 3
    assert counts['open'] == (3 if preference == 6 else 5)
    assert len(panel['rows']) == (6 if preference == 6 else 8)


def test_all_program_caps_suppress_question_without_becoming_global_cap(tmp_path, monkeypatch):
    goal = goal_with_count(second=True)
    goal['design_programs'][1]['candidate_limit'] = 2
    assert size_question(goal, True) is None
    panel, counts = export(goal, tmp_path, monkeypatch)
    assert counts == {'bounded': 3, 'open': 2}
    assert len(panel['rows']) == 5


def test_global_cap_remains_global_across_programs(tmp_path, monkeypatch):
    goal = goal_with_count(second=True)
    goal['design_programs'][0].pop('candidate_limit')
    goal['candidate_limit'] = 3
    panel, _ = export(goal, tmp_path, monkeypatch)
    assert len(panel['rows']) == 3


def test_route_limits_cannot_override_a_program_cap(tmp_path, monkeypatch):
    goal = goal_with_count(second=True)
    goal['design_programs'][0]['candidate_limit'] = 1
    goal.update(candidate_limit_per_site=2, candidate_limit=4,
                request_text='Return at most two candidates per route.')
    panel, counts = export(goal, tmp_path, monkeypatch)
    assert counts['bounded'] == 1 and counts['open'] == 3
    routes = Counter(row['route_label'] for row in panel['rows'])
    assert routes == {'route_0': 2, 'route_1': 2}
    assert panel['candidate_selection']['eligible_count'] == 10
    assert panel['candidate_selection']['unselected_count'] == 6


@pytest.mark.parametrize('key', ['candidate_limit_per_site', 'per_site_candidate_limit'])
def test_noncanonical_program_site_field_does_not_suppress_question(key):
    goal = goal_with_count()
    goal['design_programs'][0].pop('candidate_limit')
    goal['design_programs'][0][key] = 3
    assert size_question(goal, True)['restart_stage'] == 'report'
