"""Saved activity reaches every report view without new inference or joins by sequence."""
from copy import deepcopy
import json

import pytest

from aso_chemistry import encode_idt_oligo
from aso_molecule import molecule_record
from aso_agents.report_activity import activity_index, activity_join_key, project_activity
from aso_agents.report_explorer_data import project_run, project_records
from aso_agents.readable_report import basic_document


SEQ = 'ACGTCAGACTTAGCACTGCA'


def test_legacy_report_preview_is_bounded_and_keeps_full_evidence_untouched():
    from aso_agents.report_activity import evaluation_preview
    from aso_agents.html_report import render_goal_intake_html
    row = candidate()
    saved = entry(row)
    saved['contexts'][0]['rna_context'] = 'PRIVATE_RNA_CONTEXT'
    saved['contexts'][0]['provenance'] = {'reference_path': '/reference/genome.fa'}
    evaluation = {'candidates': [deepcopy(saved) for _ in range(100)], 'candidate_count': 100,
                  'predicted_count': 100, 'artifact_json': '/run/full-evidence.json'}
    original = deepcopy(evaluation)
    preview = evaluation_preview(evaluation)
    assert len(preview['candidates']) == 40 and preview['candidate_preview_truncated']
    assert preview['candidate_count'] == preview['predicted_count'] == 100
    assert preview['artifact_json'] == '/run/full-evidence.json'
    document = render_goal_intake_html({'oligoai_evaluation': evaluation, 'oligoai_evaluations':[evaluation]})
    assert 'PRIVATE_RNA_CONTEXT' not in document and '/reference/genome.fa' not in document
    assert '53.44' in document
    assert evaluation == original


def candidate(architecture='5-10-5', job='j', start=100):
    notation = encode_idt_oligo(SEQ, architecture)
    return {'synthesized_antisense_sequence': SEQ, 'source_modality': 'gapmer',
            'molecule_specification': molecule_record(SEQ, architecture, notation),
            'program_id': 'p', 'job_id': job, 'strand': '+',
            'binding_intervals_bed0_half_open': [['chr1', start, start+20]]}


def entry(row, value=53.435787200927734, dose=100):
    return {'candidate_id': 'candidate_1', 'candidate_number': 1,
            'physical_molecule_sha256': row['molecule_specification']['physical_molecule_sha256'],
            'antisense_5to3': SEQ, 'status': 'predicted',
            'conditions': {'dosage_nm': dose, 'transfection_method': 'Lipofection',
                           'dose_basis': 'assumed_reference_scenario', 'method_basis': 'explicit_structured_input'},
            'contexts': [{'transcript_id': 'TRANSCRIPT.1', 'species': 'human', 'assembly': 'TEST',
                          'context_basis': 'genomic_pre_mrna', 'reference_sequence_sha256': 'reference-one',
                          'binding_interval_bed0_half_open': ['chr1', 100, 120],
                          'status': 'predicted', 'predicted_inhibition_percent': value, 'input_id': 'input-one',
                          'rna_context': 'FULL_RNA_SEQUENCE_MUST_STAY_OUT',
                          'provenance': {'genome_fasta': '/reference/never-open.fa'}}]}


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_saved_selected_predictions_join_full_pool_by_molecule_job_and_geometry(tmp_path):
    row = candidate()
    others = [candidate('4-12-4'), candidate(job='second'), candidate(start=200)]
    saved_row = {**row, 'oligoai_evaluation': entry(row)}
    evaluation = {'model_sha256': 'saved-model', 'source_revision': 'saved-source',
                  'candidates': [saved_row['oligoai_evaluation']]}
    save(tmp_path, 'artifacts/intake/structured_goal.json', {
        'gene_symbol': 'TEST', 'design_programs': [{'program_id': 'p', 'modality': 'gapmer'}],
        'gapmer_design_jobs': [{'job_id': job, 'program_id': 'p'} for job in ('j', 'second')]})
    save(tmp_path, 'artifacts/candidates/candidate_assessments.json', {'rows': [row, *others]})
    save(tmp_path, 'artifacts/candidates/generated_aso_set.json', {'rows': [saved_row], 'oligoai_evaluation': evaluation})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}
    data = project_run(tmp_path)
    assert len(data['records']) == 4
    predicted = [r for r in data['records'] if r['oligoai']]
    assert len(predicted) == 1
    activity = predicted[0]['oligoai'][0]
    assert activity['value'] == 53.435787200927734
    assert activity['dosage_nm'] == 100 and activity['dose_basis'] == 'assumed_reference_scenario'
    assert activity['model_sha256'] == 'saved-model'
    assert all(r['oligoai_applicable'] for r in data['records'])
    document = basic_document('Gapmers', '# Results\nSaved activity.', candidate_data=data)
    assert 'OligoAI RNA reduction' in document and '53.4%' in document
    assert 'FULL_RNA_SEQUENCE_MUST_STAY_OUT' not in document and '/reference/never-open.fa' not in document
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_multiple_conditions_and_references_keep_separate_values():
    row = candidate()
    first, second = entry(row, 0, 100), entry(row, 110.5, 25)
    second['contexts'].append({**second['contexts'][0], 'transcript_id': 'TRANSCRIPT.2',
                               'predicted_inhibition_percent': -3.25, 'input_id': 'input-two'})
    before = deepcopy([row, first, second])
    result = project_activity(row, [(first, {}), (second, {}), (first, {})])
    assert [a['value'] for a in result] == [0, 110.5, -3.25]
    assert len({a['scenario_id'] for a in result}) == 3
    assert result[1]['conditions_id'] == result[2]['conditions_id']
    assert result[0]['conditions_id'] != result[1]['conditions_id']
    assert [row, first, second] == before


@pytest.mark.parametrize('status', ['error', 'unsupported', 'unavailable', 'pending', 'disabled'])
def test_noncompletion_never_becomes_a_value(status):
    row = candidate()
    saved = entry(row)
    saved['contexts'][0]['status'] = status
    assert project_activity(row, [(saved, {})])[0]['value'] is None
    saved['contexts'] = []
    saved.update(status=status, reason='explicitly_disabled' if status=='disabled' else 'saved_failure')
    result = project_activity(row, [(saved, {})])
    assert result[0]['status'] == status and result[0]['value'] is None and result[0]['reason']


def test_molecular_identity_contradictions_cannot_acquire_predictions():
    row = candidate()
    saved = entry(row)
    assert project_activity(candidate('4-12-4'), [(saved, {})]) == []
    saved['antisense_5to3'] = 'T' * 20
    assert project_activity(row, [(saved, {})]) == []


def test_explicit_candidate_number_and_hash_bind_detached_saved_evidence():
    row = candidate()
    index = activity_index([row], {'candidates': [entry(row)]})
    assert project_activity(row, index[activity_join_key(row)])[0]['value'] == 53.435787200927734
    assert activity_join_key(candidate(job='other')) not in index


def test_supplied_table_chemistry_and_activity_use_the_recorded_molecule():
    row = candidate()
    supplied = {'source': 'provided', 'declared_sequence_role': 'antisense_oligo_sequence',
                'sequence': SEQ, 'input_metadata': {'molecule_specification': row['molecule_specification']},
                'oligoai_evaluation': entry(row)}
    data = project_records([supplied], activity_evaluation={'model_sha256': 'saved-model'})
    assert data['records'][0]['oligoai'][0]['value'] == 53.435787200927734
    assert data['records'][0]['oligoai'][0]['model_sha256'] == 'saved-model'


def test_skipped_supplied_rows_keep_the_original_candidate_numbers():
    first,second = candidate(),candidate(start=200)
    a,b = entry(first,12,100),entry(second,88,25)
    a['candidate_number'],b['candidate_number'] = 2,3
    data = project_records([{'sequence':'ACGT','declared_sequence_role':'target_sequence'},first,second],
                           activity_evaluation={'candidates':[a,b]})
    assert [[a['value'] for a in r['oligoai']] for r in data['records']] == [[12],[88]]


def test_identical_supplied_molecules_keep_their_own_evaluation_conditions():
    first,second = candidate(),candidate()
    first.update(input_row_id='first',oligoai_evaluation=entry(first,12,100))
    second.update(input_row_id='second',oligoai_evaluation=entry(second,88,25))
    data = project_records([first,second])
    assert [[a['value'] for a in r['oligoai']] for r in data['records']] == [[12],[88]]
