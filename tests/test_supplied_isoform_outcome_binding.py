from copy import deepcopy

import pytest
import run_goal_intake as native
from aso_agents.isoform_outcome_contract import complete_isoform_outcomes


def target(label, **extra):
    return {'target_type': 'isoform', 'objective_type': 'primary_outcome',
            'label': label, 'desired_delta': 'increase', **extra}


def test_artifact_binding_is_representable_in_the_canonical_contract():
    from aso_agents.goal_contract import canonical_structured_goal_schema
    schema = canonical_structured_goal_schema()
    for name in ('evaluation_targets', 'splice_isoform_objectives'):
        assert schema['properties'][name]['items']['properties']['artifact_ref_path']['type'] == 'string'


def test_named_outcomes_are_not_dropped_when_other_objectives_already_exist():
    goal = {'gene_symbol': 'NEUTRAL', 'splice_isoform_objectives': [
        {'isoform_label': 'one', 'desired_abundance': 'decrease'}],
        'evaluation_targets': [target('one', isoform_label='one', desired_delta='decrease'),
                               target('custom isoform', artifact_ref_path='/data/custom.gb')]}
    assert complete_isoform_outcomes(goal)
    assert [(x['isoform_label'], x['desired_abundance']) for x in goal['splice_isoform_objectives']] == [
        ('one', 'decrease'), ('custom isoform', 'increase')]
    assert goal['splice_isoform_objectives'][1]['artifact_ref_path'] == '/data/custom.gb'
    assert goal['evaluation_targets'][1]['isoform_label'] == 'custom isoform'
    before = deepcopy(goal)
    assert not complete_isoform_outcomes(goal)
    assert goal == before


def test_monitoring_gene_and_foreign_outcomes_do_not_create_local_isoform_goals():
    goal = {'gene_symbol': 'NEUTRAL', 'evaluation_targets': [
        target('monitor', objective_type='safeguard'), target('gene', target_type='gene'),
        target('foreign', gene_symbol='ANOTHER')]}
    assert not complete_isoform_outcomes(goal)
    assert 'splice_isoform_objectives' not in goal


def test_conflicting_outcome_directions_are_not_silently_overwritten():
    goal = {'splice_isoform_objectives': [{'isoform_label': 'one', 'desired_abundance': 'decrease'}],
            'evaluation_targets': [target('one')]}
    before = deepcopy(goal)
    with pytest.raises(ValueError, match='direction_conflict'):
        complete_isoform_outcomes(goal)
    assert goal == before


def test_reference_versions_remain_distinct():
    goal = {'splice_isoform_objectives': [{'isoform_label': 'old', 'transcript_id': 'TX.1',
                                         'desired_abundance': 'increase'}],
            'evaluation_targets': [target('new', transcript_id='TX.2')]}
    assert complete_isoform_outcomes(goal)
    assert [x['transcript_id'] for x in goal['splice_isoform_objectives']] == ['TX.1', 'TX.2']


def test_opposing_products_can_share_a_reference_without_becoming_one_identity():
    goal = {'splice_isoform_objectives': [
        {'isoform_label': 'alternative acceptor product', 'reference_accession': 'NM_123456.7',
         'objective_scope': 'specific_isoform', 'desired_abundance': 'increase'},
        {'objective_scope': 'aggregate_class', 'transcript_id': 'NM_123456.7',
         'aggregate_class': {'basis': 'exon_presence', 'exon_label': 6, 'presence': 'excluded'},
         'desired_abundance': 'decrease'}],
        'evaluation_targets': [target('alternative acceptor product',
            isoform_label='alternative acceptor product', reference_transcript_id='NM_123456.7')]}
    before = deepcopy(goal)
    assert not complete_isoform_outcomes(goal)
    assert goal == before
    assert 'transcript_id' not in goal['splice_isoform_objectives'][0]


def test_a_reference_anchor_does_not_turn_an_outcome_into_the_reference_transcript():
    goal = {'evaluation_targets': [target('new splice product', reference_transcript_id='NM_123456.7')]}
    complete_isoform_outcomes(goal)
    objective = goal['splice_isoform_objectives'][0]
    assert objective['isoform_label'] == 'new splice product'
    assert objective['reference_accession'] == 'NM_123456.7'
    assert objective['objective_scope'] == 'specific_isoform'
    assert 'transcript_id' not in objective


def test_same_identified_transcript_still_rejects_opposing_outcomes():
    goal = {'splice_isoform_objectives': [{'isoform_label': 'first label', 'transcript_id': 'TX.1',
                                         'desired_abundance': 'decrease'}],
            'evaluation_targets': [target('second label', transcript_id='TX.1')]}
    with pytest.raises(ValueError, match='direction_conflict'):
        complete_isoform_outcomes(goal)


@pytest.mark.parametrize('surround', [('(', ')'), ('[', ']'), ('<', '>'), ('"', '"'), ("'", "'")])
def test_supplied_path_punctuation_preserves_actual_file(tmp_path, surround):
    path = tmp_path/'construct.gb'
    path.write_text('a supplied annotation')
    assert native._detect_snapgene_paths(f'Use {surround[0]}{path}{surround[1]}.') == [str(path)]


def test_quoted_spaces_and_literal_parentheses_in_filenames(tmp_path):
    path = tmp_path/'my construct (edited).genbank'
    path.write_text('a supplied annotation')
    assert native._detect_snapgene_paths(f'Use "{path}" as the supplied isoform.') == [str(path)]


def test_typed_artifact_outcome_is_ingested_without_model_guessing(tmp_path, monkeypatch):
    path = tmp_path/'a construct.gb'; path.write_text('annotation')
    goal = {'species': 'human', 'use_supplied_sequence_variants': False,
            'snapgene_resolution_errors': {str(path): 'previous failure'},
            'artifact_refs': [{'path': str(path), 'artifact_type': 'genbank'}],
            'evaluation_targets': [target('custom', artifact_ref_path=str(path))]}
    calls = []
    def ingest(p, **kwargs):
        calls.append((p, kwargs['derive_patient_variants']))
        return {'snapgene_path': p, 'exons': []}
    monkeypatch.setattr(native, 'ingest_snapgene_file', ingest)
    monkeypatch.setattr(native, '_select_snapgene_isoform_labels_with_llm',
                        lambda *a: pytest.fail('Explicit target/file bindings need no model guess'))
    monkeypatch.setattr(native, '_build_snapgene_transcript_entry', lambda a,l: {
        'transcript_id': 'SNAPGENE::neutral', 'metadata': {'source': 'snapgene'},
        'exons_tx': [{'start': 10, 'end': 40}], 'chrom': 'chr1', 'strand': '+'})
    final = {'snapgene_resolution_errors': {str(path): 'previous failure'}}
    result = native._prepare_snapgene_objectives(goal, final, str(tmp_path))
    assert result['preseeded_isoforms'] == ['custom']
    assert calls == [(str(path), True)]
    assert goal['splice_isoform_objectives'][0]['snapgene_path'] == str(path)
    assert 'patient_reference' not in goal
    assert 'snapgene_resolution_errors' not in goal
    assert 'snapgene_resolution_errors' not in final


def test_unavailable_declared_artifact_cannot_pass_as_no_file(tmp_path):
    path = tmp_path/'missing.dna'
    with pytest.raises(ValueError, match='supplied_isoform_reference_unavailable'):
        native._prepare_snapgene_objectives({'artifact_refs': [{'path': str(path)}]}, {})


def test_ingestion_failure_cannot_be_accepted_as_resolved(tmp_path, monkeypatch):
    path = tmp_path/'construct.gb'; path.write_text('broken annotation')
    goal = {'evaluation_targets': [target('custom', artifact_ref_path=str(path))]}
    def fail(*a, **k):
        raise ValueError('invalid annotation')
    monkeypatch.setattr(native, 'ingest_snapgene_file', fail)
    with pytest.raises(ValueError, match='supplied_isoform_reference_unresolved'):
        native._prepare_snapgene_objectives(goal, {})
    assert 'preseeded_isoform_transcripts' not in goal
