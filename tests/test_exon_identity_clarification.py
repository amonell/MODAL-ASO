"""Unresolved names pause execution; explicit answers reach physical targeting."""
from copy import deepcopy
import json
from types import SimpleNamespace

import jsonschema
import pytest

from aso_agents import curated_exon_nomenclature as naming
from aso_agents import exon_nomenclature_context as identity
from aso_agents import pipeline_tools as pipeline
from aso_agents.exon_identity_clarification import (
    ExonIdentityClarification, coordinate_exon, execution_exon_identity, require_resolved_exon_identities,
)
from aso_agents.goal_contract import canonical_structured_goal_schema
from test_curated_exon_nomenclature import fixture, mock_sources
from test_exon_nomenclature_lifecycle import goal_for


def unavailable(monkeypatch):
    monkeypatch.setattr(naming, '_read', lambda *_: (_ for _ in ()).throw(OSError('source unavailable')))


def define_group(goal):
    goal['splice_isoform_objectives'] = [{'objective_scope': 'aggregate_class',
        'reference_accession': 'ENST00000000001.2', 'desired_abundance': 'decrease',
        'aggregate_class': {'basis': 'exon_presence', 'exon_label': 2, 'presence': 'included'}}]


def test_constructed_model_reuses_attested_parent_metadata_for_explicit_coordinates():
    from aso_agents.report_exon_names import report_reference_catalog, project_exon_names
    parent={'transcript_id':'TX.2','chrom':'chr8','strand':'+','metadata':{'assembly_name':'GRCh38'},
            'exons_tx':[{'index':1,'start':101,'end':150},{'index':2,'start':501,'end':550}]}
    model={'transcript_id':'TX::constructed','chrom':'chr8','strand':'+',
           'metadata':{'explicit_target_modified':True,'base_isoform_label':'parent',
                       'base_reference_transcript_id':'TX.2','base_reference_snapshot':parent['metadata']},
           'exons_tx':[parent['exons_tx'][0],{'index':2,'start':301,'end':350},{'index':3,'start':501,'end':550}]}
    goal={'isoform_transcripts':{'parent':parent,'constructed':model},
          'explicit_splice_targets':[{'transcript_id':'TX.2','isoform_label':'constructed',
              'requested_exon_label':'1x','chrom':'chr8','start':301,'end':350,'strand':'+','genome_build':'GRCh38'}]}
    before=deepcopy(goal)
    feature=identity.resolved_exon_requests(goal)[0]
    assert feature['status']=='explicit_feature_coordinates'
    assert feature['reference_transcript_id']=='TX::constructed'
    projected=report_reference_catalog(goal)
    assert project_exon_names(projected['constructed'])['exons'][1]['label']=='Requested exon 1x'
    assert goal==before
    model['metadata']['base_reference_transcript_id']='TX.3'
    assert identity.resolved_exon_requests(goal)[0]['status']=='unresolved'


@pytest.mark.parametrize('modality', ['gapmer', 'steric'])
def test_unknown_name_requests_one_question_for_repeated_feature(monkeypatch, modality):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry, modality=modality)
    define_group(goal)
    goal['evaluation_targets'] = [{'exon_label': 2, 'reference_transcript_id': 'ENST00000000001.2'}]
    identity.prepare_goal_exon_nomenclature(goal)
    with pytest.raises(ExonIdentityClarification) as raised:
        require_resolved_exon_identities(goal)
    assert len(raised.value.features) == 3
    assert len(raised.value.questions) == 1
    question = raised.value.questions[0]
    assert 'exon 2' in question and 'TEST' in question and 'ENST00000000001.2' in question
    assert 'coordinates' in question and '5′ end' in question and 'assembly' in question
    assert goal['design_programs'][0]['exon_label'] == 2


def test_gene_wide_design_and_offline_browsing_need_no_naming_answer(monkeypatch):
    from aso_agents.report_exon_names import project_exon_names
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['design_programs'][0].pop('exon_label')
    identity.prepare_goal_exon_nomenclature(goal)
    require_resolved_exon_identities(goal)
    assert project_exon_names(entry)['exons'][1]['label'] == 'second annotated exon'


def test_verified_public_name_reaches_execution_without_question(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    require_resolved_exon_identities(goal)
    saved = execution_exon_identity(entry, 2)
    assert saved['status'] == 'verified_biological_label' and saved['resolved_transcript_ordinal'] == 3


@pytest.mark.parametrize('negative', [False, True])
def test_explicit_ordinal_answer_reaches_group_and_physical_region(monkeypatch, negative):
    import run_goal_intake as native
    unavailable(monkeypatch)
    _, entry = fixture(negative=negative); goal = goal_for(entry)
    define_group(goal)
    goal['design_programs'][0]['exon_index'] = 3
    identity.prepare_goal_exon_nomenclature(goal)
    require_resolved_exon_identities(goal)
    requests = identity.resolved_exon_requests(goal)
    assert all(item['status'] == 'explicit_transcript_ordinal' for item in requests)
    assert all(item['resolved_transcript_ordinal'] == 3 for item in requests)
    assert all(not item['biological_label_verified'] for item in requests)
    assert native._resolve_authoritative_transcript_exon_ordinal(entry, 2, goal) == 3
    rows, _, _ = native._build_exon_constraint_rows(constraint={'exon_label': 2},
                                                   iso_label='requested RNA', transcript_entry=entry)
    assert rows[0][1:3] == [entry['exons_tx'][2]['start']-1, entry['exons_tx'][2]['end']]
    ordinal, _ = native._resolve_splice_relative_effective_exon(constraint={}, structured_goal=goal,
        iso_label='requested RNA', transcript_entry=entry, isoform_transcripts=goal['isoform_transcripts'],
        requested_exon_label=2, requested_site='acceptor')
    assert ordinal == 3
    assert goal['design_programs'][0]['exon_label'] == 2
    assert not entry['metadata']['curated_exon_nomenclature']['status'] == 'verified'


@pytest.mark.parametrize('negative', [False, True])
def test_supplied_complete_exon_span_needs_no_public_mapping(monkeypatch, negative):
    unavailable(monkeypatch)
    _, entry = fixture(negative=negative); goal = goal_for(entry, label='2a')
    exon = entry['exons_tx'][2]
    goal['explicit_splice_targets'] = [{'exon_label': '2a', 'reference_transcript_id': 'ENST00000000001.2',
        'chrom': 'chr3', 'start': exon['start'], 'end': exon['end'], 'strand': entry['strand'], 'genome_build': 'hg38'}]
    identity.prepare_goal_exon_nomenclature(goal)
    require_resolved_exon_identities(goal)
    assert all(item['status'] == 'explicit_feature_coordinates' for item in identity.resolved_exon_requests(goal))
    assert execution_exon_identity(entry, '2a')['resolved_transcript_ordinal'] == 3


@pytest.mark.parametrize('change', ['assembly', 'chromosome', 'strand', 'partial_span', 'version', 'ordinal'])
def test_invalid_answers_cannot_release_the_gate(monkeypatch, change):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['explicit_splice_targets'] = [{'exon_label': 2, 'reference_transcript_id': 'ENST00000000001.2',
        'chrom': 'chr3', 'start': 1101, 'end': 1120, 'strand': '+', 'genome_build': 'GRCh38'}]
    target = goal['explicit_splice_targets'][0]
    if change == 'assembly': target['genome_build'] = 'GRCh37'
    if change == 'chromosome': target['chrom'] = 'chr4'
    if change == 'strand': target['strand'] = '-'
    if change == 'partial_span': target['start'] += 1
    if change == 'version': target['reference_transcript_id'] = 'ENST00000000001.1'
    if change == 'ordinal':
        goal.pop('explicit_splice_targets')
        goal['design_programs'][0]['exon_index'] = 40
    identity.prepare_goal_exon_nomenclature(goal)
    with pytest.raises(ExonIdentityClarification):
        require_resolved_exon_identities(goal)


def test_binding_windows_cannot_supply_exon_identity(monkeypatch):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['design_programs'][0].update(start=1101, end=1120)
    goal['region_constraints'] = [{'exon_label': 2, 'intervals_bed0_half_open': [['chr3', 1100, 1120]]}]
    identity.prepare_goal_exon_nomenclature(goal)
    with pytest.raises(ExonIdentityClarification): require_resolved_exon_identities(goal)


def test_constructed_model_checks_its_recorded_parent_assembly():
    _, entry = fixture()
    entry['explicit_target_modified'] = True
    entry['metadata']['base_reference_snapshot'] = {'assembly_name': entry['metadata'].pop('assembly_name')}
    physical = {'chrom': 'chr3', 'start': 1101, 'end': 1120, 'assembly': 'GRCh38'}
    assert coordinate_exon(physical, entry)['index'] == 3
    assert coordinate_exon({**physical, 'assembly': 'GRCh37'}, entry) is None


def test_conflicting_scoped_answers_require_clarification(monkeypatch):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['design_programs'][0]['exon_index'] = 2
    other = {**goal['design_programs'][0], 'program_id': 'q', 'exon_index': 3}
    goal['design_programs'].append(other)
    identity.prepare_goal_exon_nomenclature(goal)
    with pytest.raises(ExonIdentityClarification): require_resolved_exon_identities(goal)


def test_conflicting_repeated_answers_produce_one_question(monkeypatch):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['design_programs'][0]['exon_index'] = 2
    goal['evaluation_targets'] = [{'exon_label': 2, 'exon_index': 3, 'reference_transcript_id': 'ENST00000000001.2'}]
    identity.prepare_goal_exon_nomenclature(goal)
    with pytest.raises(ExonIdentityClarification) as raised: require_resolved_exon_identities(goal)
    assert len(raised.value.questions) == 1


def test_execution_answer_is_bound_to_the_reference_structure(monkeypatch):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['design_programs'][0]['exon_index'] = 3
    identity.prepare_goal_exon_nomenclature(goal); require_resolved_exon_identities(goal)
    entry['exons_tx'][2]['start'] += 1
    assert execution_exon_identity(entry, 2) is None


@pytest.mark.parametrize('conflicting_answer', [False, True])
def test_observed_transcript_conversion_supplies_identity_or_detects_conflict(monkeypatch, conflicting_answer):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry, label=10, modality='steric')
    goal['design_programs'][0]['desired_effect'] = 'skipping'
    other = deepcopy(entry)
    other['transcript_id'] = 'ENST00000000002'
    other['metadata']['versioned_id'] = 'ENST00000000002.1'
    other['exons_tx'] = [other['exons_tx'][0], {**other['exons_tx'][2], 'index': 2}]
    goal['isoform_transcripts']['skipped RNA'] = other
    goal['splice_isoform_objectives'] = [
        {'transcript_id': 'ENST00000000001.2', 'desired_abundance': 'decrease'},
        {'transcript_id': 'ENST00000000002.1', 'desired_abundance': 'increase'}]
    if conflicting_answer: goal['design_programs'][0]['exon_index'] = 3
    identity.prepare_goal_exon_nomenclature(goal)
    if conflicting_answer:
        with pytest.raises(ExonIdentityClarification): require_resolved_exon_identities(goal)
    else:
        require_resolved_exon_identities(goal)
        resolved = identity.resolved_exon_requests(goal)[0]
        assert resolved['status'] == 'verified_structural_event' and resolved['resolved_transcript_ordinal'] == 2


def test_resolution_tool_converts_identity_blocker_into_a_user_question(monkeypatch, tmp_path):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['target_context'] = 'general human baseline'
    run_dir = tmp_path/'run'; run_dir.mkdir()
    state_path = tmp_path/'state.json'
    state = {'status': 'goal_schema_ready', 'phase': 'goal_schema_ready', 'request': 'Target exon 2',
             'run_dir': str(run_dir), 'active_runner': {'tool': 'tool_resolve_isoforms_and_sites'}, 'force_rerun_stages': []}
    state_path.write_text(json.dumps(state))
    context = {'payload': {}, 'structured_goal': goal, 'final_payload': {'structured_goal': goal}, 'request_envelope': {}}
    monkeypatch.setattr(pipeline, 'provider_credentials_available', lambda: False)
    monkeypatch.setattr(pipeline, '_prepare_execution_stage', lambda **_: {'status': 'ok', 'state': state, 'run_dir': str(run_dir)})
    monkeypatch.setattr(pipeline, '_load_goal_context_from_run_dir', lambda _: context)
    def resolve(*_, **__): require_resolved_exon_identities(goal)
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: SimpleNamespace(_resolve_isoform_transcripts=resolve))
    result = pipeline.tool_resolve_isoforms_and_sites(str(state_path))
    assert result['needs_user_input'] and 'exon 2' in result['questions'][0]
    assert result['clarification_kind'] == 'goal_intake'
    assert pipeline._load_state(str(state_path))['clarification']['restart_stage'] == 'intake'


def test_verified_external_accession_alias_reuses_the_same_answer(monkeypatch):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    entry['metadata']['reference_accession_resolution'] = {
        'exact_accession_verified': True, 'requested_accession': 'NM_999999.4'}
    goal['design_programs'][0].update(transcript_id='NM_999999.4', exon_index=3)
    identity.prepare_goal_exon_nomenclature(goal); require_resolved_exon_identities(goal)
    assert execution_exon_identity(entry, 2)['reference_transcript_id'] == 'ENST00000000001.2'
    goal['design_programs'][0]['transcript_id'] = 'NM_999999.3'
    with pytest.raises(ExonIdentityClarification): require_resolved_exon_identities(goal)


def test_compiler_schema_accepts_explicit_identity_answers_without_renaming():
    schema = canonical_structured_goal_schema()['properties']
    for collection in ('design_programs', 'evaluation_targets', 'splice_isoform_objectives'):
        assert 'exon_index' in schema[collection]['items']['properties']
    jsonschema.validate({'basis': 'exon_presence', 'exon_label': 10, 'exon_index': 9, 'presence': 'included'},
        schema['splice_isoform_objectives']['items']['properties']['aggregate_class'])
    jsonschema.validate({'chrom': 'chr3', 'start': 1101, 'end': 1120, 'exon_label': 2,
                        'reference_transcript_id': 'ENST00000000001.2', 'genome_build': 'GRCh38'},
                       schema['explicit_splice_targets']['items'])


@pytest.mark.parametrize('with_label', [False, True])
def test_ordinal_survives_intake_and_region_normalization(monkeypatch, with_label):
    import run_goal_intake as native
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    define_group(goal)
    specification = goal['splice_isoform_objectives'][0]['aggregate_class']
    specification['exon_index'] = 3
    if not with_label: specification.pop('exon_label')
    native._normalize_splice_isoform_objectives_in_goal(goal)
    assert goal['splice_isoform_objectives'][0]['aggregate_class']['exon_index'] == 3
    assert ('exon_label' in goal['splice_isoform_objectives'][0]['aggregate_class']) == with_label
    constraints = [{'source_type': 'exon', 'exon_index': 3, 'transcript_id': 'ENST00000000001.2'},
        {'source_type': 'splice_relative_window', 'splice_relative_window': {'exon_index': 3,
         'reference_transcript_id': 'ENST00000000001.2', 'anchor_site': 'acceptor',
         'relative_start_nt': -3, 'relative_end_nt': 3}}]
    if with_label:
        constraints[0]['exon_label'] = 2
        constraints[1]['splice_relative_window']['exon_label'] = 2
    normalized = native._normalize_region_constraints(constraints)
    assert normalized[0]['exon_index'] == 3
    assert normalized[1]['splice_relative_window']['exon_index'] == 3
    rows, _ = native._build_splice_relative_window_rows(constraint=normalized[1],
        iso_label='requested RNA', transcript_entry=entry, structured_goal=goal,
        isoform_transcripts=goal['isoform_transcripts'])
    assert rows and rows[0][1:3] == [1097, 1104]


@pytest.mark.parametrize('locked', [False, True])
def test_native_resolution_pauses_before_strategy(monkeypatch, locked):
    import run_goal_intake as native
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    goal['splice_isoform_objectives'] = [{'isoform_label': 'requested RNA',
        'transcript_id': 'ENST00000000001.2', 'desired_abundance': 'decrease'}]
    goal['isoform_resolution_locked'] = locked
    monkeypatch.setattr(native, 'list_transcripts', lambda **_: {'transcripts': [
        {'id': 'ENST00000000001.2', 'biotype': 'protein_coding'}]})
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda _: {
        **entry['metadata'], 'id': 'ENST00000000001.2', 'seq_region_name': 'chr3', 'strand': 1,
        'exons_tx': deepcopy(entry['exons_tx'])})
    monkeypatch.setattr(native, '_compute_splice_site_plan', lambda *_: pytest.fail('Strategy ran with ambiguous identity'))
    with pytest.raises(ExonIdentityClarification):
        native._resolve_isoform_transcripts(goal, {}, allow_interactive=False)


def test_resumed_report_asks_before_shortlist_and_answer_restarts_intake(monkeypatch, tmp_path):
    unavailable(monkeypatch)
    _, entry = fixture(); goal = goal_for(entry)
    run_dir = tmp_path/'run'; run_dir.mkdir()
    state_path = tmp_path/'state.json'
    state = {'run_dir': str(run_dir), 'request': 'Design ASOs against exon 2', 'status': 'report_ready',
             'phase': 'report_ready', 'active_runner': {'tool': 'tool_build_report_payload'}, 'force_rerun_stages': []}
    state_path.write_text(json.dumps(state))
    pipeline._save_goal_payload(str(run_dir), {}, goal, {'structured_goal': goal})
    monkeypatch.setattr(pipeline, '_prepare_execution_stage', lambda **_: {'status': 'ok', 'state': state, 'run_dir': str(run_dir)})
    monkeypatch.setattr(pipeline, '_emit_v2_generated_design_artifacts', lambda *_, **__: pytest.fail('Shortlist ran'))
    result = pipeline.tool_build_report_payload(str(state_path))
    assert result['needs_user_input'] and len(result['questions']) == 1
    assert not (run_dir/'report.html').exists()
    assert pipeline._load_state(str(state_path))['status'] == 'needs_user_input'
    answer = 'Use the third exon counted from the 5 prime end of ENST00000000001.2.'
    assert pipeline.submit_clarification_answers(str(state_path), [answer])['restart_stage'] == 'intake'
    assert pipeline.apply_clarification_to_goal(str(state_path))['restart_stage'] == 'intake'
    assert pipeline.restart_from_stage(str(state_path), 'intake', 'Apply confirmed exon identity')['status'] == 'ok'
    saved = pipeline._load_state(str(state_path))
    assert saved['request'] == state['request'] and saved['clarification_qa_history'][-1]['answer'] == answer
    # The existing evidence-aware compiler has a typed destination for that
    # answer; replay its output to check the gate and physical consumer together.
    goal['design_programs'][0]['exon_index'] = 3
    require_resolved_exon_identities(goal)
    assert execution_exon_identity(entry, 2)['resolved_transcript_ordinal'] == 3
