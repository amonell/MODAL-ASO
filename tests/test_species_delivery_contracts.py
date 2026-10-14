"""Reference and delivery invariants using neutral, non-benchmark fixtures."""
from copy import deepcopy

import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools as pipeline
from aso_agents.species_targets import scope_contract_errors, resolve_explicit_target
from aso_agents.design_delivery import design_delivery
from aso_agents.analysis_runtime import describe
from scripts import encode_rbp_peaks as encode


def goal():
    return {'species': 'human', 'gene_symbol': 'GENE', 'species_targets': [
        {'species': 'human', 'transcript_id': 'TX_A.2'},
        {'species': 'mouse', 'transcript_id': 'TX_B.4'}],
        'transcript_lock': {'kind': 'explicit_id', 'value': 'TX_A.2'}}


def structure(tid):
    organism = {'TX_A.2': 'homo_sapiens', 'TX_B.4': 'mus_musculus'}[tid]
    return {'id': tid.split('.')[0], 'versioned_id': tid, 'species': organism,
            'gene_id': 'GENE_'+tid.split('.')[0], 'seq_region_name': '7', 'strand': 1,
            'exons_tx': [{'index': 1, 'start': 11, 'end': 30}], 'reference_source': 'neutral_annotation'}


def gene_lookup(gene_id):
    return {'gene_id': gene_id, 'display_name': 'GENE',
            'species': 'human' if gene_id.endswith('A') else 'mouse'}


def test_scalar_anchor_and_lock_cannot_join_separate_reference_identities():
    value = goal()
    assert not scope_contract_errors(value)
    value['species'] = 'human_mouse'
    assert 'species' in scope_contract_errors(value)
    value['species'] = 'human'
    value['transcript_lock']['value'] = 'TX_A.2; TX_B.4'
    assert 'transcript_lock.value' in scope_contract_errors(value)
    value['transcript_lock']['value'] = 'TX_A'
    assert not scope_contract_errors(value)


def test_contract_does_not_whitelist_organisms_or_replace_optional_comparators():
    assert not scope_contract_errors({'species': 'custom_organism', 'cross_species_requirement': {'species': ['mouse'], 'required': False}})
    value = {'species': 'custom_organism', 'species_targets': [{'species': 'custom_organism'}, {'species': 'mouse'}]}
    assert not scope_contract_errors(value)


def test_canonical_boundary_returns_repair_paths():
    value = goal()
    value.update(species='human_mouse', design_programs=[{'program_id': 'p', 'label': 'p', 'modality': 'gapmer', 'intent': 'knockdown', 'objective': 'knockdown', 'desired_delta': 'decrease'}])
    assert 'context.structured_goal.species' in pipeline._canonical_design_goal_contract_issues(value)


def test_explicit_reference_beats_stale_negative_homology_cache(monkeypatch):
    value = goal()
    value['gene_id'] = 'GENE_TX_A'
    value['cross_species_targets'] = {key: {'status': 'no_ortholog'} for key in ['human', 'mouse']}
    monkeypatch.setattr(native, '_fetch_transcript_structure', structure)
    monkeypatch.setattr(native, 'lookup_gene_by_id', gene_lookup)
    monkeypatch.setattr(native, 'resolve_ortholog_gene', lambda *_: pytest.fail('Explicit reference replaced by homology'))
    result = native._resolve_cross_species_gene_targets(value, ['human', 'mouse'])
    assert result['mouse']['transcript_ids'] == ['TX_B.4']
    assert all(item['status'] == 'resolved' for item in result.values())
    assert result['mouse']['resolution_source'] == 'explicit_species_reference'


@pytest.mark.parametrize('mutation,error', [({'species': 'human'}, 'species'), ({'versioned_id': 'TX_B.5'}, 'version'), ({'gene_id': None}, 'parent')])
def test_wrong_explicit_reference_never_becomes_homology_or_default(mutation, error):
    def fetch(tid):
        return {**structure(tid), **mutation}
    result = resolve_explicit_target(goal(), 'mouse', fetch, gene_lookup)
    assert result['status'] == 'error' and error in result['error']


def test_explicit_cdna_scope_bypasses_canonical_and_retains_organism(monkeypatch):
    monkeypatch.setattr(native, 'list_transcripts', lambda **_: pytest.fail('Explicit transcript replaced with canonical'))
    monkeypatch.setattr(native, '_fetch_transcript_structure', structure)
    observed = {}
    def build(isoforms, species):
        observed.update(isoforms)
        return {key: 'ACGT' * 5 for key in isoforms}
    monkeypatch.setattr(native, '_build_isoform_cdna_sequences', build)
    sequences, error = native._resolve_cross_species_cdna_sequences('GENE', 'mouse', gene_id='GENE_TX_B', transcript_ids=['TX_B.4'])
    assert error is None and list(sequences) == ['TX_B.4']
    assert observed['TX_B.4']['metadata']['species'] == 'mus_musculus'
    assert observed['TX_B.4']['metadata']['gene_id'] == 'GENE_TX_B'


def test_explicit_cdna_scope_rejects_wrong_organism(monkeypatch):
    monkeypatch.setattr(native, '_fetch_transcript_structure', structure)
    monkeypatch.setattr(native, '_build_isoform_cdna_sequences', lambda *_: pytest.fail('Extracted a mismatched reference'))
    seq, error = native._resolve_cross_species_cdna_sequences('GENE', 'human', transcript_ids=['TX_B.4'])
    assert not seq and 'species' in error


@pytest.mark.parametrize('minimum,expected', [(None, 'unfulfilled'), (1, 'unfulfilled'), (0, 'fulfilled')])
def test_empty_result_permission_is_explicit(minimum, expected):
    value = {} if minimum is None else {'candidate_minimum': minimum}
    result = design_delivery(value, {}, {'steric': {'requested': True, 'status': 'no_candidates_after_constraints'}})
    assert result['status'] == expected


@pytest.mark.parametrize('status', ['unknown', 'skipped', 'evidence_unavailable'])
def test_zero_minimum_does_not_excuse_missing_execution_or_evidence(status):
    result = design_delivery({'candidate_minimum': 0}, {}, {'steric': {'requested': True, 'status': status}})
    assert result['status'] == 'unfulfilled'


def test_failed_rbp_search_never_becomes_completed_empty_search():
    value = {'intent': 'binding', 'rbp_binding_result': {'error': 'encode_query_failed'}}
    outcomes = pipeline._build_branch_outcomes_for_report(value, {})
    assert outcomes['steric']['status'] == 'evidence_unavailable'


def test_assay_resolution_uses_available_vocabulary_and_rejects_ambiguity():
    assert encode.resolve_assay_title('Provider assay X', ['assay X', 'assay Y']) == 'assay X'
    assert encode.resolve_assay_title('Provider assay X and assay Y', ['assay X', 'assay Y']) is None
    assert encode.resolve_assay_title('unknown', ['assay X']) is None


def test_assay_lookup_resolves_from_real_provider_labels(monkeypatch):
    calls = []
    def search(rbp, assay_title, include_non_released, encode_base):
        calls.append(assay_title)
        return [] if assay_title else [{'assay_title': 'assay X'}]
    monkeypatch.setattr(encode, '_search_experiments', search)
    monkeypatch.setattr(encode, '_collect_candidates', lambda *args, **kwargs: [{'score': 1}])
    result = encode.select_best_dataset('FACTOR', assay_title='Provider assay X')
    assert result['selected'] and calls == ['Provider assay X', None]
    assert result['assay_resolution']['resolved'] == 'assay X'


def test_no_eligible_assay_is_unresolved_not_biological_absence(monkeypatch):
    monkeypatch.setattr(encode, '_search_experiments', lambda *args, **kwargs: [])
    result = encode.select_best_dataset('FACTOR', assay_title='unknown')
    assert result['error'] == 'encode_assay_unresolved'


def test_unrelated_search_hit_cannot_become_an_rbp_dataset(monkeypatch):
    monkeypatch.setattr(encode, '_ensure_experiment_detail', lambda exp, base: exp)
    experiment = {'target': {'label': 'OTHER'}, 'files': [{'file_format': 'bed', 'output_type': 'peaks'}]}
    assert encode._collect_candidates([experiment], 'FACTOR', None, None, None, False, 'https://neutral.invalid', True) == []


def test_analysis_diagnostics_preserve_field_shapes_without_leaf_values():
    record = describe({'length_nt': 20, 'rows': [{'name': 'private_value', 'score': 0.9}]})
    assert record['fields']['length_nt']['type'] == 'builtins.int'
    assert 'private_value' not in str(record)
    cyclic = {}; cyclic['self'] = cyclic
    assert describe(cyclic)['fields']['self']['recursive']


def test_explicit_rows_require_actual_intervals_and_exon_constraints_remain_typed():
    from jsonschema import Draft202012Validator
    from aso_agents.goal_contract import _REGION_CONSTRAINT_SCHEMA
    validator = Draft202012Validator(_REGION_CONSTRAINT_SCHEMA)
    invalid = {'source_type': 'explicit_rows', 'aso_overlap_requirement': 'full_containment', 'intervals_bed0_half_open': []}
    assert list(validator.iter_errors(invalid))
    assert not list(validator.iter_errors({'source_type': 'exon', 'exon_label': 3, 'aso_overlap_requirement': 'full_containment'}))
    assert not list(validator.iter_errors({**invalid, 'rows': [['chr7', 10, 30, 'site', 0, '+']]}))


def test_future_design_outputs_satisfy_intake_but_unrelated_designs_do_not():
    envelope = {'tasks': [{'task_id': 'design', 'task_type': 'design_new_asos', 'context': {}},
                          {'task_id': 'evaluate', 'task_type': 'evaluate_existing_aso_set', 'context': {'species': 'human', 'gene_symbol': 'GENE'}}],
                'task_dependencies': [{'upstream_task_id': 'design', 'downstream_task_id': 'evaluate'}]}
    assert 'provided_asos_missing' not in pipeline._request_validation_summary(envelope)['tasks']['evaluate']['issues']
    envelope['task_dependencies'] = []
    assert 'provided_asos_missing' in pipeline._request_validation_summary(envelope)['tasks']['evaluate']['issues']


def test_downstream_evaluation_loads_only_declared_molecules_in_synthesis_orientation(tmp_path, monkeypatch):
    import json
    path = tmp_path / 'generated.json'
    path.write_text(json.dumps({'rows': [{'sequence': 'AACCGG', 'synthesized_antisense_sequence': 'CCGGTT'}]}))
    tasks = [{'task_id': 'design', 'task_type': 'design_new_asos'},
             {'task_id': 'other', 'task_type': 'design_new_asos'},
             {'task_id': 'evaluate', 'task_type': 'evaluate_existing_aso_set', 'depends_on': ['design']}]
    monkeypatch.setattr(pipeline, '_task_graph_tasks', lambda _: tasks)
    used = []
    def source(state, task):
        used.append(task); return path
    monkeypatch.setattr(pipeline, '_generated_aso_set_path_for_task', source)
    result = pipeline._generated_evaluation_inputs({}, 'evaluate')
    assert used == ['design']
    assert result['rows'][0]['sequence'] == 'CCGGTT'
    assert result['rows'][0]['source_sequence_identity']['target_binding_sequence'] == 'AACCGG'
    path.write_text(json.dumps({'rows': [{'sequence': 'AACCGG'}]}))
    assert pipeline._generated_evaluation_inputs({}, 'evaluate')['error'] == 'upstream_candidate_synthesis_identity_unresolved'


def test_completed_empty_rbp_search_retains_dataset_and_reference_evidence(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(native, '_select_encode_rbp_dataset', lambda **_: {'selected': {
        'file_format': 'bed', 'download_url': 'https://neutral.invalid/peaks.bed',
        'file_accession': 'neutral', 'assembly': 'GRCh38'}})
    monkeypatch.setattr(native, '_download_encode_peak_file', lambda url, path: Path(path).write_text('chr9\t1\t9\tpeak\t1\t+\n'))
    monkeypatch.setattr(native, '_ensure_isoform_transcripts_for_target_region', lambda _: None)
    monkeypatch.setattr(native, '_compute_gene_span', lambda _: {'chrom': 'chr7', 'start0': 10, 'end': 30})
    monkeypatch.setattr(native, '_build_utr_bed_rows', lambda *args: [['chr7', 10, 30, 'utr', 0, '+']])
    monkeypatch.setattr(native, '_select_isoform_for_utr', lambda *args: ('TX', {'transcript_id': 'TX'}))
    rows, summary = native._materialize_rbp_binding_rows({'rbp': 'FACTOR'},
        {'species': 'human', 'gene_symbol': 'GENE', 'target_region': 'utr3'}, str(tmp_path))
    assert not rows and not summary.get('error')
    assert summary['search_status'] == 'completed' and summary['status'] == 'no_overlapping_sites'
    assert summary['selected_dataset']['file_accession'] == 'neutral'
    assert summary['feature_scope']['reference_transcript'] == 'TX'
    assert Path(summary['bed_path']).exists()


def test_local_annotation_retains_parent_gene_in_resolved_structure():
    from aso_agents.transcript_chooser_agent import _transcript_structure_from_payload
    payload = {'id': 'TX', 'version': 2, 'Parent': 'GENE', 'species': 'custom_species',
               'start': 10, 'end': 20, 'strand': 1, 'seq_region_name': '7',
               'Exon': [{'start': 10, 'end': 20}]}
    resolved = _transcript_structure_from_payload(payload, 'TX.2')
    assert resolved['gene_id'] == 'GENE' and resolved['species'] == 'custom_species'


def test_zero_minimum_requires_an_actual_empty_search_outcome():
    result = design_delivery({'candidate_minimum': 0}, {}, {'gapmer': {'requested': True, 'status': 'outputs_present'}})
    assert 'empty_result_search_completion_unestablished' in result['errors']


def test_failed_generic_constraint_materialization_cannot_claim_an_empty_search():
    goal = {'intent': 'knockdown', 'materialized_region_constraints': {'global': [
        {'materialization_summary': {'status': 'unresolved', 'error': 'reference_unavailable'}}]}}
    outcomes = pipeline._build_branch_outcomes_for_report(goal, {})
    assert outcomes['gapmer']['status'] == 'evidence_unavailable'


def test_explicit_primary_reference_does_not_depend_on_symbol_lookup(monkeypatch):
    monkeypatch.setattr(native, '_fetch_transcript_structure', structure)
    monkeypatch.setattr(native, 'lookup_gene_by_id', gene_lookup)
    monkeypatch.setattr(native, 'lookup_gene_id', lambda *_: pytest.fail('Explicit transcript required a symbol lookup'))
    value = goal()
    result = native._resolve_cross_species_gene_targets(value, ['human', 'mouse'])
    assert all(item['status'] == 'resolved' for item in result.values())


def test_finalizer_preserves_conflicting_anchors_for_repair_instead_of_polluting_species_scope():
    value = goal()
    value.update(gene_symbol='GENE', intent='knockdown', target_context='cultured cells',
                 cross_species_requirement={'species': ['human', 'mouse'], 'required': True},
                 design_programs=[{'program_id': 'p', 'label': 'p', 'modality': 'gapmer', 'intent': 'knockdown', 'objective': 'decrease RNA', 'desired_delta': 'decrease'}])
    context = {'species': 'human;mouse', 'gene_symbol': 'GENE', 'target_context': 'cultured cells', 'structured_goal': value}
    task = pipeline._finalize_design_task_contract({'task_id': 'design', 'task_type': 'design_new_asos', 'context': context}, {})
    final = task['context']['structured_goal']
    assert final['species'] == 'human'
    assert final['cross_species_requirement']['species'] == ['human', 'mouse']
    assert len(final['species_targets']) == 2
    validation = pipeline._request_validation_summary({'tasks': [task]})
    assert 'context.species' in validation['tasks']['design']['repairable_issues']


def test_aliases_of_the_same_organism_are_not_anchor_conflicts():
    from aso_agents.species_targets import anchor_conflict
    assert not anchor_conflict({'species': 'Homo sapiens'}, {'species': 'human'})
    assert not anchor_conflict({}, {'species': 'custom_species'})


@pytest.mark.parametrize('kind', ['general', 'artifact', 'goal_contract', 'target_context'])
def test_attachment_paths_in_unrelated_answers_cannot_reconfigure_references(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pipeline, '_parse_reference_provisioning_decision', lambda *_: pytest.fail('Unrelated answer interpreted as provisioning'))
    state = {'clarification': {'kind': kind, 'last_answers': [{'answer': f'Read the supplied inputs in {tmp_path}.'}]}}
    assert pipeline._reference_root_from_scoped_clarification(state) == ''


@pytest.mark.parametrize('action,expected', [('use_existing', '/declared/reference'), ('continue', ''), ('download', ''), ('ask', '')])
def test_only_typed_existing_reference_decision_changes_root(monkeypatch, action, expected):
    monkeypatch.setattr(pipeline, '_parse_reference_provisioning_decision', lambda *_: {'action': action, 'reference_data_root': '/declared/reference'})
    assert pipeline._reference_root_from_scoped_clarification({'clarification': {'kind': 'reference_preflight'}}) == expected


def test_finalizer_cannot_legitimize_invalid_anchor_by_inventing_a_species_target():
    value = goal()
    value['species'] = 'combined_organisms'
    task = pipeline._finalize_design_task_contract({'task_type': 'design_new_asos', 'context': {
        'species': 'combined_organisms', 'structured_goal': value}}, {})
    final = task['context']['structured_goal']
    assert len(final['species_targets']) == 2
    assert 'species' in scope_contract_errors(final)


@pytest.mark.parametrize('layout', ['nested', 'legacy', 'nested_with_stale_legacy'])
def test_integrity_reads_actual_goal_before_accepting_report_only_run(tmp_path, layout):
    import json
    value = {'intent': 'binding', 'candidate_minimum': 0}
    payload = {'final_payload': {}}
    if layout != 'legacy':
        payload['final_payload']['structured_goal'] = value
    if layout != 'nested':
        payload['structured_goal'] = value if layout == 'legacy' else {'gene_symbol': 'STALE'}
    (tmp_path / 'goal_intake.json').write_text(json.dumps(payload))
    result = pipeline._extract_pipeline_integrity(str(tmp_path),
        {'exists': {'report_md': True, 'report_html': True}}, skip_router=True)
    assert not result['ok']
    assert 'requested_design_not_executed:steric' in result['blocking_errors']
    assert 'empty_result_search_completion_unestablished' in result['blocking_errors']


def test_integrity_accepts_real_delivered_molecule_with_nested_goal(tmp_path):
    import json
    payload = {'final_payload': {'structured_goal': {'intent': 'binding', 'candidate_minimum': 1},
                               'tiling_results': {'row_count': 1}}}
    (tmp_path / 'goal_intake.json').write_text(json.dumps(payload))
    (tmp_path / 'generated_aso_set.json').write_text(json.dumps({'rows': [
        {'sequence': 'AACCGG', 'synthesized_antisense_sequence': 'CCGGTT'}]}))
    result = pipeline._extract_pipeline_integrity(str(tmp_path),
        {'exists': {'report_md': True, 'report_html': True}}, skip_router=True)
    assert result['ok'] and result['design_delivery']['distinct_oligos'] == 1


def test_empty_executed_search_exports_inventory_and_visible_provenance(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from aso_agents.report_explainer import render_report_markdown
    program = {'program_id': 'p', 'modality': 'steric', 'intent': 'binding'}
    value = {'intent': 'binding', 'candidate_minimum': 0, 'design_programs': [program]}
    summary = {'search_status': 'completed', 'status': 'no_overlapping_sites', 'peak_count': 0,
               'selected_dataset': {'file_accession': 'SOURCE_FILE', 'biosample_term_name': 'recorded_cells'}}
    final = {'structured_goal': value, 'steric_constraints_exhausted': True,
             'steric_design_jobs': [{'job_id': 'p', 'region_rows': []}],
             'materialized_region_constraints': {'by_program': {'p': [{'source_type': 'rbp_binding', 'materialization_summary': summary}]}}}
    (tmp_path / 'goal_intake.json').write_text(json.dumps({'final_payload': final}))
    monkeypatch.setattr(pipeline, '_collect_generated_stage_sets', lambda _: {'stage_sets': {}})
    monkeypatch.setattr(pipeline, '_collect_generated_candidate_rows', lambda _run_dir, **_kwargs: [])
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    candidate_set = json.loads(Path(artifacts['generated_aso_set_json']).read_text())
    assert candidate_set['rows'] == []
    branches = candidate_set['empty_search_outcomes']
    assert design_delivery(value, candidate_set, branches)['status'] == 'fulfilled'
    report = render_report_markdown({}, {'structured_goal': value, 'branch_outcomes': branches})
    assert 'SOURCE_FILE' in report and 'recorded_cells' in report
    assert 'unsearched datasets' in report
    assert Path(artifacts['design_panel_csv']).is_file()


def test_report_stage_cannot_skip_unexecuted_design(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, '_load_state', lambda _: {'run_dir': str(tmp_path)})
    monkeypatch.setattr(pipeline, '_reject_disallowed_tool_call', lambda **_: None)
    monkeypatch.setattr(pipeline, '_mark_last_tool', lambda *_: None)
    monkeypatch.setattr(pipeline, '_relocate_agentic_state_into_run_dir', lambda state, path, run: path)
    monkeypatch.setattr(pipeline, '_load_goal_context_from_run_dir', lambda _: {
        'structured_goal': {'intent': 'binding'}, 'final_payload': {}})
    result = pipeline._prepare_execution_stage(state_path=str(tmp_path / 'state.json'),
        tool_name='tool_build_report_payload', stage_name='report', force_rerun=False, issue_summary='',
        invalidation_stage='report', invalidation_default_summary='')
    assert result['error'] == 'design_execution_pending'
    assert result['allowed_next_tools'] == ['tool_build_steric_regions']


def test_aggregate_candidates_bind_only_to_one_unambiguous_producer():
    from aso_agents.candidate_presentation import bind_candidate_program_groups
    value = {'design_programs': [{'program_id': 'p', 'modality': 'gapmer', 'label': 'Panel'}]}
    row = {'source_modality': 'gapmer', 'site_label': 'one_molecule_binding_site'}
    assert bind_candidate_program_groups([row], value)[0]['candidate_group_label'] == 'Panel'
    assert bind_candidate_program_groups([{**row, 'job_id': 'unrecognized'}], value)[0].get('program_id') is None
    assert bind_candidate_program_groups([{**row, 'source_modality': 'steric'}], value)[0].get('program_id') is None
    value['design_programs'].append({'program_id': 'q', 'modality': 'gapmer', 'label': 'Other panel'})
    assert bind_candidate_program_groups([row], value)[0].get('program_id') is None


def test_new_synthesis_declares_count_bounds_without_invalidating_legacy_goals():
    from jsonschema import Draft202012Validator
    from aso_agents.structured_outputs import DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA
    from aso_agents.goal_contract import canonical_structured_goal_schema
    program = {'program_id': 'p', 'label': 'p', 'modality': 'steric', 'intent': 'binding', 'objective': 'block binding'}
    goal_value = {'design_programs': [program]}
    assert not list(Draft202012Validator(canonical_structured_goal_schema()).iter_errors(goal_value))
    output = {'status': 'ready', 'questions': [], 'reason': None, 'structured_goal': goal_value}
    validator = Draft202012Validator(DESIGN_GOAL_SYNTHESIS_OUTPUT_SCHEMA)
    assert list(validator.iter_errors(output))
    for minimum in [0, 1, 7]:
        goal_value['candidate_minimum'] = minimum
        assert list(validator.iter_errors(output))
        goal_value['candidate_limit'] = None
        assert not list(validator.iter_errors(output))
        goal_value['candidate_limit'] = 9
        assert not list(validator.iter_errors(output))
        del goal_value['candidate_limit']


def test_composite_recovery_cannot_accept_child_reports_after_failed_delivery(tmp_path, monkeypatch):
    import json
    (tmp_path / 'report.md').write_text('Partial report')
    (tmp_path / 'report.html').write_text('<html>Partial report</html>')
    (tmp_path / 'goal_intake.json').write_text(json.dumps({'final_payload': {'structured_goal': {'intent': 'binding'}}}))
    monkeypatch.setattr(pipeline, '_emit_v2_generated_design_artifacts', lambda _: pytest.fail('Incomplete child accepted'))
    result = pipeline._fallback_design_task_payload({'task_id': 'design', 'context': {}}, {'child_run_dir': str(tmp_path)})
    assert result == {}


@pytest.mark.parametrize('deliver', [True, False])
def test_single_delegated_design_checks_child_without_requiring_duplicate_parent_pipeline(tmp_path, monkeypatch, deliver):
    import json
    child = tmp_path / 'child_runs' / 'one_design'
    child.mkdir(parents=True)
    (child / 'report.md').write_text('Report')
    (child / 'report.html').write_text('<html>Report</html>')
    (child / 'goal_intake.json').write_text(json.dumps({'final_payload': {
        'structured_goal': {'intent': 'binding', 'candidate_minimum': 1}, 'tiling_results': {'row_count': int(deliver)}}}))
    if deliver:
        (child / 'generated_aso_set.json').write_text(json.dumps({'rows': [{'sequence': 'AACCGG', 'synthesized_antisense_sequence': 'CCGGTT'}]}))
    envelope = {'schema_version': 'v3', 'workflow_name': 'design_new_asos', 'tasks': [{'task_id': 'd', 'task_type': 'design_new_asos'}]}
    (tmp_path / 'goal_intake.json').write_text(json.dumps({'final_payload': {
        'request_envelope': envelope, 'structured_goal': {}}}))
    result_path = pipeline._write_run_path(str(tmp_path), 'v3_tasks', 'd', 'design_task_result.json')
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({'task_id': 'd', 'child_status': 'run_complete', 'child_run_dir': str(child)}))
    monkeypatch.setattr(pipeline, '_resolve_run_splice_router', lambda _: True)
    monkeypatch.setattr(pipeline, '_routing_required_after_splice_eval', lambda _: False)
    integrity = pipeline._extract_pipeline_integrity(str(tmp_path), {'exists': {'report_md': True, 'report_html': True}}, skip_router=True)
    assert integrity['ok'] is deliver
    assert integrity['scoped_design_integrity']['design_delivery']['distinct_oligos'] == int(deliver)
