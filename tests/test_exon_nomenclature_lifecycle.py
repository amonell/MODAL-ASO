"""Biological names survive reference resolution, selection and final saving."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aso_agents import curated_exon_nomenclature as naming
from aso_agents import exon_nomenclature_context as identity
from aso_agents import pipeline_tools as pipeline
from aso_agents.report_exon_names import project_exon_names
from aso_agents.shortlist_evidence import evidence_catalog, review_context
from test_curated_exon_nomenclature import fixture, mock_sources


def goal_for(entry, *, label=2, modality='gapmer'):
    return {'gene_symbol': 'TEST', 'species': 'human', 'intent': 'knockdown',
        'isoform_transcripts': {'requested RNA': entry},
        'design_programs': [{'program_id': 'p', 'transcript_id': 'ENST00000000001.2',
            'modality': modality, 'exon_label': label, 'focus_exons_only': True,
            'targeting_policy': 'default_exploratory'}]}


def candidate(start, end, *, strand='+', reference=None):
    row = {'program_id': 'p', 'synthesized_antisense_sequence': 'ACGTTGCATGCA',
        'binding_intervals_bed0_half_open': [['chr3', start-1, end]], 'strand': strand,
        'candidate_assessment': {'selection_eligible': True}}
    if reference:
        row['executed_reference'] = {'transcript_id': reference}
    return row


@pytest.mark.parametrize('negative', [False, True])
@pytest.mark.parametrize('modality', ['gapmer', 'steric'])
def test_public_names_reach_review_plot_and_physical_ranking(monkeypatch, negative, modality):
    root, entry = fixture(negative=negative)
    calls = mock_sources(monkeypatch, root)
    goal = goal_for(entry, modality=modality)
    identity.prepare_goal_exon_nomenclature(goal)
    identity.prepare_goal_exon_nomenclature(goal)
    assert len(calls) == 2
    assert goal['design_programs'][0]['exon_label'] == 2
    assert identity.resolved_exon_requests(goal)[0]['resolved_transcript_ordinal'] == 3
    ordinal_two, biological_two = entry['exons_tx'][1:]
    rows = [candidate(e['start'], e['end'], strand=entry['strand']) for e in (ordinal_two, biological_two)]
    pipeline._annotate_requested_exon_proximity(rows, goal)
    identity.annotate_candidate_exon_identity(rows, goal)
    assert [r['requested_exon_proximity_tier'] for r in rows] == [1, 0]
    assert [r['exon_identity_evidence']['requested_features'][0]['overlap_nt'] for r in rows] == [0, 20]
    assert [r['exon_identity_evidence']['binding_exons'][0]['display_name'] for r in rows] == ['Exon 2a', 'Exon 2']
    context = review_context(goal)
    assert context['exon_identity']['requested_features'][0]['biological_label_verified']
    assert context['exon_identity']['references']['requested RNA']['exons'] == project_exon_names(entry)['exons']
    evidence = next(iter(evidence_catalog(rows).values()))
    assert evidence['overview'][0]['exon_identity'] == rows[0]['exon_identity_evidence']
    assert evidence['facts']['/contexts/1/exon_identity_evidence/requested_features/0/overlap_nt'] == 20


@pytest.mark.parametrize('label', [2, '2a'])
def test_source_failure_keeps_biological_feature_unknown_and_all_candidates(monkeypatch, label):
    _, entry = fixture()
    monkeypatch.setattr(naming, '_read', lambda *_: (_ for _ in ()).throw(OSError('unavailable')))
    goal = goal_for(entry, label=label)
    identity.prepare_goal_exon_nomenclature(goal)
    row = candidate(1051, 1070)
    before = deepcopy(row)
    pipeline._annotate_requested_exon_proximity([row], goal)
    identity.annotate_candidate_exon_identity([row], goal)
    assert 'requested_exon_proximity_tier' not in row
    feature = row['exon_identity_evidence']['requested_features'][0]
    assert feature['status'] == 'unresolved' and feature['overlap_nt'] is None
    assert 'start' not in identity.resolved_exon_requests(goal)[0]
    assert row['exon_identity_evidence']['binding_exons'][0]['display_name'] == 'second annotated exon'
    assert all(row[k] == v for k, v in before.items())


def test_ambiguous_public_naming_does_not_supply_a_feature_by_ordinal(monkeypatch):
    root, entry = fixture()
    other = deepcopy(root.find('./fixed_annotation/transcript'))
    other.set('name', 't2')
    root.find('fixed_annotation').append(other)
    root.find('.//transcript[@accession="ENST00000000001.2"]').attrib.pop('fixed_id')
    mock_sources(monkeypatch, root)
    goal = goal_for(entry)
    identity.prepare_goal_exon_nomenclature(goal)
    request = identity.resolved_exon_requests(goal)[0]
    assert request['status'] == 'unresolved' and 'start' not in request
    row = candidate(1051, 1070)
    pipeline._annotate_requested_exon_proximity([row], goal)
    assert 'requested_exon_proximity_tier' not in row


def test_non_numeric_names_resolve_and_explicit_ordinals_stay_explicit(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry, label='2a')
    identity.prepare_goal_exon_nomenclature(goal)
    assert identity.resolved_exon_requests(goal)[0]['resolved_transcript_ordinal'] == 2
    goal['design_programs'][0]['exon_index'] = 3
    request = identity.resolved_exon_requests(goal)[0]
    assert request['status'] == 'explicit_transcript_ordinal'
    assert request['resolved_transcript_ordinal'] == 3 and not request['biological_label_verified']


def test_aggregate_class_uses_its_reference_accession_without_renaming_the_group(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry)
    goal['splice_isoform_objectives'] = [{'objective_scope': 'aggregate_class',
        'reference_accession': 'ENST00000000001.2', 'isoform_label': 'exon-containing RNAs',
        'aggregate_class': {'basis': 'exon_presence', 'exon_label': 2, 'presence': 'included'}}]
    identity.prepare_goal_exon_nomenclature(goal)
    request = next(r for r in identity.resolved_exon_requests(goal) if r['request_path'].endswith('/aggregate_class'))
    assert request['biological_label_verified'] and request['resolved_transcript_ordinal'] == 3
    assert goal['splice_isoform_objectives'][0]['isoform_label'] == 'exon-containing RNAs'


@pytest.mark.parametrize('change', ['version', 'coordinates', 'modified', 'ambiguous'])
def test_stale_or_ambiguous_references_cannot_reuse_names(monkeypatch, change):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry)
    identity.prepare_goal_exon_nomenclature(goal)
    if change == 'version': entry['metadata']['versioned_id'] = 'ENST00000000001.3'
    if change == 'coordinates': entry['exons_tx'][2]['end'] += 1
    if change == 'modified': entry['explicit_target_modified'] = True
    if change == 'ambiguous':
        extra = deepcopy(entry); extra['exons_tx'][2]['end'] += 1
        goal['isoform_transcripts']['conflicting reference'] = extra
    request = identity.resolved_exon_requests(goal)[0]
    assert request['status'] == 'unresolved' and not request['biological_label_verified']
    assert 'start' not in request


def test_render_and_review_projection_use_saved_data_without_network(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    before = deepcopy(goal)
    monkeypatch.setattr(naming, '_read', lambda *_: pytest.fail('Offline projection queried a source'))
    row = candidate(1101, 1120)
    identity.annotate_candidate_exon_identity([row], goal)
    assert review_context(goal)['exon_identity']['references']['requested RNA']['naming_verified']
    assert project_exon_names(entry)['exon_naming']['verified']
    assert goal == before


def test_complete_saved_naming_receipt_resolves_requested_feature_offline(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    naming.ensure_entry_nomenclature(entry)
    goal = goal_for(entry)
    assert 'exon_nomenclature_resolutions' not in entry['metadata']
    monkeypatch.setattr(naming, '_read', lambda *_: pytest.fail('Saved names require no new lookup'))
    before = deepcopy(goal)
    assert identity.resolved_exon_requests(goal)[0]['resolved_transcript_ordinal'] == 3
    assert goal == before


def test_candidate_reference_version_overrides_program_and_remains_exact(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    row = candidate(1101, 1120, reference='ENST00000000001.1')
    identity.annotate_candidate_exon_identity([row], goal)
    assert row['exon_identity_evidence'] == {'status': 'unresolved_reference'}
    row['executed_reference'] = {'transcript_id': 'ENST00000000001', 'versioned_id': 'ENST00000000001.1'}
    identity.annotate_candidate_exon_identity([row], goal)
    assert row['exon_identity_evidence'] == {'status': 'unresolved_reference'}


def test_stale_catalog_key_cannot_override_the_record_version():
    _, entry = fixture()
    assert identity.selected_reference({'ENST00000000001.1': entry}, 'ENST00000000001.1') == (None, None)


def test_candidate_execution_structure_is_checked_before_assigning_names(monkeypatch):
    from aso_agents.execution_references import executed_reference
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    row = candidate(1101, 1120)
    row['executed_reference'] = executed_reference({}, 'requested RNA', entry)
    identity.annotate_candidate_exon_identity([row], goal)
    assert row['exon_identity_evidence']['binding_exons'][0]['display_name'] == 'Exon 2'
    entry['exons_tx'][0]['start'] += 1
    identity.annotate_candidate_exon_identity([row], goal)
    assert row['exon_identity_evidence'] == {'status': 'unresolved_reference'}


@pytest.mark.parametrize('change', ['strand', 'chromosome', 'missing'])
def test_unmatched_binding_geometry_retains_unknown_feature_overlap(monkeypatch, change):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    row = candidate(1101, 1120)
    if change == 'strand': row['strand'] = '-'
    if change == 'chromosome': row['binding_intervals_bed0_half_open'][0][0] = 'chr4'
    if change == 'missing': row.pop('binding_intervals_bed0_half_open')
    identity.annotate_candidate_exon_identity([row], goal)
    assert row['exon_identity_evidence']['binding_exons'] == []
    assert row['exon_identity_evidence']['requested_features'][0]['overlap_nt'] is None


def test_overlapping_saved_segments_do_not_double_count_exon_coverage(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    row = candidate(1101, 1120)
    row['binding_intervals_bed0_half_open'] = [['chr3', 1100, 1115], ['chr3', 1110, 1120]]
    identity.annotate_candidate_exon_identity([row], goal)
    assert row['exon_identity_evidence']['requested_features'][0]['overlap_nt'] == 20


def test_junction_overlap_counts_each_bound_exon_without_the_intervening_intron(monkeypatch):
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); identity.prepare_goal_exon_nomenclature(goal)
    row = candidate(1066, 1105)
    row['binding_intervals_bed0_half_open'] = [['chr3', 1065, 1070], ['chr3', 1100, 1105]]
    identity.annotate_candidate_exon_identity([row], goal)
    assert [e['overlap_nt'] for e in row['exon_identity_evidence']['binding_exons']] == [5, 5]
    assert row['exon_identity_evidence']['requested_features'][0]['overlap_nt'] == 5


def test_resumed_report_resolves_before_shortlist_and_persists_before_render(monkeypatch, tmp_path):
    root, entry = fixture(); calls = mock_sources(monkeypatch, root)
    goal = goal_for(entry)
    final = {'structured_goal': goal}; payload = {'final_payload': final}
    pipeline._save_goal_payload(str(tmp_path), payload, goal, final)
    monkeypatch.setattr(pipeline, '_prepare_execution_stage', lambda **_: {
        'status': 'ok', 'state': {}, 'run_dir': str(tmp_path)})
    monkeypatch.setattr(pipeline, '_promote_atomic_status', lambda **kw: kw)
    observed = []
    def review(run_dir, **kwargs):
        stored = pipeline._goal_payload_context(run_dir)['structured_goal']
        observed.extend(identity.resolved_exon_requests(stored))
        return {}
    monkeypatch.setattr(pipeline, '_emit_v2_generated_design_artifacts', review)
    monkeypatch.setattr(pipeline, '_build_report_payload_manifest', lambda **kw: {'ready': True})
    pipeline.tool_build_report_payload(str(tmp_path/'state.json'))
    assert len(calls) == 2 and observed[0]['resolved_transcript_ordinal'] == 3
    saved = pipeline._goal_payload_context(str(tmp_path))['structured_goal']
    assert project_exon_names(saved['isoform_transcripts']['requested RNA'])['exons'][2]['label'] == 'Exon 2'


@pytest.mark.parametrize('locked', [False, True])
def test_reference_resolution_supplies_names_before_strategy(monkeypatch, locked):
    import run_goal_intake as native
    root, entry = fixture(); calls = mock_sources(monkeypatch, root)
    goal = goal_for(entry)
    goal['splice_isoform_objectives'] = [{'isoform_label': 'requested RNA',
        'transcript_id': 'ENST00000000001.2', 'desired_abundance': 'decrease'}]
    goal['isoform_resolution_locked'] = locked
    monkeypatch.setattr(native, 'list_transcripts', lambda **_: {'transcripts': [
        {'id': 'ENST00000000001.2', 'biotype': 'protein_coding'}]})
    monkeypatch.setattr(native, '_fetch_transcript_structure', lambda _: {
        **entry['metadata'], 'id': 'ENST00000000001.2', 'seq_region_name': 'chr3', 'strand': 1,
        'exons_tx': deepcopy(entry['exons_tx'])})
    checked = []
    def strategy(_objectives, catalog, goal):
        features = identity.resolved_exon_requests(goal)
        assert features[0]['resolved_transcript_ordinal'] == 3
        checked.append(True)
        return [], []
    monkeypatch.setattr(native, '_compute_splice_site_plan', strategy)
    monkeypatch.setattr(native, '_should_skip_splice_strategy', lambda _: False)
    monkeypatch.setattr(native, '_compute_steric_regions', lambda *_: None)
    monkeypatch.setattr(native, '_generate_splice_strategy_analysis', lambda *_, **__: None)
    result = native._resolve_isoform_transcripts(goal, {}, allow_interactive=False)
    assert checked == [True] and len(calls) == 2
    assert result['isoform_transcripts']['requested RNA']['metadata']['curated_exon_nomenclature']['status'] == 'verified'


def test_final_artifact_publication_preserves_native_nomenclature(monkeypatch, tmp_path):
    import run_goal_intake as native
    root, entry = fixture(); mock_sources(monkeypatch, root)
    goal = goal_for(entry); final = {'structured_goal': goal}
    pipeline._save_goal_payload(str(tmp_path), {'request': 'Design ASOs', 'final_payload': final}, goal, final)
    (tmp_path/'report.md').write_text('Report')
    (tmp_path/'report.html').write_text('<html>Report</html>')
    monkeypatch.setattr(pipeline, '_prepare_execution_stage', lambda **_: {
        'status': 'ok', 'state': {'request': 'Design ASOs'}, 'run_dir': str(tmp_path)})
    monkeypatch.setattr(pipeline, '_goal_runtime_module', lambda: SimpleNamespace(_persist_outputs=native._persist_outputs))
    monkeypatch.setattr(pipeline, '_emit_v2_generated_design_artifacts', lambda *_: {'generated_aso_set_json': str(tmp_path/'generated.json')})
    monkeypatch.setattr(pipeline, '_promote_atomic_status', lambda **kw: kw)
    pipeline.tool_persist_run_outputs(str(tmp_path/'state.json'))
    saved = pipeline._goal_payload_context(str(tmp_path))['structured_goal']
    assert saved['isoform_transcripts']['requested RNA']['metadata']['curated_exon_nomenclature']['status'] == 'verified'
    assert saved['generated_aso_set_json'].endswith('generated.json')
