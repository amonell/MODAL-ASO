"""Required splice geometry remains available without optional effect scores."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime


def goal():
    return {
        'gene_symbol': 'NEUTRAL', 'species': 'human', 'intent': 'splice_modulation',
        'target_region': 'cds', 'focus_exons_only': True,
        'transcript_policy': 'user_locked',
        'transcript_lock': {'kind': 'explicit_id', 'value': 'ENST00000000001.3', 'version': 3},
        'splice_isoform_objectives': [{'objective_scope': 'default_exon_context',
            'exon_label': 2, 'desired_effect': 'skipping', 'desired_abundance': 'decrease', 'role': 'context'}],
        'design_programs': [{'program_id': 'neutral-splice', 'modality': 'steric',
            'intent': 'splice_modulation', 'desired_effect': 'skipping',
            'run_splice_eval': False, 'run_splice_router': True}],
    }


@pytest.mark.parametrize('intent', ['splice_modulation', 'knockdown'])
def test_reference_requirement_does_not_depend_on_optional_prediction(intent):
    current = goal()
    current['intent'] = intent
    assert runtime._should_seed_default_splice_context(current)
    assert runtime._resolve_run_splice_eval(current) is False
    assert runtime._resolve_run_splice_router(current, '') is True


@pytest.mark.parametrize('strand', [1, -1])
def test_excluded_prediction_keeps_locked_reference_geometry(monkeypatch, strand):
    current = goal()
    before = deepcopy(current)
    reference_id = current['transcript_lock']['value']
    transcript = {'id': reference_id, 'version': 3, 'is_canonical': False, 'length': 150}
    alternative = {'id': 'ENST00000000002.1', 'is_canonical': True, 'length': 200}
    structure = {'id': reference_id, 'version': 3, 'seq_region_name': '1', 'strand': strand,
        'exons_tx': [{'index': i + 1, 'start': start, 'end': start + 49, 'contains_coding': True}
                     for i, start in enumerate([101, 201, 301] if strand == 1 else [301, 201, 101])]}
    monkeypatch.setattr(runtime, '_default_splice_context_transcript_listing',
        lambda *a, **k: {'transcripts': [alternative, transcript], 'structures': {reference_id: structure}})
    def fail_fetch(*args, **kwargs):
        raise AssertionError('The explicitly locked supplied reference should be used without another fetch.')
    monkeypatch.setattr(runtime, '_fetch_transcript_structure', fail_fetch)
    monkeypatch.setattr(runtime, '_cached_authoritative_exon_nomenclature_mapping',
        lambda entry, label: {'status': 'not_attempted', 'source': 'transcript_exon_index',
                             'requested_exon_label': label})
    monkeypatch.setattr(runtime, '_generate_splice_strategy_analysis', lambda *a, **k: {})
    payload = {}
    result = runtime._ensure_default_splice_context(current, payload, user_request='Design exon-skipping ASOs; skip effect prediction.')
    assert result['resolved_transcript_strategy']['transcript_ids'] == [reference_id]
    resolved = current['isoform_transcripts'][reference_id]
    assert resolved['exons_tx'] == structure['exons_tx']
    assert resolved['strand'] == ('+' if strand == 1 else '-')
    assert payload['isoform_transcripts'][reference_id] == resolved
    assert current['transcript_lock'] == before['transcript_lock']
    assert current['design_programs'] == before['design_programs']
    assert runtime._resolve_run_splice_eval(current) is False


@pytest.mark.parametrize('update', [
    {'intent': 'binding', 'splice_isoform_objectives': [], 'design_programs': []},
    {'intent': 'knockdown', 'splice_isoform_objectives': [], 'design_programs': [{'modality': 'gapmer'}]},
    {'target_region': 'utr3'},
    {'splice_site_plan': [{'existing': True}]},
    {'explicit_splice_targets': [{'existing': True}]},
    {'splice_isoform_objectives': [{'isoform_label': 'named-isoform', 'desired_effect': 'skipping'}]},
])
def test_other_scope_guards_still_prevent_default_reference_selection(update):
    assert not runtime._should_seed_default_splice_context({**goal(), **update})
