from copy import deepcopy
import pytest
import run_goal_intake as intake
from aso_agents import pipeline_tools
from aso_agents.evaluation_feature_identity import transcript_identifier, transcript_matches
from aso_agents.explicit_target_identity import reconcile_clarified_targets
from aso_agents.splice_recommendations import preserve_splice_site_evidence


def reference(strand='+'):
    return {'transcript_id': 'TX_NEUTRAL', 'chrom': 'chr2', 'strand': strand,
            'metadata': {'id': 'TX_NEUTRAL', 'version': 3,
                         'versioned_id': 'TX_NEUTRAL.3', 'is_canonical': True,
                         'is_mane_select': True, 'translation_id': 'PROTEIN_ORIGINAL',
                         'reference_source': 'local_gtf'},
            'exons_tx': [{'index': 1, 'start': 101, 'end': 140, 'seq_region_name': 'chr2'},
                         {'index': 2, 'start': 201, 'end': 240, 'seq_region_name': 'chr2'},
                         {'index': 3, 'start': 301, 'end': 340, 'seq_region_name': 'chr2'}]}


def target(**updates):
    value = {'chrom': 'chr2', 'start': 201, 'end': 240, 'strand': '+',
             'isoform_label': 'TX_NEUTRAL', 'base_isoform_label': 'TX_NEUTRAL',
             'desired_delta': 'increase', 'exon_label': 2}
    value.update(updates)
    return value


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('explicit_strand', [True, False])
def test_clarified_physical_feature_replaces_wrong_inferred_orientation(strand, explicit_strand):
    inferred = target(strand='-' if strand == '+' else '+',
                      isoform_label='TX_NEUTRAL::explicit_target_model')
    confirmed = target(source='clarification_answers', transcript_id='TX_NEUTRAL', strand=strand)
    if not explicit_strand:
        confirmed.pop('strand')
    payload = {'explicit_splice_targets': [inferred, confirmed],
               'isoform_transcripts': {'TX_NEUTRAL': reference(strand)}}
    before = deepcopy(payload)
    intake._ensure_explicit_splice_target_context(payload, {})
    assert before['explicit_splice_targets'][0] == inferred
    targets = payload['explicit_splice_targets']
    assert len(targets) == 1 and targets[0]['strand'] == strand
    assert targets[0]['isoform_label'] == 'TX_NEUTRAL'
    assert targets[0]['superseded_target_interpretations'][0]['strand'] == inferred['strand']
    assert set(payload['isoform_transcripts']) == {'TX_NEUTRAL'}
    sites = intake._build_splice_sites_for_eval(payload, None, 10)
    assert len(sites) == 2
    pair = [sites[0][k]['position'] for k in ('acceptor', 'donor')]
    assert pair == ([201, 240] if strand == '+' else [240, 201])
    after = deepcopy(payload)
    intake._ensure_explicit_splice_target_context(payload, {})
    assert payload['explicit_splice_targets'] == after['explicit_splice_targets']


@pytest.mark.parametrize('fault', ['different_transcript', 'different_version', 'other_interval',
                                  'other_chromosome', 'opposite_objective', 'explicit_user_target'])
def test_clarification_does_not_erase_independent_or_conflicting_user_target(fault):
    inferred = target(transcript_id='TX_NEUTRAL.3')
    clarified = target(source='clarification_answers', transcript_id='TX_NEUTRAL.3')
    if fault == 'different_transcript':
        clarified.update(transcript_id='TX_OTHER', base_isoform_label='TX_OTHER', isoform_label='TX_OTHER')
    elif fault == 'different_version':
        clarified['transcript_id'] = 'TX_NEUTRAL.4'
    elif fault == 'other_interval':
        clarified.update(start=205, end=245)
    elif fault == 'other_chromosome':
        clarified['chrom'] = 'chr3'
    elif fault == 'opposite_objective':
        clarified['desired_delta'] = 'decrease'
    elif fault == 'explicit_user_target':
        inferred.update(source='user_provided', strand='-')
    items = [inferred, clarified]
    assert reconcile_clarified_targets(items) == items


def test_missing_clarification_strand_is_not_defaulted_to_positive():
    state = {'request': 'Design splice ASOs for exon 2 inclusion.',
             'request_envelope': {'default_context': {
                 'intent': 'splice_modulation', 'strand': '+',
                 'splice_isoform_objectives': [{'transcript_id': 'ENST00000000001',
                     'isoform_label': 'ENST00000000001', 'desired_abundance': 'increase'}]}}}
    preseed = pipeline_tools._explicit_splice_target_preseed_from_clarifications(
        state, [{'question': 'What is the exon interval?', 'answer': 'Exon 2 is chr2:201-240 in GRCh38.'}])
    assert preseed
    assert 'strand' not in preseed['structured_goal_overlay']['explicit_splice_targets'][0]
    assert 'strand' not in preseed['structured_goal_overlay']['evaluation_targets'][0]


@pytest.mark.parametrize('text,expected', [('plus strand', '+'), ('minus strand', '-'),
    ('strand +', '+'), ('strand -', '-'), ('(+)', '+'), ('(-)', '-'),
    ('reverse complement the ASO', ''), ('offsets -4 to +12', '')])
def test_strand_wording_is_explicit_without_offset_or_sequence_guess(text, expected):
    assert pipeline_tools._strand_from_clarification_text(text) == expected


@pytest.mark.parametrize('strand', ['+', '-'])
def test_explicit_conflict_is_rejected_before_custom_model_creation(strand):
    ref = reference(strand)
    inconsistent = target(strand='-' if strand == '+' else '+', source='user_provided')
    payload = {'explicit_splice_targets': [inconsistent],
               'isoform_transcripts': {'TX_NEUTRAL': ref}}
    with pytest.raises(ValueError, match='strand.*conflicts'):
        intake._ensure_explicit_splice_target_context(payload, {})
    assert set(payload['isoform_transcripts']) == {'TX_NEUTRAL'}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_custom_model_keeps_its_own_identity_geometry_and_unknown_annotation(strand):
    original = reference(strand)
    requested = target(start=207, end=253, strand=strand)
    payload = {'explicit_splice_targets': [requested],
               'isoform_transcripts': {'TX_NEUTRAL': deepcopy(original)}}
    intake._ensure_explicit_splice_target_context(payload, {})
    model_id = 'TX_NEUTRAL::explicit_target_model'
    model = payload['isoform_transcripts'][model_id]
    assert transcript_identifier(model) == model_id
    assert model['metadata']['base_reference_transcript_id'] == 'TX_NEUTRAL.3'
    assert model['metadata']['base_reference_snapshot']['is_mane_select'] is True
    assert model['metadata']['is_mane_select'] is None
    assert 'versioned_id' not in model['metadata']
    assert 'translation_id' not in model['metadata']
    assert model['metadata']['cdna_length'] == 127
    assert original['metadata']['is_canonical'] is True
    sites = intake._build_splice_sites_for_eval(payload, None, 10)
    assert [sites[0][k]['position'] for k in ('acceptor', 'donor')] == ([207, 253] if strand == '+' else [253, 207])
    saved = preserve_splice_site_evidence({'splice_site_deltas': sites}, payload['isoform_transcripts'])
    assert all(s['reference_transcript_id'] == model_id for s in saved)
    assert all(s['reference_binding_status'] == 'resolved_catalog_identity' for s in saved)


@pytest.mark.parametrize('version', ['', '.3'])
def test_synthetic_alias_cannot_match_annotated_reference(version):
    model = 'TX_NEUTRAL' + version + '::explicit_target_model'
    assert not transcript_matches('TX_NEUTRAL', model)
    assert not transcript_matches('TX_NEUTRAL.3', model)
    assert not transcript_matches(model, 'TX_NEUTRAL.3')
    assert transcript_matches(model, model)


def test_legacy_custom_metadata_cannot_steal_base_reference_identity():
    model = reference()
    model.update(transcript_id='TX_NEUTRAL.3::explicit_target_model', explicit_target_modified=True)
    assert transcript_identifier(model) == 'TX_NEUTRAL.3::explicit_target_model'


@pytest.mark.parametrize('strand', ['+', '-'])
def test_reversed_pair_never_gets_resolved_catalog_attestation(strand):
    ref = reference(strand)
    correct_pair = [201, 240] if strand == '+' else [240, 201]
    sites = []
    for pair in (correct_pair, correct_pair[::-1]):
        sites.append({'isoform_label': 'TX_NEUTRAL', 'exon_index': 2,
            'acceptor': {'chrom': 'chr2', 'strand': strand, 'position': pair[0]},
            'donor': {'chrom': 'chr2', 'strand': strand, 'position': pair[1]},
            'delta_inclusion_deletion': .3})
    raw = {'splice_site_deltas': sites}
    before = deepcopy(raw)
    saved = preserve_splice_site_evidence(raw, {'TX_NEUTRAL': ref})
    assert raw == before
    assert saved[0]['reference_binding_status'] == 'resolved_catalog_identity'
    assert saved[1]['reference_binding_status'] == 'conflicting'
    assert [s['delta_inclusion_deletion'] for s in saved] == [.3, .3]


def test_overlay_reconciliation_preserves_reference_versions():
    first = target(transcript_id='TX_NEUTRAL.3')
    second = target(transcript_id='TX_NEUTRAL.4', source='clarification_answers')
    assert len(pipeline_tools._merge_explicit_splice_target_lists([second], [first])) == 2


def test_overlay_missing_strand_does_not_inherit_inferred_guess():
    first = target(strand='-')
    second = target(source='clarification_answers', transcript_id='TX_NEUTRAL')
    second.pop('strand')
    merged = pipeline_tools._merge_explicit_splice_target_lists([second], [first])
    assert len(merged) == 1 and 'strand' not in merged[0]
    assert merged[0]['superseded_target_interpretations'][0]['strand'] == '-'
