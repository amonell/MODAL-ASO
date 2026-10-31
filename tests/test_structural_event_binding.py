"""Named events must survive reference annotation differences into real windows."""
from copy import deepcopy

import pytest

import run_goal_intake as runtime


def conversion_fixture(strand='+'):
    physical = [(100, 130), (1000, 1030), (2000, 2030), (3000, 3030), (4000, 4030)]
    if strand == '-':
        physical.reverse()

    def transcript(identifier, intervals):
        exons = [{'index': i, 'start': start, 'end': end, 'contains_coding': True}
                 for i, (start, end) in enumerate(intervals, 1)]
        return {'transcript_id': identifier, 'chrom': 'chr8', 'strand': strand,
            'metadata': {'id': identifier, 'versioned_id': identifier + '.4'},
            'exons_tx': exons,
            'sites': [{'exon_index': e['index'], 'chrom': 'chr8', 'strand': strand,
                'acceptor': e['start'] if strand == '+' else e['end'],
                'donor': e['end'] if strand == '+' else e['start'], 'coding': True}
                for e in exons]}

    included = transcript('TXIN', physical)
    excluded = transcript('TXOUT', physical[:2] + physical[3:])
    # Terminal annotation differences must not obscure the internal splice event.
    terminal = excluded['exons_tx'][-1]
    terminal['end' if strand == '+' else 'start'] += 5 if strand == '+' else -5
    goal = {'intent': 'splice_modulation', 'desired_effect': 'skipping',
        'exon_label': 5, 'focus_exons_only': True,
        'transcript_lock': {'kind': 'explicit_id', 'value': 'TXIN.4'},
        'isoform_transcripts': {'included': included, 'excluded': excluded},
        'splice_isoform_objectives': [
            {'isoform_label': 'included', 'transcript_id': 'TXIN.4', 'desired_abundance': 'decrease'},
            {'isoform_label': 'excluded', 'transcript_id': 'TXOUT.4', 'desired_abundance': 'increase'}]}
    program = {'program_id': 'conversion', 'modality': 'steric', 'intent': 'splice_modulation',
        'transcript_id': 'TXIN.4', 'exon_label': 5, 'desired_effect': 'skipping', 'site_kind': 'both'}
    goal['design_programs'] = [program]
    return goal, program


@pytest.fixture(autouse=True)
def no_reference_evidence_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('MODAL_ASO_CACHE_DIR', str(tmp_path / 'empty-cache'))
    monkeypatch.setattr(runtime, 'resolve_transcript_exon_nomenclature_sync',
                        lambda **kwargs: {'status': 'unresolved', 'reason': 'No nomenclature evidence supplied'})


@pytest.mark.parametrize('strand', ['+', '-'])
def test_conversion_uses_internal_event_instead_of_terminal_ordinal(strand):
    goal, program = conversion_fixture(strand)
    declared = deepcopy(goal['splice_isoform_objectives'])
    plan, _ = runtime._compute_splice_site_plan(declared, goal['isoform_transcripts'], goal)
    goal['splice_site_plan'] = plan
    sites = runtime._collect_program_splice_plan_entries(goal, program)
    assert {(s['site'], s['position']) for s in sites} == {
        ('acceptor', 2000 if strand == '+' else 2030),
        ('donor', 2030 if strand == '+' else 2000)}
    assert all(s['resolved_exon_ordinals'] == {'included': 3} for s in sites)
    rows, metadata = runtime._build_steric_program_site_window_rows(goal, program)
    assert len(rows) == 2
    assert all(row[1] < 2030 and row[2] > 2000 for row in rows)
    assert goal['exon_label'] == program['exon_label'] == 5
    assert goal['splice_isoform_objectives'] == declared


def test_single_reference_keeps_requested_ordinal():
    goal, program = conversion_fixture()
    goal['isoform_transcripts'].pop('excluded')
    goal['splice_isoform_objectives'] = goal['splice_isoform_objectives'][:1]
    plan, _ = runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    goal['splice_site_plan'] = plan
    assert {s['position'] for s in runtime._collect_program_splice_plan_entries(goal, program)} == {4000}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_site_record_cannot_disagree_with_its_transcript_exon(strand):
    goal, _ = conversion_fixture(strand)
    goal['isoform_transcripts']['included']['sites'][2]['donor'] += 7
    with pytest.raises(ValueError, match='splice_site_reference_conflict'):
        runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)


@pytest.mark.parametrize('strand', ['+', '-'])
def test_inclusion_uses_same_physical_event_with_opposite_abundance_objectives(strand):
    goal, program = conversion_fixture(strand)
    goal['desired_effect'] = program['desired_effect'] = 'inclusion'
    for objective in goal['splice_isoform_objectives']:
        objective['desired_abundance'] = 'increase' if objective['isoform_label'] == 'included' else 'decrease'
    plan, _ = runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    goal['splice_site_plan'] = plan
    assert {site['position'] for site in runtime._collect_program_splice_plan_entries(goal, program)} == {2000, 2030}


@pytest.mark.parametrize('conflict', ['nomenclature', 'ordinal'])
def test_explicit_conflicting_evidence_requires_resolution(conflict):
    goal, program = conversion_fixture()
    if conflict == 'ordinal':
        program['exon_ordinal'] = 4
    else:
        goal['isoform_transcripts']['included']['metadata']['exon_nomenclature_resolutions'] = {
            '5': {'status': 'verified_alternate_mapping', 'resolved_transcript_ordinal': 4}}
    with pytest.raises(ValueError, match='exon_conversion_reference_conflict'):
        runtime._resolve_authoritative_transcript_exon_ordinal(goal['isoform_transcripts']['included'], 5, goal)


def test_two_different_cassette_events_are_not_guessed():
    goal, _ = conversion_fixture()
    alternate = deepcopy(goal['isoform_transcripts']['included'])
    alternate['transcript_id'] = 'OTHER'
    alternate['metadata'] = {'id': 'OTHER', 'versioned_id': 'OTHER.4'}
    alternate['exons_tx'].pop(1)
    for index, exon in enumerate(alternate['exons_tx'], 1):
        exon['index'] = index
    goal['isoform_transcripts']['other'] = alternate
    goal['splice_isoform_objectives'].append({'transcript_id': 'OTHER.4', 'desired_abundance': 'increase'})
    with pytest.raises(ValueError, match='more than one internal cassette event'):
        runtime._resolve_authoritative_transcript_exon_ordinal(goal['isoform_transcripts']['included'], 5, goal)


def test_changed_conversion_does_not_reuse_prior_event():
    goal, _ = conversion_fixture()
    entry = goal['isoform_transcripts']['included']
    assert runtime._resolve_authoritative_transcript_exon_ordinal(entry, 5, goal) == 3
    goal['splice_isoform_objectives'][1]['desired_abundance'] = 'decrease'
    assert runtime._resolve_authoritative_transcript_exon_ordinal(entry, 5, goal) == 5
