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


def test_replanning_explicit_windows_preserves_only_current_conversion_evidence(monkeypatch):
    goal, _ = conversion_fixture()
    entry = goal['isoform_transcripts']['included']
    runtime._resolve_authoritative_transcript_exon_ordinal(entry, 5, goal)
    original = deepcopy(goal['exon_conversion_bindings'])
    # Explicit, already-derived windows need no further ordinal lookup inside
    # the native planner. Their original physical-event evidence is still valid.
    monkeypatch.setattr(runtime, '_compute_native_splice_site_plan', lambda *a: ([], []))
    runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    assert goal['exon_conversion_bindings'] == original
    goal['splice_isoform_objectives'][1]['desired_abundance'] = 'decrease'
    runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    assert goal['exon_conversion_bindings'] == []


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('effect', ['inclusion', 'skipping'])
@pytest.mark.parametrize('representation', ['global', 'program', 'objective'])
def test_unlocked_conversion_resolves_before_planning(strand, effect, representation):
    goal, program = conversion_fixture(strand)
    goal.pop('transcript_lock')
    program.pop('transcript_id')
    goal['desired_effect'] = program['desired_effect'] = effect
    for objective in goal['splice_isoform_objectives']:
        included = objective['isoform_label'] == 'included'
        objective['desired_abundance'] = 'increase' if included == (effect == 'inclusion') else 'decrease'
    if representation == 'program':
        goal.pop('exon_label')
        goal.pop('desired_effect')
    elif representation == 'objective':
        goal['splice_isoform_objectives'][0].update(exon_label=5, desired_effect=effect)
        goal.pop('exon_label')
        goal.pop('desired_effect')
        program.pop('exon_label')
        program.pop('desired_effect')
    # Mimic a partially coding annotation that suppresses a default donor.
    goal['isoform_transcripts']['included']['sites'][2]['donor_plan_ok'] = False
    declared = deepcopy(goal['splice_isoform_objectives'])
    plan, _ = runtime._compute_splice_site_plan(declared, goal['isoform_transcripts'], goal)
    goal['splice_site_plan'] = plan
    chosen = runtime._collect_program_splice_plan_entries(goal, program)
    assert {(site['site'], site['position']) for site in chosen} == {
        ('acceptor', 2000 if strand == '+' else 2030),
        ('donor', 2030 if strand == '+' else 2000)}
    assert goal['exon_conversion_bindings']
    assert goal['splice_isoform_objectives'] == declared
    from aso_agents.pipeline_tools import _derive_exploratory_steric_programs
    derived, _ = _derive_exploratory_steric_programs(structured_goal=goal, base_program=program)
    assert {site for p in derived for site in p['candidate_sites']} == {
        'acceptor@' + str(2000 if strand == '+' else 2030),
        'donor@' + str(2030 if strand == '+' else 2000)}


def test_missing_response_site_cannot_pass_from_one_surviving_job():
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, _ = conversion_fixture()
    goal.pop('transcript_lock')
    goal['design_programs'][0].pop('transcript_id')
    goal['min_asos_per_site'] = 2
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    goal['steric_design_jobs'] = [{'program_id': 'conversion', 'job_id': 'one_surviving_job',
        'region_rows': [['chr8', 1900, 2100]], 'tile_overlap_requirement': 'any_overlap'}]
    rows = [{'Synthesis': sequence, 'IDT': 'declared:' + sequence,
        'binding_intervals_bed0_half_open': [['chr8', 1990, 2010]],
        'selected_site_chrom': 'chr8', 'selected_site_strand': '+',
        'selected_site_kind': 'acceptor', 'selected_site_position': 2000,
        'program_id': 'conversion'} for sequence in ['ACGT' * 5, 'TGCA' * 5]]
    result = region_minimum_delivery(goal, {'rows': rows}, {},
        lambda row, **kwargs: row['binding_intervals_bed0_half_open'])
    assert result['status'] == 'unfulfilled'
    missing = [group for group in result['groups'] if group.get('site_kind') == 'donor']
    assert len(missing) == 1 and missing[0]['distinct_molecules'] == 0
    assert missing[0]['shortfall'] == 2


def response_fixture():
    from aso_agents.design_event_consistency import requested_response_minimums
    goal, _ = conversion_fixture()
    goal['min_asos_per_site'] = 2
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    rows = [{'Synthesis': sequence, 'IDT': 'declared:' + sequence,
        'selected_site_chrom': 'chr8', 'selected_site_strand': '+',
        'selected_site_kind': kind, 'selected_site_position': position,
        'program_id': 'conversion'} for kind, position, sequence in [
            ('acceptor', 2000, 'ACGT' * 5), ('acceptor', 2000, 'TGCA' * 5),
            ('donor', 2030, 'AGCT' * 5), ('donor', 2030, 'TCGA' * 5)]]
    return goal, rows, requested_response_minimums(goal)


def test_response_minimum_survives_missing_jobs_and_counts_each_molecule_once():
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, rows, _ = response_fixture()
    result = region_minimum_delivery(goal, {'rows': rows}, {}, lambda *a, **k: [])
    assert result['status'] == 'fulfilled'
    result = region_minimum_delivery(goal, {'rows': rows[:3] + rows[2:3]}, {}, lambda *a, **k: [])
    assert result['status'] == 'unfulfilled'
    assert sorted(g['distinct_molecules'] for g in result['groups']) == [1, 2]


@pytest.mark.parametrize('field,value', [
    ('selected_site_chrom', 'chr9'), ('selected_site_strand', '-'),
    ('selected_site_position', 2031), ('selected_site_position', 2030.5),
    ('selected_site_kind', 'acceptor')])
def test_response_membership_requires_exact_site_identity(field, value):
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, rows, _ = response_fixture()
    rows[-1][field] = value
    result = region_minimum_delivery(goal, {'rows': rows}, {}, lambda *a, **k: [])
    assert result['status'] == 'unfulfilled'
    assert next(g for g in result['groups'] if g['site_kind'] == 'donor')['shortfall'] == 1


def test_missing_or_stale_event_receipt_cannot_be_replaced_by_route_labels():
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, rows, _ = response_fixture()
    bindings = deepcopy(goal['exon_conversion_bindings'])
    for replacement in [[], [{**b, 'reference_structure_sha256': 'stale'} for b in bindings]]:
        goal['exon_conversion_bindings'] = replacement
        result = region_minimum_delivery(goal, {'rows': rows}, {}, lambda *a, **k: [])
        assert result['status'] == 'unfulfilled'
        assert 'requested_exon_event_reference_unresolved' in result['errors']


def test_selected_panel_must_itself_deliver_both_response_sites():
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, rows, _ = response_fixture()
    goal['candidate_limit'] = 3
    result = region_minimum_delivery(goal, {'rows': rows[:3]}, {'sets': [{'rows': rows}]},
                                    lambda *a, **k: [])
    assert result['status'] == 'unfulfilled' and result['count_scope'] == 'selected_panel'


def test_bounded_selection_reserves_lower_ranked_response_site_candidates():
    from aso_agents.region_panel_selection import select_region_minimum_panel
    goal, rows, groups = response_fixture()
    extra = {**rows[0], 'Synthesis': 'AACC' * 5, 'IDT': 'declared:' + 'AACC' * 5}
    chosen, summary = select_region_minimum_panel(rows[:2] + [extra] + rows[2:], groups,
        lambda *a, **k: [], global_limit=4)
    assert {r['Synthesis'] for r in chosen} == {r['Synthesis'] for r in rows}
    assert len(chosen) == 4


def test_unscoped_multiple_physical_events_require_clarification():
    goal, program = conversion_fixture()
    goal.pop('transcript_lock')
    program.pop('transcript_id')
    for name in ['included', 'excluded']:
        other = deepcopy(goal['isoform_transcripts'][name])
        other['transcript_id'] += 'OTHER'
        other['metadata'] = {'id': other['transcript_id'], 'versioned_id': other['transcript_id'] + '.4'}
        other['chrom'] = 'chr9'
        for site in other['sites']:
            site['chrom'] = 'chr9'
        goal['isoform_transcripts'][name + '_other'] = other
        goal['splice_isoform_objectives'].append({'isoform_label': name + '_other',
            'transcript_id': other['transcript_id'] + '.4',
            'desired_abundance': 'decrease' if name == 'included' else 'increase'})
    with pytest.raises(ValueError, match='more than one physical cassette'):
        runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)


def test_explicit_version_is_not_rebound_to_another_version():
    goal, program = conversion_fixture()
    goal['transcript_lock']['value'] = program['transcript_id'] = 'TXIN.5'
    plan, _ = runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    goal['splice_site_plan'] = plan
    assert not runtime._collect_program_splice_plan_entries(goal, program)
    assert not goal['exon_conversion_bindings']


@pytest.mark.parametrize('global_defaults', [True, False])
def test_distinct_program_references_keep_separate_events(global_defaults):
    goal, program = conversion_fixture()
    goal.pop('transcript_lock')
    if not global_defaults:
        goal.pop('exon_label')
        goal.pop('desired_effect')
    other_program = {**program, 'program_id': 'other_conversion', 'transcript_id': 'TXINOTHER.4'}
    goal['design_programs'].append(other_program)
    for name in ['included', 'excluded']:
        other = deepcopy(goal['isoform_transcripts'][name])
        other['transcript_id'] += 'OTHER'
        other['metadata'] = {'id': other['transcript_id'], 'versioned_id': other['transcript_id'] + '.4'}
        other['chrom'] = 'chr9'
        for site in other['sites']:
            site['chrom'] = 'chr9'
        goal['isoform_transcripts'][name + '_other'] = other
        goal['splice_isoform_objectives'].append({'isoform_label': name + '_other',
            'transcript_id': other['transcript_id'] + '.4',
            'desired_abundance': 'decrease' if name == 'included' else 'increase'})
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    for chosen_program, chrom in [(program, 'chr8'), (other_program, 'chr9')]:
        sites = runtime._collect_program_splice_plan_entries(goal, chosen_program)
        assert {(s['chrom'], s['site'], s['position']) for s in sites} == {
            (chrom, 'acceptor', 2000), (chrom, 'donor', 2030)}


@pytest.mark.parametrize('program_only', [False, True])
@pytest.mark.parametrize('program_minimum', [False, True])
def test_derived_program_cannot_erase_original_response_obligations(program_only, program_minimum):
    from aso_agents.design_event_consistency import requested_response_minimums
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, program = conversion_fixture()
    goal.pop('transcript_lock')
    goal['min_asos_per_site'] = 2
    if program_minimum:
        program['min_asos_per_site'] = 3
    if program_only:
        goal.pop('exon_label')
        goal.pop('desired_effect')
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    program.update(program_id='conversion_explore', parent_program_id='conversion',
                   targeting_policy='exploratory_derived', site_kind='acceptor')
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    groups = requested_response_minimums(goal)
    assert {g['site_kind'] for g in groups} == {'acceptor', 'donor'}
    assert {g['minimum'] for g in groups} == {3 if program_minimum else 2}
    _, rows, _ = response_fixture()
    for row in rows:
        row['program_id'] = 'conversion_explore'
    result = region_minimum_delivery(goal, {'rows': rows[:2]}, {}, lambda *a, **k: [])
    assert result['status'] == 'unfulfilled'
    assert next(g for g in result['groups'] if g['site_kind'] == 'donor')['distinct_molecules'] == 0


def test_nullable_program_fields_inherit_explicit_global_constraints():
    from aso_agents.design_event_consistency import requested_response_minimums
    goal, program = conversion_fixture()
    goal.pop('transcript_lock')
    goal['min_asos_per_site'] = 2
    program['min_asos_per_site'] = None
    goal['splice_site_plan'], _ = runtime._compute_splice_site_plan(
        goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)
    assert {(g['site_kind'], g['minimum']) for g in requested_response_minimums(goal)} == {
        ('acceptor', 2), ('donor', 2)}
    goal['exon_ordinal'] = 4
    with pytest.raises(ValueError, match='explicit transcript ordinal disagrees'):
        runtime._compute_splice_site_plan(goal['splice_isoform_objectives'], goal['isoform_transcripts'], goal)


def test_native_annotation_fields_support_response_membership():
    from aso_agents.candidate_minimums import region_minimum_delivery
    goal, rows, _ = response_fixture()
    for row in rows:
        row['source_annotations'] = {key: row[key] for key in ['selected_site_chrom', 'selected_site_strand']}
        row['selected_site_chrom'] = row['selected_site_strand'] = None
    assert region_minimum_delivery(goal, {'rows': rows}, {}, lambda *a, **k: [])['status'] == 'fulfilled'
