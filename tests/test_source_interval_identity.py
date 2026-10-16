"""Source identity and program provenance control physical-region selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aso_agents.candidate_minimums import belongs_to_region, requested_region_minimums
from aso_agents.candidate_region_gate import filter_final_rows_by_regions
from aso_agents.region_interval_summary import nominated_region_interval_summary
from aso_agents.region_projection import genomic_sources, source_intervals


def intervals(row, **_):
    return row['intervals']


def fixture(strand='+'):
    exons = [{'start': 1001, 'end': 1020}, {'start': 2001, 'end': 2060}]
    if strand == '-':
        exons.reverse()
    source = {'constraint_id': 'windows', 'label': 'shared_name', 'source_type': 'explicit_rows',
        'coordinate_space': 'cdna', 'operator': 'union', 'aso_overlap_requirement': 'any_overlap',
        'minimum_overlap_bp': 8, 'materialized_rows': [
            ['TX.3', 12, 35, 'shared_name'], ['TX.3', 28, 50, 'shared_name']]}
    goal = {'min_asos_per_site': 2, 'isoform_transcripts': {
        'main': {'transcript_id': 'TX.3', 'chrom': 'chrFixture', 'strand': strand, 'exons_tx': exons}},
        'steric_design_jobs': [{'job_id': 'job_one', 'program_id': 'panel'}],
        'design_programs': [{'program_id': 'panel', 'modality': 'steric'}],
        'materialized_region_constraints': {'by_program': {'panel': [source]}}}
    return goal, source


@pytest.mark.parametrize('strand', ['+', '-'])
def test_same_label_sites_remain_separate_after_projection_and_obey_each_threshold(strand):
    goal, source = fixture(strand)
    original = deepcopy(goal)
    groups = requested_region_minimums(goal)
    assert len(groups) == 2 and len({group['group_id'] for group in groups}) == 2
    assert len({group['label'] for group in groups}) == 2
    def row(a, b):
        projected = genomic_sources(goal, {**source, 'materialized_rows': [['TX.3', a, b]]})[0]
        return {'intervals': [block[:3] for block in projected['genomic_rows']]}
    assert [belongs_to_region(row(12, 20), group, intervals) for group in groups] == [True, False]
    assert [belongs_to_region(row(40, 48), group, intervals) for group in groups] == [False, True]
    assert not any(belongs_to_region(row(28, 35), group, intervals) for group in groups)
    assert goal == original


@pytest.mark.parametrize('strand', ['+', '-'])
def test_one_source_split_over_exons_counts_unique_overlap_across_its_blocks(strand):
    goal, source = fixture(strand)
    source['minimum_overlap_bp'] = 12
    source['materialized_rows'] = [['TX.3', 10, 70, 'single']]
    group = requested_region_minimums(goal)[0]
    assert len(group['selectors'][0]['rows']) == 2
    boundary = 20 if strand == '+' else 60
    def row(a, b):
        blocks = genomic_sources(goal, {**source, 'materialized_rows': [['TX.3', a, b]]})[0]['genomic_rows']
        return {'intervals': [block[:3] for block in blocks] * 2}
    assert belongs_to_region(row(boundary - 6, boundary + 6), group, intervals)
    assert not belongs_to_region(row(boundary - 5, boundary + 6), group, intervals)


def test_source_identity_ignores_display_name_order_and_exact_duplicate_rows():
    _, source = fixture()
    ids = [item['source_interval_id'] for item in source_intervals(source)]
    source['materialized_rows'].reverse()
    source['materialized_rows'][0][3] = 'renamed'
    source['materialized_rows'].append(deepcopy(source['materialized_rows'][0]))
    assert [item['source_interval_id'] for item in source_intervals(source)] == ids[::-1]
    source['coordinate_space'] = 'genomic'
    assert set(ids).isdisjoint(item['source_interval_id'] for item in source_intervals(source))


def test_summary_keeps_same_named_intervals_across_constraints_and_deduplicates_physical_sites():
    goal, source = fixture()
    duplicate = deepcopy(source)
    duplicate['constraint_id'] = 'duplicate'
    goal['materialized_region_constraints']['by_program']['panel'].append(duplicate)
    result = nominated_region_interval_summary(goal)[0]
    assert len(result['source_regions']) == len(result['source_interval_ids']) == 2
    assert result['union_intervals'] == [[12, 50]] and result['union_length_nt'] == 38
    assert result['intersection_of_all_regions'] == [[28, 35]] and result['intersection_length_nt'] == 7
    assert result['coordinate_convention'] == '0based_half_open'


def test_aggregate_projection_is_not_distributed_among_unresolved_source_sites():
    goal, source = fixture()
    goal['isoform_transcripts'] = {}
    job = goal['steric_design_jobs'][0]
    job['selection_meta'] = {'region_constraints': {'tile_constraint_filters': [
        {'rows': [['chrFixture', 100, 120, 'shared_name'], ['chrFixture', 200, 230, 'shared_name']]}]}}
    groups = requested_region_minimums(goal)
    assert len(groups) == 2
    assert all(not group['selectors'][0]['explicit_scope_resolved'] for group in groups)
    source['materialized_rows'] = source['materialized_rows'][:1]
    group = requested_region_minimums(goal)[0]
    assert group['selectors'][0]['explicit_scope_resolved']
    assert len(group['selectors'][0]['rows']) == 2


def test_program_gate_filters_aggregate_pool_without_changing_source_rows():
    goal, source = fixture()
    rows = [{'name': str(a), 'source_modality': 'steric', 'intervals': [['chrFixture', a, a + 12]]}
            for a in [1005, 1012, 2030, 2043]]
    original = deepcopy(rows)
    kept, receipt = filter_final_rows_by_regions(rows, goal, intervals)
    assert [row['name'] for row in kept] == ['1012']
    assert receipt['status'] == 'complete' and receipt['excluded_count'] == 3
    assert all(row['program_id'] == 'panel' for row in kept)
    assert rows == original


def test_program_gate_keeps_separate_rules_and_does_not_infer_identity_from_location():
    goal, source = fixture()
    other = {'constraint_id': 'other_region', 'materialized_rows': [['chrFixture', 3000, 3050]],
             'aso_overlap_requirement': 'full_containment'}
    goal['design_programs'].append({'program_id': 'other', 'modality': 'steric'})
    goal['materialized_region_constraints']['by_program']['other'] = [other]
    rows = [{'name': name, 'program_id': program, 'intervals': [['chrFixture', a, a + 10]]}
        for name, program, a in [('first', 'panel', 1012), ('second', 'other', 3001),
                                 ('wrong_scope', 'panel', 3001), ('unbound', '', 1012),
                                 ('unknown', 'absent', 1012)]]
    kept, receipt = filter_final_rows_by_regions(rows, goal, intervals)
    assert [row['name'] for row in kept] == ['first', 'second', 'unbound', 'unknown']
    assert receipt['status'] == 'unresolved'
    assert [row['row_index'] for row in receipt['unresolved_rows']] == [3, 4]
    assert kept[-2]['program_id'] == '' and kept[-1]['program_id'] == 'absent'


def test_program_rules_preserve_compiled_global_intersection_and_exclusion_order():
    goal, source = fixture()
    source['operator'] = 'intersect'
    goal['materialized_region_constraints']['global'] = [{'materialized_rows': [['chrFixture', 1000, 2000]]}]
    goal['materialized_region_constraints']['by_program']['panel'].append({
        'operator': 'exclude', 'materialized_rows': [['chrFixture', 1018, 1020]]})
    rows = [{'intervals': [['chrFixture', a, b]], 'program_id': 'panel'}
            for a, b in [(1010, 1020), (1012, 1019), (2008, 2020)]]
    assert filter_final_rows_by_regions(rows, goal, intervals)[0] == []


@pytest.mark.parametrize('local', [[{'materialized_rows': []}], ['invalid'], {'invalid': True}])
def test_incomplete_local_rules_are_explicitly_unverified(local):
    goal, _ = fixture()
    goal['materialized_region_constraints']['by_program']['panel'] = local
    rows = [{'program_id': 'panel', 'intervals': [['chrFixture', 3000, 3020]]}]
    kept, receipt = filter_final_rows_by_regions(rows, goal, intervals)
    assert kept == rows and receipt['status'] == 'unresolved'
    assert receipt['unresolved_rows'][0]['reason'] == 'materialization_incomplete'


def test_native_export_enforces_two_same_named_program_intervals_and_total_panel_size(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    from aso_chemistry import encode_idt_oligo
    goal, source = fixture()
    goal.update(candidate_minimum=4, candidate_limit=4)
    source['coordinate_space'] = 'genomic'
    source['materialized_rows'] = [['chrFixture', 100, 130, 'shared'], ['chrFixture', 200, 230, 'shared']]
    rows = []
    for index, a in enumerate([98, 100, 102, 104, 198, 200, 225]):
        synthesis = 'ACGT' * 4 + 'ACGT'[index // 4] + 'ACGT'[index % 4] + 'AC'
        rows.append({'Synthesis': synthesis, 'Sequence': synthesis.translate(str.maketrans('ACGT','TGCA'))[::-1],
                     'IDT': encode_idt_oligo(synthesis, 'steric-hindrance'), 'source_modality': 'steric',
                     'selection_rank': index + 1, 'Chrom': 'chrFixture', 'Start': a, 'End': a + 20,
                     'coordinate_system': '0based_half_open', 'Strand': '+'})
    monkeypatch.setattr(pipeline, '_collect_generated_candidate_rows', lambda *a, **k: rows)
    monkeypatch.setattr(pipeline, '_collect_generated_stage_sets', lambda *a: {})
    monkeypatch.setattr(pipeline, '_goal_payload_context', lambda *a: {'structured_goal': goal})
    files = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    panel = json.loads(Path(files['generated_aso_set_json']).read_text())
    assert len(panel['rows']) == 4
    assert {row['selection_rank'] for row in panel['rows']} == {1, 2, 5, 6}
    assert panel['candidate_selection']['status'] == 'complete'
    assert [group['selected_count'] for group in panel['candidate_selection']['groups']] == [2, 2]
