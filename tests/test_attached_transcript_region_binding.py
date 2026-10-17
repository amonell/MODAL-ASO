"""Attachment intervals constrain designs even when intake omitted their rows."""

import json

from aso_agents.attached_regions import bind_attached_transcript_intervals
from aso_agents.candidate_minimums import belongs_to_region, requested_region_minimums


def test_distinct_attached_transcript_windows_bind_only_their_named_programs(tmp_path):
    public = tmp_path / 'windows.json'
    public.write_text(json.dumps({'features': [
        {'feature_id': 'zone_alpha', 'reference_transcript': 'TX.3',
         'start': 21, 'end': 80, 'coordinate_system': 'mature_transcript_1based_inclusive'},
        {'feature_id': 'zone_beta', 'reference_transcript': 'TX.3',
         'start': 121, 'end': 180, 'coordinate_system': 'mature_transcript_1based_inclusive'},
    ]}))
    goal = {'artifact_refs': [{'path': str(public)}], 'design_programs': [
        {'program_id': 'first', 'targeting_policy': 'explicit_user_region',
         'objective': 'Two molecules in zone_alpha', 'transcript_id': 'TX.3'},
        {'program_id': 'second', 'targeting_policy': 'explicit_user_region',
         'objective': 'Two molecules in zone_beta', 'transcript_id': 'TX.3'},
    ]}
    updated, evidence = bind_attached_transcript_intervals(goal, tmp_path)
    assert evidence['bound_programs'] == ['first', 'second']
    assert updated['design_programs'][0]['region_constraints'][0]['rows'][0][:3] == ['TX.3', 20, 80]
    assert updated['design_programs'][1]['region_constraints'][0]['rows'][0][:3] == ['TX.3', 120, 180]
    assert not goal['design_programs'][0].get('region_constraints')


def test_attachment_binding_keeps_existing_constraints_and_rejects_ambiguous_ids(tmp_path):
    public = tmp_path / 'windows.json'
    public.write_text(json.dumps({'features': [
        {'feature_id': 'site1', 'reference_transcript': 'TX.3', 'start': 1, 'end': 40,
         'coordinate_system': 'mature_transcript_1based_inclusive'},
        {'feature_id': 'site2', 'reference_transcript': 'TX.3', 'start': 50, 'end': 90,
         'coordinate_system': 'mature_transcript_1based_inclusive'},
    ]}))
    existing = [{'constraint_id': 'already_decided'}]
    goal = {'artifact_refs': [{'path': str(public)}], 'design_programs': [
        {'program_id': 'p1', 'targeting_policy': 'explicit_user_region',
         'objective': 'site1 and site2', 'transcript_id': 'TX.3'},
        {'program_id': 'p2', 'targeting_policy': 'explicit_user_region',
         'objective': 'site1', 'transcript_id': 'TX.3', 'region_constraints': existing},
    ]}
    updated, evidence = bind_attached_transcript_intervals(goal, tmp_path)
    assert evidence['bound_programs'] == []
    assert evidence['unresolved_programs'] == ['p1']
    assert updated['design_programs'][1]['region_constraints'] == existing


def test_unresolved_explicit_region_cannot_satisfy_per_site_minimum_by_broad_gene_overlap():
    goal = {'min_asos_per_site': 2, 'steric_design_jobs': [
        {'program_id': 'site_a', 'targeting_policy': 'explicit_user_region',
         'region_rows': [['chr7', 100, 300]], 'selection_meta': {
             'region_constraints': {'constraint_count': 0}}},
    ]}
    group = requested_region_minimums(goal)[0]
    row = {'binding_intervals_bed0_half_open': [['chr7', 120, 141]],
           'synthesized_antisense_sequence': 'ACGT' * 5}
    assert not belongs_to_region(row, group,
                                 lambda value, **_: value['binding_intervals_bed0_half_open'])
