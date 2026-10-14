from copy import deepcopy
import csv

import pytest
import run_goal_intake as native
from aso_agents.design_delivery import _resolved_transcript_site, write_design_panel
from aso_agents.region_overlap import constraint_execution_summary, matches_constraint_filters


@pytest.mark.parametrize('space', ['cdna', 'genomic'])
@pytest.mark.parametrize('start,end', [(1, 1), (11, 25)])
def test_inclusive_rows_normalize_once_and_preserve_base_count(tmp_path, space, start, end):
    raw = {'constraint_id':'window','source_type':'explicit_rows', 'coordinate_space':space,
           'coordinate_convention':'one_based_inclusive', 'rows':[['TX',start,end]],
           'aso_overlap_requirement':'full_containment'}
    normalized = native._normalize_region_constraints([raw])
    assert normalized[0]['rows'][0][1:3] == [start-1,end]
    assert normalized[0]['coordinate_convention'] == 'bed0_half_open'
    assert native._normalize_region_constraints(normalized) == normalized
    materialized = native._materialize_region_constraint_entry(
        structured_goal={'species':'human'}, final_payload={}, run_dir=str(tmp_path), constraint=raw)
    assert materialized['rows'] == raw['rows']
    assert materialized['materialized_rows'][0][1:3] == [start-1,end]
    assert materialized['materialized_coordinate_convention'] == 'bed0_half_open'


def test_execution_summary_exposes_or_without_changing_legitimate_alternatives():
    filters = [
        {'operator':'union','rows':[['T',10,50]],'aso_overlap_requirement':'full_containment'},
        {'operator':'union','rows':[['T',24,27]],'aso_overlap_requirement':'contains_feature'},
    ]
    goal = {'design_programs':[{'region_constraints':filters}]}
    before = deepcopy(goal)
    summary = constraint_execution_summary(goal)[0]
    assert summary['constraints_allowed_to_fail_when_all_others_match'] == summary['constraint_pointers']
    assert summary['constraints_sufficient_alone_for_admission'] == summary['constraint_pointers']
    assert matches_constraint_filters([('T',5,30)], filters)
    assert goal == before
    filters[1]['operator'] = 'intersect'
    summary = constraint_execution_summary(goal)[0]
    assert summary['constraints_allowed_to_fail_when_all_others_match'] == []
    assert summary['constraints_sufficient_alone_for_admission'] == []
    assert not matches_constraint_filters([('T',5,30)], filters)
    assert matches_constraint_filters([('T',10,30)], filters)


@pytest.mark.parametrize('strand', ['+', '-'])
def test_export_maps_contiguous_mature_site_across_exons(tmp_path, strand):
    exons = [{'index':1,'start':101,'end':120,'cdna_start':1},
             {'index':2,'start':201,'end':220,'cdna_start':21}]
    if strand == '-':
        exons = [{**exons[1],'index':1,'cdna_start':1}, {**exons[0],'index':2,'cdna_start':21}]
    record = {'transcript_id':'TX','chrom':'chr8','strand':strand,'exons_tx':exons,
              'metadata':{'versioned_id':'TX.2'}}
    goal = {'transcript_id':'TX.2','isoform_transcripts':{'TX':record}}
    blocks = [['chr8',110,120],['chr8',200,210]]
    row = {'chrom':'TX.2','start':11,'end':30,'length':20,'strand':strand,
           'binding_intervals_bed0_half_open':blocks,
           'synthesized_antisense_sequence':'ACGT'*5,'target_binding_sequence':'ACGT'*5}
    site = _resolved_transcript_site(goal,row)
    assert (site['cdna_start'],site['cdna_end']) == (11,30)
    assert site['exon_ordinal'] is None
    write_design_panel({'rows':[row]},goal,tmp_path)
    delivered = list(csv.DictReader((tmp_path/'design_panel.csv').open()))[0]
    assert delivered['genomic_chrom'] == 'chr8'
    assert delivered['transcript_cdna_start'] == '11'
    assert delivered['transcript_cdna_end'] == '30'
    # An intervening unbound exon portion cannot become one continuous target.
    row['binding_intervals_bed0_half_open'] = [['chr8',110,118],['chr8',200,210]]
    assert _resolved_transcript_site(goal,row)['cdna_start'] is None
    # A genomic intron-containing span must not be labeled mature RNA.
    row['binding_intervals_bed0_half_open'] = [['chr8',110,210]]
    assert _resolved_transcript_site(goal,row)['cdna_start'] is None


def test_focused_audit_preserves_evidence_and_confines_patch_surface():
    from aso_agents import pipeline_tools as tools
    goal = {'gene_symbol':'GENERIC','species':'human','design_programs':[{
        'program_id':'binding','modality':'steric','region_constraints':[
            {'operator':'union','source_type':'explicit_rows','rows':[['TX',9,45]],
             'coordinate_space':'cdna','aso_overlap_requirement':'full_containment'},
            {'operator':'union','source_type':'explicit_rows','rows':[['TX',20,23]],
             'coordinate_space':'cdna','aso_overlap_requirement':'contains_feature'}]}]}
    envelope={'tasks':[{'task_id':'d','task_type':'design_new_asos',
                        'context':{'structured_goal':goal}}]}
    pointer='/tasks/0/context/structured_goal/design_programs/0/region_constraints'
    evidence=[{'evidence_id':'request','kind':'user_request','text':'Keep either alternative.'}]
    before=deepcopy(envelope)
    prompt=tools._goal_audit_prompt(request_envelope=envelope,validation_issues=[],
        issue_records=[],evidence_sources=evidence,semantic_audit=True,
        semantic_scope_pointers=[pointer])
    protocol=tools._goal_audit_protocol_payload(prompt)
    assert protocol['current_request_envelope']==before
    assert protocol['evidence_sources'][0]['text']==evidence[0]['text']
    paths={item['json_pointer'] for item in tools._semantic_goal_audit_patch_destinations(protocol)}
    assert pointer+'/1/operator' in paths
    assert paths and all(path==pointer or path.startswith(pointer+'/') for path in paths)
    assert '/tasks/0/context/structured_goal/gene_symbol' not in paths
    assert protocol['region_constraint_execution'][0]['scopes'][0]['constraints_sufficient_alone_for_admission']
    assert envelope==before
