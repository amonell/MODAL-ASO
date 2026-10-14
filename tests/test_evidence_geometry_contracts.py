from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

import run_goal_intake as native
from aso_agents import pipeline_tools, repeat_risk
from aso_agents.mapping_evidence import annotate_reference_junctions
from aso_agents.regulatory_feature_scope import restrict_source_rows
from aso_agents.report_coordinates import attach_genomic_marker


def transcript(strand='+', name='TX.4'):
    bounds = [(101,130),(201,230),(301,330)]
    if strand == '-':
        bounds.reverse()
    return {'transcript_id':name,'chrom':'chr8','strand':strand,
            'metadata':{'versioned_id':name,'seq_region_name':'chr8'},
            'exons_tx':[{'index':i+1,'start':a,'end':b} for i,(a,b) in enumerate(bounds)]}


@pytest.mark.parametrize('strand,expected', [('+',125),('-',306)])
def test_cdna_marker_uses_reference_projection_and_preserves_source(strand, expected):
    target = dict(chrom='TX',region_start=20,region_end=30,position=25,
                  coordinate_system='bed0_half_open',site_kind='region')
    original = deepcopy(target)
    attach_genomic_marker(target, {'TX':transcript(strand)}, native._map_cdna_range_to_genomic_segments, 'chr8')
    assert target['genomic_map_position'] == expected
    assert target['source_region']['position'] == 25
    assert all(target[k] == v for k,v in original.items())


def test_spliced_marker_lies_on_exon_and_unknown_version_is_not_plotted():
    target = dict(chrom='TX',region_start=20,region_end=40,position=30,
                  coordinate_system='bed0_half_open',site_kind='region')
    attach_genomic_marker(target, {'TX':transcript()}, native._map_cdna_range_to_genomic_segments, '8')
    assert target['genomic_map_position'] == 130
    assert target['genomic_map_segments'] == [dict(chrom='chr8',start=121,end=130),dict(chrom='chr8',start=201,end=210)]
    target['chrom'] = 'TX.5'
    attach_genomic_marker(target, {'TX':transcript()}, native._map_cdna_range_to_genomic_segments, '8')
    assert target['genomic_map_position'] is None
    target['chrom'] = 'TX';target['region_end'] = 400
    attach_genomic_marker(target, {'TX':transcript()}, native._map_cdna_range_to_genomic_segments, '8')
    assert target['genomic_map_position'] is None


def test_native_report_marks_both_coordinate_spaces_on_the_same_genome():
    goal = {'isoform_transcripts':{'TX':transcript()},'gapmer_bed':{'rows':[['TX',20,30,'cdna',0,'+']]},
            'intent':'binding','steric_bed':{'rows':[['chr8',120,130,'genomic',0,'+']]}}
    targets = native._build_targets(goal, {}, [], [])
    assert len(targets) == 2
    assert {t['genomic_map_position'] for t in targets} == {125}


def repeat_row(start, *, sequence='ACGTACGT', **kwargs):
    return {'Sequence':sequence,'Genomic_segments':[dict(chrom='chr8',start=start,end=start+7)],
            'Strand':'+','coordinate_system':'1based_inclusive','source':'gapmer_cdna',**kwargs}


def classify(rows):
    return repeat_risk.classify_repeat_hits_for_rows(rows, {'isoform_transcripts':{'TX':transcript()}},
                                                   primary_transcript_id='TX.4', modality='gapmer')


def test_overlapping_repeat_placements_are_positive_evidence_not_exhaustive_counts():
    rows = [repeat_row(105), repeat_row(108), repeat_row(108)]
    original = deepcopy(rows)
    summary = classify(rows)
    assert summary['incomplete_inventory_row_count'] == 3
    assert all(r['repeat_observed_hit_count_exact'] == 1 for r in rows)
    assert all(r['repeat_observed_hit_count_same_transcript'] == 1 for r in rows)
    assert all(r['repeat_hit_count_exact'] is None for r in rows)
    assert all(r['repeat_inventory_status'] == 'partial' for r in rows)
    assert [r['Genomic_segments'] for r in rows] == [r['Genomic_segments'] for r in original]
    assert all(r['repeat_risk_class'] != 'none' for r in rows)


@pytest.mark.parametrize('mapping', [None,'','overflow','3+matches','chr8:105-112,broken'])
def test_absent_capped_or_malformed_inventory_never_claims_no_repeats(mapping):
    row = repeat_row(105)
    if mapping is not None: row['Mapped_to'] = mapping
    classify([row])
    assert row['repeat_hit_count_exact'] is None
    assert row['repeat_splice_eval_status'] != 'no_repeat_hits'


def test_complete_mapping_inventory_counts_only_alternative_physical_loci():
    rows = [repeat_row(105, Mapped_to='chr8:105-112'),
            repeat_row(205, sequence='GGCCAATT',Mapped_to='chr8:205-212,chr8:208-215,chr8:208-215')]
    classify(rows)
    assert rows[0]['repeat_hit_count_exact'] == 0
    assert rows[0]['repeat_inventory_status'] == 'complete'
    assert rows[1]['repeat_hit_count_exact'] == 1
    assert rows[1]['repeat_hit_count_same_transcript'] == 1


def test_a_reported_placement_missing_from_mapper_invalidates_completeness():
    rows = [repeat_row(105, Mapped_to='chr8:105-112'),repeat_row(108)]
    classify(rows)
    assert rows[0]['repeat_hit_count_exact'] is None
    assert rows[0]['repeat_observed_hit_count_exact'] == 1
    assert 'omits' in rows[0]['repeat_inventory_evidence']['basis']


def test_later_complete_mapper_inventory_replaces_derived_unknown_without_stale_zero():
    row = repeat_row(105)
    classify([row])
    assert row['repeat_hit_count_exact'] is None
    row['Mapped_to'] = 'chr8:105-112,chr8:108-115'
    classify([row])
    assert row['repeat_inventory_status'] == 'complete'
    assert row['repeat_hit_count_same_transcript'] == 1
    classify([row])
    assert row['repeat_hit_count_same_transcript'] == 1


def test_declared_partial_structured_inventory_is_not_upgraded_by_parsability():
    row=repeat_row(105,Mapped_loci=[dict(chrom='chr8',start=105,end=112)],
                   repeat_inventory_status='partial')
    classify([row])
    assert row['repeat_inventory_status'] != 'complete'
    assert row['repeat_hit_count_exact'] is None
    classify([row])
    assert row['repeat_hit_count_same_transcript'] is None


def test_repeat_summary_exposes_missingness_and_observed_hits(tmp_path):
    rows = [repeat_row(105),repeat_row(108),repeat_row(205,sequence='GGCCAATT')]
    classify(rows)
    path=tmp_path/'tiles.json';path.write_text(json.dumps({'rows':rows}))
    summary=repeat_risk.summarize_repeat_risk_for_tiles_json(str(path))
    assert summary['incomplete_inventory_row_count'] == 3
    assert summary['repeat_row_count'] == 2
    assert summary['same_transcript_hit_count'] == 2


@pytest.mark.parametrize('strand',['+','-'])
def test_source_feature_projection_preserves_original_peaks_and_boundary_semantics(strand):
    source = [['chr8',80,110,'crosses',0,strand],['chr8',140,155,'inside',0,strand],
              ['chr8',150,170,'opposite',0,'-' if strand=='+' else '+'],
              ['chr8',180,200,'touches',0,strand],['chr9',120,140,'other_chrom',0,strand]]
    original = deepcopy(source)
    regions = [['8',100,180,'feature',0,strand]]
    rows,audit=restrict_source_rows(source,regions,'utr3',reference_transcript='TX.4')
    assert rows == [['chr8',100,110,'crosses',0,strand],['chr8',140,155,'inside',0,strand]]
    assert audit['source_rows_bed0_half_open'] == source == original
    assert audit['excluded_source_row_count'] == 3
    assert audit['retained_source_row_count'] == 2


def test_missing_feature_reference_is_unresolved_not_an_empty_verified_feature():
    rows,audit=restrict_source_rows([['chr8',10,20]],[],'utr5')
    assert not rows and audit['status'] == 'unresolved_reference_feature'


def test_discontiguous_reference_feature_does_not_fill_introns_or_duplicate_intersections():
    rows,audit=restrict_source_rows([['chr8',10,80,'peak']],
        [['chr8',10,20],['chr8',15,20],['chr8',60,70]],'utrs')
    assert [r[1:3] for r in rows] == [[10,20],[60,70]]
    assert audit['retained_source_row_count'] == 1


@pytest.mark.parametrize('assembly,compatible', [('GRCh38',True),('hg38',True),(None,False),('GRCh37',False)])
def test_native_rbp_materializer_uses_source_assembly_and_selected_feature(tmp_path,monkeypatch,assembly,compatible):
    source={'file_format':'bed','download_url':'https://source.invalid/peaks.bed',
            'file_accession':'neutral_peaks','assembly':assembly}
    monkeypatch.setattr(native,'_select_encode_rbp_dataset',lambda **_: {'selected':source})
    monkeypatch.setattr(native,'_download_encode_peak_file',lambda _,dest: Path(dest).write_text(
        'chr8\t90\t115\tcrosses\t10\t+\nchr8\t140\t155\tinside\t10\t+\nchr8\t190\t210\toutside\t10\t+\n'))
    monkeypatch.setattr(native,'_compute_gene_span',lambda _: {'chrom':'chr8','start0':80,'end':250})
    monkeypatch.setattr(native,'_build_utr_bed_rows',lambda goal,region: [['chr8',100,180,region,0,'+']])
    monkeypatch.setattr(native,'_select_isoform_for_utr',lambda *_: ('TX',{'transcript_id':'TX.4'}))
    rows,summary=native._materialize_rbp_binding_rows({'rbp':'FACTOR'},
        {'species':'human','target_region':'utr3'},str(tmp_path))
    if compatible:
        assert sorted(r[1:3] for r in rows)==[[100,115],[140,155]]
        assert summary['source_peak_count']==3 and summary['peak_count']==2
        assert summary['feature_scope']['reference_transcript']=='TX.4'
    else:
        assert not rows and summary['error']=='source_feature_assembly_mismatch'


def test_broad_transcript_and_unresolved_named_feature_do_not_select_all_boundaries():
    sites = pipeline_tools._build_transcript_splice_sites_for_eval(transcript())
    targets = [{'transcript_id':'TX.4','target_type':'transcript','objective_type':'primary_outcome'},
               {'transcript_id':'TX.4','target_type':'exon','label':'short alternative exon'}]
    assert pipeline_tools._filter_splice_sites_by_targets(sites,targets) == []
    targets.append({'transcript_id':'TX.4','exon_label':2,'site_kind':'donor'})
    selected=pipeline_tools._filter_splice_sites_by_targets(sites,targets)
    assert len(selected)==1 and selected[0]['donor']['position']==230
    assert pipeline_tools._filter_splice_sites_by_targets(sites,[{'transcript_id':'OTHER','exon_label':2}]) == []
    assert pipeline_tools._filter_splice_sites_by_targets(sites,[{'transcript_id':'TX.5','exon_label':2}]) == []


def test_evaluation_primary_outcome_precedes_safeguard_default_but_not_explicit_lock():
    catalog={'A':transcript(name='A.4'),'B':transcript(name='B.2')}
    goal={'transcript_id':'B.2','evaluation_targets':[
        {'transcript_id':'B.2','objective_type':'monitoring_safeguard'},
        {'transcript_id':'A.4','objective_type':'primary_outcome'}]}
    assert pipeline_tools._choose_eval_transcript(goal,catalog,{},SimpleNamespace())[0]=='A.4'
    goal['transcript_lock']={'value':'B.2'}
    assert pipeline_tools._choose_eval_transcript(goal,catalog,{},SimpleNamespace())[0]=='B.2'
    goal['transcript_lock']={'value':'B.3'}
    assert pipeline_tools._choose_eval_transcript(goal,catalog,{},SimpleNamespace())[2] is None


@pytest.mark.parametrize('strand',['+','-'])
def test_reference_junction_geometry_retains_actual_ordinals_and_intervening_exons(strand):
    full=transcript(strand, 'FULL.1'); skipped=deepcopy(full)
    skipped['transcript_id']='SKIP.1';skipped['metadata']['versioned_id']='SKIP.1'
    skipped['exons_tx'].pop(1);skipped['exons_tx'][1]['index']=2
    blocks = [dict(chrom='chr8',start=126,end=130),dict(chrom='chr8',start=301,end=305)] if strand=='+' else [
        dict(chrom='chr8',start=301,end=305),dict(chrom='chr8',start=126,end=130)]
    hit={'transcript_id':'SKIP','mapping_scope':'spliced_cdna','genomic_segments':blocks}
    row={'mapping_hits':[hit]}
    annotate_reference_junctions(row, {'FULL':full,'SKIP':skipped})
    junction=hit['reference_junctions_spanned'][0]
    assert junction['upstream_transcript_ordinal']==1 and junction['downstream_transcript_ordinal']==2
    assert {junction['donor'],junction['acceptor']} == {130,301}
    assert junction['catalog_intervening_exon_context'][0]['intervening_annotated_exons'][0]['start']==201
    from aso_agents.evaluation_evidence import attach_measurement_references
    other = {'input_row_id':'other_observation', 'name':'other', 'mapping_hits':[]}
    row.update(input_row_id='junction_observation', name='junction', requested_feature_membership=[
        {'identity_resolution':{'status':'unresolved'}, 'observations':[]}])
    summary = attach_measurement_references({'provided_asos':[row,other]})
    available = row['evidence_inventory']['feature_evidence_availability']
    assert available['requested_site_register']['status']=='not_recorded'
    assert available['independent_reference_junction_geometry']=={
        'status':'recorded','source_record_count':1,'source_fields':['mapping_hits[0]']}
    assert other['evidence_inventory']['feature_evidence_availability']['independent_reference_junction_geometry']['status']=='not_recorded'
    token=row['reference_tokens']['evidence_inventory'].strip('{}')
    assert summary['measurement_evidence'][token]['value']['feature_evidence_availability']==available
    hit['mapping_scope']='pre_mrna_locus'
    annotate_reference_junctions(row, {'FULL':full,'SKIP':skipped})
    assert hit['reference_junctions_spanned']==[]
    attach_measurement_references(summary)
    assert row['evidence_inventory']['feature_evidence_availability']['independent_reference_junction_geometry']['status']=='not_recorded'


def test_shared_site_has_no_selective_direction_from_opposing_abundance_goals():
    a,b=transcript(name='A'),transcript(name='B')
    for entry in (a,b):
        for exon in entry['exons_tx']:
            exon['contains_coding']=True
        entry['sites']=[{'chrom':'chr8','exon_index':2,'acceptor':201,'donor':230,'coding':True,'strand':'+'}]
    objectives=[{'isoform_label':'A','desired_abundance':'increase'},
                {'isoform_label':'B','desired_abundance':'decrease'}]
    plan,_=native._compute_splice_site_plan(objectives,{'A':a,'B':b},{'desired_effect':'inclusion'})
    assert plan and all(p['delta_direction']=='context' for p in plan)
    assert all(p['isoform_abundance_objectives']=={'A':'increase','B':'decrease'} for p in plan)
    assert all(p['delta_per_isoform']==p['isoform_abundance_objectives'] for p in plan)
    assert all(p['delta_per_isoform_scope']=='desired_objectives_not_predicted_effects' for p in plan)
    objectives[1]['desired_abundance']='increase'
    plan,_=native._compute_splice_site_plan(objectives,{'A':a,'B':b},{'desired_effect':'inclusion'})
    assert plan and all(p['delta_direction']=='increase' for p in plan)


def test_structure_prefers_configured_reference_but_honors_explicit_version(monkeypatch):
    import importlib
    chooser=importlib.import_module('aso_agents.transcript_chooser_agent')
    source={'id':'TX','version':4,'versioned_id':'TX.4','reference_source':'local_gtf'}
    monkeypatch.setattr(chooser,'_lookup_local_gtf_id',lambda _:deepcopy(source))
    calls=[]
    def remote(identifier,expand=1):
        calls.append(identifier)
        return {'id':'TX','version':5,'versioned_id':'TX.5','Exon':[{}]}
    monkeypatch.setattr(chooser,'_lookup_id',remote)
    monkeypatch.setattr(chooser,'_transcript_structure_from_payload',lambda data,requested: data)
    assert chooser.get_transcript_structure('TX')['version']==4 and not calls
    assert chooser.get_transcript_structure('TX.4')['version']==4 and not calls
    assert chooser.get_transcript_structure('TX.5')['version']==5 and calls==['TX.5']
