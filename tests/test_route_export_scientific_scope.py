from copy import deepcopy
import csv
import json

import run_goal_intake as runtime
from aso_agents import pipeline_tools as pipeline
from test_route_objective_evidence_scope import goal


def write_route(tmp_path):
    path=tmp_path/'route.json'
    row={'Sequence':'ACGTACGTACGTACGT','Synthesis':'ACGTACGTACGTACGT',
         'Chrom':'chr8','Start':80,'End':96,'Strand':'+',
         'selected_label':'acceptor@100','selected_direction':'decrease',
         'recommendation_status':'incomplete_prediction_evidence',
         'recommendation_note':'Exploratory: requested exon identity remains unresolved.',
         'recommendation_basis':'reference_bound_splice_prediction',
         'exon_objective_evidence':[{'status':'unresolved_identity'}],
         'safeguard_report':[],'safeguards_passed':True}
    path.write_text(json.dumps({'rows':[row],'safeguards':[],
        'primary_targets':[{'label':'acceptor@100','direction':'decrease'}]}))
    return path


def test_native_csv_and_workbook_retain_unresolved_scientific_caveats(tmp_path):
    path=write_route(tmp_path)
    csv_path,error=runtime._write_strategy_route_csv(str(path))
    assert error is None
    with open(csv_path) as stream:row=next(csv.DictReader(stream))
    assert row['recommendation_status']=='incomplete_prediction_evidence'
    assert 'unresolved' in row['recommendation_note']
    assert json.loads(row['exon_objective_evidence'])==[{'status':'unresolved_identity'}]
    import openpyxl
    book=openpyxl.load_workbook(path.with_suffix('.xlsx'),read_only=True,data_only=True)
    try:
        headers={cell for sheet in book for cell in next(sheet.values)}
        assert {'recommendation_status','recommendation_note','exon_objective_evidence'} <= headers
    finally:book.close()


def test_no_configured_safeguards_is_not_exported_as_a_pass(tmp_path):
    path=write_route(tmp_path);csv_path,error=runtime._write_strategy_route_csv(str(path))
    assert error is None
    with open(csv_path) as stream:row=next(csv.DictReader(stream))
    assert row['safeguards_status']=='not_configured'
    normalized=pipeline._artifact_candidate_record(row)
    assert normalized is not None and normalized['passes_basic_filters'] is True
    assert pipeline._artifact_candidate_record({**row,'safeguards_status':'fail'})['passes_basic_filters'] is False


def test_native_annotation_and_csv_retain_checked_shared_site_scope(tmp_path):
    path=write_route(tmp_path);value=goal()
    value['splice_site_plan']=[{'site':'acceptor','position':100,'chrom':'chr8','strand':'+',
                               'present_in':['reduced'],'absent_in':['raised']}]
    pipeline._annotate_route_json_with_site_context(json_path=str(path),structured_goal=value)
    row=json.loads(path.read_text())['rows'][0]
    assert row['reference_scope']['present_in']==['REF_A','REF_B']
    assert row['shared_with_beneficial'] is True
    assert row['conflict_between_objectives'] is True
    csv_path,error=runtime._write_strategy_route_csv(str(path));assert error is None
    with open(csv_path) as stream:export=next(csv.DictReader(stream))
    assert json.loads(export['reference_scope'])['complete_for_gene'] is False
    assert 'unestablished' in export['route_interpretation']


def test_site_direction_hint_cannot_reverse_declared_isoform_abundance():
    value=goal()
    for entry in value['isoform_transcripts'].values():
        entry['sites']=[{'chrom':'chr8','strand':'+','donor':40,'exon_index':1,'coding':True},
                        {'chrom':'chr8','strand':'+','acceptor':100,'exon_index':2,'coding':True}]
        entry['exon_count']=2
        for exon in entry['exons_tx']:exon['contains_coding']=True
    value['desired_effect']='skipping'
    value['site_direction_hints']=[{'isoform_label':'raised','desired_delta':'decrease'}]
    before=deepcopy(value)
    plan,_=runtime._compute_splice_site_plan(value['splice_isoform_objectives'],value['isoform_transcripts'],value)
    assert plan
    for site in plan:
        assert site['isoform_abundance_objectives']=={'raised':'increase','reduced':'decrease'}
    assert value['splice_isoform_objectives']==before['splice_isoform_objectives']


def test_safeguard_note_cannot_claim_unique_pathogenic_when_reference_is_shared(monkeypatch):
    value=goal()
    plan=[{'site':'acceptor','position':100,'chrom':'chr8','strand':'+'},
          {'site':'donor','position':40,'chrom':'chr8','strand':'+'}]
    candidates=[{'label':label,'present_in':['reduced'],'absent_in':['raised'],
                 'pathogenic_isoforms':['reduced'],'unique_to_pathogenic':True,
                 'suggested_guard':'no_increase','suggested_guard_reason':'pathogenic_only'}
                for label in ['acceptor@100','donor@40']]
    monkeypatch.setattr(runtime,'resolve_ambiguous_safeguards',lambda *a,**k:{})
    selected=runtime._build_safeguard_candidates_by_target(plan,candidates,value)
    guards=runtime._default_safeguards_from_candidates(selected['acceptor@100'])
    assert len(guards)==1 and 'shared' in guards[0]['note'].lower()
    assert 'pathogenic-only' not in guards[0]['note']


def test_native_router_absent_guards_are_not_a_positive_assessment(tmp_path):
    import importlib
    router=importlib.import_module('aso_agents.splice_strategy_router_agent')
    source=tmp_path/'scored.json';output=tmp_path/'selected.route.json'
    source.write_text(json.dumps({'rows':[{'Sequence':'ACGTACGTACGTACGT',
        'Chrom':'chr8','Start':80,'End':96,'Strand':'+','GC':.5,
        'delta_names':'acceptor@100','delta_deletion':'.2','delta_replacement':'.2'}]}))
    result=router.auto_threshold_constrained_site_metrics(json_path=str(source),
        primary_label='acceptor@100',primary_direction='increase',
        safeguards_json='[]',min_count=1,min_gc=.1,max_gc=.9,output_path=str(output))
    assert result['count']==1
    assert result['safeguards_pass_count']==0
    row=json.loads(output.read_text())['rows'][0]
    assert row['safeguards_passed'] is None
    assert row['safeguards_status']=='not_configured'
    assert pipeline._artifact_candidate_record(row)['passes_basic_filters'] is True


def test_secondary_guard_scope_survives_flat_export_without_changing_bounds(tmp_path):
    path=write_route(tmp_path);value=goal()
    value['splice_site_plan']=[{'site':'acceptor','position':100,'chrom':'chr8','strand':'+'},
                              {'site':'donor','position':40,'chrom':'chr8','strand':'+'}]
    data=json.loads(path.read_text())
    guard={'label':'donor@40','max_increase':.1,'max_decrease':None,
           'max_abs_delta':None,'passes':True,'values':[-.8,-.7],
           'note':'One-sided computational limit; preservation not established.'}
    data['safeguards']=[deepcopy(guard)];data['rows'][0]['safeguard_report']=[deepcopy(guard)]
    path.write_text(json.dumps(data))
    pipeline._annotate_route_json_with_site_context(json_path=str(path),structured_goal=value)
    row=json.loads(path.read_text())['rows'][0]
    actual=row['safeguard_report'][0]
    assert actual['shared_with_beneficial'] is True
    assert actual['unbounded_directions']==['decrease']
    assert 'does not establish' in actual['assessment_scope']
    assert all(actual[k]==v for k,v in guard.items())
    csv_path,error=runtime._write_strategy_route_csv(str(path));assert error is None
    with open(csv_path) as stream:export=next(csv.DictReader(stream))
    assert json.loads(export['safeguard_report'])==row['safeguard_report']
    assert export['safeguards_status']=='pass'
    assert pipeline._artifact_candidate_record(export)['passes_basic_filters'] is True
