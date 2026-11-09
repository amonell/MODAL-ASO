"""The report is a scoped projection, with exact portable ordering evidence."""
from copy import deepcopy
from html.parser import HTMLParser
import json

from aso_agents.report_explorer import explorer_html, pack_data
from aso_agents.report_explorer_data import project_run, chemistry, coordinates, project_records, safeguard_summary
from aso_agents.readable_report import basic_document
from aso_agents.candidate_export_evidence import export_context, named_rank_components, with_native_ranks
from aso_agents.design_qualification import applied_filter_thresholds


class Packet(HTMLParser):
    def __init__(self, document):
        super().__init__(); self.active = False; self.text = ''; self.feed(document)
    def handle_starttag(self, tag, attrs):
        if tag=='script' and dict(attrs).get('class')=='report-explorer-data': self.active = True
    def handle_data(self, data):
        if self.active: self.text += data
    def handle_endtag(self, tag):
        if tag=='script': self.active = False


def save(root, path, value):
    target = root/path; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value)); return target


def native_row(sequence='ACGT', start=100, notation='A*C*G*T', rank=1):
    return {'Synthesis':sequence,'IDT':notation,'Chrom':'chr1','Start':start,'End':start+4,
            'Strand':'+','GC':.5,'off_target_search_status':'complete','Off_target_exact_count':0,
            'Off_target_1mm_count':0,'Off_target_2mm_count':1,'route_rank':rank,
            'splice_site_deltas':[{'canonical_label':'acceptor@110','site_kind':'acceptor',
                'acceptor':{'chrom':'chr1','position':110}, 'delta_acceptor_deletion':.1,
                'delta_acceptor_replacement':.3,'reference_binding_status':'resolved_catalog_identity',
                'exon_index':2}]}


def run_fixture(root):
    goal = {'gene_symbol':'TEST','design_programs':[{'program_id':'p','desired_delta':'increase','desired_effect':'inclusion','exon_label':2}],
            'steric_design_jobs':[{'job_id':'j','program_id':'p'}]}
    raw = [native_row(), native_row('CGTA',101,'C*G*T*A',2), native_row('GTAC',102,'G*T*A*C',3)]
    pool = [{**raw[0],'job_id':'j','program_id':'p','route_label':'acceptor@110',
             'candidate_assessment':{'selection_eligible':True}}]
    save(root,'artifacts/intake/structured_goal.json',goal)
    save(root,'artifacts/candidates/candidate_assessments.json',{'rows':pool})
    save(root,'artifacts/design/steric_jobs/j/scored_tiles.json',{'rows':raw})
    save(root,'artifacts/candidates/recommended_shortlist.json',{'records':[{
        'antisense_5to3':'ACGT','molecule_identity':['ACGT',json.dumps('A*C*G*T')],'recommendation_order':1}]})
    return goal, raw, pool


def test_full_pool_panel_identity_and_predictions_survive_rendering(tmp_path):
    run_fixture(tmp_path)
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    data = project_run(tmp_path)
    assert data['kind']=='design' and len(data['records'])==3
    assert sum(r['panel_rank'] is not None for r in data['records'])==1
    assert all(abs(r['effect']-.2)<1e-9 for r in data['records'])
    assert data['records'][0]['intervals']==[{'chrom':'chr1','start':101,'end':104}]
    document = basic_document('Test', '# Results\nA recorded finding.', candidate_data=data)
    packed = json.loads(Packet(document).text)
    assert packed['encoding']=='columns-v1' and len(packed['records'])==3
    assert 'A recorded finding.' in document
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_jobs_chemistry_and_opposite_objectives_stay_separate(tmp_path):
    goal, raw, pool = run_fixture(tmp_path)
    goal['design_programs'].append({'program_id':'q','desired_delta':'decrease'})
    goal['steric_design_jobs'].append({'job_id':'k','program_id':'q'})
    second = {**raw[0],'IDT':'ACGT','job_id':'k','program_id':'q','route_label':'acceptor@110'}
    save(tmp_path,'artifacts/intake/structured_goal.json',goal)
    save(tmp_path,'artifacts/candidates/candidate_assessments.json',{'rows':pool+[second]})
    save(tmp_path,'artifacts/design/steric_jobs/k/scored_tiles.json',{'rows':[second]})
    data = project_run(tmp_path)
    assert len(data['contexts'])==2
    a,b = [r for r in data['records'] if r['aso']=='ACGT']
    assert a['molecule']!=b['molecule'] and a['effect']==-b['effect']
    assert a['panel_rank']==1 and b['panel_rank'] is None


def test_missing_second_prediction_and_unknown_screen_are_kept(tmp_path):
    goal, raw, pool = run_fixture(tmp_path)
    raw[-1]['splice_site_deltas'][0]['delta_acceptor_replacement']=None
    raw[-1]['off_target_search_status']='incomplete'; raw[-1]['Off_target_1mm_count']=None
    save(tmp_path,'artifacts/design/steric_jobs/j/scored_tiles.json',{'rows':raw})
    row=project_run(tmp_path)['records'][-1]
    assert row['effect'] is None and row['predictions']==[.1,None]
    assert not row['off_target_complete'] and row['off_targets'][1] is None


def test_junction_intervals_and_undeclared_coordinate_space():
    assert coordinates({'chrom':'chr1','start':100,'end':110})==[]
    assert coordinates({'binding_intervals_bed0_half_open':[['chr1',10,12],['chr1',20,22]]})==[
        {'chrom':'chr1','start':11,'end':12},{'chrom':'chr1','start':21,'end':22}]


def test_moe_c_token_encodes_methyl_c_and_real_contradictions_are_visible():
    from aso_molecule import parse_ordering_notation
    notation='/52MOErA/*/i2MOErC/*/i2MOErG/*/32MOErT/'
    parsed=parse_ordering_notation(notation)
    row={'antisense_5to3':'ACGT','ordering_notation':notation,'molecule_specification':{
        'platform':'steric_blocker','nucleotide_sugars_5to3':parsed['sugars_5to3'],
        'base_modifications_5to3':parsed['base_modifications_5to3'],'linkages_5to3':parsed['linkages_5to3']}}
    assert chemistry(row)['issues']==[]
    row['molecule_specification']['linkages_5to3']=['PO']*3
    result=chemistry(row)
    assert result['notation']==notation
    assert result['issues']==['The ordering notation differs from the recorded backbone chemistry.']


def test_packing_is_lossless_compact_and_safe_in_script_context():
    data=project_records([{'antisense_5to3':'ACGT','ordering_notation':'A*C*G*T','name':'</script><img onerror=alert(1)>'}]*100)
    packed=pack_data(data)
    restored=[]
    for values in packed['records']:
        row={c['name']:(c['values'][value] if 'values' in c else value) for c,value in zip(packed['record_columns'],values)}
        row['chemistry']={**row['chemistry'],'notation':row.pop('notation')}; restored.append(row)
    assert restored==data['records']
    assert len(json.dumps(packed))<len(json.dumps(data))*.5
    document=explorer_html(data)
    assert '</script><img' not in document
    assert json.loads(Packet(document).text)==packed


def test_exported_rank_provenance_and_delta_keep_recorded_scope():
    row={**native_row(), 'selection_rank':7, 'route_label':'acceptor@110','job_id':'j','program_id':'p',
         'recommendation_rank_key':[4,2,0,.25,4,.2],
         'selected_effect_evidence':[{'label':'acceptor@110','prediction':{'direction':'increase','values':[.1,.3],'directional_mean':.2}}]}
    view=export_context(row)
    assert (view['start'],view['end'],view['strand'])==(101,104,'+')
    assert view['native_rank']==7 and view['native_rank_basis']=='selection_rank'
    assert view['primary_objective_delta_deletion']==.1
    assert view['recommendation_rank_components']['site_directional_mean']==.2
    assert named_rank_components({'recommendation_rank_key':[1,2]})=={'uninterpreted_saved_values':[1,2]}


def test_strategy_rank_never_becomes_a_candidate_rank_and_scopes_do_not_mix():
    rows=[{'job_id':'j','route_label':'acceptor@110','route_rank':1,'recommendation_rank_key':[4,.1,4,.1]},
          {'job_id':'j','route_label':'acceptor@110','route_rank':1,'recommendation_rank_key':[4,.5,4,.5]},
          {'job_id':'j','route_label':'donor@120','route_rank':2,'recommendation_rank_key':[4,.9,4,.9]},
          {'job_id':'j','route_rank':1}]
    before=deepcopy(rows); ranked=with_native_ranks(rows)
    assert [r.get('native_rank') for r in ranked]==[2,1,1,None]
    assert export_context(ranked[0])['native_rank_basis']=='recommendation_key_descending_within_route'
    assert export_context(rows[-1])['native_rank'] is None
    assert rows==before


def test_filter_receipts_copy_execution_and_never_claim_unexecuted_defaults():
    assert applied_filter_thresholds({'parameter_constraints':{'gc_min':.3}})['status']=='unavailable'
    bounds={'gc_min_percent':30,'gc_max_percent':65,'gc_min_exclusive':False,'gc_max_exclusive':True}
    goal={'gapmer_tiling_results_by_job':{'j':{'initial_filtering':{'constraint_summary':bounds}}}}
    before=deepcopy(goal); result=applied_filter_thresholds(goal)
    assert result['stages'][0]['thresholds']==bounds and goal==before


def test_raw_genomic_strand_joins_assessed_gapmer_once(tmp_path):
    goal, raw, pool=run_fixture(tmp_path)
    row=raw[0]; row['Genomic_strand']=row.pop('Strand')
    save(tmp_path,'artifacts/design/steric_jobs/j/scored_tiles.json',{'rows':[row]})
    data=project_run(tmp_path)
    assert len(data['records'])==1 and data['records'][0]['strand']=='+'


def test_safeguard_reason_uses_recorded_review_and_preserves_a_zero_bound():
    site={'donor@120':{'site_kind':'donor','position':120,'verified_exon_label':2}}
    guard={'label':'donor@120','passes':False,'max_increase':0,'values':[.08,.03],
           'review':{'status':'reviewed','rationale':'This junction belongs to a protected isoform.'}}
    result=safeguard_summary({'safeguard_report':[guard]},site)
    assert result['summary']=='Exon 2 donor rose 0.08; limit 0.'
    assert result['details'][0]['reason']=='This junction belongs to a protected isoform.'
    guard.update(policy='observe',passes=None)
    guard['review']['rationale']='The paired junction supports the requested exon outcome.'
    result=safeguard_summary({'safeguard_report':[guard]},site)
    assert result['status']=='observed' and result['details'][0]['reason'].startswith('The paired junction')


def test_unknown_and_unsearched_mismatch_bins_are_never_completed_zeros():
    data=project_records([{'antisense_5to3':'ACGT','ordering_notation':'ACGT',
        'Off_target_mismatch_budget':0,'Off_target_exact_count':0,'Off_target_1mm_count':0,
        'off_target_search_status':'complete','off_target_mapping_capped':True}])
    row=data['records'][0]
    assert row['off_targets']==[0,None,None] and not row['off_target_complete']


def test_variant_fact_uses_the_matching_chromosome_and_alternate_allele(tmp_path):
    goal,_,_=run_fixture(tmp_path)
    vcf=tmp_path/'patient.vcf'
    vcf.write_text('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPATIENT\n'
                   'chr2\t100\t.\tA\tG\t.\t.\t.\tGT\t0/1\n'
                   'chr1\t100\t.\tA\tC,G\t.\t.\t.\tGT\t2|2\n')
    goal.update(patient_variant={'notation':'c.10A>G','chr':'chr1','pos':100,'ref':'A','alt':'G'},
                patient_reference={'vcf_path':str(vcf)})
    save(tmp_path,'artifacts/intake/structured_goal.json',goal)
    fact=next(f for f in project_run(tmp_path)['facts'] if f['label']=='Patient variant')
    assert fact['value']=='c.10A>G' and fact['note']=='both copies · chr1:100'


def test_executed_gc_receipt_retains_precision_and_open_bounds(tmp_path):
    import run_goal_intake as runtime
    path=save(tmp_path,'tiles.json',{'rows':[{'Sequence':'ACGT','GC':.5,'No_homopolymer':True}]})
    low,high=.333333333,.700000001
    summary=runtime._apply_initial_tiling_filters(str(path),{'gc_content':{
        'min':low,'max':high,'min_inclusive':False,'max_inclusive':True}})
    bounds=summary['constraint_summary']['gc_fraction']
    # Execution normalizes constraints to six decimal places. Record those
    # actual bounds, preserving more precision than the legacy percent labels.
    assert abs(bounds['min']-.333333)<1e-12 and abs(bounds['max']-.7)<1e-12
    assert bounds['min_inclusive'] is False and bounds['max_inclusive'] is True
