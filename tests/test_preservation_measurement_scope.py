from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import run_goal_intake as runtime
from aso_agents import pipeline_tools as p
from aso_agents.preservation import include_preserved_exon_boundaries


def fixture(strand='+'):
    def entry(name, altered):
        spans = [(101, 130), (201, 250 if altered else 230), (301, 330)]
        if strand == '-': spans.reverse()
        return {'transcript_id': name, 'chrom': '8', 'strand': strand,
            'exons_tx': [{'index': i+1, 'start': a, 'end': b} for i, (a,b) in enumerate(spans)]}
    changed, normal = entry('TX_CHANGED', True), entry('TX_NORMAL', False)
    plan = [{'site': kind, 'position': pos, 'chrom': 'chr8', 'strand': strand,
             'present_in': ['changed'], 'exon_indices': {'changed': 2}, 'delta_direction': 'decrease'}
            for kind,pos in ([('acceptor',201),('donor',250)] if strand=='+' else [('acceptor',250),('donor',201)])]
    return {'gene_symbol':'NEUTRAL','isoform_transcripts':{'changed':changed,'normal':normal},
        'splice_site_plan':plan,'preservation_targets':[{'action':'preserve', 'preservation_target_type':'isoform',
            'isoform_label':'normal', 'transcript_id':'TX_NORMAL'}]}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_overlapping_protected_exon_adds_its_alternative_boundary_without_retiling_or_direction(strand):
    goal=fixture(strand);before=deepcopy(goal)
    # An unrelated transcript ordinal must not control physical feature matching.
    for exon in goal['isoform_transcripts']['normal']['exons_tx']:
        exon['index'] *= 10
    before=deepcopy(goal)
    sites=runtime._build_splice_sites_for_eval(goal,None,10)
    extra=[s for s in sites if s.get('monitoring_purpose')=='preservation']
    assert len(extra)==1
    assert extra[0]['canonical_label']==('donor@230' if strand=='+' else 'acceptor@230')
    assert extra[0]['delta_direction']=='context'
    assert extra[0]['exon_index']==20
    assert goal==before and goal['splice_site_plan']==before['splice_site_plan']
    assert include_preserved_exon_boundaries(goal,sites,window_bp=10)==sites


@pytest.mark.parametrize('change',['none','different_chromosome','different_strand','nonoverlap','other_gene'])
def test_no_unrequested_or_unrelated_measurement_sites(change):
    goal=fixture()
    if change=='none':goal['preservation_targets']=[]
    elif change=='different_chromosome':goal['isoform_transcripts']['normal']['chrom']='9'
    elif change=='different_strand':goal['isoform_transcripts']['normal']['strand']='-'
    elif change=='other_gene':goal['preservation_targets'][0]['gene_symbol']='UNRELATED'
    else:
        for e in goal['isoform_transcripts']['normal']['exons_tx']:e['start']+=1000;e['end']+=1000
    sites=runtime._build_splice_sites_for_eval(goal,None,10)
    assert not any(s.get('monitoring_purpose')=='preservation' for s in sites)


def test_missing_alternative_boundary_is_a_required_guard_not_a_new_route(tmp_path,monkeypatch):
    goal=fixture();entry={'label':'donor@250','site':'donor','position':250,'decision_direction':'decrease'}
    (tmp_path/'scores.json').write_text(json.dumps({'rows':[{
        'delta_names':'donor@250','delta_deletion':'-.8','delta_replacement':'-.8'}]}))
    monkeypatch.setattr(p,'_routing_target_entries',lambda *a,**k:[entry])
    monkeypatch.setattr(p,'_rank_routing_entries_with_llm',lambda **k:{'ranked_entries':[entry]})
    result=p._build_atomic_routing_plan(structured_goal=goal,scored_path=str(tmp_path/'scores.json'),run_dir=str(tmp_path),flags={})
    steps=result['planner_plan']['steps'];assert len(steps)==1 and steps[0]['args']['primary_label']=='donor@250'
    guards=json.loads(steps[0]['args']['safeguards_json'])
    assert {g['label'] for g in guards if g.get('required')}=={'acceptor@201','donor@230'}


def test_unmeasured_legacy_zero_cannot_pass_a_required_guard(tmp_path,monkeypatch):
    from test_splice_recommendation_evidence import site,goal
    pair=site((0,0),canonical_label='acceptor@201',site_kind='acceptor',
        evaluation_status='outside_evaluated_window',delta_acceptor_deletion=0,delta_acceptor_replacement=0)
    raw={'results':[{'splice_site_deltas':[pair],'delta_inclusion_deletion':None,'delta_inclusion_replacement':None}]}
    monkeypatch.setattr(p,'evaluate_splice_shifts_on_tiles',lambda **k:deepcopy(raw))
    monkeypatch.setattr(p,'_annotate_repeat_risk_json',lambda **k:{})
    path=tmp_path/'tiles.json';path.write_text(json.dumps({'rows':[{'Sequence':'AACGT','Chrom':'chr8','Start':180,'End':185,'Strand':'+'}]}))
    result=p._run_splice_eval_on_tiles_json(initial_tiles_path=str(path),structured_goal=goal(),transcript_id='TX_NEUTRAL.2',patient_reference_fasta=None,splice_sites_json=None,runtime=SimpleNamespace(filter_tiles_json=runtime.filter_tiles_json))
    assert 'error' not in result
    row=json.loads((tmp_path/'scored_tiles.json').read_text())['rows'][0]
    assert row['delta_deletion']=='NA' and row['delta_replacement']=='NA'


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('missing', ['membership', 'exon_indices', 'both'])
def test_minimal_plan_recovers_preservation_measurements_from_physical_junction(strand, missing):
    goal = fixture(strand)
    for entry in goal['splice_site_plan']:
        if missing in {'membership', 'both'}:
            entry.pop('present_in')
        if missing in {'exon_indices', 'both'}:
            entry.pop('exon_indices')
    before = deepcopy(goal)
    sites = runtime._build_splice_sites_for_eval(goal, None, 10)
    extra = [site for site in sites if site.get('monitoring_purpose') == 'preservation']
    assert len(extra) == 1
    assert extra[0]['canonical_label'] == ('donor@230' if strand == '+' else 'acceptor@230')
    assert goal == before
    _, required = runtime._routing_preservation_context(goal, goal['splice_site_plan'])
    assert required == ({'acceptor@201', 'donor@230'} if strand == '+' else {'donor@201', 'acceptor@230'})


@pytest.mark.parametrize('mismatch', ['chromosome', 'strand', 'position', 'terminal', 'ambiguous'])
def test_minimal_plan_does_not_invent_reference_membership(mismatch):
    goal = fixture()
    site = {'site': 'donor', 'position': 250, 'chrom': 'chr8', 'strand': '+'}
    if mismatch == 'chromosome':
        site['chrom'] = 'chr9'
    elif mismatch == 'strand':
        site['strand'] = '-'
    elif mismatch == 'position':
        site['position'] = 249
    elif mismatch == 'terminal':
        site['position'] = 330
    else:
        changed = goal['isoform_transcripts']['changed']['exons_tx']
        changed.insert(2, {**changed[1], 'index': 3})
        changed[-1]['index'] = 4
    goal['splice_site_plan'] = [site]
    before = deepcopy(goal)
    assert runtime._build_splice_sites_for_eval(goal, None, 10) == []
    assert goal == before
