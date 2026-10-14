"""Neutral controls for reference geometry, chemistry and evidence contracts."""
from copy import deepcopy
from types import SimpleNamespace
import importlib

import pytest

import run_goal_intake as cli
from aso_agents import pipeline_tools as pipeline
from aso_agents.mapping_evidence import cdna_near_matches, requested_feature_membership
from aso_agents.region_overlap import candidate_matches_regions
from aso_agents.reporting_contracts import exact_gene_count


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('other_coding', [False, True])
def test_shared_junction_identity_includes_exons_excluded_by_coding_policy(strand, other_coding):
    exons = [{'index':1, 'start':1, 'end':40},
             {'index':2, 'start':101, 'end':140, 'contains_coding':True},
             {'index':3, 'start':201, 'end':240}]
    site = {'exon_index':2, 'chrom':'chr2', 'strand':strand,
            'acceptor':101 if strand=='+' else 140, 'donor':140 if strand=='+' else 101,
            'coding':True, 'acceptor_plan_ok':True, 'donor_plan_ok':True}
    target = {'transcript_id':'TX_A', 'chrom':'chr2', 'strand':strand, 'exons_tx':exons, 'sites':[site]}
    other = deepcopy(target)
    other['transcript_id'] = 'TX_B'
    other['exons_tx'][1]['contains_coding'] = other_coding
    other['sites'] = [{**site, 'acceptor_plan_ok':False, 'donor_plan_ok':False}] if other_coding else []
    transcripts = {'TX_A':target, 'TX_B':other}
    before = deepcopy(transcripts)
    plan, _ = cli._compute_splice_site_plan(
        [{'isoform_label':'TX_A', 'desired_abundance':'decrease'},
         {'isoform_label':'TX_B', 'desired_abundance':'increase'}], transcripts)
    assert plan
    assert all(s['present_in']==['TX_A','TX_B'] and not s['absent_in'] for s in plan)
    assert all(s['uniqueness']=='shared' for s in plan)
    assert transcripts == before
    # A different genomic boundary must remain distinct.
    other['exons_tx'][1].update(start=111, end=150)
    other['sites'] = []
    separate, _ = cli._compute_splice_site_plan([], transcripts)
    assert all(s['present_in']==['TX_A'] for s in separate)


@pytest.mark.parametrize('source', ['gapmer_cdna', 'junction_gapmer'])
def test_native_cdna_tiles_keep_inclusive_first_base(source):
    row = {'Chrom':'chr3','Start':101,'End':120,'source':source, 'coordinate_system':'1based_inclusive'}
    assert ('chr3',100,120) in pipeline._tile_row_intervals(row)
    assert candidate_matches_regions(pipeline._tile_row_intervals(row), [['chr3',100,120]], 'full_containment')
    assert not candidate_matches_regions(pipeline._tile_row_intervals(row), [['chr3',101,120]], 'full_containment')
    assert ('chr3',100,101) in pipeline._tile_row_intervals({**row,'End':101})


def test_full_containment_does_not_accept_a_junction_with_one_outside_block():
    blocks = [('chr3',100,110),('3',100,110),('chr3',200,210),('3',200,210)]
    assert not candidate_matches_regions(blocks, [['chr3',100,120]], 'full_containment')
    assert candidate_matches_regions(blocks, [['chr3',100,120]], 'any_overlap')
    assert candidate_matches_regions(blocks, [['chr3',100,210]], 'full_containment')


@pytest.mark.parametrize('row,expected', [
    ({},None), ({'Off_target_0mm_loci':0},None),
    ({'Off_target_0mm_genes':[]},0), ({'Off_target_0mm_genes':['A','A','B']},2),
    ({'Off_target_genes_by_mismatch':{'0mm':[]}},0),
    ({'off_target_count_status':'incomplete','Off_target_0mm_genes':[]},None),
    ({'Off_target_exact_count':3,'off_target_exact_count_unit':'genes'},3),
])
def test_gene_count_does_not_conflate_missing_data_or_loci(row, expected):
    assert exact_gene_count(row) == expected


@pytest.mark.parametrize('name,expected', [('', 'eCLIP'), ('ENCODE RBP annotations','eCLIP'),
    ('RNA binding protein peaks','eCLIP'), ('iCLIP','iCLIP'), ('PAR-CLIP','PAR-CLIP')])
def test_database_description_uses_default_assay_but_named_assays_are_preserved(name,expected):
    assert cli._normalize_rbp_assay(name) == expected


def test_encode_empty_search_is_not_a_missing_resource_error(monkeypatch):
    module = importlib.import_module('scripts.encode_rbp_peaks')
    class Response:
        status_code = 404
        def json(self): return {'@graph':[], 'total':0}
        def raise_for_status(self): raise RuntimeError('404')
    monkeypatch.setattr(module.requests,'get',lambda *a,**k:Response())
    assert module._encode_get_json('/search/')['total']==0
    with pytest.raises(RuntimeError,match='404'):
        module._encode_get_json('/experiments/MISSING/')


def test_explicit_version_replaces_live_alias_and_unavailable_version_stays_unavailable():
    old = {'transcript_id':'ENST00000000001','metadata':{'version':10}, 'exons_tx':[]}
    goal = {'transcript_lock':{'value':'ENST00000000001.9'},'isoform_transcripts':{'TX':old},
            'all_transcripts_full':{'TX':deepcopy(old)}, 'evaluation_targets':None}
    def resolve(tid):
        assert tid=='ENST00000000001.9'
        return {'id':'ENST00000000001','version':9,'versioned_id':tid,'seq_region_name':'2',
                'strand':1,'exons_tx':[{'index':1,'start':101,'end':120}],'reference_source':'local_gtf'}
    runtime = SimpleNamespace(get_transcript_structure=resolve)
    isoforms, all_tx, notes = pipeline._ensure_eval_transcript_catalog(goal,runtime)
    assert not notes and isoforms['TX']['metadata']['version']==9
    assert pipeline._choose_eval_transcript(goal,isoforms,all_tx,runtime)[2]['metadata']['version']==9
    runtime.get_transcript_structure = lambda _: {'error':'requested version unavailable'}
    isoforms, all_tx, notes = pipeline._ensure_eval_transcript_catalog(goal,runtime)
    assert notes and not isoforms and not all_tx and not goal['isoform_transcripts']
    assert pipeline._choose_eval_transcript(goal,isoforms,all_tx,runtime)[2] is None


@pytest.mark.parametrize('strand',[1,-1])
def test_near_junction_mapping_preserves_mismatch_positions_and_exon_membership(strand):
    transcript = {'transcript_id':'TX', 'chrom':'chr4','strand':strand,
        'exons_tx':[{'index':1,'start':101 if strand==1 else 201,'end':104 if strand==1 else 204},
                    {'index':2,'start':201 if strand==1 else 101,'end':204 if strand==1 else 104}]}
    entry = {'sequence':'TCGTAACC', 'reverse_complement_sequence':'GGTTACGA'}
    # One mismatch relative to a mature RNA sequence spanning both exons.
    matches = cdna_near_matches(entry,{'TX':transcript},{'TX':'GGTTACGT'},
        cli._map_cdna_range_to_genomic_segments,'TX')
    match = next(m for m in matches if m['match_orientation']=='reverse_complement')
    assert match['mismatch_count']==1
    assert match['mismatch_positions_in_supplied_sequence_1based']==[1]
    assert len(match['genomic_segments'])==2
    features = requested_feature_membership({'near_match_candidates':[match]},
        [{'exon_index':2,'transcript_id':'TX'}],transcript,'TX')
    evidence = features[0]['observations'][0]
    assert evidence['overlap_bp']==4 and evidence['overlaps'] and not evidence['fully_contained']
    assert evidence['mismatch_count']==1


def test_unstated_interval_coordinate_convention_is_not_assumed():
    tx = {'transcript_id':'TX','chrom':'chr1','exons_tx':[]}
    target = {'transcript_id':'TX','chrom':'chr1','start':100,'end':120}
    assert requested_feature_membership({},[target],tx,'TX')==[]
    assert requested_feature_membership({},[{**target,'coordinate_system':'bed'}],tx,'TX')[0]['start']==101
