import gzip
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
from aso_agents.coordinate_conversion import _chain_alignments, map_interval, convert_rows
from aso_agents import pipeline_tools
import run_goal_intake as runtime


def chain(tmp_path, text):
    path = tmp_path/'test.chain.gz'
    path.write_bytes(gzip.compress(text.encode()))
    return _chain_alignments(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('orientation,expected', [('+',[210,220,'+']), ('-',[780,790,'-'])])
def test_complete_chain_conversion_keeps_bed_and_strand(tmp_path, orientation, expected):
    data = chain(tmp_path, f'chain 900 chr2 1000 + 100 200 chr3 1000 {orientation} 200 300 7\n100\n\n')
    row, evidence = map_interval(data, ['chr2',110,120,'site',0,'+'])
    assert row == ['chr3',*expected[:2],'site',0,expected[2]]
    assert evidence['chain_id'] == '7' and evidence['status'] == 'converted'


@pytest.mark.parametrize('start,end', [(90,110),(150,160),(120,180),(205,220)])
def test_partial_gap_spanning_and_unmapped_intervals_never_become_boxes(tmp_path, start, end):
    data = chain(tmp_path, 'chain 900 chr2 1000 + 100 200 chr3 1000 + 200 300 7\n50 10 10\n40\n\n')
    row, evidence = map_interval(data, ['chr2',start,end,'site',0,'+'])
    assert row is None and evidence['status'] == 'ambiguous_or_incomplete_mapping'


def test_multiple_chains_do_not_choose_highest_score(tmp_path):
    text = 'chain {score} chr2 1000 + 100 200 chr3 1000 + {a} {b} {id}\n100\n\n'
    data = chain(tmp_path, text.format(score=900,a=200,b=300,id=7)+text.format(score=100,a=400,b=500,id=8))
    assert map_interval(data,['chr2',110,120,'site',0,'+'])[0] is None


def test_conversion_requires_permission_and_preserves_independent_native_work(tmp_path):
    original = {'intervals_bed0_half_open':[{'genome_build':'hg19'},{'genome_build':'hg38'}]}
    rows = [['chr2',110,120,'old',0,'+'],['chr2',210,220,'new',0,'+']]
    before = deepcopy(original)
    converted, evidence = convert_rows(rows, original, {'species':'human'}, tmp_path)
    assert converted == rows[1:] and original == before
    assert evidence['coordinate_conversions'][0]['original_build'] == 'GRCh37'
    assert evidence['coordinate_conversions'][0]['status'] == 'conversion_requires_confirmation'


def test_materialization_does_not_reuse_source_rows_when_mapping_is_unresolved():
    constraint = {'source_type':'explicit_rows','coordinate_space':'genomic',
                  'rows':[['chr2',110,120,'old',0,'+']], 'materialized_rows':[]}
    rows = runtime._constraint_rows_in_output_space(structured_goal={}, constraint=constraint,
        output_space='genomic',base_rows=[],isoform_transcripts={},iso_label_primary=None)
    assert rows == []


def test_outside_gene_interval_is_evidenced_and_does_not_block_usable_site(tmp_path):
    goal = {'gene_symbol':'TEST','species':'human','isoform_transcripts':{
        'TX':{'chrom':'chr2','transcript_id':'TX.3','exons_tx':[{'start':200,'end':300}]}}}
    rows = [['chr2',110,120,'outside',0,'+'],['chr2',210,220,'inside',0,'+']]
    result, evidence = convert_rows(rows, {'genome_build':'hg38'}, goal, tmp_path)
    assert result == rows[1:]
    first = evidence['coordinate_conversions'][0]
    assert first['design_status'] == 'outside_resolved_gene_reference'
    assert first['mapped_row_bed0_half_open'] == rows[0]


def test_admission_requires_typed_authorization_and_does_not_ignore_other_conflicts():
    goal = {'species':'human','region_constraints':[{'source_type':'explicit_rows','coordinate_space':'genomic',
        'genome_build':'hg19','coordinate_convention':'BED0','allow_coordinate_conversion':True,
        'intervals_bed0_half_open':[{'chrom':'chr2','start':110,'end':120,
                                   'label':'Original hg19 chr2:110-120'}]}]}
    envelope = {'request_text':'Use hg19 chr2:110-120 on installed hg38.'}
    before = deepcopy(goal)
    validate = lambda: pipeline_tools._mixed_genome_build_validation_entry(envelope, {}, {'structured_goal':goal})
    assert validate() is None
    assert goal == before
    goal['patient_variant'] = {'genome_build':'hg19'}
    assert validate()['issue'] == 'mixed_genome_build_coordinates'
    goal.pop('patient_variant')
    goal['region_constraints'][0]['allow_coordinate_conversion'] = False
    assert validate()['issue'] == 'mixed_genome_build_coordinates'


@pytest.mark.parametrize('strand', ['+', '-'])
def test_only_last_transcript_exon_is_terminal_donor(strand):
    entry = {'strand':strand,'exons_tx':[{'index':i} for i in (1,2,3)]}
    assert not runtime._is_terminal_donor(entry,1)
    assert runtime._is_terminal_donor(entry,3)
