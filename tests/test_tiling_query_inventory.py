"""Search success must not decide which supplied-reference tiles exist."""
import gzip
import importlib.util
from pathlib import Path
import subprocess
import sys

from Bio import SeqIO
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / 'aso_agents/resources/tiling/filter_mapped.py'


def module():
    spec = importlib.util.spec_from_file_location('mapping_inventory_controls', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_fasta(path, records):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt') as handle:
        for name, sequence in records:
            handle.write(f'>{name}\n{sequence}\n')


def run_converter(tmp_path, queries, exact=(), near=(), unaligned=(), near_unaligned=(), overflow=(), near_overflow=()):
    paths = {name: tmp_path / name for name in ['queries.fasta.gz','exact.sam','near.sam',
        'unaligned.fasta','near_unaligned.fasta','overflow.fasta','near_overflow.fasta','features.txt','result.fasta']}
    write_fasta(paths['queries.fasta.gz'], queries)
    for name, rows in [('exact.sam',exact),('near.sam',near)]:
        paths[name].write_text(''.join('\t'.join(map(str,[q,flag,chrom,pos,255,f'{len(seq)}M','*',0,0,seq,'*',f'NM:i:{mm}']))+'\n'
                                     for q,seq,flag,chrom,pos,mm in rows))
    for name, records in [('unaligned.fasta',unaligned),('near_unaligned.fasta',near_unaligned),
                          ('overflow.fasta',overflow),('near_overflow.fasta',near_overflow)]:
        write_fasta(paths[name],records)
    paths['features.txt'].write_text(''.join(f'{q}\tUnassigned\t0\tNone\n' for q,_ in queries))
    result = subprocess.run([sys.executable,str(SCRIPT),'-s',str(paths['exact.sam']),'-f',str(paths['features.txt']),
        '-o',str(paths['result.fasta']),'--query-fasta',str(paths['queries.fasta.gz']),
        '--near-sam',str(paths['near.sam']),'--exact-unaligned',str(paths['unaligned.fasta']),
        '--near-unaligned',str(paths['near_unaligned.fasta']),'--overflow',str(paths['overflow.fasta']),
        '--near-overflow',str(paths['near_overflow.fasta']),'--same-reference','--max-mismatches','2'],
        text=True,capture_output=True)
    records = list(SeqIO.parse(paths['result.fasta'],'fasta')) if paths['result.fasta'].exists() else []
    return result, records


@pytest.mark.parametrize('strand,flag,aligned_sequence', [('+',0,'AACCGGTA'),('-',16,'TACCGGTT')])
def test_reference_specific_query_survives_without_exact_placement(tmp_path,strand,flag,aligned_sequence):
    q = f'id=neutral|pos=chr8:101-108:{strand}'
    result, records = run_converter(tmp_path,[(q,'AACCGGTA')],
        near=[(q,aligned_sequence,flag,'chr8',101,1),(q,aligned_sequence,flag,'chr9',201,2)],
        unaligned=[(q,'AACCGGTA')])
    assert result.returncode == 0, result.stderr
    assert len(records) == 1 and str(records[0].seq) == 'AACCGGTA'
    header = records[0].description
    assert 'mappednum=0|mappedto=None' in header
    assert 'off_target_exact_count=0|' in header
    assert 'off_target_1mm_count=0|' in header  # Intended coordinate is excluded.
    assert 'off_target_2mm_count=1|' in header
    assert 'off_target_search_status=complete|' in header


def test_absent_or_censored_evidence_never_becomes_zero(tmp_path):
    queries=[('missing','AACCGGTA'),('censored','AACCAGTA')]
    result,records=run_converter(tmp_path,queries,overflow=[queries[1]],near_overflow=[queries[1]])
    assert result.returncode == 0,result.stderr
    assert len(records)==2
    assert 'mappednum=NA|' in records[0].description
    assert 'mappednum=>2000|' in records[1].description
    assert all('off_target_exact_count=NA|' in r.description and 'off_target_search_status=incomplete_or_censored|' in r.description for r in records)


def test_explicit_no_hits_are_zero_only_within_searched_bins(tmp_path):
    queries=[('no_hits','AACCGGTA')]
    result,records=run_converter(tmp_path,queries,unaligned=queries,near_unaligned=queries)
    assert result.returncode==0,result.stderr
    header=records[0].description
    assert all(f'off_target_{label}_count=0|' in header for label in ['exact','1mm','2mm'])
    assert 'off_target_3mm_count=NA|' in header
    assert 'off_target_search_status=complete|' in header


def test_query_sequence_disagreement_is_an_error(tmp_path):
    result,_=run_converter(tmp_path,[('q','AACCGGTA')],unaligned=[('q','AACCGGTT')])
    assert result.returncode!=0 and 'does not match the supplied query inventory' in result.stderr


def test_contradictory_mapping_and_no_hit_evidence_is_an_error(tmp_path):
    query=('q','AACCGGTA')
    result,_=run_converter(tmp_path,[query],exact=[(*query,0,'chr8',101,0)],unaligned=[query])
    assert result.returncode!=0 and 'Conflicting aligned' in result.stderr


def test_duplicate_identical_queries_collapse_but_distinct_records_remain(tmp_path):
    queries=[('one','AACCGGTA'),('one','AACCGGTA'),('two','AACCGGTA')]
    result,records=run_converter(tmp_path,queries,unaligned=queries,near_unaligned=queries)
    assert result.returncode==0,result.stderr
    assert len(records)==2


def test_conflicting_duplicate_query_identity_is_rejected(tmp_path):
    result,_=run_converter(tmp_path,[('one','AACCGGTA'),('one','AACCGGTT')])
    assert result.returncode!=0 and 'Conflicting sequences for query identity' in result.stderr


def test_empty_query_inventory_is_valid(tmp_path):
    result,records=run_converter(tmp_path,[])
    assert result.returncode==0,result.stderr
    assert not records


def test_exact_no_hit_proof_does_not_fill_censored_near_bins():
    m=module();key=('q','AACCGGTA')
    values=m.reconcile_mapping_metadata(pd.DataFrame(),pd.DataFrame(),[key],2,
        near_available=True,near_overflow={'q'},exact_unaligned=[key])[key]
    assert values['off_target_exact_count']=='0'
    assert values['off_target_1mm_count']==values['off_target_2mm_count']=='NA'
    assert values['off_target_search_status']=='incomplete_or_censored'
    other=m.reconcile_mapping_metadata(pd.DataFrame(),pd.DataFrame(),[key],2,
        near_available=False,exact_unaligned=[key],same_reference=False)[key]
    assert other['off_target_exact_count']=='NA'
