"""Whole-record placement must preserve file flanks and strand geometry."""
from random import Random
from types import SimpleNamespace

import pytest
from pyfaidx import Fasta
from aso_agents import snapgene_ingestion as ingestion


def reference(tmp_path, sequence):
    path = tmp_path/'reference.fa'
    path.write_text('>chr2\n'+sequence+'\n')
    return Fasta(str(path), sequence_always_upper=True)


def rc(seq):
    return seq.translate(str.maketrans('ACGT','TGCA'))[::-1]


@pytest.mark.parametrize('strand,feature_strand', [('+',1), ('-',1), ('+',-1), ('-',-1)])
def test_file_local_exons_anchor_long_flanks_on_both_strands(tmp_path, strand, feature_strand):
    rng = Random(9)
    sequence = ''.join(rng.choice('ACGT') for _ in range(2400))
    query = list(sequence);query[1300] = 'C' if sequence[1300] != 'C' else 'T';query=''.join(query)
    start = 101
    file = query if strand == '+' else rc(query)
    fasta = reference(tmp_path, 'G'*100 + sequence + 'T'*100)
    local_start, local_end = 1700, 1780
    genomic_start = start + (local_start if strand == '+' else len(file)-local_end)
    exon_sign = (1 if strand == '+' else -1) * feature_strand
    record = {'chrom':'chr2','genomic_start':genomic_start,'genomic_end':genomic_start+79,
              'local_start':local_start,'local_end':local_end,'feature_strand':feature_strand,
              'strand':'+' if exon_sign == 1 else '-'}
    found = ingestion._fallback_full_sequence_alignment(file,[record],'chr2',strand,fasta)
    assert (found.start,found.end,found.strand) == (start,start+len(file)-1,strand)
    variants = ingestion._variants_from_aligned_sequence(found,file,fasta)
    assert variants == [{'chrom':'chr2','pos':start+1300,'ref':sequence[1300],'alt':query[1300]}]


def test_unsupported_divergence_and_tied_placements_remain_unresolved(tmp_path):
    sequence = 'ACGTTGCA'*10
    fasta = reference(tmp_path,sequence+'TTTT'+sequence)
    records = [{'chrom':'chr2','genomic_start':start,'genomic_end':start+19,
                'local_start':0,'local_end':20,'feature_strand':1,'strand':'+'}
               for start in (1,len(sequence)+5)]
    assert ingestion._fallback_full_sequence_alignment(sequence,records,'chr2','+',fasta) is None
    assert ingestion._fallback_full_sequence_alignment('G'*80,records[:1],'chr2','+',fasta) is None
    assert ingestion._fallback_full_sequence_alignment(sequence,[{'genomic_start':1}],'chr2','+',fasta) is None


def test_bowtie_whole_record_uses_supported_mismatch_argument(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(ingestion,'_write_sequence_fasta',lambda *args,**kwargs:tmp_path/'query.fa')
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='')
    monkeypatch.setattr(ingestion.subprocess,'run',run)
    assert ingestion._align_full_sequence('ACGT',tmp_path/'idx',10,None) is None
    assert calls[0][calls[0].index('-v')+1] == '3'


def test_whole_ungapped_variant_call_does_not_use_quadratic_sequence_matcher(tmp_path,monkeypatch):
    sequence='ACGT'*2000
    fasta=reference(tmp_path,sequence)
    query=sequence[:5555]+'A'+sequence[5556:]
    monkeypatch.setattr(ingestion.difflib,'SequenceMatcher',lambda *a,**k:pytest.fail('Ungapped whole-record substitutions require no sequence realignment'))
    alignment=ingestion.SnapExonAlignment('chr2',1,len(sequence),'+',255,str(len(sequence))+'M')
    assert ingestion._variants_from_aligned_sequence(alignment,query,fasta) == [{'chrom':'chr2','pos':5556,'ref':'T','alt':'A'}]


def test_exon_mnv_and_whole_record_snps_cannot_duplicate_reference_bases():
    variants=[{'chrom':'chr2','pos':2,'ref':'C','alt':'T'},
              {'chrom':'2','pos':2,'ref':'CG','alt':'TA'},
              {'chrom':'chr2','pos':3,'ref':'G','alt':'A'}]
    merged=ingestion.merge_substitution_variants(variants)
    assert len(merged)==2
    assert ingestion._apply_variants_to_sequence('ACGT',merged)=='ATAT'


def test_conflicting_file_backgrounds_are_not_silently_combined():
    variants=[{'chrom':'chr2','pos':2,'ref':'C','alt':'T'},
              {'chrom':'chr2','pos':2,'ref':'C','alt':'G'}]
    with pytest.raises(ValueError,match='variant_conflict'):
        ingestion.merge_substitution_variants(variants)
