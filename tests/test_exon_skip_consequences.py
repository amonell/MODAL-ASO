import pytest
from aso_agents.exon_skip_consequences import simulate_exon_skips


def fixture(strand, middle_length):
    sequences = ['ATG', 'C', 'TAA'+'A'*(middle_length-3), 'TAA']
    starts = [101,501,1001,2001] if strand == '+' else [2001,1001,501,101]
    genome = list('N'*2500); exons=[]
    for i,(start,sequence) in enumerate(zip(starts,sequences),1):
        source = sequence if strand == '+' else sequence.translate(str.maketrans('ACGT','TGCA'))[::-1]
        end=start+len(sequence)-1;genome[start-1:end]=source
        exons.append({'index':i, 'start':start, 'end':end, 'contains_coding':True,
                      'coding_start_genomic':start, 'coding_end_genomic':end})
    return {'chrom':'8', 'strand':strand, 'exons_tx':exons}, {'chr8':''.join(genome)}


@pytest.mark.parametrize('strand', ['+', '-'])
@pytest.mark.parametrize('middle_length,expected,distance', [(65,True,62),(56,False,53)])
def test_skipped_transcript_junction_distance_and_strand(strand,middle_length,expected,distance):
    transcript,genome=fixture(strand,middle_length)
    results=simulate_exon_skips(transcript,genome)
    result=next(v for v in results.values() if v['skipped_exon_index']==2)
    assert result['status']=='computed' and result['premature_stop'] is True
    assert result['removed_coding_length']==1 and result['coding_frame_shift']
    assert result['stop_pos_nt']==3
    assert result['stop_end_to_last_junction_nt']==distance
    assert result['nmd_predicted'] is expected
    start=next(v for v in results.values() if v['skipped_exon_index']==1)
    assert start['status']=='translation_start_removed' and start['nmd_predicted'] is None


def test_incomplete_or_discordant_reference_is_not_an_nmd_negative():
    transcript,genome=fixture('+',65)
    genome['chr8']=genome['chr8'][:100]+'CCC'+genome['chr8'][103:]
    results=simulate_exon_skips(transcript,genome)
    assert all(v['status']=='unverified_reference_cds' and v['nmd_predicted'] is None for v in results.values())
    transcript['exons_tx'][0]['coding_start_genomic']=1
    with pytest.raises(ValueError,match='outside'):
        simulate_exon_skips(transcript,genome)
