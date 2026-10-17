import json

import pytest

from aso_agents.analysis_helpers import shared_analysis_helpers, validate_shared_helper_bindings
from aso_agents.spliced_coordinates import SplicedSequenceCoordinates


EXONS = [(31, 35), (8, 10), (20, 23)]
BASE_COORDINATES = [8, 9, 10, 20, 21, 22, 23, 31, 32, 33, 34, 35]
GENOME = dict(zip(BASE_COORDINATES, 'TGGCTAAACGTA'))


def fetch(chrom, start, end):
    assert chrom == 'contig_test'
    return ''.join(GENOME[p] for p in range(start, end + 1))


@pytest.mark.parametrize('strand,sequence', [('+', 'TGGCTAAACGTA'), ('-', 'TACGTTTAGCCA')])
def test_every_interval_against_explicit_base_map_and_reference_sequence(strand, sequence):
    frame = SplicedSequenceCoordinates(chromosome='contig_test', strand=strand, exons=EXONS)
    expected = BASE_COORDINATES if strand == '+' else BASE_COORDINATES[::-1]
    assert frame.length == len(expected) == len(sequence)
    for index, coordinate in enumerate(expected, 1):
        assert frame.to_genomic(index) == coordinate
        assert frame.to_mature(coordinate) == index
    for start in range(1, len(sequence) + 1):
        for end in range(start, len(sequence) + 1):
            blocks = frame.project_interval(start, end)
            observed = []
            for block in blocks:
                assert block['chrom'] == 'contig_test' and block['strand'] == strand
                lo, hi = block['genomic_start1'], block['genomic_end1']
                assert lo <= hi
                assert hi - lo == block['mature_end1'] - block['mature_start1']
                bases = list(range(lo, hi + 1))
                observed.extend(bases if strand == '+' else bases[::-1])
            assert observed == expected[start - 1:end]
            assert frame.extract(start, end, fetch_genomic=fetch) == sequence[start - 1:end]


def test_source_order_does_not_override_strand_and_input_is_not_mutated():
    exons = [[31, 35], [8, 10], [20, 23]]
    frame = SplicedSequenceCoordinates(chromosome='ref', strand='-', exons=exons)
    assert frame.exons == ((31, 35), (20, 23), (8, 10))
    exons[0][0] = 999
    assert frame.exons[0] == (31, 35)
    assert frame.project_interval(5, 6) == [
        {'chrom': 'ref', 'strand': '-', 'genomic_start1': 31, 'genomic_end1': 31,
         'mature_start1': 5, 'mature_end1': 5, 'exon_ordinal': 1},
        {'chrom': 'ref', 'strand': '-', 'genomic_start1': 23, 'genomic_end1': 23,
         'mature_start1': 6, 'mature_end1': 6, 'exon_ordinal': 2},
    ]


@pytest.mark.parametrize('exons', [[], [(0, 2)], [(4, 3)], [(2, 4), (4, 7)],
                                 [(2, 4), (2, 4)], [(1, True)], [('1', 2)], [(1, 2, 3)]])
def test_invalid_exons_are_not_silently_reinterpreted(exons):
    with pytest.raises((TypeError, ValueError)):
        SplicedSequenceCoordinates(chromosome='ref', strand='+', exons=exons)


@pytest.mark.parametrize('chromosome,strand', [('', '+'), (None, '+'), ('ref', '.'), ('ref', -1)])
def test_missing_reference_frame_is_not_inferred(chromosome, strand):
    with pytest.raises(ValueError):
        SplicedSequenceCoordinates(chromosome=chromosome, strand=strand, exons=[(5, 9)])


def test_out_of_range_intron_and_noninteger_queries_are_rejected():
    frame = SplicedSequenceCoordinates(chromosome='ref', strand='+', exons=EXONS)
    for start, end in [(0, 1), (1, 13), (5, 4), (True, 2), (1, 2.0)]:
        with pytest.raises((TypeError, ValueError)):
            frame.project_interval(start, end)
    for position in [7, 11, 19, 24, 36, True, '8']:
        with pytest.raises((TypeError, ValueError)):
            frame.to_mature(position)


def test_genome_fetch_bounds_alphabet_and_orientation_are_explicit():
    frame = SplicedSequenceCoordinates(chromosome='ref', strand='-', exons=[(5, 19)])
    assert frame.extract(1, 15, fetch_genomic=lambda *args: 'acgtrymkbdhvswn') == 'NWSBDHVMKRYACGT'
    for bad in ['', 'A' * 14, 'A' * 16, b'A' * 15, 'A' * 14 + 'U', 'A' * 14 + '-']:
        with pytest.raises(ValueError):
            frame.extract(1, 15, fetch_genomic=lambda *args: bad)


def test_native_import_binding_and_generated_program_projection(tmp_path):
    from aso_agents.analysis_execution import execute_analysis_program
    assert any(h['name'] == 'SplicedSequenceCoordinates' for h in shared_analysis_helpers())
    with pytest.raises(ValueError, match='shared analysis helper'):
        validate_shared_helper_bindings('class SplicedSequenceCoordinates: pass')

    def generate(payload):
        contract = payload['execution_contract']
        return {'python_code': (
            'from aso_agents.spliced_coordinates import SplicedSequenceCoordinates\n'
            'from pathlib import Path\nimport json\n'
            "frame = SplicedSequenceCoordinates(chromosome='ref', strand='-', exons=[(10, 13), (40, 44)])\n"
            'blocks = frame.project_interval(4, 7)\n'
            f"Path({contract['result_json']!r}).write_text(json.dumps({{'blocks': blocks}}))\n"
            f"Path({contract['summary_markdown']!r}).write_text('Projected four mature bases across two exons.')\n"
        )}

    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
                                      required_outputs=[], max_attempts=1)
    assert result['ok'], result
    assert [(b['genomic_start1'], b['genomic_end1']) for b in result['result']['blocks']] == [(40, 41), (12, 13)]
