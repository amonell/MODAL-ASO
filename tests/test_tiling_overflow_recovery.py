"""Overflow recovery must count every locus without expanding report previews."""
import importlib.util
import io
import json
import gzip
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1] / 'aso_agents/resources/tiling'


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sam(name, seq, pos, mm=0, flag=0, chrom='chr1'):
    return '\t'.join(map(str, [name, flag, chrom, pos, 0, f'{len(seq)}M', '*', 0, 0, seq, '*', f'NM:i:{mm}'])) + '\n'


def test_distinct_loci_reverse_strand_and_genes_beyond_preview(tmp_path):
    recovery = module('recover_overflow')
    db = sqlite3.connect(tmp_path / 'hits.db')
    lines = (sam('q0', 'AACCGGTA', 101) * 2 + sam('q0', 'TACCGGTT', 201, 1, 16)
             + sam('q0', 'AACCGGTA', 301, 1) + sam('q0', 'AACCGGTA', 401, 2))
    recovery.aggregate_sam(io.StringIO(lines), db, {'q0': 'AACCGGTA'}, 2)
    annotation = tmp_path / 'genes.gtf'
    annotation.write_text('chr1\ttest\tgene\t300\t320\t.\t-\t.\tgene_id "g1"; gene_name "BeyondPreview";\n')
    query = 'id=one|pos=chr1:101-108:+'
    other = 'id=two|pos=chr1:201-208:-'
    rows = recovery.summarize(db, {query: 'AACCGGTA', other: 'AACCGGTA'}, {'AACCGGTA': 'q0'}, 2,
                              recovery.gene_index(annotation), 1)
    first, second = [row['metadata'] for row in rows]
    assert [first[f'off_target_{label}_count'] for label in ('exact', '1mm', '2mm', '3mm')] == ['0','2','1','NA']
    assert first['off_target_1mm_positions'] == 'chr1:201-208'
    assert first['off_target_1mm_genes'] == 'BeyondPreview'
    assert first['off_target_positions_truncated'] == 'true'
    assert second['off_target_exact_count'] == second['off_target_1mm_count'] == '1'
    converter = module('filter_mapped')
    metadata = converter.reconcile_mapping_metadata(pd.DataFrame(), pd.DataFrame(), [(query, 'AACCGGTA')], 2,
        near_available=True, near_overflow=[query], near_recovered={(query, 'AACCGGTA'): rows[0]})
    assert metadata[(query, 'AACCGGTA')]['off_target_search_status'] == 'complete'


@pytest.mark.parametrize('lines', ['', sam('wrong', 'AACCGGTA', 1), sam('q0', 'AACCGGTT', 1),
                                   sam('q0', 'AACCGGTA', 1, 3)])
def test_missing_mismatched_or_out_of_scope_records_are_rejected(tmp_path, lines):
    recovery = module('recover_overflow')
    with sqlite3.connect(tmp_path / 'hits.db') as db, pytest.raises(ValueError):
        recovery.aggregate_sam(io.StringIO(lines), db, {'q0': 'AACCGGTA'}, 2)


def test_failed_recovery_keeps_censored_counts_unknown(tmp_path):
    recovery, converter = module('recover_overflow'), module('filter_mapped')
    overflow = tmp_path / 'overflow.fasta'; overflow.write_text('>q\nAACCGGTA\n')
    annotation = tmp_path / 'genes.gtf'; annotation.write_text('')
    output = tmp_path / 'recovery.json'
    receipt = recovery.recover(overflow, 'missing_index', 2, annotation, output, bowtie='/missing/bowtie')
    assert receipt['status'] == 'failed' and receipt['rows'] == []
    assert converter.read_recovery(output, overflow, {'q': 'AACCGGTA'}, 2, 'missing_index') == {}
    values = converter.reconcile_mapping_metadata(pd.DataFrame(), pd.DataFrame(), [('q', 'AACCGGTA')], 2,
        near_available=True, near_overflow=['q'])[('q','AACCGGTA')]
    assert values['off_target_2mm_count'] == 'NA'
    assert values['off_target_search_status'] == 'incomplete_or_censored'
    assert not list(tmp_path.glob('.overflow-*'))


def test_aligner_failure_after_valid_records_discards_partial_results(tmp_path):
    recovery = module('recover_overflow')
    overflow = tmp_path / 'overflow.fasta'; overflow.write_text('>q\nAACCGGTA\n')
    annotation = tmp_path / 'genes.gtf'; annotation.write_text('')
    executable = tmp_path / 'interrupted-bowtie'
    executable.write_text(f'#!{sys.executable}\nimport sys\nsys.stdout.write({sam("q0", "AACCGGTA", 101)!r})\nsys.exit(1)\n')
    executable.chmod(0o755)
    receipt = recovery.recover(overflow, 'index', 2, annotation, tmp_path / 'recovery.json', bowtie=str(executable))
    assert receipt['returncode'] == 1
    assert receipt['status'] == 'failed' and receipt['rows'] == []


def test_full_gene_annotations_survive_short_previews_and_repeated_sequences(tmp_path, monkeypatch):
    from aso_agents import tiling_agent, pipeline_tools
    monkeypatch.setattr(tiling_agent, 'get_species_config', lambda *_: SimpleNamespace(gtf_path=tmp_path / 'absent.gtf'))
    sequence = 'AACCGGTA'
    bed, table = [], ['Chrom\tStart\tEnd\tStrand\tSequence\tOff_target_exact_count\tOff_target_1mm_count\tOff_target_2mm_count\tOff_target_mismatch_budget']
    for start, gene in [(100, 'FirstGene'), (200, 'SecondGene')]:
        metadata = (f'seq={sequence}|off_target_1mm_genes={gene}|off_target_1mm_gene_ids=id_{gene}'
                    '|off_target_search_status=complete|off_target_positions_truncated=true'
                    '|off_target_position_preview_limit=100')
        bed.append(f'chr1\t{start}\t{start+8}\t{metadata}\t1000\t+')
        table.append(f'chr1\t{start}\t{start+8}\t+\t{sequence}\t0\t2001\t0\t2')
    (tmp_path / 'query.scored.bed').write_text('\n'.join(bed) + '\n')
    (tmp_path / 'query.scored.idt.tsv').write_text('\n'.join(table) + '\n')
    result = tiling_agent.collect_results_json(str(tmp_path), species='human')
    rows = json.loads(Path(result['json_path']).read_text())['rows']
    assert [row['Off_target_1mm_genes'] for row in rows] == ['FirstGene', 'SecondGene']
    assert all(row['off_target_positions_truncated'] is True for row in rows)
    assert all(row['Off_target_1mm_count'] == 2001 for row in rows)
    assert pipeline_tools._normalized_generated_candidate_row(rows[0])['off_target_position_preview_limit'] == 100


@pytest.mark.skipif(not shutil.which('bowtie') or not shutil.which('bowtie-build'), reason='Native Bowtie required')
def test_real_aligner_recovers_over_2000_hits_and_matches_exhaustive_search(tmp_path):
    recovery, converter = module('recover_overflow'), module('filter_mapped')
    sequence = 'ACGTCAGTACGA'
    reference = (sequence + 'NNNN') * 2105 + 'ACGTCAGTACGTNNNNACGTCAGTATGT'
    fasta = tmp_path / 'reference.fasta'; fasta.write_text('>chr1\n' + reference + '\n')
    index = tmp_path / 'index'
    subprocess.run(['bowtie-build', str(fasta), str(index)], capture_output=True, check=True)
    query = 'id=repeat|pos=chr1:1-12:+'
    queries = tmp_path / 'queries.fasta'; queries.write_text(f'>{query}\n{sequence}\n')
    overflow = tmp_path / 'overflow.fasta'
    capped = subprocess.run(['bowtie', '-a', '-f', '-v', '2', '-y', '-m', '2000', '--sam', '--sam-nohead',
                             '--max', str(overflow), str(index), str(queries)], capture_output=True, text=True)
    assert capped.returncode == 0 and overflow.read_text()
    annotation = tmp_path / 'genes.gtf'; annotation.write_text('chr1\ttest\tgene\t33000\t33708\t.\t+\t.\tgene_id "g"; gene_name "Tail";\n')
    output = tmp_path / 'recovered.json'
    receipt = recovery.recover(overflow, index, 2, annotation, output, preview_limit=3)
    assert receipt['status'] == 'complete', receipt
    row = receipt['rows'][0]
    expected = [set() for _ in range(3)]
    reverse = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    for offset in range(len(reference) - len(sequence) + 1):
        window = reference[offset:offset + len(sequence)]
        if 'N' in window or offset == 0:
            continue
        for query_sequence in (sequence, reverse):
            mismatches = sum(a != b for a,b in zip(window, query_sequence))
            if mismatches <= 2:
                expected[mismatches].add(offset + 1)
    for mm, label in enumerate(('exact','1mm','2mm')):
        assert int(row['metadata'][f'off_target_{label}_count']) == len(expected[mm])
    assert row['exact_mapped_count'] == 2105
    assert row['metadata']['off_target_0mm_genes'] == 'Tail'
    assert len(row['metadata']['off_target_0mm_positions'].split(',')) == 3
    assert len(output.read_bytes()) < 5000
    recovered = converter.read_recovery(output, overflow, {query: sequence}, 2, str(index))
    assert (query, sequence) in recovered
    assert converter.read_recovery(output, overflow, {query: sequence}, 1, str(index)) == {}
    assert converter.read_recovery(output, overflow, {query: sequence}, 2, 'other_index') == {}
    overflow.write_text(overflow.read_text() + '\n')
    assert converter.read_recovery(output, overflow, {query: sequence}, 2, str(index)) == {}


@pytest.mark.skipif(any(not shutil.which(tool) for tool in ('bowtie', 'bowtie-build', 'samtools', 'featureCounts'))
                    or importlib.util.find_spec('snakemake') is None, reason='Native tiling runtime required')
def test_native_workflow_delivers_recovered_counts(tmp_path):
    from Bio import SeqIO
    sequence = 'ACGTCAGTACGA'
    reference = tmp_path / 'reference.fasta'
    reference.write_text('>chr1\n' + (sequence + 'NNNN') * 2105 + '\n')
    index = tmp_path / 'index'
    subprocess.run(['bowtie-build', str(reference), str(index)], capture_output=True, check=True)
    annotation = tmp_path / 'genes.gtf'
    annotation.write_text('chr1\ttest\tgene\t1\t33680\t.\t+\t.\tgene_id "g"; gene_name "Repeat";\n'
                         'chr1\ttest\texon\t1\t33680\t.\t+\t.\tgene_id "g"; gene_name "Repeat"; transcript_id "t";\n')
    output = tmp_path / 'tiles'; pipeline = output / 'pipeline_outputs'; pipeline.mkdir(parents=True)
    with gzip.open(pipeline / 'repeat.fasta.gz', 'wt') as handle:
        handle.write(f'>id=repeat|pos=chr1:1-12:+\n{sequence}\n')
    config = {'OUTPUT_FOLDER': str(output), 'INPUT_FOLDER': str(tmp_path), 'REGIONS': ['repeat.bed'],
              'GENOME': str(reference), 'BOWTIE_INDEX': str(index), 'SECONDARY_INDEX': str(index),
              'SECONDARY_ANNOTATION': str(annotation), 'TILE_LEN': 12, 'NUM_MISMATCHES': 2,
              'PHASTCONS': '', 'PHYLOP': '', 'CHROM_SIZES': '', 'UCSC_SPECIES': 'test',
              'CONSERVATION_MODE': 'disabled', 'PHASTCONS_THRESHOLD': 0, 'PHYLOP_THRESHOLD': 0,
              'GC_MIN': 0, 'GC_MAX': 1, 'AVOID': []}
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    target = pipeline / 'repeat.mapped.fasta'
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']}
    run = subprocess.run([sys.executable, '-m', 'snakemake', str(target), '-s', str(ROOT / 'Snakefile'),
        '--cores', '1', '--scheduler', 'greedy', '--configfile', str(config_path), '--allowed-rules',
        'map', 'map_secondary', 'recover_exact_overflow', 'recover_near_overflow', 'annotate_secondary',
        'convert_to_seq'], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    records = list(SeqIO.parse(target, 'fasta'))
    assert len(records) == 1
    header = records[0].description
    assert 'mappednum=2105|' in header
    assert 'off_target_exact_count=2104|' in header
    assert 'off_target_search_status=complete|' in header
    assert 'off_target_positions_truncated=true|' in header
    assert 'off_target_0mm_genes=Repeat|' in header
    assert 'overflow=TRUE' not in header
