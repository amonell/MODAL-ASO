"""Exercise the real Snakefile/CLI boundary, then native table/filter consumers."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from aso_agents import tiling_agent
import run_goal_intake as runtime


@pytest.mark.parametrize(('motifs', 'failed_names'), [
    (['AAAAA', 'CCCCC', 'GGGGG', 'TTTTT'], {'A5', 'C5', 'G5', 'T5'}),
    (['GGGG'], {'C4', 'C5'}),
    ([], set()),
    (['AAA', 'CCCCCC'], {'T4', 'T5'}),
])
def test_actual_score_rule_preserves_motif_list_and_filter_scope(tmp_path, monkeypatch, motifs, failed_names):
    source = Path(__file__).parents[1] / 'aso_agents/resources/tiling'
    (tmp_path / 'filter_seqs.py').symlink_to(source / 'filter_seqs.py')
    output = tmp_path / 'output'
    (output / 'pipeline_outputs').mkdir(parents=True)
    sequences = {}
    for base, suffix in [('A', 'CGT'), ('C', 'AGT'), ('G', 'ACT'), ('T', 'ACG')]:
        for run in (4, 5):
            sequences[base + str(run)] = base * run + (suffix * 6)[:20-run]
    sequences['mixed'] = 'ACGT' * 5
    fasta = output / 'pipeline_outputs/neutral.folded.fasta'
    fasta.write_text(''.join(
        f'>id={name}|pos=chr8:{101+i*30}-{120+i*30}:+|gc=0.5|pc=0.5|ph=0.5|nostruct=YES|mappednum=1|secondary_hit=None\n{seq}\n'
        for i, (name, seq) in enumerate(sequences.items())
    ))
    configuration = {
        'INPUT_FOLDER': str(tmp_path), 'REGIONS': ['neutral.bed'],
        'OUTPUT_FOLDER': str(output), 'TILE_LEN': 20, 'CHEMISTRY': 'steric-hindrance',
        'GENOME': 'unused.fa', 'PHASTCONS': 'unused.bw', 'PHYLOP': 'unused.bw',
        'BOWTIE_INDEX': 'unused', 'SECONDARY_INDEX': 'unused',
        'SECONDARY_ANNOTATION': 'unused.gtf', 'CHROM_SIZES': 'unused.sizes',
        'UCSC_SPECIES': 'neutral', 'CONSERVATION_MODE': 'disabled',
        'PHASTCONS_THRESHOLD': 0, 'PHYLOP_THRESHOLD': 0, 'NUM_MISMATCHES': 0,
        'GC_MIN': 0, 'GC_MAX': 1, 'AVOID': motifs,
    }
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(configuration))
    target = output / 'neutral.scored.idt.tsv'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(tmp_path / 'mpl'), XDG_CACHE_HOME=str(tmp_path / 'cache'),
               XDG_CONFIG_HOME=str(tmp_path / 'config'), TMPDIR=str(tmp_path),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    command = [sys.executable, '-m', 'snakemake', '--snakefile', str(source / 'Snakefile'),
               '--directory', str(tmp_path), '--configfile', str(config), '--cores', '1',
               '--scheduler', 'greedy', '--nolock', '--allowed-rules', 'score', '--', str(target)]
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    (tmp_path / 'snakemake.stdout').write_text(completed.stdout)
    (tmp_path / 'snakemake.stderr').write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    with target.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    by_sequence = {row['Sequence']: row for row in rows}
    assert set(by_sequence) == set(sequences.values())
    for name, seq in sequences.items():
        row = by_sequence[seq]
        assert row['No_homopolymer'] == str(name not in failed_names)
        assert json.loads(row['Homopolymer_excluded_motifs']) == motifs
        assert row['Homopolymer_sequence_role'] == 'synthesized_antisense_sequence'
        assert row['Homopolymer_screen_status'] == ('not_requested' if not motifs else 'failed' if name in failed_names else 'passed')
        assert row['Score'] == ('800' if name in failed_names else '900')

    pdf_text = subprocess.check_output(['pdftotext', str(output / 'neutral.upset.pdf'), '-'], text=True)
    assert 'Raw candidate scoring' in pdf_text
    assert 'Later filtering is recorded separately.' in pdf_text.replace('\n', ' ')
    if not motifs:
        assert 'screen not performed' in pdf_text.replace('\n', ' ')
        assert 'no_homopolymer' not in pdf_text

    monkeypatch.setattr(tiling_agent, 'get_species_config', lambda *_: SimpleNamespace(gtf_path=tmp_path/'missing.gtf'))
    collected = tiling_agent.collect_results_json(output_dir=str(output))
    collected = json.loads(collected) if isinstance(collected, str) else collected
    payload = json.loads(Path(collected['json_path']).read_text())
    assert len(payload['rows']) == len(rows)
    assert all('Homopolymer_excluded_motifs' in row for row in payload['rows'])
    # A later, explicitly different threshold must not rewrite raw annotations.
    constraints = {'gc_min': 0, 'gc_max': 1, 'homopolymer_max_run': 4}
    summary = runtime._apply_initial_tiling_filters(collected['json_path'], constraints)
    rejected = json.loads(Path(summary['filtered_no_homopolymer']['path']).read_text())['rows']
    assert {row['Sequence'] for row in rejected} == {sequences[base+'5'] for base in 'ACGT'}
    assert all(json.loads(row['Homopolymer_excluded_motifs']) == motifs for row in rejected)
