import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pyBigWig
import pytest

from aso_agents import tiling_agent
import run_goal_intake as runtime


def script(name):
    path = Path(__file__).parents[1] / 'aso-designer-main/aso-designer-main' / (name + '.py')
    spec = importlib.util.spec_from_file_location('neutral_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_bigwig_distinguishes_zero_negative_partial_and_absent_coverage(tmp_path):
    path = tmp_path / 'neutral.bw'
    writer = pyBigWig.open(str(path), 'w')
    writer.addHeader([('chr8', 40)])
    writer.addEntries(['chr8'] * 3, [0, 10, 20], ends=[10, 20, 30], values=[0.0, -2.0, 0.6])
    writer.close()
    module = script('tile_seqs')
    source = module.Conservation(str(path))
    try:
        assert source.summary('chr8', 0, 10, '+') == (0.0, 10, 10)
        assert source.summary('8', 10, 20, '-') == (-2.0, 10, 10)
        average, covered, total = source.summary('chr8', 20, 40, '+')
        assert average == pytest.approx(0.6) and (covered, total) == (10, 20)
        average, covered, total = source.summary('chr8', 30, 40, '+')
        assert math.isnan(average) and (covered, total) == (0, 10)
        assert math.isnan(source.avg_value('chr9', 0, 10, '+'))
        disabled = module.NeutralConservation()
        assert math.isnan(disabled.avg_value('chr8', 0, 10, '+'))
        assert disabled.summary('chr8', 0, 10, '+')[1:] == (0, 10)
    finally:
        source.phastcon.close()


def test_missing_conservation_survives_filter_and_json_collection_as_null(tmp_path, monkeypatch):
    module = script('filter_seqs')
    fasta = tmp_path / 'input.fasta'
    fasta.write_text('>id=neutral|pos=chr8:101-120:+|gc=0.5|pc=nan|ph=-2|pc_covered_bases=0|ph_covered_bases=20|conservation_interval_bases=20|nostruct=YES|mappednum=1|secondary_hit=None\nACGTACGTACGTACGTACGT\n')
    bed = tmp_path / 'neutral.scored.bed'
    tsv = tmp_path / 'neutral.scored.idt.tsv'
    flags = module.filter(str(fasta), str(bed), str(tsv), [], 0.0, 1.0, 0.0, -3.0, 'steric-hindrance')
    row_flags = flags['ACGTACGTACGTACGTACGT']
    assert row_flags['phastcons_measured'] is False and row_flags['pass_phastcon'] is False
    assert row_flags['phylop_measured'] is True and row_flags['pass_phylop'] is True
    monkeypatch.setattr(tiling_agent, 'get_species_config', lambda *args: SimpleNamespace(gtf_path=tmp_path/'absent.gtf'))
    result = tiling_agent.collect_results_json(output_dir=str(tmp_path))
    result = json.loads(result) if isinstance(result, str) else result
    payload = json.loads(Path(result['json_path']).read_text())
    row = payload['rows'][0]
    assert row['Phastcon'] is None and row['PhyloP'] == -2.0
    assert (row['pc_covered_bases'], row['ph_covered_bases'], row['conservation_interval_bases']) == (0, 20, 20)
    json.dumps(payload, allow_nan=False)


def test_conservation_tie_break_preserves_measured_zero_and_does_not_reward_missing():
    base = {'isoform_uniqueness_class': 'gene_wide', 'isoform_uniqueness_score': 1.0,
            'GC': 0.5, 'Phastcon': 0.0}
    rows = [{**base, 'PhyloP': value} for value in [None, 0.0, -2.0, float('nan')]]
    order = sorted(range(len(rows)), key=lambda index: runtime._gapmer_uniqueness_sort_key(rows[index]))
    assert order == [1, 2, 0, 3]
    zero_alias = {**base, 'PhyloP': 0.0, 'phylop': 10.0}
    assert runtime._gapmer_uniqueness_sort_key(zero_alias) == runtime._gapmer_uniqueness_sort_key(rows[1])
