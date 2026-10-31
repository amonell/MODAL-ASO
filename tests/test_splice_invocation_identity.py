from copy import deepcopy
import json

import pytest

from aso_agents.splice_invocation_identity import InvocationIdentity
from test_structural_event_binding import conversion_fixture


def test_every_scientific_input_invalidates_within_invocation_reuse(tmp_path):
    goal, _ = conversion_fixture()
    paths = [tmp_path / name for name in ('tiles', 'reference', 'weights')]
    for path in paths:
        path.write_text(path.name)
    identity = InvocationIdentity()
    inputs = dict(tiles=paths[0], reference=paths[1], assets=[paths[2]],
                  goal=goal, transcript='TXIN.4', sites={'donor': 2030}, backend={'backend': 'fixture'})
    first = identity.key(**inputs)
    assert first and identity.key(**inputs) == first
    for path in paths:
        before = path.read_text()
        path.write_text(before + 'changed')
        assert identity.key(**inputs) != first
        path.write_text(before)
    for key, replacement in [('sites', {'donor': 2031}), ('backend', {'backend': 'other'})]:
        assert identity.key(**{**inputs, key: replacement}) != first
    altered = deepcopy(goal)
    altered['isoform_transcripts']['included']['exons_tx'][2]['end'] += 1
    assert identity.key(**{**inputs, 'goal': altered}) != first
    paths[2].unlink()
    assert identity.key(**inputs) is None


@pytest.mark.parametrize('strand', ['+', '-'])
def test_scoring_uses_current_run_reference_without_another_structure_lookup(tmp_path, monkeypatch, strand):
    import importlib
    scoring = importlib.import_module('aso_agents.splicing_agent')
    goal, _ = conversion_fixture(strand)
    goal['exon_label'] = 3
    tiles = tmp_path / 'tiles.json'
    tiles.write_text(json.dumps({'rows': [{'Sequence': 'ACGT', 'Chrom': 'chr8', 'Start': 2000, 'End': 2004}]}))
    monkeypatch.setattr(scoring, 'resolve_splice_eval_backend', lambda *_: {'backend': 'openspliceai', 'species': 'mouse', 'model': 'fixture'})
    def unexpected_lookup(*a, **k):
        pytest.fail('The attested reference was resolved again instead of being used')
    monkeypatch.setattr(scoring, 'get_transcript_structure', unexpected_lookup)
    monkeypatch.setattr(scoring, '_evaluate_with_openspliceai', lambda **kwargs: kwargs)
    result = scoring.evaluate_splice_shifts_on_tiles(str(tiles), 'TXIN.4', json.dumps(goal), ref_fasta_path=str(tmp_path / 'reference'))
    assert result['fixed_acc'] == (2000 if strand == '+' else 2030)
    assert result['fixed_don'] == (2030 if strand == '+' else 2000)
