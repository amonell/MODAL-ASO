"""Exercise the actual exporter from a clean run and invalidate changed evidence."""
import json
from pathlib import Path

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.artifact_io import write_json
from aso_chemistry import encode_idt_oligo


@pytest.fixture
def run(tmp_path, monkeypatch):
    goal = {'species': 'human', 'candidate_limit': 2,
            'design_programs': [{'program_id': 'neutral', 'modality': 'steric', 'length': 20}]}
    context = {'structured_goal': goal, 'final_payload': {'structured_goal': goal}}
    monkeypatch.setattr(pipeline, '_goal_payload_context', lambda *_: context)
    path = pipeline._write_run_path(tmp_path, 'tiling', 'steric_scored_uniqueness.json')
    rows = []
    for index in range(4):
        seq = 'ACGT' * 4 + 'ACG' + 'ACGT'[index]
        rows.append({'Synthesis': seq, 'Sequence': seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1],
            'IDT': encode_idt_oligo(seq, 'steric-hindrance'), 'Chrom': 'chr8',
            'Start': 100 + index * 30, 'End': 120 + index * 30, 'Strand': '+',
            'program_id': 'neutral', 'source_modality': 'steric', 'selection_rank': index + 1})
    write_json(path, {'rows': rows})
    original, builds = pipeline._build_v2_generated_design_artifacts, []
    def build(root, **kwargs):
        builds.append(root)
        return original(root, **kwargs)
    monkeypatch.setattr(pipeline, '_build_v2_generated_design_artifacts', build)
    return tmp_path, context, path, rows, builds


def test_unchanged_export_reuses_verified_outputs_after_publication(run):
    root, context, path, rows, builds = run
    first = pipeline._emit_v2_generated_design_artifacts(str(root))
    before = {key: Path(value).read_bytes() for key, value in first.items() if isinstance(value, str)}
    pipeline._publish_generated_artifacts(context['structured_goal'], context['final_payload'], first)
    assert pipeline._emit_v2_generated_design_artifacts(str(root)) == first
    assert len(builds) == 1
    assert all(Path(first[key]).read_bytes() == content for key, content in before.items())


@pytest.mark.parametrize('change', ['count', 'candidate', 'reference', 'output_corruption', 'missing_output'])
def test_changed_inputs_or_outputs_force_real_rebuild(run, change):
    root, context, path, rows, builds = run
    artifacts = pipeline._emit_v2_generated_design_artifacts(str(root))
    if change == 'count':
        context['structured_goal']['candidate_limit'] = 3
    elif change == 'candidate':
        rows[0]['selection_rank'] = 100
        write_json(path, {'rows': rows})
    elif change == 'reference':
        context['structured_goal']['isoform_transcripts'] = {'ref': {'transcript_id': 'TX.3'}}
    elif change == 'output_corruption':
        Path(artifacts['generated_aso_set_json']).write_text('{}')
    else:
        Path(artifacts['recommended_shortlist_csv']).unlink()
    pipeline._emit_v2_generated_design_artifacts(str(root))
    assert len(builds) == 2
    panel = json.loads(Path(artifacts['generated_aso_set_json']).read_text())
    assert panel['row_count'] == (3 if change == 'count' else 2)
    assert json.loads(Path(artifacts['candidate_assessments_json']).read_text())['pool_row_count'] == 4


def test_source_edit_and_newly_available_input_change_identity(tmp_path):
    from aso_agents.report_build import build_identity
    source = tmp_path / 'source'
    source.mkdir()
    code = source / 'algorithm.py'
    code.write_text('version = 1')
    present, missing = tmp_path / 'present', tmp_path / 'missing'
    present.write_text('input')
    first = build_identity({}, [present, missing], source)
    code.write_text('version = 2')
    second = build_identity({}, [present, missing], source)
    assert first != second
    missing.write_text('new evidence')
    assert build_identity({}, [present, missing], source) != second


def test_reviewed_membership_reaches_all_native_exports_and_reopens_offline(run, monkeypatch):
    from aso_agents import shortlist_review
    from aso_agents.candidate_presentation import molecule_key
    from test_shortlist_review import model
    root, context, source, source_rows, _ = run
    before = source.read_bytes()
    monkeypatch.setattr(shortlist_review, '_model_call', model)
    paths = pipeline._emit_v2_generated_design_artifacts(str(root), review_recommendations=True)
    read = lambda key: json.loads(Path(paths[key]).read_text())
    generated = read('generated_aso_set_json')
    assert generated['recommendation_review']['status'] == 'completed'
    assessed = read('candidate_assessments_json')['rows']
    expected = {molecule_key(row) for row in assessed[-2:]}
    assert {molecule_key(row) for row in generated['rows']} == expected
    assert all(row['recommendation_review']['why_chosen'] for row in generated['rows'])
    shortlist = read('recommended_shortlist_json')
    assert shortlist['review']['status'] == 'completed'
    assert len(shortlist['records']) == 2
    chemistry = read('chemical_identity_json')
    assert {row['antisense_5to3'] for row in chemistry['records']} == {key[0] for key in expected}
    import csv
    with Path(paths['design_panel_csv']).open() as handle:
        panel = list(csv.DictReader(handle))
    sequences = [row['synthesized_antisense_sequence'] for row in generated['rows']]
    assert [row['antisense_5to3'] for row in panel] == sequences
    assert [row['antisense_5to3'] for row in shortlist['records']] == sequences
    assert [row['antisense_5to3'] for row in chemistry['records']] == sequences
    assert all(row['recommendation_reason'] and row['recommendation_tradeoff'] for row in panel)
    assert source.read_bytes() == before and len(assessed) == 4
    monkeypatch.setattr(shortlist_review, '_model_call', lambda *a: pytest.fail('Accepted review must reopen offline'))
    again = pipeline._emit_v2_generated_design_artifacts(str(root), review_recommendations=True)
    assert json.loads(Path(again['generated_aso_set_json']).read_text())['rows'] == generated['rows']
