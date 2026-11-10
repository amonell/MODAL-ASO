"""Full-library activity and requested conditions survive the final export."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from aso_agents import oligoai
from aso_agents.report_explorer_data import project_run
from test_oligoai import predictor, reference, row, model_input


def test_final_panel_limit_keeps_full_pool_predictions(tmp_path, monkeypatch):
    from aso_agents import pipeline_tools as pipeline
    rows = []
    for index in range(17):
        sequence = "ACGTCAGACTTAGCAC" + ''.join('ACGT'[(index // 4**i) % 4] for i in range(4))
        candidate = row(sequence)
        candidate.update(program_id='gapmers', job_id='job', selection_rank=index + 1,
                         chrom='chrFixture', start=100 + 30 * index, end=120 + 30 * index, strand='+')
        rows.append(candidate)
    original = deepcopy(rows)
    goal = {'intent': 'knockdown', 'species': 'human', 'candidate_limit': 4,
            'oligoai_evaluation': {'dosage_nm': 25, 'transfection_method': 'Gymnosis',
                                  'unmodeled_conditions': ['HEK293 cells', '48 hours']},
            'design_programs': [{'program_id': 'gapmers', 'modality': 'gapmer'}],
            'gapmer_design_jobs': [{'program_id': 'gapmers', 'job_id': 'job'}]}
    monkeypatch.setattr(pipeline, '_collect_generated_candidate_rows', lambda *a, **k: deepcopy(rows))
    monkeypatch.setattr(pipeline, '_collect_generated_stage_sets', lambda *a: {})
    monkeypatch.setattr(pipeline, '_goal_payload_context', lambda *a: {'structured_goal': goal})
    ref = reference()
    ref['sequence'] = 'G' * 50 + ''.join(oligoai.reverse_complement(r['synthesized_antisense_sequence']) + 'C' * 100 for r in rows)
    ref.update(site_end0=len(ref['sequence'])-50, genomic_end0=100+len(ref['sequence']))
    monkeypatch.setattr(oligoai, 'target_references', lambda *a: [ref])
    seen = []
    def observe(inputs, directory):
        seen.extend(inputs)
        return predictor(inputs, directory)
    monkeypatch.setattr(oligoai, 'released_model_predict', observe)

    artifacts = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    pool = json.loads(Path(artifacts['candidate_assessments_json']).read_text())
    panel = json.loads(Path(artifacts['generated_aso_set_json']).read_text())
    evaluation = panel['oligoai_evaluation']
    assert len(panel['rows']) == 4
    assert len(pool['rows']) == evaluation['predicted_count'] == len(rows) == len(seen)
    assert evaluation['candidate_scope'] == 'assessed_pool_before_panel_selection'
    assert json.loads(Path(evaluation['artifact_json']).read_text()) == evaluation
    assert all(i['dosage_nm'] == 25 and i['transfection_method'] == 'Gymnosis' for i in seen)
    assert all(r['oligoai_evaluation']['status'] == 'predicted' for r in pool['rows'])
    assert all(r['oligoai_evaluation']['conditions']['unmodeled_conditions'] == ['HEK293 cells', '48 hours'] for r in pool['rows'])
    assert rows == original
    data = project_run(tmp_path, goal=goal)
    assert len(data['records']) == 17
    assert all(len(r['oligoai']) == 1 and r['oligoai'][0]['value'] == 73.125 for r in data['records'])
    assert len({r['oligoai'][0]['scenario_id'] for r in data['records']}) == 1
    assert all(r['oligoai'][0]['model_sha256'] == oligoai.MODEL_SHA256 for r in data['records'])

    # Changing only panel size keeps the same model inputs and evidence directory.
    goal['candidate_limit'] = 9
    again = pipeline._emit_v2_generated_design_artifacts(str(tmp_path))
    new_panel = json.loads(Path(again['generated_aso_set_json']).read_text())
    assert len(new_panel['rows']) == 9
    assert new_panel['oligoai_evaluation']['artifact_json'] == evaluation['artifact_json']
    assert seen[:17] == seen[17:]
    assert [r['synthesized_antisense_sequence'] for r in panel['rows']] == [r['synthesized_antisense_sequence'] for r in new_panel['rows'][:4]]


def test_unmodeled_experiment_is_preserved_without_becoming_model_inputs(tmp_path):
    options = {'dosage_nm': 25, 'transfection_method': 'Lipofection',
               'unmodeled_conditions': ['HEK293 cells', '48 hours', 'Lipid reagent supplied by user']}
    seen = []
    def observe(inputs, directory):
        seen.extend(inputs)
        return predictor(inputs, directory)
    result = oligoai.evaluate_candidates([row()], {'oligoai_evaluation': options}, tmp_path,
                                        references=[reference()], predictor=observe)
    assert seen[0]['dosage_nm'] == 25 and seen[0]['transfection_method'] == 'Lipofection'
    assert 'unmodeled_conditions' not in seen[0]
    assert result['candidates'][0]['conditions']['unmodeled_conditions'] == options['unmodeled_conditions']
    exported = list(csv.DictReader(Path(result['artifact_csv']).open()))
    assert exported[0]['unmodeled_conditions'] == '; '.join(options['unmodeled_conditions'])
    assert '48 hours' in oligoai.html_section(result)


def test_unresolved_explicit_dose_cannot_fall_back_to_reference_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(oligoai, 'target_references', lambda *a: pytest.fail('No reference fetch for unresolved inputs'))
    result = oligoai.evaluate_candidates([row()], {'oligoai_evaluation': {
        'unresolved_conditions': ['1 mg/kg requires an in vitro molar concentration']}}, tmp_path,
        predictor=lambda *a: pytest.fail('No default-condition inference'))
    assert result['predicted_count'] == 0
    assert '1 mg/kg' in result['candidates'][0]['reason']


@pytest.mark.parametrize('invalid', [None, [], False, 0, ''])
def test_explicit_malformed_settings_cannot_turn_into_defaults(invalid):
    with pytest.raises(ValueError):
        oligoai.evaluation_conditions({'oligoai_evaluation': invalid})


def test_legacy_and_program_conditions_match_parent_and_child_execution():
    from run_goal_intake import _design_program_goal_copy
    goal = {'constraints': {'oligoai_evaluation': {'dosage_nm': 25, 'transfection_method': 'Gymnosis'}},
            'design_programs': [{'program_id': 'p', 'oligoai_evaluation': {'dosage_nm': 50}}]}
    child = _design_program_goal_copy(goal, goal['design_programs'][0])
    assert oligoai.evaluation_conditions(child) == oligoai.candidate_conditions({'program_id': 'p'}, goal)
    assert child['oligoai_evaluation'] == {'dosage_nm': 50, 'transfection_method': 'Gymnosis'}
    goal['oligoai_evaluation'] = {}
    assert oligoai.candidate_conditions({'program_id': 'p'}, goal) == oligoai.evaluation_conditions(child)


def test_full_pool_details_have_a_bounded_static_preview(tmp_path):
    seen = []
    def observe(inputs, directory):
        seen.extend(inputs)
        return predictor(inputs, directory)
    result = oligoai.evaluate_candidates([row() for _ in range(75)], {}, tmp_path,
                                        references=[reference()], predictor=observe)
    html = oligoai.html_section(result)
    assert html.count('<tr>') == 41
    assert 'first 40 of 75' in html
    assert len(list(csv.DictReader(Path(result['artifact_csv']).open()))) == 75
    assert len(seen) == 1


def test_saved_model_results_are_reused_only_for_identical_inputs(tmp_path, monkeypatch):
    from aso_agents import oligoai_readiness
    from aso_agents.response_artifacts import digest, write_json
    monkeypatch.setattr(oligoai, 'runtime_root', lambda: tmp_path/'runtime')
    monkeypatch.setattr(oligoai_readiness, 'inspect_installation', lambda _: {'status': 'installed'})
    monkeypatch.setattr(oligoai, 'file_sha', lambda p: oligoai.MODEL_SHA256 if Path(p).name == oligoai.MODEL_FILE else 'runner-hash')
    calls = []
    def run(command, **kwargs):
        inputs = json.loads(Path(command[command.index('--input')+1]).read_text())['inputs']
        calls.append(inputs)
        result = predictor(inputs, tmp_path)
        result.update(input_digest=digest(inputs), runner_sha256='runner-hash')
        write_json(command[command.index('--output')+1], result)
    monkeypatch.setattr(oligoai.subprocess, 'run', run)
    first = model_input()
    directory = tmp_path/'prediction'
    result = oligoai.released_model_predict([first], directory)
    assert oligoai.released_model_predict([first], directory) == result
    assert len(calls) == 1
    changed = {**first, 'dosage_nm': 25, 'transfection_method': 'Gymnosis'}
    changed['input_id'] = digest({k:v for k,v in changed.items() if k != 'input_id'})
    oligoai.released_model_predict([changed], directory)
    assert len(calls) == 2
    assert calls[-1][0]['dosage_nm'] == 25
