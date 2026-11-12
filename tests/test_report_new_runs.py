"""Fresh task completion must deliver reports through production writers.

Only model responses and scientific computation are supplied as fixtures.
State transitions, report writing, discovery and final exports run normally.
"""
from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.run_layout import canonical_run_path


class ExplorerPacket(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.active = False
        self.text = ''
        self.frame_sources = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        if tag == 'iframe':
            self.frame_sources.append(dict(attrs).get('src', ''))
        if tag == 'script' and dict(attrs).get('class') == 'report-explorer-data':
            self.active = True

    def handle_data(self, data):
        if self.active:
            self.text += data

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False


def fresh_task_report(run, monkeypatch, *, kind='answer_question', count=3, mapped=True):
    run.mkdir(parents=True)
    title = 'Reading the results' if kind == 'answer_question' else 'Review supplied observations'
    destination = run/'exports'/'reader copy.html'
    request = title + '. Save the HTML report to the specified file.'
    raw = {'schema_version': 'v3', 'request_text': request, 'request_mode': 'single_task',
        'tasks': [{'task_id': 'read_results', 'task_type': kind, 'label': title,
                   'context': {'question': title, 'web_retrieval': {'needed': False}}}],
        'final_outputs': [{'task_id': 'read_results', 'kind': 'report', 'format': 'html',
                           'source_artifact': 'report_html', 'path': str(destination)}]}
    if kind == 'evaluate_existing_aso_set':
        raw['tasks'][0]['context'].update(gene_symbol='NEUTRAL', species='human',
            provided_asos=[{'name': f'Observation {i + 1}',
                'sequence': 'ACGT' + ''.join('ACGT'[(i // 4**j) % 4] for j in range(5))}
                for i in range(max(1, count))])
    envelope = pipeline._build_request_envelope(request_text=request, request_id='fresh_report',
        structured_goal={}, raw_envelope=raw, finalize_design_contract=False)
    state = {'request': request, 'status': 'request_parsed_ready', 'phase': 'request_parsed_ready',
        'run_dir': str(run), 'request_envelope': envelope, 'last_error': None,
        'clarification': {'isoform_scope_confirmed': False}, 'pending_questions': [],
        'clarification_qa_history': []}
    pipeline._ensure_v2_request_metadata(state)
    pipeline._save_goal_payload(str(run), {'request': request, 'request_envelope': envelope}, {},
                               {'status': 'ready', 'request_envelope': envelope})
    state_path = canonical_run_path(run, 'agentic_state.json')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))
    response = {'answer_summary': 'Each observation retains its recorded value.',
        'markdown': '# Main finding\nEach observation retains its recorded value.\n\n'
                    '## Supporting evidence\n[Example source](https://example.invalid/source).\n'}
    rows = []
    for i in range(count if kind == 'evaluate_existing_aso_set' else 0):
        sequence = 'ACGT' + ''.join('ACGT'[(i // 4**j) % 4] for j in range(5))
        row = {'input_row_id': f'row_{i}', 'name': f'Observation {i + 1}', 'sequence': sequence,
            'declared_sequence_role': 'antisense_oligo_sequence', 'ordering_notation': sequence,
            'mapping_hits': [], 'gc_fraction': .5, 'off_target_gene_count': 0 if i == 0 else None,
            'off_target_search_status': 'complete' if i == 0 else 'not_screened'}
        if mapped:
            row['genomic_segments'] = [{'chrom': 'chr1', 'start': 101 + i * 10, 'end': 109 + i * 10}]
        rows.append(row)
    computed = {'provided_asos': rows, 'context_summary': {'gene_symbol': 'NEUTRAL'},
                'limitations': ['Synthetic report fixture.']}
    if kind == 'evaluate_existing_aso_set':
        response['evaluation_assessments'] = [
            {'input_row_id': row['input_row_id'], 'name': row['name'],
             'mechanistic_plausibility': 'The recorded evidence is available for review.',
             'selectivity_limitations': 'Screening coverage is recorded for this observation.',
             'uncertainty': 'Cellular activity is unmeasured.'} for row in rows]
        monkeypatch.setattr(pipeline, '_evaluate_existing_asos_payload', lambda **kwargs: deepcopy(computed))
    monkeypatch.setattr(pipeline, '_run_workbench_response_sync', lambda *args, **kwargs: deepcopy(response))
    monkeypatch.setattr(pipeline, '_run_sandbox_analysis_sync',
                        lambda *args, **kwargs: pytest.fail('Native HTML delivery must use the report exporter'))
    tool = pipeline.tool_answer_question if kind == 'answer_question' else pipeline.tool_evaluate_existing_aso_set
    completed = tool(str(state_path))
    assert completed['status'] == 'ok', completed
    assert completed['task_status'] == 'report_ready', completed
    final = pipeline.finalize_run_state(completed['state_path'])
    assert final['terminal_status'] == 'run_complete', final
    saved = pipeline._load_state(final['state_path'])
    assert saved['requested_output_delivery']['ok']
    assert saved['requested_output_delivery'].get('execution_attempts', []) == []
    return {'run': run, 'report': run/'report.html', 'export': destination,
            'rows': rows, 'artifacts': completed['artifacts'], 'state': saved}


@pytest.mark.parametrize('kind,count,mapped', [
    ('answer_question', 0, False),
    ('evaluate_existing_aso_set', 3, True),
    ('evaluate_existing_aso_set', 130, True),
    ('evaluate_existing_aso_set', 3, False),
    ('evaluate_existing_aso_set', 0, False),
])
def test_fresh_tasks_generate_and_deliver_named_html(tmp_path, monkeypatch, kind, count, mapped):
    result = fresh_task_report(tmp_path/'new run', monkeypatch, kind=kind, count=count, mapped=mapped)
    document = result['report'].read_text()
    packet = ExplorerPacket(document).text
    assert 'Each observation retains its recorded value.' in document
    assert 'https://example.invalid/source' in document
    assert 'Skip to results' in document
    assert 'file unavailable in this report bundle' not in document
    if result['rows']:
        data = json.loads(packet)
        assert data['kind'] == 'evaluation'
        columns = data['record_columns']
        restored = [{column['name']: column['values'][value] if 'values' in column else value
                     for column, value in zip(columns, row)} for row in data['records']]
        assert [row['aso'] for row in restored] == [row['sequence'] for row in result['rows']]
        assert all(bool(row['intervals']) == mapped for row in restored)
        assert 'Analysis and selection evidence' in document
        assert not data.get('recommendation_review')
    else:
        assert not packet
        assert '<section class="report-explorer' not in document
        assert 'Analysis and selection evidence' not in document
    # A recipient can read the exported file after the producing run is gone.
    portable = tmp_path/'reader copy.html'
    shutil.copyfile(result['export'], portable)
    exported_packet = ExplorerPacket(portable.read_text()).text
    assert exported_packet == packet
    shutil.rmtree(result['run'])
    assert 'Each observation retains its recorded value.' in portable.read_text()


def fresh_design_report(run):
    import run_goal_intake as runtime
    from test_report_explorer import run_fixture, save
    from aso_agents.candidate_assessment import write_assessment_artifacts
    from aso_agents.shortlist_evidence import molecule_id

    goal, raw, pool = run_fixture(run)
    pool[0]['synthesized_antisense_sequence'] = pool[0]['Synthesis']
    review = {'status': 'completed', 'summary': 'A recorded observation supports this fixture panel.',
        'reviewed_molecule_count': 1, 'recommendations': {molecule_id(pool[0]): {
            'candidate_id': molecule_id(pool[0]), 'why_chosen': 'Retains the recorded observation.',
            'panel_role': 'Selected fixture.', 'tradeoff': 'Activity remains unmeasured.',
            'evidence': [], 'comparison_evidence': []}}}
    write_assessment_artifacts(pool, pool, goal, run/'artifacts/candidates', review=review)
    save(run, 'artifacts/intake/goal_intake.json', {'final_payload': {
        'structured_goal': goal, 'report_payload': {'payload_kind': 'report_manifest'}}})
    (run/'request.txt').write_text('Review this synthetic design fixture.')
    (run/'report.md').write_text('# Selection checks\nThe recorded observation is retained.\n')
    report = runtime._generate_goal_intake_html_report(str(run), goal)
    assert report is not None
    return {'run': run, 'report': Path(report), 'rows': raw}


def test_fresh_design_writer_keeps_the_panel_and_full_pool(tmp_path):
    result = fresh_design_report(tmp_path/'new design')
    document = result['report'].read_text()
    data = json.loads(ExplorerPacket(document).text)
    assert data['kind'] == 'design'
    assert len(data['records']) == len(result['rows'])
    assert data['recommendation_review']['status'] == 'completed'
    assert 'Retains the recorded observation.' in document
    assert 'The recorded observation is retained.' in document
    assert 'Skip to results' in document


def fresh_composite_report(run):
    child = fresh_design_report(run/'child_runs'/'design')
    markdown = '# Main finding\nThe combined report retains each task result.\n'
    response = {'completed_tasks': [{'task_id': 'design', 'task_type': 'design_new_asos',
        'label': 'Recorded design', 'payload': {'child_run_dir': str(child['run']),
        'child_structured_goal': {'gene_symbol': 'TEST'}, 'report_excerpt': 'Recorded design.'}}]}
    destination = run/'exports'/'combined.html'
    outputs = [{'task_id': 'combined', 'kind': 'report', 'source_artifact': 'report_html', 'path': str(destination)}]
    pipeline._write_v2_response_outputs(run_dir=str(run), markdown_text=markdown, response_payload=response,
        response_basename='composite_report', title='Combined review', requested_outputs=outputs)
    state = {'request_envelope': {'schema_version': 'v3', 'final_outputs': outputs,
        'tasks': [{'task_id': 'combined', 'task_type': 'build_composite_report'}]}}
    receipt = pipeline._materialize_requested_final_outputs(state, str(run))
    assert receipt['ok'] and receipt['execution_attempts'] == [], receipt
    return {'run': run, 'report': run/'report.html', 'export': destination, 'receipt': receipt}


def test_fresh_combined_export_keeps_child_tabs_and_refreshes_downloads(tmp_path):
    result = fresh_composite_report(tmp_path/'combined run')
    def check(path):
        document = path.read_text()
        assert 'Reports by target' in document
        assert 'child-report-frame' in document
        assert 'Recorded design' in document
        assert 'file unavailable in this report bundle' not in document
        frames = ExplorerPacket(document).frame_sources
        assert len(frames) == 1
        target = urlsplit(frames[0])
        assert target.scheme == '' and target.path
        child = (path.parent/unquote(target.path)).resolve()
        assert child.is_file()
        data = json.loads(ExplorerPacket(child.read_text()).text)
        assert len(data['records']) == 3
        assert data['recommendation_review']['status'] == 'completed'
    check(result['report'])
    check(result['export'])
    portable = tmp_path/'received'
    shutil.copytree(result['export'].parent, portable)
    shutil.rmtree(result['run'])
    check(portable/result['export'].name)
