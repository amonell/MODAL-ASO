import asyncio
from contextvars import ContextVar
from copy import deepcopy
import json
import threading

import pytest

from aso_agents import pipeline_tools as p
from aso_agents import response_assembly as assembly


def inputs():
    rows = [{'input_row_id': 'alpha', 'name': 'Alpha'},
            {'input_row_id': 'beta', 'name': 'Beta'}]
    payload = {'task_type': 'evaluate_existing_aso_set',
               'evaluation_summary': {'provided_asos': rows}}
    response = {'answer_summary': 'Both observations are assessed.',
                'limitations': ['Activity is unmeasured.'],
                'evaluation_assessments': [
                    {**row, 'ranking_rationale': 'No potency ranking is established.'}
                    for row in rows]}
    return payload, response


@pytest.mark.parametrize('fail_one_page', [False, True])
def test_blocking_provider_pages_overlap_preserve_context_and_join_before_return(monkeypatch, tmp_path, fail_one_page):
    payload, response = inputs()
    untouched = deepcopy((payload, response))
    marker = ContextVar('assembly_test_trace', default=None)
    marker.set('same-request-trace')
    barrier = threading.Barrier(2, timeout=3)
    finished = threading.Event()
    calls = []
    monkeypatch.setattr(assembly, 'narrative_pages', lambda fields: [
        {'/answer_summary': fields['/answer_summary']},
        {'/limitations/0': fields['/limitations/0']},
    ])

    async def blocking_provider(**kwargs):
        # Deliberately synchronous, like Runner.run's provider transport.
        assert marker.get() == 'same-request-trace'
        document = json.loads(kwargs['prompt'])
        calls.append((threading.get_ident(), document['source']))
        barrier.wait()
        if fail_one_page and '/answer_summary' in document['draft_narrative_fields']:
            raise RuntimeError('neutral worker failure')
        result = {'findings': []}
        kwargs['attempt_observer'](1, {'prompt': kwargs['prompt'], 'raw_response': json.dumps(result)})
        finished.set()
        return result

    monkeypatch.setattr(p, '_run_schema_validated_agent_once', blocking_provider)
    if fail_one_page:
        with pytest.raises(RuntimeError, match='neutral worker failure'):
            asyncio.run(assembly.review_assembled_response(payload, response, artifact_dir=tmp_path))
        assert not list(tmp_path.glob('components/*/accepted.json'))
    else:
        result = asyncio.run(assembly.review_assembled_response(payload, response, artifact_dir=tmp_path))
        assert {k: v for k, v in result.items() if k != 'native_assembly_review'} == response
        assert len(list(tmp_path.glob('components/*/accepted.json'))) == 1
        assert len(list(tmp_path.glob('components/*/attempt*/round-1-page-*-call-1.json'))) == 2
    assert finished.is_set()
    assert len({thread for thread, _ in calls}) == 2
    assert calls[0][1] == calls[1][1]
    assert (payload, response) == untouched
