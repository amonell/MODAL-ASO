import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from aso_agents.analysis_execution import execute_analysis_program
from aso_agents.analysis_process import analysis_timeout_seconds, run_program, TIMEOUT_ENV


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '86401', 'invalid', '', True])
def test_invalid_execution_budgets_are_rejected(value, monkeypatch):
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)
    with pytest.raises(ValueError):
        analysis_timeout_seconds(value)


def test_budget_default_override_and_payload_agree(tmp_path, monkeypatch):
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)
    assert analysis_timeout_seconds() == 300
    monkeypatch.setenv(TIMEOUT_ENV, '12.5')
    assert analysis_timeout_seconds() == 12.5
    assert analysis_timeout_seconds(3) == 3
    requests = []
    def generate(payload):
        requests.append(payload)
        return {'python_code': "from pathlib import Path\n"
                "Path('sandbox_result.json').write_text('{}')\n"
                "Path('sandbox_summary.md').write_text('Finished.')"}
    result = execute_analysis_program(directory=tmp_path, payload={'resource_limits': {'timeout_seconds': None}},
        generate=generate, required_outputs=[])
    assert result['ok']
    assert requests[0]['resource_limits']['timeout_seconds'] == 12.5
    assert result['attempts'][0]['timeout_seconds'] == 12.5


def test_timeout_preserves_logs_restores_inputs_and_stops_repair(tmp_path):
    protected = tmp_path / 'observations.txt'
    protected.write_text('original observations')
    calls = []
    def generate(payload):
        calls.append(payload)
        return {'python_code': "from pathlib import Path\nimport time\n"
                "Path('observations.txt').write_text('changed')\n"
                "print('partial output', flush=True)\n"
                "time.sleep(60)"}
    started = time.monotonic()
    result = execute_analysis_program(directory=tmp_path, payload={}, generate=generate,
        # Allow time for the interpreter and sandbox helper to start.
        required_outputs=[], readonly_outputs=[protected], timeout_seconds=2, max_attempts=5)
    assert time.monotonic() - started < 10
    assert not result['ok'] and result['error_code'] == 'analysis_timeout'
    assert len(calls) == len(result['attempts']) == 1
    assert protected.read_text() == 'original observations'
    local = tmp_path / 'execution_attempts/01'
    assert 'partial output' in (local / 'stdout.log').read_text()
    receipt = json.loads((local / 'execution.json').read_text())
    assert receipt['status'] == 'timed_out' and receipt['exit_code'] == 124
    assert receipt['restored_output_paths'] == [str(protected)]


@pytest.mark.skipif(os.name != 'posix', reason='Process sessions are supported on Linux, macOS and WSL')
def test_timeout_terminates_children_that_inherit_streams(tmp_path):
    sentinel = tmp_path / 'child-finished'
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(sentinel)!r}).touch()"
    parent = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}])\nprint('started', flush=True)\ntime.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run_program([sys.executable, '-c', parent], directory=tmp_path, timeout=.3,
                    stdout_path=tmp_path/'stdout.log', stderr_path=tmp_path/'stderr.log')
    assert 'started' in failure.value.stdout
    time.sleep(1.1)
    assert not sentinel.exists()


def test_production_analysis_advertises_enforced_budget(tmp_path, monkeypatch):
    from aso_agents import analysis_execution, pipeline_tools
    from test_analysis_user_contract import prepared_task
    monkeypatch.setenv(TIMEOUT_ENV, '21')
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {'ok': False, 'error': 'captured', 'attempts': []}
    monkeypatch.setattr(analysis_execution, 'execute_analysis_program', capture)
    pipeline_tools._execute_sandbox_analysis(prepared_task(tmp_path, monkeypatch), tool_name='analysis')
    assert calls[0]['timeout_seconds'] == calls[0]['payload']['resource_limits']['timeout_seconds'] == 21
