"""File-based host adapter for the public CLI lifecycle.

The browser supplies human answers through this adapter. The pipeline retains
all planning, validation, reference resolution and scientific decisions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import signal
import time
import uuid


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class Host:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.status = read_json(self.directory / 'status.json', {})
        self.status.update(status='running', message='Loading MODAL-ASO', question=None)
        self.save()

    def save(self):
        self.status['updated_at'] = time.time()
        write_json(self.directory / 'status.json', self.status)

    def event(self, event):
        from run_goal_intake import _cli_message_for_tool
        kind = event.get('type')
        if kind == 'model_installation':
            answers = self.clarify(questions=[f"Install {event.get('model', 'the requested model')} now?"],
                context={'reason': event.get('message', ''),
                         'recommendation': 'Installation downloads the model and prepares it in your configured model directory.'})
            return answers[0]['answer'].strip().casefold() in {'yes', 'install', 'yes, install'}
        message = None
        if kind == 'tool_call':
            message = _cli_message_for_tool(event.get('tool'), 1)
        elif kind in {'progress', 'warning'}:
            message = str(event.get('message') or '')[:4000]
        if message:
            self.status.update(message=message, stage_started_at=time.time())
            history = self.status.setdefault('progress', [])
            if not history or history[-1]['message'] != message:
                history.append({'message': message, 'time': time.time()})
                self.status['progress'] = history[-100:]
        result = event.get('result')
        if isinstance(result, dict):
            for key in ('state_path', 'run_dir'):
                if isinstance(result.get(key), str):
                    self.status[key] = result[key]
        if message or isinstance(result, dict):
            self.save()

    def clarify(self, *, questions, context):
        identifier = uuid.uuid4().hex
        self.status.update(status='needs_user_input', question={'id': identifier,
                           'questions': questions, 'context': context}, message='Waiting for your answer')
        self.save()
        answer_path = self.directory / ('answer-' + identifier + '.json')
        while True:
            answers = read_json(answer_path)
            if isinstance(answers, list) and len(answers) == len(questions) and all(
                isinstance(a, dict) and a.get('question') == q and isinstance(a.get('answer'), str)
                and a['answer'].strip() for a, q in zip(answers, questions)
            ):
                self.status.setdefault('conversation', []).append({'questions': questions, 'answers': answers})
                self.status.update(status='running', question=None, message='Continuing with your answer', stage_started_at=time.time())
                self.save()
                return answers
            time.sleep(.2)

    def finish(self, result):
        self.status.update(result=result, status=result.get('status', 'fatal'), question=None)
        for key in ('state_path', 'run_dir'):
            if result.get(key):
                self.status[key] = result[key]
        self.save()


def run(directory, *, pipeline=None):
    host = Host(directory)
    request = read_json(Path(directory) / 'request.json')
    try:
        if not isinstance(request, dict):
            raise ValueError('The saved GUI request is unavailable.')
        os.environ['ASO_RUNS_DIR'] = request['output_dir']
        os.environ.setdefault('MODAL_ASO_SKIP_RUNTIME_INSTALL', '1')
        if pipeline is None:
            from .runtime_binaries import refresh_runtime_path
            refresh_runtime_path()
            from .llm_provider import get_provider_config
            get_provider_config()
            from run_goal_intake import main
            pipeline = main
        argv = ['--raw-cli']
        if request.get('skip_shortlist'):
            argv.append('--skip-shortlist')
        if request.get('state_path'):
            argv.extend(['--agentic-state-path', request['state_path']])
        argv.extend(['--', request['request']])
        pipeline(argv=argv, clarification_resolver=host.clarify, result_callback=host.finish,
                 event_callback=host.event)
        if host.status['status'] == 'running':
            raise RuntimeError('The pipeline returned without a completion result. The saved state can be resumed.')
    except (Exception, SystemExit) as exc:
        # Tracebacks stay in the local worker log. Credentials never enter the
        # browser's exception response through a provider exception string.
        import traceback
        traceback.print_exc()
        host.status.update(status='fatal', question=None,
            message='The run stopped. Check the local worker log and installation, then resume the saved state.',
            error_type=type(exc).__name__)
        host.save()


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    run(sys.argv[1])
