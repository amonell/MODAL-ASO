"""Optional, loopback-only browser host for MODAL-ASO's existing CLI."""
from __future__ import annotations

from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import fcntl
import json
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import quote, unquote, urlsplit
import uuid
import webbrowser

from .gui_worker import read_json, write_json


class Jobs:
    def __init__(self, output_dir):
        self.output = Path(output_dir).expanduser().resolve()
        self.root = self.output / '.modal-aso-gui'
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lease = (self.root / 'server.lock').open('a')
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close()
            raise ValueError('A GUI server already uses this output directory. Open its local URL.') from None
        self.lock = threading.RLock()
        self.workers = {}

    def directory(self, identifier):
        if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Unknown run.')
        path = self.root / identifier
        if not path.is_dir() or path.is_symlink():
            raise ValueError('Unknown run.')
        return path

    def snapshot(self, identifier):
        directory = self.directory(identifier)
        data = read_json(directory / 'status.json', {})
        process = self.workers.get(identifier)
        if data.get('status') in {'running', 'needs_user_input'} and (process is None or process.poll() is not None):
            data.update(status='stopped', question=None, message='The worker stopped. You can resume from its saved state.')
        data['id'] = identifier
        data['worker_log'] = str(directory / 'worker.log')
        result = data.get('result') or {}
        data['files'] = []
        run_dir = data.get('run_dir')
        if run_dir:
            for label, filename in (result.get('artifacts') or {}).items():
                try:
                    relative = Path(filename).resolve().relative_to(Path(run_dir).resolve())
                    self.file(identifier, relative.as_posix(), status=data)
                except (ValueError, OSError, TypeError):
                    continue
                data['files'].append({'label': label, 'name': relative.name,
                    'url': '/files/' + identifier + '/' + quote(relative.as_posix())})
        return data

    def list(self):
        identifiers = sorted((p.name for p in self.root.iterdir() if p.is_dir() and not p.is_symlink()),
                             key=lambda name: (self.root / name).stat().st_mtime, reverse=True)
        result = []
        for identifier in identifiers[:50]:
            try:
                data = self.snapshot(identifier)
                result.append({key: data.get(key) for key in ('id', 'request', 'status', 'created_at')})
            except ValueError:
                continue
        return result

    def start(self, payload, *, resume=None):
        with self.lock:
            if any(p.poll() is None for p in self.workers.values()):
                raise ValueError('A run is active. Answer its questions or stop it before starting another.')
            state = None
            if resume:
                saved = self.snapshot(resume)
                state = saved.get('state_path')
                if not state or not Path(state).is_file() or not Path(state).resolve().is_relative_to(self.output):
                    raise ValueError('A resumable state is unavailable for this run.')
                payload = read_json(self.directory(resume) / 'request.json')
            request = payload.get('request')
            if not isinstance(request, str) or not request.strip() or len(request) > 100000:
                raise ValueError('Enter a request of up to 100,000 characters.')
            identifier = uuid.uuid4().hex
            directory = self.root / identifier
            directory.mkdir(mode=0o700)
            write_json(directory / 'request.json', {'request': request, 'output_dir': str(self.output),
                'state_path': state, 'skip_shortlist': payload.get('skip_shortlist') is True})
            write_json(directory / 'status.json', {'request': request, 'created_at': time.time(),
                'stage_started_at': time.time(), 'status': 'running', 'message': 'Starting MODAL-ASO',
                'state_path': state, 'progress': [], 'conversation': []})
            with (directory / 'worker.log').open('ab') as log:
                try:
                    self.workers[identifier] = subprocess.Popen(
                        [sys.executable, '-B', '-m', 'aso_agents.gui_worker', str(directory)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                except OSError:
                    write_json(directory / 'status.json', {'request': request, 'status': 'fatal',
                        'message': 'The worker could not start. Check the Python installation.'})
                    raise
            return self.snapshot(identifier)

    def answer(self, identifier, payload):
        with self.lock:
            data = self.snapshot(identifier)
            question = data.get('question') or {}
            if data['status'] != 'needs_user_input' or payload.get('question_id') != question.get('id'):
                raise ValueError('This question has changed. Refresh the run before answering.')
            answers = payload.get('answers')
            questions = question.get('questions') or []
            if not isinstance(answers, list) or len(answers) != len(questions) or any(
                not isinstance(answer, str) or not answer.strip() or len(answer) > 100000 for answer in answers
            ):
                raise ValueError('Answer each displayed question before continuing.')
            path = self.directory(identifier) / ('answer-' + question['id'] + '.json')
            if path.exists():
                raise ValueError('Your answer has already been submitted.')
            write_json(path, [{'question': q, 'answer': a} for q, a in zip(questions, answers)])

    def stop(self, identifier):
        with self.lock:
            process = self.workers.get(identifier)
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            directory = self.directory(identifier)
            data = read_json(directory / 'status.json', {})
            if data.get('status') in {'running', 'needs_user_input'}:
                data.update(status='stopped', question=None, message='Stopped at your request. Saved outputs remain available.')
                write_json(directory / 'status.json', data)

    def close(self):
        for identifier in list(self.workers):
            self.stop(identifier)
        self.lease.close()

    def file(self, identifier, relative, *, status=None):
        data = status or read_json(self.directory(identifier) / 'status.json', {})
        root = Path(data.get('run_dir') or '/unavailable').resolve()
        relative = Path(relative)
        if not root.is_relative_to(self.output) or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Output file unavailable.')
        path = root / relative
        if any((root.joinpath(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError('Linked reference files are available through the local filesystem.')
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Output file unavailable.')
        return path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Session tokens and requests stay out of access logs.

    def send(self, code, body, content_type='application/json', *, cookie=False):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        if cookie:
            self.send_header('Set-Cookie', 'modal_aso_session=' + self.server.token + '; HttpOnly; SameSite=Strict; Path=/files/')
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, *, file=False):
        if self.headers.get('Host') != self.server.authority:
            return False
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.server.authority:
            return False
        token = self.headers.get('Authorization', '').removeprefix('Bearer ')
        if file:
            cookies = SimpleCookie(self.headers.get('Cookie', ''))
            if 'modal_aso_session' in cookies:
                token = cookies['modal_aso_session'].value
        return secrets.compare_digest(token, self.server.token)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/gui.js': ('gui.js', 'text/javascript; charset=utf-8'),
                  '/gui.css': ('gui.css', 'text/css; charset=utf-8')}
        try:
            if path in assets:
                name, content_type = assets[path]
                return self.send(200, (Path(__file__).parent / 'resources/gui' / name).read_bytes(), content_type)
            if not self.authorized(file=path.startswith('/files/')):
                return self.send(403, {'error': 'Open the local URL printed by modal-aso gui to connect.'})
            if path == '/api/session':
                return self.send(200, {'output_dir': str(self.server.jobs.output)}, cookie=True)
            if path == '/api/runs':
                return self.send(200, self.server.jobs.list())
            if path.startswith('/api/runs/'):
                return self.send(200, self.server.jobs.snapshot(path.removeprefix('/api/runs/')))
            if path.startswith('/files/'):
                identifier, _, relative = path.removeprefix('/files/').partition('/')
                filename = self.server.jobs.file(identifier, relative)
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(filename)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(filename.stat().st_size))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                with filename.open('rb') as handle:
                    shutil.copyfileobj(handle, self.wfile)
                return
            self.send(404, {'error': 'Page unavailable.'})
        except (ValueError, OSError):
            self.send(404, {'error': 'Run or output file unavailable.'})

    def do_POST(self):
        if not self.authorized():
            return self.send(403, {'error': 'Reconnect using the local URL printed by modal-aso gui.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1024 * 1024 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Send a JSON request of up to 1 MB.')
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('A JSON object is required.')
            parts = urlsplit(self.path).path.strip('/').split('/')
            if parts == ['api', 'runs']:
                return self.send(201, self.server.jobs.start(payload))
            if len(parts) == 4 and parts[:2] == ['api', 'runs']:
                identifier, action = parts[2:]
                if action == 'answer':
                    self.server.jobs.answer(identifier, payload)
                elif action == 'stop':
                    self.server.jobs.stop(identifier)
                elif action == 'resume':
                    return self.send(201, self.server.jobs.start({}, resume=identifier))
                else:
                    return self.send(404, {'error': 'Action unavailable.'})
                return self.send(200, {'status': 'ok'})
            self.send(404, {'error': 'Action unavailable.'})
        except (ValueError, OSError) as exc:
            self.send(400, {'error': str(exc) if isinstance(exc, ValueError) else 'The local run files could not be updated.'})


def make_server(output_dir, port=8860):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    try:
        server.jobs = Jobs(output_dir)
    except Exception:
        server.server_close()
        raise
    server.token = secrets.token_urlsafe(32)
    server.authority = '127.0.0.1:' + str(server.server_port)
    return server


def serve(output_dir, *, port=8860, open_browser=True):
    server = make_server(output_dir, port)
    url = 'http://' + server.authority + '/#' + server.token
    print('MODAL-ASO browser interface: ' + url, flush=True)
    print('Outputs: ' + str(server.jobs.output) + '\nKeep this terminal open. Ctrl+C stops the server and its active run.', flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.close()
        server.server_close()
    return 0
