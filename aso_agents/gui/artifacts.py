"""Serve one recorded run locally, keeping relative report links usable."""
from functools import lru_cache
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import atexit
import mimetypes
import secrets
import shutil
import threading


class ArtifactHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.headers.get('Host') != self.server.authority:
            self.send_error(403)
            return
        token, _, relative = unquote(urlsplit(self.path).path).lstrip('/').partition('/')
        with self.server.guard:
            root = self.server.roots.get(token)
        if root is None:
            self.send_error(404)
            return
        parts = Path(relative)
        path = root / parts
        if (not relative or parts.is_absolute() or '..' in parts.parts
                or any(root.joinpath(*parts.parts[:i]).is_symlink() for i in range(1, len(parts.parts) + 1))
                or not path.resolve().is_relative_to(root) or not path.is_file()):
            self.send_error(404)
            return
        try:
            with path.open('rb') as handle:
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(path.stat().st_size))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if path.suffix.lower() in {'.csv', '.tsv', '.json', '.xlsx', '.zip', '.fa', '.fasta'}:
                    self.send_header('Content-Disposition', 'attachment')
                self.end_headers()
                shutil.copyfileobj(handle, self.wfile)
        except (OSError, BrokenPipeError):
            return


def register(server, root):
    root = Path(root).resolve()
    with server.guard:
        token = next((key for key, value in server.roots.items() if value == root), None)
        if token is None:
            token = secrets.token_urlsafe(32)
            server.roots[token] = root
        server.roots.move_to_end(token)
        while len(server.roots) > 50:
            server.roots.popitem(last=False)
        return token


def artifact_server(root=None):
    server = ThreadingHTTPServer(('127.0.0.1', 0), ArtifactHandler)
    server.roots = OrderedDict()
    server.guard = threading.RLock()
    if root is not None:
        server.token = register(server, root)
    server.authority = '127.0.0.1:' + str(server.server_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@lru_cache(maxsize=1)
def shared_server():
    server = artifact_server()
    def close():
        server.shutdown()
        server.server_close()
    atexit.register(close)
    return server


def report_url(root):
    server = shared_server()
    token = register(server, root)
    return 'http://' + server.authority + '/' + token + '/report.html'
