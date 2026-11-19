"""Stream registered run outputs through the browser UI's own origin."""
from collections import OrderedDict
from html import escape
from pathlib import Path
import secrets
import threading


ROUTE = "_modal_aso_artifacts"
DOWNLOAD_TYPES = {".csv", ".tsv", ".json", ".xlsx", ".zip", ".fa", ".fasta"}
_roots: OrderedDict[str, Path] = OrderedDict()
_guard = threading.Lock()


def register(root):
    root = Path(root).resolve()
    with _guard:
        token = next((key for key, value in _roots.items() if value == root), None)
        if token is None:
            token = secrets.token_urlsafe(32)
            _roots[token] = root
        _roots.move_to_end(token)
        while len(_roots) > 50:
            _roots.popitem(last=False)
        return token


def artifact_path(token, relative):
    with _guard:
        root = _roots.get(token)
    parts = Path(relative)
    if root is None or not parts.parts or parts.is_absolute() or any(part.startswith(".") for part in parts.parts):
        return None
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            return None
    try:
        return path if path.resolve().is_relative_to(root) and path.is_file() else None
    except (OSError, ValueError):
        return None


def serve_artifact(request):
    from starlette.responses import FileResponse, Response

    path = artifact_path(request.path_params["token"], request.path_params["relative"])
    if path is None:
        return Response(status_code=404)
    return FileResponse(
        path,
        filename=path.name if path.suffix.lower() in DOWNLOAD_TYPES else None,
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox allow-scripts allow-downloads allow-popups; frame-ancestors 'self'",
        },
    )


def route_prefix(base_url_path):
    return "/" + "/".join(part for part in (base_url_path.strip("/"), ROUTE) if part)


def report_frame(root, base_url_path):
    source = escape(f"{route_prefix(base_url_path)}/{register(root)}/report.html", quote=True)
    return (f'<iframe title="Run report" src="{source}" '
            'style="display:block;border:0;width:100%;height:100vh"></iframe>')
