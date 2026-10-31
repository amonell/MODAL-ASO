"""Verify reuse of this run's exports. No reference cache is read or distributed."""
import json
from pathlib import Path

from .artifact_io import file_sha256, json_sha256, write_json


def _paths(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _paths(child)
    elif isinstance(value, str) and value.startswith('/'):
        path = Path(value)
        if path.is_file():
            yield path


def build_identity(context, dependencies, source_root, prior=None):
    """Include absent input paths so newly appearing evidence invalidates reuse."""
    artifacts = (prior or {}).get('artifacts') or {}
    # Only this builder's own published fields are feedback, not scientific input.
    scopes = {key: {name: value for name, value in (context.get(key) or {}).items()
                   if name not in artifacts} for key in ('structured_goal', 'final_payload')}
    if isinstance(scopes['final_payload'].get('structured_goal'), dict):
        scopes['final_payload']['structured_goal'] = scopes['structured_goal']
    paths = sorted(set(Path(path) for path in dependencies), key=str)
    if not any(path.is_file() for path in paths):
        return None
    source_root = Path(source_root)
    source_paths = sorted({*source_root.glob('*.py'), *source_root.joinpath('aso_agents').rglob('*.py')})
    return {'version': 1, 'scope': 'current_run_generated_exports',
            'context_sha256': json_sha256(scopes),
            'source_sha256': json_sha256({str(path.relative_to(source_root)): file_sha256(path)
                                         for path in source_paths}),
            'inputs': {str(path): file_sha256(path) if path.is_file() else None for path in paths}}


def read_receipt(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def verified_artifacts(receipt, identity):
    if identity is None or receipt.get('identity') != identity or not receipt.get('outputs'):
        return None
    try:
        if any(not Path(path).is_file() or file_sha256(path) != expected
               for path, expected in receipt['outputs'].items()):
            return None
    except OSError:
        return None
    return receipt['artifacts']


def write_receipt(path, identity, artifacts):
    if identity is not None and artifacts:
        outputs = {str(path): file_sha256(path) for path in _paths(artifacts)}
        write_json(path, {'identity': identity, 'artifacts': artifacts, 'outputs': outputs})
