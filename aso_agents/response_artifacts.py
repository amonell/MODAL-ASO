"""Run-local, complete report provenance and reuse of accepted components."""
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


@lru_cache(maxsize=1)
def implementation_identity():
    directory = Path(__file__).parent
    names = ('pipeline_tools.py', 'response_artifacts.py', 'response_contracts.py',
             'response_context.py', 'response_assembly.py', 'evaluation_response_audit.py',
             'evaluation_evidence.py', 'evaluation_report.py', 'workbench_response_agent.py',
             'llm_provider.py', 'report_recovery.py')
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def report_configuration(agents):
    from .llm_provider import get_provider_config
    config = get_provider_config()
    # Explicit allowlist: credentials and request headers are never serialized.
    provider = {key: getattr(config, key) for key in (
        'name', 'base_url', 'continuation_mode', 'provider_preferences',
        'model_override', 'reasoning_effort_override', 'max_output_tokens_cap',
        'structured_output_mode')}
    configurations = []
    for agent in agents:
        settings = agent.model_settings
        configurations.append({'name': agent.name, 'model': str(agent.model),
            'resolved_model': config.resolve_model(str(agent.model)),
            'instructions': agent.instructions,
            'model_settings': asdict(settings) if is_dataclass(settings) else vars(settings)})
    return {'provider': provider, 'agents': configurations, 'implementation': implementation_identity()}


class ResponseArtifacts:
    def __init__(self, root, payload, configuration):
        self.key = digest({'payload': payload, 'configuration': configuration})
        self.root = Path(root) / 'components' / self.key
        self.attempt = self.root / ('attempt-' + str(time.time_ns()) + '-' + uuid.uuid4().hex[:8])
        self.attempt.mkdir(parents=True)
        self.record('source', payload)
        self.record('configuration', configuration)

    def record(self, name, value):
        write_json(self.attempt / (name + '.json'), value)

    def accepted(self):
        path = self.root / 'accepted.json'
        if not path.is_file():
            return None
        saved = json.loads(path.read_text())
        if saved.get('component_key') != self.key or digest(saved.get('response')) != saved.get('response_sha256'):
            raise ValueError('Accepted report component has invalid provenance: ' + str(path))
        self.record('reused', {'accepted_path': str(path), 'response_sha256': saved['response_sha256']})
        return saved['response']

    def accept(self, response):
        self.record('accepted-response', response)
        write_json(self.root / 'accepted.json', {'component_key': self.key, 'response': response,
            'response_sha256': digest(response), 'attempt_path': str(self.attempt)})
