"""Bounded repair of unsupported quote strings, never extracted requirements."""
from copy import deepcopy
import json

from agents import Agent
from .goal_intake_agent import GOAL_INTAKE_MODEL, GOAL_INTAKE_SETTINGS


evidence_quote_repair_agent = Agent(
    name='ASO Evidence Quote Repair',
    instructions=(
        'Repair only the listed invalid evidence quotes in a request-obligation audit. '
        'The original request and clarification answers are the only sources. Each replacement '
        'must be one contiguous verbatim substring that actually supports the existing extracted '
        'requirement, including its scope, quantity, identity and qualifiers. Do not add or remove '
        'requirements, change extracted values, splice fragments, invent words or use ellipses. '
        'Return JSON {"repairs":[{"path":string,"quote":string}],"unresolved_paths":[string]}. '
        'Use exactly the provided JSON-pointer paths. If no supporting quote exists, list that '
        'path as unresolved instead of substituting unrelated text. No tools or scientific analysis.'
    ),
    tools=[], model=GOAL_INTAKE_MODEL, model_settings=GOAL_INTAKE_SETTINGS,
)


def invalid_quote_paths(value, source, path=''):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            pointer = path + '/' + str(key).replace('~', '~0').replace('/', '~1')
            if key in {'quote', 'references_quote'}:
                if (not isinstance(child, str) or not child.strip()
                        or child.strip().casefold() not in source.casefold()):
                    result.append({'path': pointer, 'quote': child})
            else:
                result.extend(invalid_quote_paths(child, source, pointer))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(invalid_quote_paths(child, source, path + '/' + str(index)))
    return result


async def repair_audit_quotes(audit, source, evidence, run, parse):
    invalid = invalid_quote_paths(audit, source)
    if not invalid:
        return audit, {'status': 'not_required', 'model_calls': 0}
    receipt = {'status': 'unresolved', 'model_calls': 1, 'invalid_quotes': invalid}
    try:
        response = await run(evidence_quote_repair_agent, input=json.dumps({
            'user_evidence': evidence, 'audit': audit, 'invalid_quotes': invalid,
        }, ensure_ascii=False), max_steps=2)
        proposed = parse(response.final_output)
        if not isinstance(proposed, dict) or proposed.get('unresolved_paths'):
            return audit, receipt
        repairs = proposed.get('repairs')
        paths = {item['path'] for item in invalid}
        if (not isinstance(repairs, list) or len(repairs) != len(paths)
                or any(not isinstance(item, dict) or set(item) != {'path', 'quote'} for item in repairs)
                or {item['path'] for item in repairs} != paths):
            return audit, receipt
        repaired = deepcopy(audit)
        for item in repairs:
            quote = item['quote']
            if not isinstance(quote, str) or not quote.strip() or quote.strip().casefold() not in source.casefold():
                return audit, receipt
            keys = [key.replace('~1', '/').replace('~0', '~') for key in item['path'].split('/')[1:]]
            parent = repaired
            for key in keys[:-1]:
                parent = parent[int(key)] if isinstance(parent, list) else parent[key]
            parent[keys[-1]] = quote.strip()
        receipt.update(status='repaired', repairs=repairs)
        return repaired, receipt
    except Exception as exc:
        receipt['error_type'] = type(exc).__name__
        return audit, receipt
