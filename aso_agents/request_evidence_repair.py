"""Bounded evidence repair, including retraction of unrequested optional audits."""
from copy import deepcopy
import json

from agents import Agent
from .goal_intake_agent import GOAL_INTAKE_MODEL, GOAL_INTAKE_SETTINGS


# These audit objects are nullable in the request-obligation protocol. Lists
# and partially supported objects cannot be removed by quote repair.
_NULLABLE_AUDIT_FIELDS = frozenset({
    'candidate_count', 'per_site_count', 'supplied_observation_count',
    'candidate_site_diversity', 'candidate_genomic_spacing',
    'candidate_panel_ranking', 'candidate_role_ranking', 'target_feature',
    'target_window', 'target_accessibility', 'shared_binding_cohort',
})


evidence_quote_repair_agent = Agent(
    name='ASO Evidence Quote Repair',
    instructions=(
        'Repair only the listed invalid evidence quotes in a request-obligation audit. '
        'The original request and clarification answers are the only sources. Each replacement '
        'must be one contiguous verbatim substring that actually supports the existing extracted '
        'requirement, including its scope, quantity, identity and qualifiers. Do not change '
        'extracted values, splice fragments, invent words or use ellipses. An audit can also '
        'invent an optional obligation that the user never requested. Only for a path in '
        'eligible_retractions, you may retract that entire object to null after checking all '
        'user evidence and finding that obligation was not requested. Do not retract a real '
        'but underspecified requirement or one whose quotation merely needs repair. '
        'Return JSON {"repairs":[{"path":string,"quote":string}],'
        '"retractions":[{"path":string,"reason":"not_requested"}],'
        '"unresolved_paths":[string]}. Use exactly the provided JSON-pointer paths. '
        'Every invalid quote must be repaired, covered by an eligible retraction, or listed '
        'as unresolved. Never repair and retract the same object. If the user requests the '
        'obligation but its extracted details cannot be supported, leave it unresolved '
        'instead of removing it or substituting unrelated text. No tools or scientific analysis.'
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


def _eligible_retractions(audit, source):
    eligible = {}
    for name, value in audit.items():
        if name not in _NULLABLE_AUDIT_FIELDS or not isinstance(value, dict):
            continue
        pointer = '/' + name
        invalid = {item['path'] for item in invalid_quote_paths(value, source, pointer)}
        # With empty source every quote is invalid. Equality means no verified
        # quote, including a nested reference quote, would be discarded.
        all_quotes = {item['path'] for item in invalid_quote_paths(value, '', pointer)}
        if invalid and invalid == all_quotes:
            eligible[pointer] = invalid
    return eligible


async def repair_audit_quotes(audit, source, evidence, run, parse):
    invalid = invalid_quote_paths(audit, source)
    if not invalid:
        return audit, {'status': 'not_required', 'model_calls': 0}
    eligible = _eligible_retractions(audit, source)
    receipt = {'status': 'unresolved', 'model_calls': 1, 'invalid_quotes': invalid,
               'eligible_retractions': sorted(eligible)}
    try:
        response = await run(evidence_quote_repair_agent, input=json.dumps({
            'user_evidence': evidence, 'audit': audit, 'invalid_quotes': invalid,
            'eligible_retractions': sorted(eligible),
        }, ensure_ascii=False), max_steps=2)
        proposed = parse(response.final_output)
        if (not isinstance(proposed, dict) or proposed.get('unresolved_paths')
                or set(proposed) - {'repairs', 'retractions', 'unresolved_paths'}):
            return audit, receipt
        repairs = proposed.get('repairs')
        retractions = proposed.get('retractions', [])
        paths = {item['path'] for item in invalid}
        if (not isinstance(repairs, list) or not isinstance(retractions, list)
                or any(not isinstance(item, dict) or set(item) != {'path', 'quote'}
                       or not isinstance(item['path'], str) for item in repairs)
                or any(not isinstance(item, dict) or set(item) != {'path', 'reason'}
                       or not isinstance(item['path'], str) or item['path'] not in eligible
                       or item['reason'] != 'not_requested' for item in retractions)):
            return audit, receipt
        repaired_paths = {item['path'] for item in repairs}
        retracted_paths = {item['path'] for item in retractions}
        covered = set().union(*(eligible[path] for path in retracted_paths))
        if (len(repaired_paths) != len(repairs) or len(retracted_paths) != len(retractions)
                or repaired_paths & covered or repaired_paths | covered != paths):
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
        for item in retractions:
            repaired[item['path'][1:]] = None
        if invalid_quote_paths(repaired, source):
            return audit, receipt
        receipt.update(status='repaired', repairs=repairs, retractions=retractions)
        return repaired, receipt
    except Exception as exc:
        receipt['error_type'] = type(exc).__name__
        return audit, receipt
