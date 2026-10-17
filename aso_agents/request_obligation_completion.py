"""Bounded completion of omitted fields in an independently quoted contract."""
from copy import deepcopy
import json
import re

from agents import Agent
from jsonschema import Draft7Validator

from .goal_contract import CANONICAL_GOAL_FIELD_SCHEMAS
from .request_obligation_agent import request_obligation_agent


PANEL_OBLIGATION_SCHEMA = {
    'type': 'object',
    'required': ['applies_to_all_candidates', 'scope', 'criteria', 'quote'],
    'properties': {
        'applies_to_all_candidates': {'const': True},
        'scope': {'const': 'joint_panel'},
        'criteria': deepcopy(CANONICAL_GOAL_FIELD_SCHEMAS['candidate_panel_objectives']),
        'quote': {'type': 'string', 'minLength': 1},
    },
    'additionalProperties': False,
}


def _preserves_existing(before, after):
    if isinstance(before, dict):
        return (isinstance(after, dict) and before.keys() <= after.keys()
                and all(_preserves_existing(value, after[key]) for key, value in before.items()))
    if isinstance(before, list):
        return (isinstance(after, list) and len(before) == len(after)
                and all(_preserves_existing(a, b) for a, b in zip(before, after)))
    return type(before) is type(after) and before == after


def _added_paths(before, after, path=()):
    if isinstance(before, dict):
        return {path + (key,) for key in after if key not in before} | {
            addition for key in before
            for addition in _added_paths(before[key], after[key], path + (key,))
        }
    if isinstance(before, list):
        return {addition for index, (a, b) in enumerate(zip(before, after))
                for addition in _added_paths(a, b, path + (index,))}
    return set()


async def complete_panel_obligation(audit, source_text, evidence, run, parse):
    """Add missing contract fields once; never replace a recorded scientific value."""
    panel = audit.get('candidate_panel_ranking') if isinstance(audit, dict) else None
    receipt = {'status': 'not_needed', 'attempts': 0}
    if panel is None:
        return audit, receipt
    errors = list(Draft7Validator(PANEL_OBLIGATION_SCHEMA).iter_errors(panel))
    if not errors:
        return audit, receipt
    receipt['validation_errors'] = [
        {'path': list(error.absolute_path), 'message': error.message} for error in errors
    ]
    # Completion is narrower than reinterpreting a request: all existing
    # values, priorities and quoted evidence must survive byte-for-byte.
    if (not isinstance(panel, dict) or any(error.validator != 'required' for error in errors)
            or not isinstance(panel.get('quote'), str) or not panel['quote'].strip()
            or panel['quote'].strip() not in source_text):
        receipt['status'] = 'not_completable'
        return audit, receipt
    allowed_additions = {tuple(error.absolute_path) + (key,)
                         for error in errors for key in error.validator_value
                         if isinstance(error.instance, dict) and key not in error.instance}
    agent = Agent(
        name='ASO Obligation Contract Completion',
        instructions=(
            'Complete missing required fields in the supplied independently quoted obligation. '
            'Return only the complete panel_obligation JSON object. Preserve every existing '
            'key, value, quote, array member and order exactly. Add only missing fields required '
            'by required_contract, using original_evidence. Transcript references must be exact '
            'complete identifiers from the evidence, including supplied versions. Do not invent '
            'a reference, infer an unstated choice, relax constraints or replace ranking metrics. '
            'If the evidence cannot establish a missing field, return the original object unchanged.'
        ),
        tools=[], model=request_obligation_agent.model,
        model_settings=request_obligation_agent.model_settings,
    )
    prompt = json.dumps({'original_evidence': evidence, 'panel_obligation': panel,
                         'validation_errors': receipt['validation_errors'],
                         'required_contract': PANEL_OBLIGATION_SCHEMA}, ensure_ascii=False)
    receipt['attempts'] = 1
    try:
        response = await run(agent, input=prompt, max_steps=4)
        completed = parse(response.final_output)
    except Exception as exc:
        receipt.update(status='unresolved', error_type=type(exc).__name__)
        return audit, receipt
    if (not Draft7Validator(PANEL_OBLIGATION_SCHEMA).is_valid(completed)
            or not _preserves_existing(panel, completed)):
        receipt.update(status='unresolved', reason='invalid_completion_or_existing_value_changed')
        return audit, receipt
    if not _added_paths(panel, completed) <= allowed_additions:
        receipt.update(status='unresolved', reason='unrequested_optional_fields_added')
        return audit, receipt
    added_refs = []
    for index, after in enumerate(completed['criteria']):
        previous = panel.get('criteria', [])
        before = previous[index] if index < len(previous) else {}
        if 'reference_transcript_id' in after and 'reference_transcript_id' not in before:
            reference = after['reference_transcript_id']
            # A sentence-ending period is punctuation; a dot followed by a
            # word character continues an accession/version and must not match.
            if not re.search(r'(?<![\w.\-])' + re.escape(reference) + r'(?![\w\-]|\.\w)', source_text):
                receipt.update(status='unresolved', reason='added_reference_not_in_source_evidence')
                return audit, receipt
            added_refs.append(reference)
    result = deepcopy(audit)
    result['candidate_panel_ranking'] = completed
    receipt.update(status='completed', existing_values_preserved=True,
                   added_references=added_refs,
                   added_paths=[list(path) for path in sorted(allowed_additions, key=str)])
    return result, receipt
