"""Structural delivery checks; factual prose is checked against source evidence.

Natural language cannot reliably distinguish search settings, observed counts,
and missing evidence by matching phrases. The mandatory native consistency
review checks those claims. This layer checks citations and observation identity.
"""
from .input_rows import assessment_coverage_errors
from .evaluation_evidence import assessment_measurement_errors
from .citation_tokens import citation_placeholders



def response_contract_errors(response, payload):
    if not isinstance(response, dict):
        return ['Expected one JSON object.']
    if response.get('error'):
        return [str(response.get('error')) + ': ' + str(response.get('reason') or '')]
    if payload.get('task_type') != 'evaluate_existing_aso_set':
        errors = [] if str(response.get('markdown') or '').strip() else ['The requested response markdown is missing.']
        # Composite responses have no measurement binder. A context identifier in
        # template syntax is therefore an unresolved citation, even when the
        # structured assessment can be rendered as Markdown.
        def unresolved(value):
            if isinstance(value, dict):
                return any(unresolved(item) for key, item in value.items()
                           if key not in {'completed_tasks', 'completed_task_context'})
            if isinstance(value, list):
                return any(unresolved(item) for item in value)
            return bool(citation_placeholders(value))
        if unresolved(response):
            errors.append('Unresolved citation placeholder in the response.')
        return errors
    errors = assessment_measurement_errors(response, payload.get('evaluation_summary') or {})
    if not isinstance(response.get('evaluation_assessments'), list) or not response['evaluation_assessments']:
        errors.append('Provide the requested per-molecule evaluation_assessments.')
    else:
        rows = (payload.get('evaluation_summary') or {}).get('provided_asos') or []
        if rows:
            errors.extend(assessment_coverage_errors(response['evaluation_assessments'], rows))
    return list(dict.fromkeys(errors))
