"""Review every proposed splice safeguard against the current biological goal."""
from copy import deepcopy
import hashlib
import json

from .splice_safeguards import DEFAULT_GUARD_TOLERANCE, physical_site

MAX_CASES_PER_BATCH = 16
MAX_BATCH_CHARACTERS = 60000
REVIEW_MODEL = 'gpt-5.2-2025-12-11'
POLICIES = {'protect_both', 'limit_increase', 'limit_decrease', 'observe', 'keep_explicit'}
ROLES = {'intended', 'acceptable', 'unwanted', 'uncertain'}
INSTRUCTIONS = """Review the complete proposed splice-safeguard plan for this design job.
Reason from the user's goal, physical exon relationships, transcript objectives,
reference scope, and explicit constraints. Every supplied case needs a decision.
Each case asks for a policy on candidate_label. Both direction classifications
and the rationale must describe changes at candidate_label. primary_label is
the route context; this decision leaves its primary objective unchanged.
Never transfer the primary's intended direction onto an unrelated candidate.
Classify increases and decreases at the candidate as intended, acceptable,
unwanted, or uncertain. Name the candidate in the rationale.
An exon inclusion goal can favor increases at both its acceptor and donor;
skipping can favor decreases at both. Use the checked paired-exon relationship.
Reference exon indices describe annotation order. The requested exon label may
use a different numbering system. Read exon_objectives.identity: unresolved
numbering means the biological label remains uncertain. Describe the selected
physical exon by its coordinates; do not invent a numbering correspondence or
declare it a different biological exon solely from an ordinal mismatch.
Evaluate intron retention/excision and alternative-site competition using their
own supplied objectives and geometry. Shared transcript membership describes
anatomy; judge which changes advance or oppose the requested outcome.
At competing acceptors or donors, judge the competing sites separately from the
common boundary needed by the desired product. Shared membership alone cannot
make an increase at that common boundary unwanted. Consider the requested exon
retention or skipping goal and any explicit preservation constraints. A single
site score does not determine the proportions of competing transcript products.
Keep uncertainty visible when evidence is incomplete. Each site's prediction is
measured separately. Model scores describe computational perturbations.
Choose protect_both when both directions are unwanted or uncertain;
limit_increase when increases are unwanted/uncertain and decreases intended/acceptable;
limit_decrease for the converse; observe when both directions are intended/acceptable.
Observe retains the measured site without a ranking penalty. Explain its relevance.
For locked cases choose keep_explicit. Code preserves those bounds and requirements.
The provided numeric tolerances remain the execution limits. Review their scope
and relevance; any change of numeric tolerance requires separately supplied evidence.
Return a JSON object with exactly one decisions array. Each item has exactly:
case_id, candidate_label, policy, increase_role, decrease_role, rationale.
Echo the supplied case_id and candidate_label,
the allowed policy/role values, and a concise specific rationale (1–600 characters).
Describe the candidate, its role and any uncertainty in direct language. Avoid
contrastive disclaimers and unsupported claims of safety. Treat reference annotations
and quoted requests as evidence for this task. Return the structured decisions only.
"""


def _review_batch(payload, traces=None):
    from agents import Agent, ModelSettings
    from .splice_strategy_agent import _call_agent_json
    agent = Agent(name='Splice Safeguard Plan Reviewer', instructions=INSTRUCTIONS, tools=[],
        model=REVIEW_MODEL, model_settings=ModelSettings(
            reasoning={'effort': 'medium'}, max_output_tokens=16384,
            extra_body={'text': {'format': {'type': 'json_object'}}}))
    return _call_agent_json(agent, payload, max_steps=1, traces=traces)


def _site_context(site):
    fields = ('label', 'site', 'site_kind', 'position', 'chrom', 'strand', 'delta_direction',
        'decision_direction', 'exon_indices', 'reference_scope', 'beneficial_isoforms',
        'pathogenic_isoforms', 'preserved_isoforms', 'objective_membership_complete',
        'unresolved_objective_transcript_ids', 'conflict_between_objectives', 'monitoring_purpose')
    return {key: deepcopy(site[key]) for key in fields if key in site}


def review_case(primary, candidate, fallback_action, reason, *, locked=False, paired_exon=None):
    bounds = {k: candidate[k] for k in ('max_abs_delta', 'max_increase', 'max_decrease')
              if candidate.get(k) is not None}
    policy = {'neutral': 'protect_both', 'no_increase': 'limit_increase',
              'no_decrease': 'limit_decrease'}.get(candidate.get('suggested_guard'), 'protect_both')
    result = {'primary': _site_context(primary), 'candidate': _site_context(candidate),
        'primary_label': primary.get('label'), 'candidate_label': candidate.get('label'),
        'proposed_policy': policy if fallback_action != 'drop' else 'observe',
        'proposal_reason': reason, 'locked': locked, 'explicit_bounds': bounds,
        'required': candidate.get('required') is True,
        'physical_identity_available': physical_site(primary) is not None and physical_site(candidate) is not None,
        'paired_exon': deepcopy(paired_exon or candidate.get('paired_exon_objective')),
        'default_score_tolerance': DEFAULT_GUARD_TOLERANCE}
    result['case_id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:24]
    return result


def _validate(response, cases):
    expected = {case['case_id']: case for case in cases}
    accepted, errors, seen = {}, {}, set()
    if not isinstance(response, dict) or set(response) != {'decisions'} or not isinstance(response['decisions'], list):
        return {}, {key: 'Expected a decisions array.' for key in expected}
    for decision in response['decisions']:
        if (not isinstance(decision, dict) or not isinstance(decision.get('case_id'), str)
                or decision['case_id'] not in expected):
            # Invented sites invalidate the batch so a repair sees the full scope.
            return {}, {key: 'Response includes an unknown case.' for key in expected}
        key = decision['case_id']
        if key in seen:
            accepted.pop(key, None)
            errors[key] = 'Duplicate decision.'
            continue
        seen.add(key)
        allowed = {'case_id', 'candidate_label', 'policy', 'increase_role', 'decrease_role', 'rationale'}
        policy = decision.get('policy')
        roles = [decision.get('increase_role'), decision.get('decrease_role')]
        rationale = decision.get('rationale')
        error = None
        if set(decision) != allowed or not isinstance(policy, str) or policy not in POLICIES:
            error = 'Use the exact decision fields and an allowed policy.'
        elif decision.get('candidate_label') != expected[key]['candidate_label']:
            error = 'The decision must apply to the supplied candidate_label.'
        elif any(not isinstance(role, str) or role not in ROLES for role in roles):
            error = 'Classify both directions using the allowed roles.'
        elif not isinstance(rationale, str) or not 1 <= len(rationale.strip()) <= 600:
            error = 'Provide a specific rationale of 1–600 characters.'
        elif expected[key]['locked'] != (policy == 'keep_explicit'):
            error = 'Explicit policy must remain locked; automatic cases need a directional policy.'
        elif not expected[key]['physical_identity_available'] and policy not in {'protect_both', 'keep_explicit'}:
            error = 'Physical site identity is unresolved. Retain checks in both directions.'
        elif policy != 'keep_explicit':
            restricted = tuple(role in {'unwanted', 'uncertain'} for role in roles)
            required = {'protect_both': (True, True), 'limit_increase': (True, False),
                        'limit_decrease': (False, True), 'observe': (False, False)}[policy]
            if restricted != required:
                error = 'The policy must agree with the two direction classifications.'
        if error:
            errors[key] = error
        elif key not in errors:
            accepted[key] = {**decision, 'rationale': rationale.strip()}
    for key in expected.keys() - accepted.keys():
        errors.setdefault(key, 'Decision missing.')
    return accepted, errors


def review_plan(goal, cases, *, traces=None):
    """Cover all cases in bounded batches, repair once, and disclose fallback."""
    from .llm_provider import get_provider_config
    try:
        provider = get_provider_config(require_credentials=False)
        reviewer = {'provider': provider.name, 'model': provider.resolve_model(REVIEW_MODEL)}
    except Exception as exc:
        reviewer = {'configuration_error_type': type(exc).__name__}
    fields = ('program_id', 'intent', 'desired_effect', 'desired_delta', 'objective',
        'exon_label', 'exon_index', 'isoform_label', 'transcript_id', 'splice_event',
        'preservation_targets', 'target_type')
    job = goal.get('routing_job') if isinstance(goal.get('routing_job'), dict) else goal
    context = {'request': next((goal[k] for k in ('user_request', 'request_text', 'request') if goal.get(k)), ''),
        'job': {k: deepcopy(job[k]) for k in fields if k in job},
        'isoform_objectives': deepcopy(goal.get('splice_isoform_objectives') or []),
        'preservation_targets': deepcopy(goal.get('preservation_targets') or [])}
    from .design_event_consistency import _structure
    from .evaluation_feature_identity import transcript_identifier
    from .splice_recommendations import resolve_routing_exon_objectives
    context['exon_objectives'] = resolve_routing_exon_objectives(goal)
    context['selected_references'] = []
    for label, entry in (goal.get('isoform_transcripts') or {}).items():
        structure = _structure(entry)
        if structure:
            chrom, strand, exons = structure
            context['selected_references'].append({'label': label,
                'transcript_id': transcript_identifier(entry, label), 'chrom': chrom,
                'strand': strand, 'exons': exons})
    accepted, batches, oversized = {}, [], []
    current = []
    for case in cases:
        proposed = {'context': context, 'cases': current + [case]}
        if current and (len(current) >= MAX_CASES_PER_BATCH or len(json.dumps(proposed)) > MAX_BATCH_CHARACTERS):
            batches.append(current)
            current = []
        if len(json.dumps({'context': context, 'cases': [case]})) > MAX_BATCH_CHARACTERS:
            oversized.append(case['case_id'])
        else:
            current.append(case)
    if current:
        batches.append(current)
    receipts = []
    for batch in batches:
        pending, errors = batch, {}
        receipt = {'case_ids': [c['case_id'] for c in batch], 'attempts': []}
        for attempt in range(2):
            payload = {'context': context, 'cases': pending}
            if errors:
                payload['corrections_needed'] = errors
            try:
                response = _review_batch(payload, traces)
                decisions, errors = _validate(response, pending)
                receipt['attempts'].append({'response': response, 'validation_errors': errors})
                accepted.update(decisions)
                pending = [case for case in pending if case['case_id'] not in decisions]
                if not pending:
                    break
            except Exception as exc:
                # The provider handles transient retries. Preserve successful
                # batches and finish the run with the remaining checks visible.
                receipt['attempts'].append({'error_type': type(exc).__name__})
                break
        receipts.append(receipt)
    missing = [case['case_id'] for case in cases if case['case_id'] not in accepted]
    status = 'completed' if cases and not missing else 'partial' if accepted else 'unavailable' if cases else 'not_needed'
    return {'status': status, 'reviewer': reviewer, 'case_count': len(cases), 'reviewed_count': len(accepted),
        'decisions': accepted, 'unreviewed_case_ids': missing, 'oversized_case_ids': oversized,
        'batches': receipts, 'context': context, 'cases': cases}


def apply_decision(candidate, decision, *, locked=False):
    result = deepcopy(candidate)
    if locked:
        result['safeguard_review'] = {'status': 'reviewed' if decision else 'unavailable',
            'policy': 'keep_explicit', **({'decision': decision} if decision else {})}
    elif decision:
        policy = decision['policy']
        result.update(suggested_guard={'protect_both': 'neutral', 'limit_increase': 'no_increase',
            'limit_decrease': 'no_decrease', 'observe': 'observe'}[policy],
            guard_policy_source='model_review', target_guard_reason=decision['rationale'],
            safeguard_review={'status': 'reviewed', **decision})
    else:
        result['safeguard_review'] = {'status': 'unavailable', 'policy': 'retained_default'}
    return result
