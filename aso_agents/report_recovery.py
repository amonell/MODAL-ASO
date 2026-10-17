"""Resume report delivery from recorded evidence without rerunning analysis.

A retained draft is never treated as accepted across changed implementation:
it receives the current citation/coverage checks and a fresh native audit.
"""
from copy import deepcopy
import json
from pathlib import Path

from .response_artifacts import digest, report_configuration, write_json
from .response_context import evaluation_response_batches


def _assembly_scientific_projection(response):
    """Remove only the native assembly stage's permitted narrative edit fields."""
    from .response_assembly import delivered_narrative_fields
    projected = deepcopy(response)
    projected.pop('native_assembly_review', None)
    for path in delivered_narrative_fields(response):
        tokens = [token.replace('~1', '/').replace('~0', '~') for token in path[1:].split('/')]
        parent = projected
        for token in tokens[:-1]:
            parent = parent[int(token)] if isinstance(parent, list) else parent[token]
        parent[int(tokens[-1]) if isinstance(parent, list) else tokens[-1]] = None
    return projected


def retained_assembled_report(assembled, history_dir):
    """Reuse prior narrative edits only from an attested assembly of these components.

    The result is a draft for a fresh assembly review, never a current acceptance.
    Scientific component acceptance is verified separately by the caller.
    """
    expected = digest(assembled)
    paths = sorted((Path(history_dir) / 'assembly').glob('components/*/accepted.json'),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in paths:
        saved = json.loads(path.read_text())
        attempt = path.parent / Path(saved.get('attempt_path') or '').name
        source = json.loads((attempt / 'source.json').read_text())
        if digest(source.get('assembled_response')) != expected:
            continue
        config = json.loads((attempt / 'configuration.json').read_text())
        response = saved.get('response')
        if (not isinstance(response, dict) or digest(response) != saved.get('response_sha256')
                or saved.get('component_key') != path.parent.name
                or saved.get('component_key') != digest({'payload': source, 'configuration': config})):
            raise ValueError('Retained assembly provenance is invalid.')
        receipt = response.get('native_assembly_review') or {}
        if receipt.get('method') == 'deterministic_scoped_component_assembly':
            # Rebuild inexpensive labels from the separately attested components.
            # This receipt does not claim a free-form native assembly review.
            continue
        draft = {key: value for key, value in response.items() if key != 'native_assembly_review'}
        if (receipt.get('status') != 'no_request_scope_contradiction_flagged'
                or receipt.get('source_sha256') != digest(source.get('assembly_source'))
                or receipt.get('original_draft_sha256') != expected
                or receipt.get('accepted_draft_sha256') != digest(draft)
                or _assembly_scientific_projection(draft) != _assembly_scientific_projection(assembled)):
            raise ValueError('Retained assembly changes or fails to attest the accepted scientific components.')
        return {'draft': deepcopy(draft), 'source_path': str(path), 'draft_sha256': digest(draft),
                'policy': 'Previous assembly narrative edits retained as a draft; current native assembly review remains mandatory.'}
    return None


def retained_accepted_components(payload, history_dir, configuration):
    """Reuse scientific reviews only when their entire dependency identity matches.

    Assembly and recovery orchestration are outside the component science review.
    Every other implementation file, the exact source, agents and provider must
    match, and both the accepted artifact and native review receipt are checked.
    """
    from .response_contracts import response_contract_errors
    batches = evaluation_response_batches(payload)
    if len(batches) < 2:
        raise ValueError('Assembly-only recovery requires a multi-component report.')
    wanted = {digest(batch): batch for batch in batches}
    candidates = {}
    assembly_files = {'response_assembly.py', 'report_recovery.py'}
    for path in sorted(Path(history_dir).glob('components/*/attempt-*/source.json')):
        source = json.loads(path.read_text())
        key = digest(source)
        if key not in wanted:
            continue
        config = json.loads((path.parent / 'configuration.json').read_text())
        old_identity, current_identity = config.get('implementation'), configuration.get('implementation')
        if (not isinstance(old_identity, dict) or not isinstance(current_identity, dict)
                or old_identity.keys() != current_identity.keys()
                or not (old_identity.keys() - assembly_files)
                or any(old_identity[name] != current_identity[name]
                       for name in old_identity.keys() - assembly_files)
                or any(config.get(k) != configuration.get(k) for k in ('provider', 'agents'))):
            raise ValueError('Scientific review dependencies changed; a fresh component review is required.')
        accepted_path = path.parent.parent / 'accepted.json'
        if not accepted_path.is_file():
            continue
        accepted = json.loads(accepted_path.read_text())
        response = accepted.get('response')
        expected_key = digest({'payload': source, 'configuration': config})
        if (accepted.get('component_key') != expected_key or path.parent.parent.name != expected_key
                or digest(response) != accepted.get('response_sha256')):
            raise ValueError('Accepted component provenance is invalid.')
        receipt = response.get('native_consistency_review') or {}
        draft = {k: v for k, v in response.items() if k != 'native_consistency_review'}
        if (receipt.get('status') != 'no_factual_contradiction_flagged'
                or receipt.get('source_sha256') != key
                or receipt.get('draft_sha256') != digest(draft)
                or response_contract_errors(response, source)):
            raise ValueError('Accepted component lacks a valid matching scientific review.')
        candidates[key] = {'response': response, 'source_path': str(accepted_path),
            'response_sha256': digest(response), 'configuration_sha256': digest(config)}
    if wanted.keys() - candidates.keys():
        raise ValueError('No accepted evidence-matched scientific review for every component.')
    return candidates


def retained_report_drafts(payload, history_dir, configuration):
    """Recover only drafts with the exact same evidence and model configuration."""
    wanted = {digest(batch) for batch in evaluation_response_batches(payload)}
    candidates = {}
    for path in sorted(Path(history_dir).glob('components/*/attempt-*/source.json')):
        source = json.loads(path.read_text())
        key = digest(source)
        if key not in wanted:
            continue
        old_config = json.loads((path.parent / 'configuration.json').read_text())
        if any(old_config.get(k) != configuration.get(k) for k in ('provider', 'agents')):
            raise ValueError('Report recovery model or instructions differ from the saved draft.')
        accepted = path.parent / 'accepted-response.json'
        audited = sorted(path.parent.glob('round-*-audited-draft.json'),
                         key=lambda p: int(p.name.split('-')[1]))
        selected = accepted if accepted.is_file() else (audited[-1] if audited else None)
        if selected:
            candidate = json.loads(selected.read_text())
            candidate.pop('native_consistency_review', None)
            findings = []
            if selected != accepted:
                audit_path = selected.with_name(selected.name.replace('-audited-draft.json', '-audit.json'))
                if audit_path.is_file():
                    findings = json.loads(audit_path.read_text()).get('findings') or []
            candidates[key] = {'draft': candidate, 'source_path': str(selected),
                               'draft_sha256': digest(candidate), 'unresolved_findings': findings}
    missing = wanted - candidates.keys()
    if missing:
        raise ValueError('No evidence-matched retained report draft for every component.')
    return candidates


async def recover_existing_aso_report(*, response_input_path, computed_evaluation_path,
                                      previous_history, output_run_dir, assembly_only=False,
                                      retain_drafts=True):
    """Native report continuation; previous artifacts remain untouched."""
    from . import pipeline_tools as p
    from .evaluation_evidence import bind_assessment_measurements
    from .workbench_response_agent import (
        workbench_response_agent, evaluation_evidence_audit_agent, evaluation_evidence_repair_agent)
    if assembly_only and not retain_drafts:
        raise ValueError('Assembly-only recovery requires retained accepted components.')
    source = json.loads(Path(response_input_path).read_text())
    computed = json.loads(Path(computed_evaluation_path).read_text())
    if source.get('task_type') != 'evaluate_existing_aso_set':
        raise ValueError('This report recovery entrypoint requires a supplied-ASO evaluation.')
    compact = p._compact_existing_aso_evaluation_for_response(deepcopy(computed))
    if digest(compact) != digest(source.get('evaluation_summary')):
        raise ValueError('Computed evaluation does not reproduce the saved response evidence.')
    output = Path(output_run_dir)
    history = output / 'response_history'
    if output.resolve() == Path(previous_history).resolve() or history.resolve() == Path(previous_history).resolve():
        raise ValueError('Report recovery requires a separate output directory.')
    if output.exists():
        raise ValueError('Report recovery requires a new output directory.')
    configuration = report_configuration((workbench_response_agent, evaluation_evidence_audit_agent,
                                          evaluation_evidence_repair_agent))
    retained = (retained_accepted_components(source, previous_history, configuration) if assembly_only
                else retained_report_drafts(source, previous_history, configuration) if retain_drafts
                else {})
    output.mkdir(parents=True)
    write_json(history / 'computed_evaluation.json', computed)
    write_json(history / 'recovery-provenance.json', {
        'previous_history': str(previous_history), 'source_sha256': digest(source),
        'computed_evaluation_sha256': digest(computed),
        'retained_drafts': {key: {k:v for k,v in item.items() if k not in {'draft', 'response'}} for key,item in retained.items()},
        'assembly_only': assembly_only,
        'retain_drafts': retain_drafts,
        'scientific_analysis_recomputed': False,
        'policy': ('Reuse only source-, configuration-, implementation- and receipt-verified accepted scientific components; review the new assembly.' if assembly_only else
            'Retain prior drafts and scientific evidence; revalidate every draft under the current implementation before acceptance. No mapping, scoring, intake or user simulation is repeated.' if retain_drafts else
            'Regenerate report components from verified unchanged scientific evidence using the current context representation and native reviews. No previous draft or acceptance is reused. No mapping, scoring, intake or user simulation is repeated.'),
    })
    if assembly_only:
        from .response_context import merge_evaluation_responses
        from .response_assembly import review_assembled_response
        write_json(history / ('response-input-' + digest(source) + '.json'), source)
        components = [retained[digest(batch)]['response'] for batch in evaluation_response_batches(source)]
        assembled = merge_evaluation_responses(components, source)
        # Original component science is already attested above. Rebuild labeled
        # sections directly; prior free-form assembly edits are not scientific
        # inputs and may have lost the original observation/search-stage scope.
        response = await review_assembled_response(source, assembled,
            artifact_dir=history / 'assembly')
    else:
        response = await p._run_workbench_response_agent_once(source, artifact_dir=history,
            initial_drafts={key:item['draft'] for key,item in retained.items()},
            initial_findings={key:item['unresolved_findings'] for key,item in retained.items()})
    response = bind_assessment_measurements(response, compact)
    markdown = str(response.get('markdown') or response.get('answer_summary') or '').strip()
    orientation = p._provided_aso_orientation_markdown(computed)
    if orientation:
        markdown = '\n\n'.join(part for part in (orientation, markdown) if part)
    final = {**response, 'evaluation_summary': computed, 'response_evaluation_summary': compact,
             'markdown': markdown, 'report_recovery': {'previous_history': str(previous_history),
                'scientific_analysis_recomputed': False, 'retained_drafts_reaudited': retain_drafts and not assembly_only,
                'components_regenerated': not retain_drafts,
                'accepted_scientific_reviews_reused': assembly_only}}
    artifacts = p._write_v2_response_outputs(run_dir=str(output), markdown_text=markdown,
        response_payload=final, response_basename='existing_aso_evaluation', title='ASO Evaluation',
        requested_outputs=(source.get('request_envelope') or {}).get('final_outputs'))
    artifacts['response_history'] = str(history)
    write_json(output / 'report_recovery_result.json', {'status': 'report_ready', 'artifacts': artifacts})
    return {'status': 'report_ready', 'artifacts': artifacts, 'run_dir': str(output)}
