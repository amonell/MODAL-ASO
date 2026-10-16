"""Preserve scope when assembling independently reviewed report components.

Attested scientific components retain their complete text in labeled sections.
Legacy unattested assembly inputs still use the restricted native scope review.
Neither path replaces the complete scientific evidence review of a component.
"""
from copy import deepcopy
import asyncio
import html
import json

from agents import Agent

from .evaluation_response_audit import (
    narrative_fields, pointer_value, response_audit_schema,
    response_audit_contract_errors, response_audit_repair_findings,
)
from .response_artifacts import ResponseArtifacts, digest, report_configuration
from .workbench_response_agent import WORKBENCH_RESPONSE_MODEL, WORKBENCH_RESPONSE_SETTINGS


assembly_audit_agent = Agent(
    name="ASO Report Assembly Scope Review",
    instructions=(
        "Check only request and delivery scope in draft_narrative_fields against source. "
        "Keys identify exact JSON Pointers in the delivered report. Source contains the "
        "the observation register and the actual delivered assessment "
        "register after component assembly, with recorded evidence availability and the "
        "final rendering contract. Check claims about supplied or assessed observations, evidence "
        "availability, panel membership, completeness, and whole-request versus component "
        "scope. A component's input subset is not the entire user's request. Respect "
        "explicitly scoped statements and scientific uncertainty. Scientific evidence "
        "was reviewed separately; only its availability metadata is present here: do not judge mapping, "
        "mechanism, citations, ranking quality, efficacy, or missing experiments. A "
        "statement that a comparison remains indeterminate is not a missing assessment. "
        "Scientific assessment fields are immutable here. Return the supplied findings "
        "schema with the applicable typed scope_kind. Each finding must cite existing pointers "
        "inside source and quote a unique exact substring of the named narrative field. "
        "Quote the smallest self-contained scope claim that needs replacement; preserve "
        "scientific clauses and citation tokens outside it. Findings must not overlap. "
        "An empty list means no concrete assembly-scope contradiction was found. "
        "lossless_shared_json_v1 uses local $ref values, $table columns/rows and $literal."
    ), tools=[], model=WORKBENCH_RESPONSE_MODEL, model_settings=WORKBENCH_RESPONSE_SETTINGS,
)

assembly_repair_agent = Agent(
    name="ASO Report Assembly Scope Repair",
    instructions=(
        "Correct the source-bound request/delivery scope findings using only the exact "
        "replacement schema. Each replacement replaces its finding's quoted span, not "
        "the whole narrative field. Preserve the surrounding sentence's grammar and "
        "every scientific claim and citation within the quoted span. Change only its "
        "incorrect request, membership, completion or component scope. Do not introduce "
        "new findings, measurements, ranks, efficacy claims or assessments. If no text "
        "is needed for a redundant incorrect span, an empty replacement is allowed. "
        "The unchanged surrounding text is supplied for context. Return JSON only. "
        "lossless_shared_json_v1 uses local $ref values, $table columns/rows and $literal."
    ), tools=[], model=WORKBENCH_RESPONSE_MODEL, model_settings=WORKBENCH_RESPONSE_SETTINGS,
)


def feature_check_outcomes(feature):
    """Count recorded Boolean outcomes, never infer an outcome from presence.

    Context groups preserve the original reference, RNA scope, orientation and
    mismatch count. Counts are source records, including repeated hypotheses;
    they are not distinct sites, binding evidence or measured activity.
    """
    records = feature.get('observations') if isinstance(feature, dict) else None
    names = ('overlap_true_record_count', 'overlap_false_record_count',
             'overlap_unrecorded_record_count')
    result = {'records_present': isinstance(records, list),
              'recorded_check_count': len(records) if isinstance(records, list) else None,
              **{name: 0 if isinstance(records, list) else None for name in names},
              'context_groups': [],
              'count_scope': 'Recorded overlap checks, including positive, negative and unknown outcomes. Source records can repeat the same hypothesis. A recorded check is not a positive overlap; geometric overlap is not antisense binding or activity.'}
    if not isinstance(records, list):
        return result
    fields = ('mapping_transcript_id', 'mapping_scope', 'match_orientation', 'mismatch_count')
    groups = {}
    for record in records:
        record = record if isinstance(record, dict) else {}
        context = {key: deepcopy(record.get(key)) for key in fields}
        key = json.dumps(context, sort_keys=True)
        group = groups.setdefault(key, {**context, **{name: 0 for name in names}})
        value = record.get('overlaps')
        name = names[0] if value is True else names[1] if value is False else names[2]
        result[name] += 1
        group[name] += 1
    result['context_groups'] = list(groups.values())
    return result


def assembly_source(payload, response):
    from .input_rows import assessment_coverage_errors, assessment_row_lookup
    from .alignment_reporting import alignment_geometry
    rows = (payload.get('evaluation_summary') or {}).get('provided_asos') or []
    assessments = response.get('evaluation_assessments') or []
    errors = assessment_coverage_errors(assessments, rows)
    if errors:
        raise ValueError('Assembly coverage failed: ' + '; '.join(errors))
    resolve = assessment_row_lookup(rows)
    evidence_register = []
    for row in rows:
        availability = {'input_row_id': row['input_row_id'], 'name': row.get('name')}
        for field in ('mapping_hits', 'near_match_candidates'):
            records = row.get(field)
            availability[field] = {'records_present': isinstance(records, list),
                'record_count': len(records) if isinstance(records, list) else None}
            if isinstance(records, list):
                geometries = [alignment_geometry(record) for record in records if isinstance(record, dict)]
                known = sum(type(geometry.get('recorded_genomic_segment_count')) is int for geometry in geometries)
                availability[field]['raw_geometry_availability'] = {
                    'raw_segment_count_known_record_count': known,
                    'raw_segment_count_unknown_record_count': len(records) - known,
                    'validated_contiguous_locus_record_count': sum(
                        isinstance(geometry.get('validated_contiguous_locus_geometry'), dict)
                        for geometry in geometries),
                    'validated_contiguous_locus_with_unknown_raw_segment_count': sum(
                        isinstance(geometry.get('validated_contiguous_locus_geometry'), dict)
                        and type(geometry.get('recorded_genomic_segment_count')) is not int
                        for geometry in geometries),
                    'scope': 'Raw segment-list completeness and derived contiguous-locus geometry are independent. A validated span can establish a contiguous unspliced window while its raw segment list and raw recorded segment count remain unknown. Window-identity completeness describes identity grouping, not raw field availability.',
                }
        availability['available_alignment_record_count'] = sum(
            len(row[field]) for field in ('mapping_hits', 'near_match_candidates')
            if isinstance(row.get(field), list))
        availability['delivery'] = 'Every available retained record is included in the assembled evidence; no observation-specific list is omitted during rendering.'
        inventory = row.get('evidence_inventory') or {}
        availability['recorded_inventory'] = {key: deepcopy(inventory[key])
            for key in ('exact_mapping_count', 'retained_near_match_count',
                        'feature_evidence_availability') if key in inventory}
        recorded_features = {
            f'requested_feature_membership[{index}]': feature
            for index, feature in enumerate(row.get('requested_feature_membership') or [])}
        feature_availability = availability['recorded_inventory'].get('feature_evidence_availability') or {}
        membership = feature_availability.get('requested_feature_membership') or {}
        for feature in membership.get('features') or []:
            feature['recorded_check_outcomes'] = feature_check_outcomes(
                recorded_features.get(feature.get('source_field')))
        windows = inventory.get('recorded_window_identity')
        if isinstance(windows, dict):
            availability['recorded_inventory']['recorded_window_identity'] = {
                key: deepcopy(windows[key]) for key in (
                    'identified_window_group_count', 'complete_for_retained_records',
                    'unresolved_source_fields', 'scope') if key in windows}
        availability['recorded_off_target_screen'] = {
            key: deepcopy(row[key]) for key in (
                'off_target_count_status', 'off_target_gene_count',
                'off_target_observed_gene_count', 'off_target_count_scope',
                'off_target_exact_count_unit') if key in row}
        availability['recorded_proxy_ranking'] = {
            key: deepcopy(row[key]) for key in (
                'knockdown_proxy_rank', 'ranking_scope', 'ranking_status') if key in row}
        evidence_register.append(availability)
    scoring = (payload.get('evaluation_summary') or {}).get('provided_aso_scoring') or {}
    return {
        'request_scope': {'observation_count': len(rows), 'observations': [
            {key: deepcopy(row[key]) for key in ('input_row_id', 'name') if key in row}
            for row in rows]},
        # This register describes the current delivered report. Historical
        # component subsets remain in original_component_responses and the
        # immutable archive; they cannot define current delivery membership.
        'delivery_scope': {'assessment_count': len(assessments),
            'assessed_input_row_ids': [resolve(item)['input_row_id'] for item in assessments]},
        'evidence_scope': {'observations': evidence_register,
            'recorded_screening_stages': {key: deepcopy(scoring[key]) for key in (
                'off_target_expression',) if key in scoring},
            'record_types': {
                'mapping_hits': 'Retained exact sequence alignment hypotheses in the searched orientations.',
                'near_match_candidates': 'Retained sequence window alignment hypotheses with recorded mismatches; they may be exact or inexact. These are alignment evidence records, not activity measurements.'},
            'scope': 'Availability of retained source records for this whole request. Counts do not establish search completeness, binding, activity, or comparative efficacy.'},
        'rendering_scope': {'narrative_perspective': 'delivered_report',
            'requested_outputs': deepcopy((payload.get('request_envelope') or {}).get('final_outputs') or []),
            'contract': 'The delivered report combines narrative, all assessment entries and every available retained evidence record. An empty component markdown field is not an empty final report.',
            'verification_boundary': 'This is the content contract for the delivered report, not a filesystem observation. Artifact existence and export success are verified after rendering; narrative review cannot infer file presence or absence from its own execution stage.'},
        'narrative_delivery_scope': {
            'global_report_fields': ['answer_summary', 'markdown', 'assumptions', 'limitations', 'suggested_followups'],
            'global_report_context': 'One assembled report for the full observation register; there are no separate observation requests in the delivered report.',
            'per_observation_field': 'ranking_rationale',
            'immutable_fields': 'All other per-observation scientific assessment fields retain their separate full-evidence reviews.'},
        'review_scope': 'Request, assessment membership and retained-evidence availability only; no scientific judgment or success grade.',
    }


def delivered_narrative_fields(response):
    # Global narrative and comparative scope are assembled across observations.
    # Scientific assessments retain their independent full-evidence review;
    # a membership/availability check cannot reinterpret their biology.
    global_fields = {'answer_summary', 'markdown', 'assumptions', 'limitations', 'suggested_followups'}
    return {path: value for path, value in narrative_fields(response).items()
        if path.split('/')[1] in global_fields or
        (len(path.split('/')) == 4 and path.split('/')[1] == 'evaluation_assessments'
         and path.split('/')[-1] == 'ranking_rationale')}


SCOPE_EVIDENCE_ROOTS = {
    'request_membership': ('/request_scope/',),
    'assessment_membership': ('/delivery_scope/',),
    'retained_evidence_availability': ('/evidence_scope/',),
    'artifact_delivery_scope': ('/rendering_scope/',),
    'component_scope': ('/delivery_scope/', '/narrative_delivery_scope/'),
}


def source_evidence_pointers(source):
    """Expose exact citations, without guessing or repairing a model's choice.

    Assembly registers contain availability metadata, not alignment records.
    Enumerate every container and scalar so evidence selection stays complete
    and grows linearly with that register. The source remains authoritative;
    choosing an existing pointer does not establish its relevance to a claim.
    """
    pointers = []
    stack = [('', source)]
    while stack:
        path, value = stack.pop()
        if path:
            pointers.append(path)
        if isinstance(value, dict):
            children = [(path + '/' + str(key).replace('~', '~0').replace('/', '~1'), item)
                        for key, item in value.items()]
        elif isinstance(value, list):
            children = [(path + '/' + str(index), item) for index, item in enumerate(value)]
        else:
            children = []
        stack.extend(reversed(children))
    return pointers


def assembly_findings_schema(response, page, source):
    schema = response_audit_schema(response)
    item = schema['properties']['findings']['items']
    item['properties']['response_path']['enum'] = list(page)
    item['required'].append('scope_kind')
    item['properties']['scope_kind'] = {'type': 'string', 'enum': list(SCOPE_EVIDENCE_ROOTS)}
    item['properties']['evidence_paths']['items']['enum'] = source_evidence_pointers(source)
    return schema


def narrative_pages(fields, *, target_bytes=12_000):
    pages, page, size = [], {}, 0
    for path, value in fields.items():
        width = len(json.dumps({path: value}, ensure_ascii=False).encode())
        if page and size + width > target_bytes:
            pages.append(page)
            page, size = {}, 0
        page[path] = value
        size += width
    if page:
        pages.append(page)
    return pages


def scope_review_context(value):
    # This context contains only bounded narrative pages and small, typed
    # registers, not the scientific alignment payload. Preserve literal JSON
    # Pointer identity: table compression otherwise exposes serialization
    # pointers that do not address the source being validated.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def scope_finding_errors(audit, response, source):
    errors = response_audit_contract_errors(audit, response, source)
    spans = {}
    for finding in audit.get('findings') or []:
        roots = SCOPE_EVIDENCE_ROOTS.get(finding.get('scope_kind'))
        # Only factual contradictions can authorize edits. A nonbinding note
        # still needs an existing source pointer and exact draft quote, but its
        # descriptive scope label must not block delivery of accepted science.
        if finding.get('kind') == 'contradicted_fact' and (
                not roots or not any(pointer.startswith(roots) for pointer in finding.get('evidence_paths') or [])):
            errors.append('An assembly finding must use a typed scope_kind and cite its scope register; scientific interpretation is outside this review.')
        path, quote = finding.get('response_path'), finding.get('quote')
        if path not in delivered_narrative_fields(response) or not isinstance(quote, str) or not quote:
            errors.append('Finding must identify a delivered narrative field and nonempty quote.')
            continue
        value = pointer_value(response, path)
        start = value.find(quote)
        if start < 0 or value.find(quote, start + 1) >= 0:
            errors.append('Quote must occur exactly once in its field; expand the quoted context.')
            continue
        end = start + len(quote)
        if any(start < right and end > left for left, right in spans.get(path, [])):
            errors.append('Scope findings in a field must not overlap.')
        spans.setdefault(path, []).append((start, end))
    return errors


def replacement_schema(findings):
    keys = [str(i) for i in range(len(findings))]
    return {'type': 'object', 'additionalProperties': False, 'required': ['replacements'],
        'properties': {'replacements': {'type': 'object', 'additionalProperties': False,
            'required': keys, 'properties': {key: {'type': 'string'} for key in keys}}}}


def apply_scope_edits(response, patch, findings, source):
    errors = scope_finding_errors({'findings': findings}, response, source)
    if any(finding.get('kind') != 'contradicted_fact' for finding in findings):
        errors.append('Only a source-bound factual contradiction authorizes an assembly edit.')
    replacements = patch.get('replacements') if isinstance(patch, dict) else None
    keys = {str(i) for i in range(len(findings))}
    if (not isinstance(patch, dict) or set(patch) != {'replacements'}
            or not isinstance(replacements, dict) or set(replacements) != keys
            or any(not isinstance(value, str) for value in replacements.values())):
        errors.append('Replace exactly the source-bound quoted spans using the supplied schema.')
    if errors:
        raise ValueError('; '.join(errors))
    result, edits = deepcopy(response), {}
    for i, finding in enumerate(findings):
        path, quote = finding['response_path'], finding['quote']
        start = pointer_value(response, path).index(quote)
        edits.setdefault(path, []).append((start, start + len(quote), replacements[str(i)]))
    for path, spans in edits.items():
        value = pointer_value(response, path)
        for start, end, replacement in sorted(spans, reverse=True):
            value = value[:start] + replacement + value[end:]
        tokens = [token.replace('~1', '/').replace('~0', '~') for token in path[1:].split('/')]
        parent = result
        for token in tokens[:-1]:
            parent = parent[int(token)] if isinstance(parent, list) else parent[token]
        parent[int(tokens[-1]) if isinstance(parent, list) else tokens[-1]] = value
    if result == response:
        raise ValueError('Scope correction did not change the flagged text.')
    return result


def scoped_component_response(payload, response):
    """Render attested component narratives without changing their scope or science.

    Each review is bound to its complete original batch, not a reduced assembly
    metadata view. Missing or invalid receipts in an otherwise attested panel
    fail closed. Legacy callers without receipts retain the existing review.
    This is delivery validation, not a new scientific assessment.
    """
    from .response_context import evaluation_response_batches, merge_evaluation_responses
    from .response_contracts import response_contract_errors

    components = response.get('original_component_responses') or []
    if not isinstance(components, list):
        raise ValueError('Component responses must be an ordered list.')
    if not any(isinstance(part, dict) and 'native_consistency_review' in part for part in components):
        return None
    batches = evaluation_response_batches(payload)
    expected = {digest(batch): batch for batch in batches}
    if len(batches) < 2 or len(expected) != len(batches) or len(components) != len(batches):
        raise ValueError('Reviewed component coverage differs from the current evidence batches.')
    by_source = {}
    for part in components:
        receipt = part.get('native_consistency_review') if isinstance(part, dict) else None
        if not isinstance(receipt, dict):
            raise ValueError('Every component requires its original scientific review receipt.')
        key = receipt.get('source_sha256')
        batch = expected.get(key)
        draft = {k: v for k, v in part.items() if k != 'native_consistency_review'}
        if (batch is None or key in by_source
                or receipt.get('status') != 'no_factual_contradiction_flagged'
                or receipt.get('draft_sha256') != digest(draft)
                or response_contract_errors(part, batch)):
            raise ValueError('Component scientific review does not attest the current evidence and response.')
        by_source[key] = part
    parts = [by_source[digest(batch)] for batch in batches]
    merged = merge_evaluation_responses(parts, payload)
    if response.get('evaluation_assessments') != merged['evaluation_assessments']:
        raise ValueError('Assembled assessments differ from the reviewed scientific components.')
    # Preserve the original archival order as well as every opaque model field.
    merged['original_component_responses'] = deepcopy(components)
    rows = (payload.get('evaluation_summary') or {}).get('provided_asos') or []
    ordinal = {row['input_row_id']: index for index, row in enumerate(rows, 1)}

    def label_for(batch):
        labels = []
        for row in batch['evaluation_summary']['provided_asos']:
            name = html.escape(' '.join(str(row.get('name') or row['input_row_id']).split()), quote=False)
            name = ''.join('\\' + c if c in '\\`*_{}[]()#+-.!|' else c for c in name)
            labels.append(f"Observation {ordinal[row['input_row_id']]} ({name})")
        return '; '.join(labels)

    sections = [(label_for(batch), part) for batch, part in zip(batches, parts)]
    for field in ('answer_summary', 'markdown'):
        fragments = []
        for label, part in sections:
            value = part.get(field)
            if value is None or value == '':
                continue
            if not isinstance(value, str):
                raise ValueError('Reviewed narrative fields must be strings.')
            fragments.append(f'### {label}\n\n' + value)
        merged[field] = '\n\n'.join(fragments)
    introduction = (f'This report contains assessments for all {len(rows)} supplied observations. '
        'Each labeled section retains the assessment of that observation; references to the '
        'current assessment or available evidence are local to that section. Explicit comparisons '
        'with other observations retain their stated scope.')
    merged['answer_summary'] = introduction + '\n\n' + merged['answer_summary']
    for field in ('assumptions', 'limitations', 'suggested_followups'):
        values = []
        for label, part in sections:
            supplied = part.get(field) or []
            if not isinstance(supplied, list) or any(not isinstance(value, str) for value in supplied):
                raise ValueError('Reviewed narrative lists must contain strings.')
            values.extend(f'**{label}:** ' + value for value in supplied)
        merged[field] = values
    errors = response_contract_errors(merged, payload)
    if errors:
        raise ValueError('Scoped component delivery failed: ' + '; '.join(errors))
    return merged


async def review_assembled_response(payload, response, *, artifact_dir=None):
    from . import pipeline_tools as p
    from .response_contracts import response_contract_errors
    response = deepcopy(response)
    source = assembly_source(payload, response)
    archive = None
    if artifact_dir is not None:
        archive = ResponseArtifacts(artifact_dir, {'assembly_source': source, 'assembled_response': response},
            report_configuration((assembly_audit_agent, assembly_repair_agent)))
        accepted = archive.accepted()
        if accepted is not None:
            return accepted
    original = deepcopy(response)
    repairs, round_number, advisory_count = {}, 0, 0
    try:
        scoped = scoped_component_response(payload, response)
        if scoped is not None:
            scoped['native_assembly_review'] = {
                'status': 'component_scopes_preserved',
                'method': 'deterministic_scoped_component_assembly',
                'source_sha256': digest(source), 'original_draft_sha256': digest(original),
                'accepted_draft_sha256': digest(scoped),
                'scientific_component_count': len(scoped['original_component_responses']),
                'native_model_calls': 0,
                'scope': 'Complete reviewed components, observation coverage and original narrative scope.',
                'edit_policy': 'Only observation labels and a delivery introduction are added. Every original scientific sentence, assessment and citation is preserved. No new scientific judgment.'}
            if archive:
                archive.accept(scoped)
            return scoped
        while True:
            round_number += 1
            fields = delivered_narrative_fields(response)
            async def audit_page(index, page):
                schema = assembly_findings_schema(response, page, source)
                agent = p._agent_with_strict_output_schema(assembly_audit_agent,
                    schema_name='aso_report_assembly_scope', schema=schema)
                prompt = scope_review_context({'source': source, 'draft_narrative_fields': page})
                stem = f'round-{round_number}-page-{index}'
                if archive:
                    archive.record(stem + '-request', {'prompt': prompt})
                audit = await p._run_schema_validated_agent_once(agent=agent, prompt=prompt, schema=schema,
                    error_prefix='report_assembly_scope', contract_validator=lambda result:
                        p._json_schema_validation_errors(result, schema) or scope_finding_errors(result, response, source),
                    attempt_observer=(lambda number, data: archive.record(stem + f'-call-{number}', data)) if archive else None)
                if audit.get('error'):
                    raise RuntimeError('report_assembly_scope_failed: ' + str(audit))
                return response_audit_repair_findings(audit, response, source)
            # Pages are independent reads of one immutable draft/source pair.
            # Await every page before any edit; provider admission controls API
            # concurrency, while archival paths and result order remain stable.
            # Runner.run wraps synchronous provider I/O. Give each independent
            # page a worker event loop, as the component composer already does.
            def run_page(index, page):
                return asyncio.run(audit_page(index, page))
            page_results = await asyncio.gather(*(asyncio.to_thread(run_page, index, page)
                for index, page in enumerate(narrative_pages(fields))), return_exceptions=True)
            for result in page_results:
                if isinstance(result, BaseException):
                    raise result
            findings = []
            for reviewed in page_results:
                # Scope review is not editorial rewriting. An ambiguity note
                # does not establish that an otherwise accepted report is false.
                # Full observations remain in the recorded native audit calls.
                findings.extend(item for item in reviewed if item.get('kind') == 'contradicted_fact')
                advisory_count += sum(item.get('kind') != 'contradicted_fact' for item in reviewed)
            if archive:
                archive.record(f'round-{round_number}-audited-draft', response)
                archive.record(f'round-{round_number}-findings', findings)
            if not findings:
                response['native_assembly_review'] = {'scope': source['review_scope'],
                    'status': 'no_request_scope_contradiction_flagged', 'rounds': round_number,
                    'source_sha256': digest(source), 'original_draft_sha256': digest(original),
                    'accepted_draft_sha256': digest(response), 'model': str(assembly_audit_agent.model),
                    'noncontradiction_observations_not_applied': advisory_count,
                    'edit_policy': 'Only source-bound factual contradictions authorize edits. Ambiguity observations remain archived.'}
                if archive:
                    archive.accept(response)
                return response
            for path in {finding['response_path'] for finding in findings}:
                repairs[path] = repairs.get(path, 0) + 1
                if repairs[path] > 3:
                    raise RuntimeError('report_assembly_scope_repair_exhausted: ' + path)
            schema = replacement_schema(findings)
            agent = p._agent_with_strict_output_schema(assembly_repair_agent,
                schema_name='aso_report_assembly_scope_replacements', schema=schema)
            repair_input = {'source': source, 'findings_by_index': dict(enumerate(findings)),
                'draft_narrative_fields': {finding['response_path']: fields[finding['response_path']] for finding in findings},
                'output_schema': schema}
            def validate_patch(patch):
                try:
                    candidate = apply_scope_edits(response, patch, findings, source)
                    return response_contract_errors(candidate, payload)
                except ValueError as exc:
                    return [str(exc)]
            patch = await p._run_schema_validated_agent_once(agent=agent,
                prompt=scope_review_context(repair_input), schema=schema,
                error_prefix='report_assembly_scope_repair', contract_validator=validate_patch,
                attempt_observer=(lambda number, data: archive.record(f'round-{round_number}-repair-call-{number}', data)) if archive else None)
            if patch.get('error'):
                raise RuntimeError('report_assembly_scope_repair_failed: ' + str(patch))
            response = apply_scope_edits(response, patch, findings, source)
    except Exception as exc:
        if archive:
            archive.record('failure', {'exception_type': type(exc).__name__, 'message': str(exc)})
        raise
