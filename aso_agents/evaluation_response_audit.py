"""Bind a native narrative-consistency review to its actual draft and evidence.

This does not score scientific success, define efficacy thresholds, or replace
the supplied measurements. Findings must quote the draft and cite real evidence.
"""
from copy import deepcopy
import hashlib
import json


_NARRATIVE_FIELDS = {
    "answer_summary", "markdown", "assumptions", "limitations",
    "suggested_followups", "mechanistic_plausibility", "selectivity_limitations",
    "uncertainty", "ranking_rationale",
}


def _escape(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def narrative_fields(response):
    result = {}

    def visit(value, path="", active=False):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + "/" + _escape(key), active or key in _NARRATIVE_FIELDS)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + "/" + str(index), active)
        elif active and isinstance(value, str) and value.strip():
            result[path] = value

    visit(response)
    return result


def pointer_value(document, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Evidence pointer must be an absolute JSON Pointer")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not token.isdigit() or str(int(token)) != token:
                raise ValueError("Invalid array index in evidence pointer")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError("Evidence pointer traverses a scalar")
    return value


def response_audit_schema(response):
    paths = list(narrative_fields(response))
    if not paths:
        raise ValueError("No narrative fields to audit")
    return {
        "type": "object", "additionalProperties": False, "required": ["findings"],
        "properties": {"findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["response_path", "quote", "kind", "evidence_paths", "reason"],
            "properties": {
                "response_path": {"type": "string", "enum": paths},
                "quote": {"type": "string", "minLength": 1},
                "kind": {"type": "string", "enum": ["contradicted_fact", "unsupported_factual_claim", "ambiguous_evidence_scope"]},
                "evidence_paths": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                "reason": {"type": "string", "minLength": 1},
            },
        }}},
    }


def splice_metric_index(source):
    """Expose typed prediction fields before large alignment inventories.

    Every value is copied from its named field at a valid source pointer. This
    index does not infer metric equivalence, aggregate sites or judge prose.
    Keep task/row paths even when molecule identifiers repeat across tasks.
    """
    fields = {
        'delta_inclusion_deletion', 'delta_inclusion_replacement',
        'delta_acceptor_deletion', 'delta_acceptor_replacement',
        'delta_donor_deletion', 'delta_donor_replacement',
        'delta_names', 'delta_deletion', 'delta_replacement',
        'primary_label', 'primary_direction', 'primary_delta_deletion',
        'primary_delta_replacement', 'selected_label', 'selected_direction',
        'selected_site_kind', 'selected_site_position',
        'selected_delta_deletion', 'selected_delta_replacement',
        'label', 'canonical_label', 'site_kind', 'isoform_label',
        'reference_transcript_id', 'reference_binding_status', 'exon_index',
        'window_bp', 'evaluation_status', 'splice_eval_status',
        'splice_eval_skip_reason',
    }
    index = []

    def visit(value, path):
        if isinstance(value, dict):
            summary = value.get('splice_eval_summary')
            if isinstance(summary, dict):
                pointer = path + '/splice_eval_summary'
                records = [(pointer, summary)]
                sites = summary.get('splice_site_deltas')
                if isinstance(sites, list):
                    records.extend((pointer + '/splice_site_deltas/' + str(i), site)
                                   for i, site in enumerate(sites) if isinstance(site, dict))
                index.append({
                    'row_source_pointer': path,
                    'row_identity': {key: deepcopy(value[key]) for key in
                                     ('input_row_id', 'name', 'transcript_id') if key in value},
                    'measurements': [
                        {'source_pointer': ref, 'recorded_fields': {
                            key: deepcopy(item) for key, item in record.items() if key in fields}}
                        for ref, record in records],
                })
            for key, item in value.items():
                if key not in {'measurement_evidence', 'splice_eval_summary'}:
                    visit(item, path + '/' + _escape(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, path + '/' + str(i))

    visit(source, '')
    return index


def response_audit_context(source, response):
    """Index recorded scope/provenance facts without replacing the full source.

    These are exact copies with valid source pointers, not inferred conclusions.
    Measurement citation dictionaries duplicate canonical records, so their
    contents remain in the complete source rather than repeating them here.
    """
    scope_fields = {
        'feature_evidence_availability', 'exon_identity_resolution',
        'identity_resolution', 'reference_context', 'transcript_version_status',
        'near_match_search', 'near_match_search_scope', 'off_target_count_scope',
        'stage_statuses', 'requested_feature_membership',
        'evaluation_targets', 'context_summary', 'splice_eval_summary',
    }
    index = []
    stack = [('', source)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, dict):
            children = []
            for key, item in value.items():
                pointer = path + '/' + _escape(key)
                if key == 'orientation_interpretation' and isinstance(item, dict):
                    for field in ('input_sequence', 'evaluated_molecule_sequence',
                                  'match_orientation', 'matched_target_sequence',
                                  'original_molecule_exact_binding', 'likely_reversed_entry',
                                  'conditional_reverse_complement_sequence'):
                        if field in item:
                            index.append({'source_pointer': pointer + '/' + _escape(field),
                                          'recorded_value': deepcopy(item[field])})
                elif key in scope_fields:
                    index.append({'source_pointer': pointer, 'recorded_value': deepcopy(item)})
                elif key != 'measurement_evidence':
                    children.append((pointer, item))
            stack.extend(reversed(children))
        elif isinstance(value, list):
            stack.extend(reversed([(path + '/' + str(i), item) for i, item in enumerate(value)]))
    return {
        'splice_metric_instructions': 'Check splice_metric_index first when the draft discusses splice predictions. Each recorded_fields entry is an exact subset of the record at source_pointer. Bind each number to its original metric, perturbation, site and task. An individual donor/acceptor delta cannot supply an absent or different inclusion proxy. Null stays unrecorded and zero stays recorded. This navigation index adds no prediction or interpretation; cite the complete source paths.',
        'splice_metric_index': splice_metric_index(source),
        'scope_index_instructions': 'Navigation only: each recorded_value exactly copies its source_pointer in the complete source. Reconcile evidence existence, requested-feature naming, search/retention/count scope and unmeasured activity separately. Cite source_pointer values, never the index itself. The complete source below remains authoritative and contains all records.',
        'evidence_scope_index': index,
        'draft_response': deepcopy(response),
        'source': deepcopy(source),
    }


def panel_comparison_audit_packet(source, response):
    """Review comparative prose against the full recorded panel, not one batch.

    Component batches can contain one assessed molecule while retaining a
    compact overview of every submitted molecule.  Keep source pointers in
    their original document so ordinary bound narrative repair can be used.
    """
    summary = source.get('evaluation_summary') or {}
    panel = summary.get('panel_comparison_overview') or []
    assessments = response.get('evaluation_assessments') or []
    if len(panel) < 2 or not assessments:
        return None
    fields = narrative_fields(response)
    comparative = {path: value for path, value in fields.items()
                   if path.endswith('/ranking_rationale') and path.startswith('/evaluation_assessments/')}
    if not comparative:
        return None
    # Give every molecule the same bounded ranking view. The complete alignment
    # source remains with the ordinary scientific audit. Retain one attested
    # example for each exact reference/context/orientation relationship while
    # recording the complete exact-group count for each molecule.
    compact = []
    comparison_keys = ('name', 'input_row_id', 'sequence', 'chemistry', 'length',
                       'gc_fraction', 'mapping_status', 'binding_evidence',
                       'selection_rank', 'selection_score_components',
                       'isoform_uniqueness_score', 'isoform_uniqueness_class',
                       'off_target_gene_count', 'off_target_count_status',
                       'off_target_count_scope', 'orientation_interpretation',
                       'transcript_id', 'phastcons', 'phylop')
    for index, row in enumerate(panel):
        if not isinstance(row, dict):
            continue
        base = f'/evaluation_summary/panel_comparison_overview/{index}'
        fields = [{'source_pointer': base + '/' + _escape(key), 'recorded_value': deepcopy(row[key])}
                  for key in comparison_keys if key in row]
        groups = (row.get('reference_alignment_summary') or {}).get('groups') or []
        exact = [(number, group) for number, group in enumerate(groups)
                 if isinstance(group, dict) and group.get('exact_mapping_record_count', 0) > 0]
        seen = set()
        for number, group in exact:
            relation = (group.get('transcript_id'), group.get('rna_context'),
                        group.get('match_orientation'))
            if relation in seen:
                continue
            seen.add(relation)
            fields.append({'source_pointer': base + f'/reference_alignment_summary/groups/{number}',
                           'recorded_value': deepcopy(group)})
        compact.append({'source_pointer': base, 'name': row.get('name'), 'fields': fields,
                        'exact_source_group_count': len(exact),
                        'exact_source_group_count_pointer': base + '/reference_alignment_summary/groups',
                        'full_source_available': True})
    panel_by_id = {row.get('input_row_id'): index for index, row in enumerate(panel)
                   if isinstance(row, dict) and row.get('input_row_id')}
    assessed_subjects = {}
    for path in comparative:
        tokens = path.strip('/').split('/')
        if len(tokens) != 3 or not tokens[1].isdigit() or int(tokens[1]) >= len(assessments):
            continue
        assessment = assessments[int(tokens[1])]
        if not isinstance(assessment, dict):
            continue
        panel_index = panel_by_id.get(assessment.get('input_row_id'))
        if panel_index is None:
            continue
        assessed_subjects[path] = {
            'input_row_id': assessment.get('input_row_id'),
            'name': panel[panel_index].get('name'),
            'source_pointer': f'/evaluation_summary/panel_comparison_overview/{panel_index}',
        }
    return {
        'review_scope': 'Check only comparative factual claims in ranking_rationale against the complete panel ranking view. Use assessed_subjects to identify the molecule for each draft field: a component assessment index is not a panel row index. The exact groups shown are a representative record for every distinct transcript/RNA-context/orientation relation with at least one exact mapping, and exact_source_group_count counts all source groups. A component may assess one molecule, but words such as only, alone, better, lower, tied and first refer to the complete supplied panel. Distinguish exact antisense binding of the submitted molecule from same-strand similarity and from a separate conditional reverse-complement molecule. Preserve uncertainty and declared measurement scope. Other omitted alignment details are not evidence of absence. Report only concrete source-contradicted or unsupported factual claims; do not grade efficacy or demand new analyses.',
        'request_text': source.get('request_text'),
        'panel_observation_count': len(panel),
        'full_panel_source': compact,
        'assessed_subjects': assessed_subjects,
        'draft_narrative_fields': comparative,
    }


def panel_comparison_finding_errors(audit, response, source):
    """Require comparative findings to cite both the subject and comparator."""
    errors = response_audit_contract_errors(audit, response, source)
    panel = (source.get('evaluation_summary') or {}).get('panel_comparison_overview') or []
    assessments = response.get('evaluation_assessments') or []
    by_id = {row.get('input_row_id'): index for index, row in enumerate(panel)
             if isinstance(row, dict)}
    for number, finding in enumerate(audit.get('findings') or []):
        path = str(finding.get('response_path') or '')
        parts = path.strip('/').split('/')
        if len(parts) != 3 or parts[0] != 'evaluation_assessments' or not parts[1].isdigit():
            errors.append(f'findings[{number}] must identify one comparative assessment field')
            continue
        index = int(parts[1])
        if index >= len(assessments) or not isinstance(assessments[index], dict):
            errors.append(f'findings[{number}] has no assessed subject')
            continue
        subject = by_id.get(assessments[index].get('input_row_id'))
        cited = set()
        for pointer in finding.get('evidence_paths') or []:
            tokens = str(pointer).strip('/').split('/')
            if (len(tokens) >= 4 and tokens[:2] == ['evaluation_summary', 'panel_comparison_overview']
                    and tokens[2].isdigit() and int(tokens[2]) < len(panel)):
                cited.add(int(tokens[2]))
        if subject is None or subject not in cited or len(cited - {subject}) < 1:
            errors.append(f'findings[{number}] must cite source evidence for both the assessed molecule and at least one other panel molecule')
    return errors


def orientation_audit_packets(source, response):
    """Small source-bound packets for the molecule/alternative distinction.

    The complete evidence review still runs separately. These packets expose
    exact pointers into that same source and draft, so findings can use the
    existing bounded narrative repair without inventing new evidence.
    """
    summary = source.get('evaluation_summary') or {}
    rows = summary.get('provided_asos') or []
    assessments = response.get('evaluation_assessments') or []
    fields = narrative_fields(response)
    packets = []
    attested = set()

    def append_packet(row, row_pointer, response_paths, *, cross_observation):
        if not isinstance(row, dict):
            return
        orientation = row.get('orientation_interpretation') or {}
        if not isinstance(orientation, dict) or orientation.get('match_orientation') != 'input_sequence':
            return
        if (not orientation.get('conditional_reverse_complement_sequence')
                or not orientation.get('input_sequence')
                or orientation.get('matched_target_sequence') != orientation.get('input_sequence')):
            return
        alternative = str(orientation['conditional_reverse_complement_sequence']).upper().replace('U', 'T')
        matched = str(orientation['matched_target_sequence']).upper().replace('U', 'T')
        if not alternative or not set(alternative) <= set('ACGT'):
            return
        if alternative.translate(str.maketrans('ACGT', 'TGCA'))[::-1] != matched:
            return
        evidence_key = (row.get('name'), row.get('sequence'), matched, alternative,
                        tuple(sorted(response_paths)), cross_observation)
        if evidence_key in attested:
            return
        attested.add(evidence_key)
        evidence = []
        for key in ('name', 'input_row_id', 'sequence', 'resolved_sequence_identity'):
            if key in row:
                path = row_pointer + '/' + _escape(key)
                evidence.append({'source_pointer': path, 'recorded_value': deepcopy(row[key])})
        for key in ('input_sequence', 'evaluated_molecule_sequence', 'match_orientation',
                    'matched_target_sequence', 'original_molecule_exact_binding',
                    'likely_reversed_entry', 'conditional_reverse_complement_sequence',
                    'interpretation_note'):
            if key in orientation:
                path = row_pointer + '/orientation_interpretation/' + _escape(key)
                evidence.append({'source_pointer': path, 'recorded_value': deepcopy(orientation[key])})
        relevant = [{'response_path': path, 'text': fields[path]}
                    for path in sorted(response_paths) if path in fields]
        if relevant:
            packets.append({'scope': 'Check only a false denial of the conditional alternative '
                                     'molecule\'s exact antisense relationship to the recorded '
                                     'matched RNA window. In a user-requested orientation comparison, '
                                     'the phrase reverse-complement interpretation or hypothesis '
                                     'may name this distinct hypothetical oligo; inspect the claim\'s '
                                     'actual meaning and source. Do not judge the written molecule\'s '
                                     'separate exact-binding result, efficacy, off-target scope, '
                                     'or other omitted evidence.',
                            'request_text': source.get('request_text'),
                            'orientation_subject': {'name': row.get('name'),
                                                    'input_row_id': row.get('input_row_id'),
                                                    'source': 'panel_comparison_overview' if cross_observation
                                                              else 'provided_asos'},
                            'verified_sequence_relation': {
                                'submitted_oligo_equals_recorded_rna_window': True,
                                'conditional_alternative_reverse_complement_equals_recorded_rna_window': True,
                                'interpretation': 'The written oligo is same-strand to this RNA window; '
                                                  'the separately hypothetical alternative oligo is '
                                                  'exactly antisense to this same window.'},
                            'orientation_evidence': evidence,
                            'draft_narrative': relevant})

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        assessment_matches = [number for number, item in enumerate(assessments)
                              if isinstance(item, dict) and item.get('input_row_id') == row.get('input_row_id')]
        if len(assessment_matches) != 1:
            continue
        assessment_index = assessment_matches[0]
        append_packet(row, f'/evaluation_summary/provided_asos/{index}',
                      {f'/evaluation_assessments/{assessment_index}/mechanistic_plausibility',
                       f'/evaluation_assessments/{assessment_index}/ranking_rationale'},
                      cross_observation=False)
    current_ids = {row.get('input_row_id') for row in rows if isinstance(row, dict)}
    panel = [row for row in summary.get('panel_comparison_overview') or []
             if isinstance(row, dict)]

    def explicit_identity_reference(text, row):
        """Only apply another molecule's fact to prose naming that molecule.

        A panel-wide ranking field can discuss a sibling, but generic phrases
        such as "the submitted molecule" refer to the assessed row. Name and
        sequence are usable only when unique in the panel. This deliberately
        misses implicit cross-row references instead of importing a sibling's
        orientation into the current molecule's assessment.
        """
        identities = [row.get('input_row_id')]
        identities.extend(row.get(key) for key in ('name', 'sequence')
                          if row.get(key) and sum(
                              item.get(key) == row.get(key) for item in panel) == 1)
        haystack = text.casefold()
        for identity in identities:
            needle = str(identity or '').strip().casefold()
            if not needle:
                continue
            start = 0
            while (index := haystack.find(needle, start)) >= 0:
                end = index + len(needle)
                if ((index == 0 or not haystack[index - 1].isalnum())
                        and (end == len(haystack) or not haystack[end].isalnum())):
                    return True
                start = end
        return False

    ranking_paths = {path for path in fields if path.endswith('/ranking_rationale')
                     and path.startswith('/evaluation_assessments/')}
    for index, row in enumerate(summary.get('panel_comparison_overview') or []):
        if not isinstance(row, dict) or row.get('input_row_id') in current_ids:
            continue
        named_paths = {path for path in ranking_paths
                       if explicit_identity_reference(fields[path], row)}
        append_packet(row, f'/evaluation_summary/panel_comparison_overview/{index}',
                      named_paths, cross_observation=True)
    return packets


def chemical_position_audit_packets(source, response):
    """Bind supplied-gapmer position claims to native numeric partitions."""
    rows = (source.get('evaluation_summary') or {}).get('provided_asos') or []
    assessments = response.get('evaluation_assessments') or []
    draft = narrative_fields(response)
    packets = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        position_evidence = row.get('chemical_position_evidence') or {}
        if not position_evidence.get('recorded_near_match_mismatch_positions'):
            continue
        matches = [(number, assessment) for number, assessment in enumerate(assessments)
                   if isinstance(assessment, dict)
                   and assessment.get('input_row_id') == row.get('input_row_id')]
        if len(matches) != 1:
            continue
        number, _ = matches[0]
        prefix = f'/evaluation_assessments/{number}/'
        relevant = [{'response_path': path, 'text': value} for path, value in draft.items()
                    if path.startswith(prefix)]
        if not relevant:
            continue
        row_pointer = f'/evaluation_summary/provided_asos/{index}'
        evidence = [{'source_pointer': row_pointer + '/' + _escape(key),
                     'recorded_value': deepcopy(row[key])}
                    for key in ('name', 'input_row_id', 'sequence', 'chemistry',
                                'chemical_position_evidence') if key in row]
        packets.append({
            'scope': 'Check only explicit claims placing a recorded 1-based mismatch '
                     'position in a chemical wing or the DNA gap of this supplied '
                     'molecule. Use the typed position evidence for this row. Do not '
                     'infer binding strength, activity, or correctness of a different '
                     'alignment from the position partition. If the draft makes no '
                     'such position-to-region claim, return no findings.',
            'request_text': source.get('request_text'),
            'subject': {'name': row.get('name'), 'input_row_id': row.get('input_row_id')},
            'position_evidence': evidence,
            'draft_narrative': relevant,
        })
    return packets


def _chemical_claim_words(value):
    current = ''
    result = []
    for char in str(value).lower():
        if char.isascii() and char.isalnum():
            current += char
        elif current:
            result.append(current)
            current = ''
    if current:
        result.append(current)
    return result


def _explicit_chemical_positions(tokens):
    positions = []
    for index, token in enumerate(tokens):
        if token not in {'position', 'positions', 'pos'}:
            continue
        cursor = index + 1
        while cursor < len(tokens):
            next_token = tokens[cursor]
            if next_token.isascii() and next_token.isdecimal():
                positions.append(int(next_token))
                cursor += 1
            elif (next_token == 'and' and positions
                  and cursor + 1 < len(tokens)
                  and tokens[cursor + 1].isascii()
                  and tokens[cursor + 1].isdecimal()):
                cursor += 1
            else:
                break
    return positions


def _chemical_region_locations(tokens):
    """Return regions explicitly introduced as a location, not mere mentions."""
    locations = set()
    for index, token in enumerate(tokens):
        if token not in {'in', 'inside', 'within'}:
            continue
        nearby = tokens[index + 1:index + 6]
        if 'gap' in nearby:
            locations.add('gap')
        if 'wing' in nearby or 'wings' in nearby:
            locations.add('wing')
    return locations


def filter_chemical_position_findings(audit, response, source):
    """Admit only source-verifiable, explicit position/region contradictions.

    A model finding is not itself evidence. For this narrow typed relation we
    can verify a single affirmative position-to-wing/gap claim without trying
    to parse arbitrary biological prose. Ambiguous claims stay with the general
    narrative audit and cannot trigger a focused repair.
    """
    rows = (source.get('evaluation_summary') or {}).get('provided_asos') or []
    assessments = response.get('evaluation_assessments') or []
    by_id = {}
    duplicate_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get('input_row_id'):
            continue
        row_id = row['input_row_id']
        if row_id in by_id:
            duplicate_ids.add(row_id)
        by_id[row_id] = (index, row)
    for row_id in duplicate_ids:
        by_id.pop(row_id, None)
    accepted, rejected = [], []

    for finding in audit.get('findings') or []:
        path = str(finding.get('response_path') or '')
        parts = path.strip('/').split('/')
        if (len(parts) < 3 or parts[0] != 'evaluation_assessments'
                or not parts[1].isdigit() or int(parts[1]) >= len(assessments)):
            rejected.append({'finding': finding, 'reason': 'No unique assessed molecule'})
            continue
        assessment = assessments[int(parts[1])]
        if not isinstance(assessment, dict) or assessment.get('input_row_id') not in by_id:
            rejected.append({'finding': finding, 'reason': 'No unique supplied input row'})
            continue
        row_index, row = by_id[assessment['input_row_id']]
        evidence_pointer = f'/evaluation_summary/provided_asos/{row_index}/chemical_position_evidence'
        if not any(str(pointer) == evidence_pointer or str(pointer).startswith(evidence_pointer + '/')
                   for pointer in finding.get('evidence_paths') or []):
            rejected.append({'finding': finding, 'reason': 'No row-bound position evidence cited'})
            continue
        positions = {item['position_1based']: item['region']
                     for item in (row.get('chemical_position_evidence') or {}).get(
                         'recorded_near_match_mismatch_positions') or []
                     if isinstance(item, dict) and isinstance(item.get('position_1based'), int)
                     and item.get('region') in {'dna_gap', '5prime_wing', '3prime_wing'}}
        quote = str(finding.get('quote') or '')
        tokens = _chemical_claim_words(quote)
        if any(token in {'not', 'outside', 'rather', 'instead', 'than', 'neither', 'except'}
               for token in tokens):
            rejected.append({'finding': finding, 'reason': 'Negated or contrastive region claim'})
            continue
        claimed_positions = _explicit_chemical_positions(tokens)
        if (not claimed_positions or len(set(claimed_positions)) != len(claimed_positions)
                or any(position not in positions for position in claimed_positions)):
            rejected.append({'finding': finding,
                             'reason': 'No unambiguous list of recorded positions stated in claim'})
            continue
        locations = _chemical_region_locations(tokens)
        wing, gap = 'wing' in locations, 'gap' in locations
        if not wing and not gap:
            field = narrative_fields(response).get(path, '')
            start = field.find(quote)
            if not quote or start < 0 or field.find(quote, start + 1) >= 0:
                rejected.append({'finding': finding,
                                 'reason': 'Quoted claim is not unique in its source field'})
                continue
            prefix = field[max(0, start - 120):start]
            # A region in a different sentence or clause cannot qualify this
            # quote. Use only the adjacent original draft, never model reason.
            for boundary in ('. ', '; ', '? ', '! ', ', ', '\n'):
                index = prefix.rfind(boundary)
                if index >= 0:
                    prefix = prefix[index + len(boundary):]
            prefix_tokens = _chemical_claim_words(prefix)
            if any(token in {'not', 'outside', 'rather', 'instead', 'than', 'neither', 'except'}
                   for token in prefix_tokens):
                rejected.append({'finding': finding, 'reason': 'Negated adjacent region claim'})
                continue
            locations = _chemical_region_locations(prefix_tokens)
            wing, gap = 'wing' in locations, 'gap' in locations
        if wing == gap:
            rejected.append({'finding': finding, 'reason': 'Chemical region claim is ambiguous or absent'})
            continue
        contradiction = all(
            (wing and positions[position] == 'dna_gap')
            or (gap and positions[position] != 'dna_gap')
            for position in claimed_positions)
        if contradiction:
            accepted.append(finding)
        else:
            rejected.append({'finding': finding,
                             'reason': 'Quoted position-to-region claim agrees with native typed evidence'})
    return accepted, rejected


def native_chemical_position_findings(source, response):
    """Find only explicit, source-checkable supplied gapmer region contradictions.

    This narrow arithmetic check does not depend on an LLM noticing the claim.
    Ambiguous prose, negations and mixed position groups remain for the general
    evidence review; this function makes no affinity or activity judgment.
    """
    rows = (source.get('evaluation_summary') or {}).get('provided_asos') or []
    assessments = response.get('evaluation_assessments') or []
    narratives = narrative_fields(response)
    findings = []
    for assessment_index, assessment in enumerate(assessments):
        if not isinstance(assessment, dict):
            continue
        matches = [(index, row) for index, row in enumerate(rows)
                   if isinstance(row, dict) and row.get('input_row_id')
                   and row.get('input_row_id') == assessment.get('input_row_id')]
        if len(matches) != 1:
            continue
        row_index, row = matches[0]
        if not (row.get('chemical_position_evidence') or {}).get(
                'recorded_near_match_mismatch_positions'):
            continue
        prefix = f'/evaluation_assessments/{assessment_index}/'
        for path, field in narratives.items():
            if not path.startswith(prefix):
                continue
            clauses = [field]
            for boundary in ('. ', '; ', '? ', '! ', '\n'):
                clauses = [part for clause in clauses for part in clause.split(boundary)]
            for clause in clauses:
                quote = clause.strip()
                if not quote:
                    continue
                candidate = {
                    'response_path': path, 'quote': quote,
                    'kind': 'contradicted_fact',
                    'evidence_paths': [
                        f'/evaluation_summary/provided_asos/{row_index}/chemical_position_evidence'],
                    'reason': 'The explicitly assigned supplied-ASO mismatch positions '
                              'disagree with the native numeric wing-gap-wing partition; '
                              'no binding or activity conclusion is inferred.',
                }
                accepted, _ = filter_chemical_position_findings(
                    {'findings': [candidate]}, response, source)
                findings.extend(accepted)
    return findings


def correct_native_chemical_position_claims(source, response):
    """Replace only explicit, typed-evidence-contradicted region sentences.

    The original response and each exact quotation belong in the response
    archive. A partial or ambiguous edit fails closed instead of asking a
    language model to remove the quoted position while retaining its claim.
    """
    findings = native_chemical_position_findings(source, response)
    if not findings:
        return response, []
    rows = (source.get('evaluation_summary') or {}).get('provided_asos') or []
    assessments = response.get('evaluation_assessments') or []
    changes = {}
    receipts = []
    for finding in findings:
        path = finding['response_path']
        parts = path.strip('/').split('/')
        assessment = assessments[int(parts[1])]
        matching = [row for row in rows if isinstance(row, dict)
                    and row.get('input_row_id') == assessment.get('input_row_id')]
        if len(matching) != 1:
            raise ValueError('Chemical claim correction has no unique supplied row')
        records = (matching[0].get('chemical_position_evidence') or {}).get(
            'recorded_near_match_mismatch_positions') or []
        regions = {item['position_1based']: item['region'] for item in records
                   if isinstance(item, dict) and isinstance(item.get('position_1based'), int)
                   and item.get('region') in {'dna_gap', '5prime_wing', '3prime_wing'}}
        quote = finding['quote']
        positions = _explicit_chemical_positions(_chemical_claim_words(quote))
        if not positions or len(set(positions)) != len(positions) or any(
                position not in regions for position in positions):
            raise ValueError('Chemical claim correction lacks unambiguous typed positions')
        names = {'dna_gap': 'DNA gap', '5prime_wing': '5′ wing',
                 '3prime_wing': '3′ wing'}
        locations = ', '.join(f'{position} ({names[regions[position]]})'
                              for position in positions)
        replacement = ('Recorded mismatch positions ' + locations
                       + ' refer to the supplied ASO; binding and activity for this '
                       'near-match remain unmeasured')
        field = changes.get(path, narrative_fields(response).get(path, ''))
        if field.count(quote) != 1:
            raise ValueError('Chemical claim correction quotation is not unique')
        changes[path] = field.replace(quote, replacement, 1)
        receipts.append({'response_path': path, 'quote': quote,
                         'replacement': replacement,
                         'evidence_paths': finding['evidence_paths'],
                         'positions': [{'position_1based': position,
                                        'region': regions[position]}
                                       for position in positions]})
    corrected = apply_bound_narrative_patch(
        response, {'corrections': changes}, findings, source)
    markdown = corrected.get('markdown')
    if isinstance(markdown, str) and markdown:
        for receipt in receipts:
            if receipt['quote'] in markdown:
                markdown = markdown.replace(receipt['quote'], receipt['replacement'])
        corrected['markdown'] = markdown
    if native_chemical_position_findings(source, corrected):
        raise ValueError('Chemical claim correction left a native contradiction')
    return corrected, receipts


def evidence_pointer_candidates(document, pointer, limit=8):
    """Suggest existing suffix matches only inside the closest valid parent.

    A suggestion establishes only path existence, never evidence relevance.
    In particular, a missing field on one row must not be filled from another.
    """
    result = {'candidates': [], 'total_candidates': 0, 'searched_parent': ''}
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        return result
    try:
        pointer_value(document, pointer)
        return result  # Valid citations do not need repair.
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    wanted = tuple(token.replace('~1', '/').replace('~0', '~')
                   for token in pointer[1:].split('/'))
    prefix = ()
    parent = document
    for token in wanted:
        candidate = prefix + (token,)
        try:
            value = pointer_value(document, '/' + '/'.join(_escape(x) for x in candidate))
        except (KeyError, IndexError, TypeError, ValueError):
            break
        prefix, parent = candidate, value
    result['searched_parent'] = '/' + '/'.join(_escape(x) for x in prefix) if prefix else ''
    matches = []
    stack = [(prefix, parent)]
    while stack:
        path, value = stack.pop()
        suffix = 0
        for left, right in zip(reversed(path), reversed(wanted)):
            if left != right:
                break
            suffix += 1
        if suffix:
            rendered = '/' + '/'.join(_escape(x) for x in path)
            matches.append((-suffix, abs(len(path) - len(wanted)), rendered))
        if isinstance(value, dict):
            stack.extend((path + (str(key),), item) for key, item in value.items())
        elif isinstance(value, list):
            stack.extend((path + (str(index),), item) for index, item in enumerate(value))
    matches.sort()
    result['total_candidates'] = len(matches)
    result['candidates'] = [item[2] for item in matches[:limit]]
    return result


def response_audit_contract_errors(audit, response, source):
    fields = narrative_fields(response)
    errors = []
    for index, finding in enumerate(audit.get("findings") or []):
        path, quote = finding.get("response_path"), finding.get("quote")
        if path not in fields or not isinstance(quote, str) or not quote or quote not in fields[path]:
            errors.append(f"findings[{index}] must quote an exact substring of the specified draft field")
        for pointer in finding.get("evidence_paths") or []:
            try:
                pointer_value(source, pointer)
            except (KeyError, IndexError, TypeError, ValueError):
                hints = evidence_pointer_candidates(source, pointer)
                message = f"findings[{index}] has an unknown source evidence pointer: {pointer}"
                if hints['candidates']:
                    message += ('; existing paths under the closest valid parent (hints only, not equivalence or evidence relevance; inspect the cited source before choosing): '
                                + json.dumps(hints, ensure_ascii=False))
                errors.append(message)
    return errors


def response_audit_repair_findings(audit, response, source):
    errors = response_audit_contract_errors(audit, response, source)
    if errors:
        raise ValueError("; ".join(errors))
    return [{
        **deepcopy(finding),
        # The complete source is already in the repair context. Bind citations
        # without duplicating potentially large alignment inventories.
        "evidence_sha256": {
            pointer: hashlib.sha256(json.dumps(pointer_value(source, pointer),
                ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            for pointer in finding["evidence_paths"]},
    } for finding in audit["findings"]]


def response_audit_receipt(response, source, *, attempts, model):
    def digest(value):
        body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    return {"status": "no_factual_contradiction_flagged", "model": model,
            "draft_sha256": digest(response), "source_sha256": digest(source),
            "reviewed_fields": list(narrative_fields(response)), "attempts": attempts,
            "scope": "Native model consistency review of supplied evidence; not independent validation or experimental evidence."}


def merge_bound_narrative_repair(previous, candidate, findings, source):
    """Apply corrections only to narrative fields identified by the prior audit.

    A factual correction must not regenerate unrelated accepted assessments,
    row identities, measurements, or artifact metadata as a side effect.
    The complete merged response still requires a new consistency review.
    """
    errors = response_audit_contract_errors({'findings': findings}, previous, source)
    if errors:
        raise ValueError('; '.join(errors))
    result = deepcopy(previous)
    for path in dict.fromkeys(finding['response_path'] for finding in findings):
        tokens = [part.replace('~1', '/').replace('~0', '~') for part in path[1:].split('/')]
        old_parent, new_parent, dest = previous, candidate, result
        try:
            for token in tokens[:-1]:
                index = int(token) if isinstance(old_parent, list) else token
                old_parent, new_parent, dest = old_parent[index], new_parent[index], dest[index]
                if isinstance(old_parent, dict):
                    for identity in ('input_row_id', 'name'):
                        if identity in old_parent and new_parent.get(identity) != old_parent[identity]:
                            raise ValueError('Factual repair changed the identity of a reviewed observation')
            index = int(tokens[-1]) if isinstance(dest, list) else tokens[-1]
            replacement = new_parent[index]
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError('Factual repair omitted the flagged narrative field: ' + path) from exc
        if not isinstance(replacement, str):
            raise ValueError('Factual repair must preserve narrative field types: ' + path)
        dest[index] = replacement
    return result


def narrative_repair_schema(findings):
    """An explicit edit protocol, limited to previously audited text fields."""
    paths = list(dict.fromkeys(item['response_path'] for item in findings))
    return {
        'type': 'object', 'additionalProperties': False, 'required': ['corrections'],
        'properties': {'corrections': {
            'type': 'object', 'additionalProperties': False, 'required': paths,
            'properties': {path: {'type': 'string', 'minLength': 1} for path in paths},
        }},
    }


def apply_bound_narrative_patch(previous, patch, findings, source):
    errors = response_audit_contract_errors({'findings': findings}, previous, source)
    if errors:
        raise ValueError('; '.join(errors))
    paths = set(item['response_path'] for item in findings)
    if not isinstance(patch, dict) or set(patch) != {'corrections'}:
        raise ValueError('Factual repair must contain only corrections.')
    changes = patch['corrections']
    if not isinstance(changes, dict) or set(changes) != paths:
        raise ValueError('Factual repair must correct exactly the flagged narrative paths.')
    result = deepcopy(previous)
    for path, replacement in changes.items():
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError('Factual repair must preserve nonempty narrative field types: ' + path)
        tokens = [part.replace('~1', '/').replace('~0', '~') for part in path[1:].split('/')]
        parent = result
        for token in tokens[:-1]:
            parent = parent[int(token)] if isinstance(parent, list) else parent[token]
        parent[int(tokens[-1]) if isinstance(parent, list) else tokens[-1]] = replacement
    return result
