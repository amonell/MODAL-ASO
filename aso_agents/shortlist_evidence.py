"""Lossless, bounded evidence packets for review of physical ASO molecules.

Repeated facts are shared within a packet. Scientific values keep their original
precision; source pointers refer to the assessed rows retained with the run.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import sys

from .candidate_presentation import molecule_key


def digest(value):
    result = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).iterencode(value):
        result.update(chunk.encode())
    return result.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if not str(k).startswith('_recommendation')}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def molecule_id(row):
    return 'aso_' + digest(molecule_key(row))[:16]


def fact_id(pointer):
    return 'f_' + digest(pointer)[:12]


def _leaves(value, path=''):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            escaped = key.replace('~', '~0').replace('/', '~1')
            yield from _leaves(child, path + '/' + escaped)
    elif isinstance(value, list) and value and any(isinstance(v, dict) for v in value):
        for index, child in enumerate(value):
            yield from _leaves(child, path + '/' + str(index))
    else:
        yield path, value


def coordinate_observations(row, geometry):
    """Interpret a contiguous declared target in RNA direction from BED intervals.

    Discontinuous targets retain their original evidence without extrapolating
    across introns. Target-strand agreement is required before deriving a fact.
    """
    intervals = geometry.get('binding_intervals_1based') or []
    if len(intervals) != 1 or geometry.get('strand') not in ('+', '-'):
        return []
    interval = intervals[0]
    if not isinstance(interval, dict):
        return []
    try:
        lo, hi = int(interval['start']) - 1, int(interval['end'])
    except (KeyError, TypeError, ValueError):
        return []
    if lo < 0 or hi <= lo:
        return []
    observations = []
    annotations = row.get('source_annotations') or {}
    memberships = annotations.get('binding_site_membership') or [] if isinstance(annotations, dict) else []
    for index, member in enumerate(memberships):
        if not isinstance(member, dict):
            continue
        segments = member.get('genomic_segments_bed0_half_open') or []
        if len(segments) != 1 or not isinstance(segments[0], (list, tuple)) or len(segments[0]) < 6:
            continue
        chrom, start, end, _, _, strand = segments[0][:6]
        if chrom != interval.get('chrom') or strand != geometry['strand']:
            continue
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            continue
        left, right = max(lo, start), min(hi, end)
        if left >= right:
            continue
        covered = [left - start + 1, right - start] if strand == '+' else [end - right + 1, end - left]
        upstream = max(0, start - lo) if strand == '+' else max(0, hi - end)
        downstream = max(0, hi - end) if strand == '+' else max(0, start - lo)
        observations.append({'target_label': member.get('label') or member.get('constraint_label'),
            'target_strand': strand, 'RNA_direction_in_genomic_coordinates': 'increasing' if strand == '+' else 'decreasing',
            'ASO_binding_interval_bed0_half_open': [chrom, lo, hi],
            'target_interval_bed0_half_open': [chrom, start, end], 'target_length_nt': end - start,
            'target_bases_covered_1based_in_RNA_order': covered,
            'bases_beyond_target_upstream_in_RNA': upstream,
            'bases_beyond_target_downstream_in_RNA': downstream,
            'source_membership_index': index})
    return observations


def panel_binding_geometry(catalog, selected_ids):
    """Compare unambiguous contiguous binding placements in RNA direction."""
    anchors, result = {}, {}
    for order, key in enumerate(selected_ids, 1):
        placements = set()
        ambiguous = False
        for context in catalog[key].get('overview', []):
            geometry = context.get('binding_and_objective') or {}
            intervals = geometry.get('binding_intervals_1based') or []
            if len(intervals) != 1 or geometry.get('strand') not in ('+', '-'):
                ambiguous = True
                continue
            interval = intervals[0]
            if not isinstance(interval, dict) or not interval.get('chrom'):
                ambiguous = True
                continue
            start, end = interval.get('start'), interval.get('end')
            if type(start) is int and type(end) is int and 1 <= start <= end:
                placements.add((interval['chrom'], geometry['strand'], start, end))
            else:
                ambiguous = True
        if ambiguous or len(placements) != 1:
            continue
        chrom, strand, start, end = placements.pop()
        rna_start, rna_end = (start, end) if strand == '+' else (-end, -start)
        anchor = anchors.setdefault((chrom, strand), (key, order, rna_start, rna_end))
        start_shift, end_shift = rna_start - anchor[2], rna_end - anchor[3]
        result[key] = {'panel_order': order, 'chrom': chrom, 'strand': strand,
            'binding_interval_1based_inclusive': [start, end],
            'relative_to_candidate_id': anchor[0], 'relative_to_panel_order': anchor[1],
            'start_shift_nt_in_RNA': start_shift, 'end_shift_nt_in_RNA': end_shift,
            'start_direction_from_anchor': 'upstream' if start_shift < 0 else 'downstream' if start_shift > 0 else 'aligned',
            'end_direction_from_anchor': 'upstream' if end_shift < 0 else 'downstream' if end_shift > 0 else 'aligned'}
    return result


def evidence_catalog(pool):
    """Include all saved fields; hash bulk sequences and local file locations.

    Genomes and full reference sequences are provenance, while the ASO and its
    short binding context remain inspectable. Nothing follows filesystem paths.
    """
    entries = {}
    for index, original in enumerate(pool):
        row = clean(original)
        key = molecule_id(row)
        entry = entries.setdefault(key, {'id': key, 'identity': list(molecule_key(row)),
            'row_indices': [], 'facts': {}, 'overview': [], 'coordinate_observations': [], 'eligible': False})
        entry['row_indices'].append(index)
        entry['eligible'] |= (row.get('candidate_assessment') or {}).get('selection_eligible') is True
        context = len(entry['row_indices']) - 1
        from .candidate_export_evidence import export_context
        assessment = row.get('candidate_assessment') or {}
        activity = row.get('oligoai_evaluation') or {}
        geometry = export_context(row)
        entry['coordinate_observations'].extend({'context': context, **item}
            for item in coordinate_observations(row, geometry))
        entry['overview'].append({'context': context,
            'binding_and_objective': {k: v for k, v in geometry.items() if v is not None},
            'length': row.get('length'), 'modality': row.get('source_modality'),
            'gc_fraction': row.get('gc_fraction'),
            'evidence_status': assessment.get('recommendation_status'),
            'native_prediction': assessment.get('native_prediction'),
            'off_target': {k: v for k, v in (assessment.get('off_target') or {}).items() if k in {
                'status', 'mismatch_budget', 'counts_by_mismatch', 'unit', 'expression_weighted_risk',
                'expression_tissue', 'incomplete_reasons', 'species_statuses'}},
            'motifs': assessment.get('motifs'),
            'splice_safeguards': {k: v for k, v in (assessment.get('splice_safeguards') or {}).items()
                                  if k in {'status', 'priority'}},
            'unavailable_evidence': assessment.get('unavailable_evidence'),
            'activity': {'status': activity.get('status'), 'conditions': activity.get('conditions'),
                'contexts': [{k: v for k, v in item.items() if k in {
                    'transcript_id', 'label', 'species', 'assembly', 'kind', 'status',
                    'context_basis', 'predicted_inhibition_percent', 'flank_padding_nt',
                    'masked_flank_nt', 'binding_interval_bed0_half_open'}}
                    for item in activity.get('contexts') or []]}})
        for path, value in _leaves(row):
            pointer = sys.intern(f'/contexts/{context}' + path)
            if isinstance(value, str) and (value.startswith('/') and len(value) > 100
                    or len(value) > 512 and set(value.upper()) <= set('ACGTUNRYKMSWBDHV-')):
                value = {'stored_value_sha256': digest(value), 'length': len(value),
                         'representation': 'bulk_sequence_or_local_artifact_location'}
            entry['facts'][pointer] = value
    return entries


def compact_packet(entries, *, citation_numbers=False, include_overviews=False):
    """Fact columns use common values and exact per-ASO exceptions.

    Paths with identical value vectors are aliases, so nested copies of the
    same measurement occupy one column. Citations use the original paths.
    """
    ids = [entry['id'] for entry in entries]
    columns = defaultdict(dict)
    for entry in entries:
        for pointer, value in entry['facts'].items():
            columns[pointer][entry['id']] = value
    groups = {}
    for pointer, values in columns.items():
        key = digest(values)
        if key not in groups:
            groups[key] = {'paths': [], 'values': values}
        groups[key]['paths'].append(pointer)
    facts, reference_number = [], 0
    for group in groups.values():
        values = group.pop('values')
        paths = {}
        for path in group['paths']:
            reference_number += 1
            paths['f_' + str(reference_number) if citation_numbers else fact_id(path)] = path
        group['paths'] = paths
        frequencies = defaultdict(list)
        for key, value in values.items():
            frequencies[json.dumps(value, sort_keys=True, ensure_ascii=False)].append(key)
        common = max(frequencies, key=lambda v: len(frequencies[v]))
        common_ids = frequencies[common]
        if len(common_ids) > 1:
            fact = {**group, 'shared': json.loads(common),
                'overrides': {str(ids.index(k)): v for k, v in values.items() if k not in common_ids}}
            if len(values) != len(ids):
                fact['available_for'] = [ids.index(k) for k in values]
        else:
            fact = {**group, 'values_in_id_order': [values.get(k) for k in ids]}
        facts.append(fact)
    return {'ids': ids, **({'candidate_overviews': [
                {'id': entry['id'], 'antisense_5to3': entry['identity'][0], 'contexts': entry['overview']}
                for entry in entries]} if include_overviews else {}), 'facts': facts,
            'reading_guide': 'paths maps short citation IDs to exact saved evidence paths. Cite the short IDs. '
                'Shared values apply to all ASO IDs '
                'unless available_for restricts them; overrides replace the shared value. '
                'overrides and available_for use zero-based indices into ids. '
                'values_in_id_order follows ids. Absent fields and null mean unavailable. '
                'contexts are separate saved binding/analysis records for the same physical molecule.'}


def _decision_facts(entry):
    """Present the normalized analysis once; retain new evidence fields.

    These are known duplicate export/provenance containers, rather than a
    whitelist of scientific metrics. Individual predictions and new attributes
    outside those containers survive without a schema update.
    """
    duplicate_roots = {'off_target_by_species', 'off_target_search_provenance_by_species',
        'off_target_expression_by_species', 'Off_target_expression_details',
        'off_target_expression_details', 'off_target_reference_coverage'}
    duplicate_assessment = {'reference_coverage': 'off_target_reference_coverage',
        'references_by_species': 'off_target_search_provenance_by_species',
        'species_evidence': 'off_target_by_species'}
    technical = {'source_path', 'physical_molecule_sha256', 'input_id', 'candidate_id',
        'candidate_number', 'reference_sequence_sha256', 'rna_context', 'nucleobases_5to3',
        'ordering_notation', 'IDT', 'Synthesis', 'artifact_json',
        'artifact_csv', 'provenance', 'genome_fasta', 'annotation_gtf', 'bowtie_index_prefix',
        'reference_annotation', 'parameter_table_sha256', 'biopython_version', 'source_revision', 'model_sha256'}
    facts = {}
    for path, value in entry['facts'].items():
        parts = path.split('/')[3:]
        context = '/'.join(path.split('/')[:3])
        if parts and parts[-1] in {'IDT', 'ordering_notation'} and not all(
                context + '/molecule_specification/' + field in entry['facts']
                for field in ('nucleotide_sugars_5to3', 'linkages_5to3')):
            facts[path] = value
            continue
        if not parts or technical.intersection(parts):
            continue
        if parts[-1] == 'nucleotide_chemistry_5to3':
            prefix = path.rsplit('/', 1)[0]
            if all(prefix + '/' + key in entry['facts'] for key in ('nucleotide_sugars_5to3', 'base_modifications_5to3')):
                continue
        if parts[0] in duplicate_roots:
            # Retain scientific scalar values that a future backend adds, and
            # the declared biological search scope. Export cache bookkeeping
            # and repeated input/reference trees stay in the full catalog.
            if parts[0] in {'off_target_by_species', 'off_target_expression_by_species'} and not any(key in parts for key in
                    ('off_target_count_scope', 'off_target_reference_coverage', 'Off_target_expression_details',
                     'off_target_expression_details', 'junction_search_summary', 'supplemental_transcript_reference')):
                facts[path] = value
            continue
        if parts[:2] == ['candidate_assessment', 'off_target'] and len(parts) > 2 and parts[2] in duplicate_assessment:
            counterpart = context + '/' + '/'.join([duplicate_assessment[parts[2]], *parts[3:]])
            if counterpart in entry['facts'] and entry['facts'][counterpart] == value:
                continue
        if parts[0] == 'source_annotations' and len(parts) > 1 and parts[1] in {
                'id', 'Name', 'source', 'executed_reference', 'repeat_inventory_evidence',
                'off_target_count_scope', 'off_target_observed_gene_ids_by_mismatch',
                'off_target_observed_genes_by_mismatch', 'Off_target_gene_ids_by_mismatch',
                'Off_target_genes_by_mismatch'}:
            continue
        facts[path] = value
    # Search scope has biological meaning even when its provenance is large.
    for path, value in entry['facts'].items():
        if '/off_target_count_scope/' in path and path.rsplit('/', 1)[-1] in {
                'count_unit', 'assembly', 'genomic_search', 'counted_genomic_rna_universe',
                'genomic_strand_policy', 'junction_search', 'junction_strand_policy',
                'mismatch_policy', 'mismatch_bin_count_unit', 'weighted_risk_definition',
                'indel_policy', 'intended_target_exclusion', 'limitations'}:
            # All duplicated copies carry the same fact; keep one canonical
            # source container for each context/species and scope attribute.
            if '/off_target_by_species/' in path and '/candidate_assessment/' not in path:
                facts[path] = value
    return facts


def screening_packet(entries, *, include_provenance=False, local_ids=True):
    """Candidate rows for nomination, with shared constants and exact values.

    Source paths and molecular identities remain in the immutable full catalog.
    First-pass packets omit file locations, duplicated model inputs and technical
    provenance. All measurements, conditions, constraints, chemistry, match
    identities, uncertainty and unrecognized future attributes remain available.
    """
    projected = entries if include_provenance else [{**entry, 'facts': _decision_facts(entry)} for entry in entries]
    packet = compact_packet(projected)
    facts = [{'paths': list(fact['paths'].values()),
              **{key: value for key, value in fact.items() if key != 'paths'}}
             for fact in packet['facts']]
    for fact in facts:
        available = [i for i, entry in enumerate(projected) if fact['paths'][0] in entry['facts']]
        if len(available) != len(entries):
            fact['available_for'] = available
    frequencies, originals = defaultdict(int), {}

    def cells(fact):
        if 'shared' in fact:
            yield fact['shared']
        yield from fact.get('values_in_id_order', [])
        yield from fact.get('overrides', {}).values()

    for fact in facts:
        for value in cells(fact):
            serialized = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
            if len(serialized) >= 32:
                frequencies[serialized] += 1
                originals[serialized] = value
    references = {value: 'v' + str(index) for index, (value, count)
                  in enumerate(frequencies.items()) if count > 1}

    def share(value):
        key = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        return {'value_ref': references[key]} if key in references else value

    for fact in facts:
        if 'shared' in fact:
            fact['shared'] = share(fact['shared'])
        if 'values_in_id_order' in fact:
            fact['values_in_id_order'] = [share(value) for value in fact['values_in_id_order']]
        if 'overrides' in fact:
            fact['overrides'] = {key: share(value) for key, value in fact['overrides'].items()}
    # Reading a whole candidate row keeps its geometry, predictions and limits
    # together. Constant columns are stated once. Exact source paths identify
    # fields, including distinct attributes that happen to share values.
    shared, columns, values, availability = [], [], [], {}
    for fact in facts:
        paths = fact.pop('paths')
        label = paths
        if 'shared' in fact and not fact.get('overrides') and 'available_for' not in fact:
            shared.append({'paths': label, 'value': fact['shared']})
            continue
        columns.append(label)
        available = set(fact.get('available_for', range(len(entries))))
        if len(available) != len(entries):
            availability[str(len(columns) - 1)] = sorted(available)
        values.append([(None if i not in available else
                        fact['values_in_id_order'][i] if 'values_in_id_order' in fact else
                        fact.get('overrides', {}).get(str(i), fact['shared']))
                       for i in range(len(entries))])
    ids = list(range(len(entries))) if local_ids else [entry['id'] for entry in entries]
    labels = []
    for paths in columns:
        parts = min(paths, key=len).split('/')[1:]
        if parts[:2] == ['contexts', '0']:
            parts = parts[2:]
        if parts and parts[0] in {'candidate_assessment', 'source_annotations'}:
            parts = parts[1:]
        label = '/'.join(parts)
        if label in labels:
            label = min(paths, key=len)
        labels.append(label)
    return {'ids': ids, 'shared_facts': shared,
        'fields': dict(zip(labels, columns)),
        'rows': [{'id': key, 'values': {labels[j]: column[i] for j, column in enumerate(values)
                  if str(j) not in availability or i in availability[str(j)]}}
                 for i, key in enumerate(ids)],
        'shared_values': {reference: originals[value] for value, reference in references.items()},
        'reading_guide': 'Each row gives one candidate ID and its named values. fields maps those names '
            'to equivalent exact source paths. Every candidate also has each shared_facts value '
            'at its listed paths. '
            'A value_ref cell uses the exact shared_values entry. Null and absent fields are missing evidence. '
            + ('Every original fact and source path is included. ' if include_provenance else
               'Technical provenance and duplicate encodings stay in the full catalog for detailed review. ')
            +
            'Read actual binding coordinates and measured effects: a program or transcript label identifies '
            'the requested objective and reference, and does not establish an exon overlap or isoform selectivity.'}


def nested_packet(entries):
    """An equivalent tree encoding for records dominated by long fact paths.

    Every leaf retains its exact path and value. Numeric keys are array indices;
    keeping them as keys permits sparse shared trees without inserting nulls.
    """
    def tree(facts):
        result = {}
        for pointer, value in facts.items():
            parts = [part.replace('~1', '/').replace('~0', '~') for part in pointer.split('/')[1:]]
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return result
    common = dict(entries[0]['facts']) if entries else {}
    for entry in entries[1:]:
        common = {path: value for path, value in common.items()
                  if path in entry['facts'] and digest(entry['facts'][path]) == digest(value)}
    return {'ids': [entry['id'] for entry in entries], 'shared_facts': tree(common),
        'candidate_facts': [{'id': entry['id'], 'facts': tree({path: value for path, value
            in entry['facts'].items() if path not in common})} for entry in entries],
        'reading_guide': 'Each candidate has ALL shared_facts plus its own facts, merged recursively. '
            'Numeric object keys are zero-based array indices. Missing fields mean unavailable. '
            'contexts are saved binding/analysis records for the same physical molecule. '
            'All saved scientific facts retain their original precision.'}


def indexed_packet(entries, *, packet=None):
    """Share pointer prefixes as well as fact values for large comparisons.

    Prefix collisions preserve both a scalar path and its descendants, which
    can occur when a field is null in one source record and structured in another.
    """
    packet = compact_packet(entries) if packet is None else packet
    tree, values = {}, {}
    for index, fact in enumerate(packet['facts'], 1):
        key = 'v' + str(index)
        values[key] = {name: value for name, value in fact.items() if name != 'paths'}
        first_path = next(iter(fact['paths'].values()))
        available = [i for i, entry in enumerate(entries) if first_path in entry['facts']]
        if len(available) != len(entries):
            values[key]['available_for'] = available
        for pointer in fact['paths'].values():
            parts = [part.replace('~1', '/').replace('~0', '~') for part in pointer.split('/')[1:]]
            node = tree
            for part in parts[:-1]:
                child = node.setdefault(part, {})
                if isinstance(child, str):
                    node[part] = child = [child, {}]
                node = child[1] if isinstance(child, list) else child
            child = node.get(parts[-1])
            node[parts[-1]] = ([key, child] if isinstance(child, dict) else
                              [key, child[1]] if isinstance(child, list) else key)
    return {'ids': packet['ids'], 'path_index': tree, 'fact_values': values,
        'reading_guide': 'path_index mirrors the complete source context tree. Each leaf names a key in '
            'fact_values. A node [value_key, child_fields] preserves both a scalar field and its children '
            'when their source types differ across ASOs. Resolve every path separately. '
            'Numeric child keys are array indices. A fact shared value applies to all IDs unless '
            'available_for restricts it; overrides replace shared values. overrides and available_for '
            'use zero-based indices into ids. values_in_id_order follows ids. Null is unavailable. '
            'All original scientific values and source paths are retained.'}


def review_context(goal):
    # Use the typed request contract, excluding generated artifacts and genomic
    # data. Keep selected-reference identity and objective geometry separately.
    from .goal_contract import canonical_structured_goal_schema
    fields = canonical_structured_goal_schema()['properties']
    result = {key: clean(value) for key, value in goal.items() if key in fields
              and key not in {'isoform_transcripts', 'patient_reference'}}
    for key in ('request', 'user_request', 'request_text'):
        if goal.get(key):
            result[key] = goal[key]
    references = {}
    for name, entry in (goal.get('isoform_transcripts') or {}).items():
        references[name] = {k: clean(v) for k, v in entry.items() if k in {
            'transcript_id', 'versioned_id', 'reference_accession', 'assembly', 'species',
            'chrom', 'strand', 'exons_tx', 'curated_exon_nomenclature', 'role'}}
        references[name]['metadata'] = {k: clean(v) for k, v in (entry.get('metadata') or {}).items()
            if k in {'versioned_id', 'species', 'assembly_name', 'is_canonical', 'is_mane_select',
                     'curated_exon_nomenclature', 'exon_nomenclature_resolutions'}}
    result['selected_references'] = references
    return result


def fact_value(entry, pointer):
    """Resolve exact saved leaves and JSON pointers inside retained arrays."""
    if pointer in entry['facts']:
        return entry['facts'][pointer]
    parts = pointer.split('/')
    for end in range(len(parts) - 1, 2, -1):
        parent = '/'.join(parts[:end])
        value = entry['facts'].get(parent)
        if not isinstance(value, list):
            continue
        try:
            for part in parts[end:]:
                token = part.replace('~1', '/').replace('~0', '~')
                if isinstance(value, list):
                    index = int(token)
                    if index < 0 or str(index) != token:
                        raise KeyError(pointer)
                    value = value[index]
                elif isinstance(value, dict):
                    value = value[token]
                else:
                    raise KeyError(pointer)
            return value
        except (KeyError, IndexError, ValueError):
            raise KeyError(pointer) from None
    raise KeyError(pointer)


def citation(entry, pointer):
    """Resolve a reference to exact saved evidence, never model-authored values."""
    value = fact_value(entry, pointer)
    parts = pointer.split('/')
    context = int(parts[2])
    return {'candidate_id': entry['id'], 'antisense_5to3': entry['identity'][0], 'path': pointer,
        'pool_row_index': entry['row_indices'][context],
        'source_pointer': '/rows/' + str(entry['row_indices'][context]) + '/' + '/'.join(parts[3:]),
        'value': deepcopy(value)}
