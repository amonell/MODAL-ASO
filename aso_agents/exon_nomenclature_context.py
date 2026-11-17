"""Reference-bound exon identities shared by design review and report rendering.

Public lookup is an execution step. Evidence projection below is offline and
never treats a requested biological name as a transcript-order index.
"""
from copy import deepcopy

from .curated_exon_nomenclature import (
    annotate_goal_exon_nomenclature, reference_context, reference_fingerprint, resolve_curated_label,
    verified_mapping_for_entry,
)
from .evaluation_feature_identity import transcript_matches
from .report_exon_names import project_exon_names


def selected_reference(catalog, requested=None):
    """Resolve one exact reference, allowing its recorded human-facing alias."""
    matches = {}
    for label, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        context = reference_context(entry)
        requested_text = str(requested or '')
        requested_base, dot, version = requested_text.rpartition('.')
        actual_base = context['transcript_id'].rpartition('.')[0]
        version_conflict = (dot and version.isdigit() and requested_base == actual_base
                            and requested_text != context['transcript_id'])
        if version_conflict:
            continue
        if requested and str(requested) != str(label) and not transcript_matches(
                str(requested), context['transcript_id']):
            continue
        signature = reference_fingerprint(context['transcript_id'], entry.get('exons_tx') or [], context)
        matches[signature] = (label, entry)
    if not matches and requested:
        # A verified RefSeq/Ensembl accession cross-reference can supply an
        # alias. Use the same version-aware attestation as transcript grouping.
        from .gapmer_selectivity import resolve_catalog_label
        alias = resolve_catalog_label(requested, catalog, catalog)
        if alias is not None:
            return alias, catalog[alias]
    return next(iter(matches.values())) if len(matches) == 1 else (None, None)


def exon_requests(goal):
    """Read typed feature declarations; request prose never supplies coordinates."""
    children = ('design_programs', 'evaluation_targets', 'region_constraints',
                'splice_isoform_objectives', 'explicit_splice_targets', 'site_direction_hints')
    def walk(node, path='', reference=None, program_id=None):
        if not isinstance(node, dict):
            return
        reference_keys = (
            'exon_numbering_reference_transcript_id', 'exon_numbering_transcript_id',
            'reference_transcript_id', 'transcript_id', 'reference_accession')
        if not (node.get('objective_scope') == 'aggregate_class' or node.get('aggregate_class')):
            reference_keys += ('isoform_label',)
        reference = next((node.get(key) for key in reference_keys if node.get(key)), reference)
        program_id = node.get('program_id', program_id)
        labels = node.get('exon_labels')
        if not isinstance(labels, list):
            labels = [node.get('requested_exon_label', node.get('exon_label'))]
        ordinal = node.get('exon_index')
        if type(ordinal) is int and ordinal > 0:
            labels = [node.get('requested_exon_label', node.get('exon_label'))]
        else:
            ordinal = None
        for label in labels:
            if label is not None or ordinal is not None:
                numbering_reference = reference
                default = goal.get('default_exon_context_reference') or {}
                if (not numbering_reference and str(label) in {str(value) for value in default.get('exon_labels') or []}
                        and (not path or node.get('objective_scope') == 'default_exon_context')):
                    numbering_reference = default.get('transcript_id') or default.get('isoform_label')
                request = {'request_path': path or '/', 'requested_exon_label': label,
                       'explicit_exon_index': ordinal, 'requested_reference': numbering_reference,
                       'program_id': program_id}
                # Only complete, explicitly declared exon spans establish identity.
                # A binding window or one splice-site coordinate cannot do so.
                if '/explicit_splice_targets/' in path:
                    request['explicit_exon_coordinates'] = {
                        key: deepcopy(node[key]) for key in ('chrom', 'start', 'end', 'strand') if key in node}
                    request['explicit_exon_coordinates']['assembly'] = (
                        node.get('genome_build') or goal.get('genome_build'))
                    # Constructed exon models retain their exact parent. The
                    # explicit target binds to that model while preserving the
                    # user's original accession as provenance.
                    catalog = goal.get('isoform_transcripts') or {}
                    model_label = node.get('isoform_label')
                    model = catalog.get(model_label) or {}
                    metadata = model.get('metadata') or {}
                    parent_label = model.get('explicit_target_original_partner') or metadata.get('base_isoform_label')
                    _, requested_entry = selected_reference(catalog, reference)
                    if (reference_context(model)['modified'] and parent_label in catalog
                            and requested_entry is catalog[parent_label]
                            and (not metadata.get('base_reference_transcript_id') or
                                 metadata['base_reference_transcript_id'] == reference_context(requested_entry)['transcript_id'])):
                        request['physical_reference'] = model_label
                yield request
        for key in children:
            values = node.get(key)
            for index, child in enumerate(values if isinstance(values, list) else []):
                yield from walk(child, f'{path}/{key}/{index}', reference, program_id)
        for key in ('splice_relative_window', 'aggregate_class'):
            if isinstance(node.get(key), dict):
                yield from walk(node[key], path+'/'+key, reference, program_id)
    lock = goal.get('transcript_lock') or {}
    return list(walk(goal, reference=lock.get('value')))


def prepare_goal_exon_nomenclature(goal):
    """Fetch once per reference and attach requested label resolutions before use."""
    annotate_goal_exon_nomenclature(goal)
    catalog = goal.get('isoform_transcripts') or {}
    for request in exon_requests(goal):
        _, entry = selected_reference(catalog, request.get('physical_reference') or request['requested_reference'])
        label = request['requested_exon_label']
        if entry is None or label is None or request['explicit_exon_index'] is not None:
            continue
        metadata = entry.setdefault('metadata', {})
        mappings = metadata.get('exon_nomenclature_resolutions')
        if not isinstance(mappings, dict):
            mappings = metadata['exon_nomenclature_resolutions'] = {}
        prior = mappings.get(str(label)) or (entry.get('exon_nomenclature_resolutions') or {}).get(str(label))
        if verified_mapping_for_entry(prior, entry):
            continue
        mappings[str(label)] = resolve_curated_label(metadata.get('curated_exon_nomenclature') or {}, label)


def resolved_exon_requests(goal):
    catalog = goal.get('isoform_transcripts') or {}
    result = []
    for request in exon_requests(goal):
        item = {**request, 'status': 'unresolved', 'biological_label_verified': False}
        _, entry = selected_reference(catalog, request.get('physical_reference') or request['requested_reference'])
        if entry is None:
            item['reason'] = 'The numbering reference is unavailable or ambiguous.'
            result.append(item)
            continue
        context = reference_context(entry)
        item.update(reference_transcript_id=context['transcript_id'], chrom=context['chrom'], strand=context['strand'])
        if request.get('physical_reference'):
            item['identity_reference_aliases'] = [request['requested_reference'], request['physical_reference'], context['transcript_id']]
        ordinal = request['explicit_exon_index']
        label = request['requested_exon_label']
        metadata = entry.get('metadata') or {}
        resolution = (entry.get('exon_nomenclature_resolutions') or
                      metadata.get('exon_nomenclature_resolutions') or {}).get(str(label))
        if not verified_mapping_for_entry(resolution, entry):
            resolution = resolve_curated_label(metadata.get('curated_exon_nomenclature') or {}, label)
        physical = request.get('explicit_exon_coordinates')
        if physical is not None:
            from .exon_identity_clarification import coordinate_exon
            exon = coordinate_exon(physical, entry)
            ordinal = exon.get('index') if exon else None
            if ordinal is not None:
                item['status'] = 'explicit_feature_coordinates'
            else:
                item['reason'] = 'The supplied exon span does not match this reference and assembly.'
        elif ordinal is not None:
            item['status'] = 'explicit_transcript_ordinal'
        elif not context['modified'] and verified_mapping_for_entry(resolution, entry):
            ordinal = resolution.get('resolved_transcript_ordinal')
            item.update(status='verified_biological_label', biological_label_verified=True,
                        evidence_refs=resolution.get('evidence_refs') or [])
        else:
            item['reason'] = ((resolution or metadata.get('curated_exon_nomenclature') or {}).get('reason')
                              or 'A public-source mapping for this exon name is unavailable.')
            # An explicit conversion between two resolved RNAs can identify a
            # single cassette even when its historical name has no public map.
            from .design_event_consistency import structural_exon_resolution
            event = structural_exon_resolution(entry, label, goal) if type(label) is int else None
            if event and event.get('status') == 'verified_structural_event':
                ordinal = event['resolved_transcript_ordinal']
                item.update(status='verified_structural_event', structural_event=event)
                item.pop('reason', None)
        matches = [e for e in entry.get('exons_tx') or [] if e.get('index') == ordinal]
        if ordinal is not None and len(matches) == 1:
            item.update(resolved_transcript_ordinal=ordinal, start=matches[0]['start'], end=matches[0]['end'],
                        coordinate_system='genomic_1based_inclusive')
        elif ordinal is not None:
            item.update(status='unresolved', biological_label_verified=False,
                        reason='The mapped exon is unavailable in this reference.')
        if ordinal is not None and type(label) is int:
            from .design_event_consistency import structural_exon_resolution
            event = structural_exon_resolution(entry, label, goal)
            if event and event.get('status') == 'verified_structural_event' and event['resolved_transcript_ordinal'] != ordinal:
                item.update(status='unresolved', biological_label_verified=False,
                            reason='The supplied exon identity conflicts with the requested transcript conversion.')
                for key in ('start', 'end', 'resolved_transcript_ordinal'):
                    item.pop(key, None)
                item['identity_conflict'] = True
        result.append(item)
    # The compiler can repeat an outcome in programs, evaluation and groups.
    # Reuse an explicit answer only for the same named feature and exact RNA.
    from .exon_identity_clarification import share_explicit_identities
    share_explicit_identities(result)
    return result


def exon_review_context(goal):
    """Supply the same names and coordinate evidence used by the HTML plot."""
    references = {}
    for label, entry in (goal.get('isoform_transcripts') or {}).items():
        if not isinstance(entry, dict):
            continue
        context = reference_context(entry)
        projection = project_exon_names(entry)
        references[label] = {**context, 'exons': projection['exons'],
                             'naming_verified': projection['exon_naming']['verified']}
    return {'references': references, 'requested_features': resolved_exon_requests(goal),
            'interpretation': 'Use verified biological labels for named-exon claims. '
                'Explicit user definitions and verified structural events establish the requested physical feature; '
                'describe these using their coordinates and annotation-order names. '
                'Unresolved labels retain the user request; their feature coordinates are unknown. '
                'Describe other exons using the supplied annotation-order display names. '
                'A transcript accession identifies a reference RNA; exon-defined isoform groups require separate membership evidence.'}


def annotate_candidate_exon_identity(rows, goal):
    """Attach exact overlap evidence, preserving scores and candidate membership."""
    from .candidate_export_evidence import export_context
    from .execution_references import executed_reference
    catalog = goal.get('isoform_transcripts') or {}
    requests = resolved_exon_requests(goal)
    programs = {p.get('program_id'): p for p in goal.get('design_programs') or [] if isinstance(p, dict)}
    projections, reference_matches = {}, {}
    for row in rows:
        source = row.get('source_annotations') or {}
        reference = row.get('executed_reference') or source.get('executed_reference') or {}
        reference = reference if isinstance(reference, dict) else {}
        program = programs.get(row.get('program_id'), {})
        requested = (reference.get('versioned_id') or reference.get('versioned_transcript_id') or reference.get('transcript_id')
                     or row.get('reference_transcript_id') or row.get('transcript_id')
                     or program.get('transcript_id') or program.get('isoform_label'))
        if requested not in reference_matches:
            reference_matches[requested] = selected_reference(catalog, requested)
        key, entry = reference_matches[requested]
        if entry is None:
            row['exon_identity_evidence'] = {'status': 'unresolved_reference'}
            continue
        if key not in projections:
            projections[key] = (reference_context(entry), project_exon_names(entry)['exons'],
                                executed_reference({}, key, entry)['structure_sha256'])
        context, exons, structure_sha = projections[key]
        if reference.get('structure_sha256') and reference['structure_sha256'] != structure_sha:
            row['exon_identity_evidence'] = {'status': 'unresolved_reference'}
            continue
        geometry = export_context(row)
        strand = {'1': '+', '-1': '-'}.get(str(geometry.get('strand')), geometry.get('strand'))
        intervals = geometry.get('binding_intervals_1based') or []
        geometry_valid = (strand == context['strand'] and bool(intervals) and all(
            str(interval.get('chrom') or '').removeprefix('chr') == str(context['chrom'] or '').removeprefix('chr')
            and type(interval.get('start')) is int and type(interval.get('end')) is int
            and 1 <= interval['start'] <= interval['end'] for interval in intervals))
        evidence = {'reference_transcript_id': context['transcript_id'], 'status': 'resolved_reference',
                    'binding_exons': [], 'requested_features': []}
        def overlap(exon):
            if not geometry_valid:
                return None
            spans = sorted((max(i['start'], exon['start']), min(i['end'], exon['end']))
                           for i in intervals if max(i['start'], exon['start']) <= min(i['end'], exon['end']))
            total, previous_end = 0, 0
            for start, end in spans:
                total += max(0, end-max(start, previous_end+1)+1)
                previous_end = max(previous_end, end)
            return total
        for exon in exons:
            amount = overlap(exon)
            if amount:
                evidence['binding_exons'].append({'annotation_order': exon['index'],
                    'biological_label': exon['verified_exon_label'], 'display_name': exon['label'],
                    'start': exon['start'], 'end': exon['end'], 'overlap_nt': amount})
        for feature in requests:
            if feature.get('reference_transcript_id') != context['transcript_id']:
                continue
            if feature.get('program_id') and feature['program_id'] != row.get('program_id'):
                continue
            item = {k: deepcopy(feature[k]) for k in ('request_path', 'requested_exon_label', 'status',
                'biological_label_verified', 'resolved_transcript_ordinal') if k in feature}
            item['overlap_nt'] = overlap(feature) if feature['status'] != 'unresolved' else None
            evidence['requested_features'].append(item)
        row['exon_identity_evidence'] = evidence
