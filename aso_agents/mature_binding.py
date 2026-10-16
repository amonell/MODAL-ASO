"""Exact, coherent target placements across explicitly resolved mature RNAs."""
from copy import deepcopy
import hashlib
import json

from .gapmer_selectivity import resolve_catalog_label
from .region_overlap import candidate_matches_region_groups, candidate_matches_regions, combine_constraint_matches
from .region_projection import genomic_sources
from .reporting_contracts import generated_sequence_fields


def reference_id(label, entry):
    return (entry.get('metadata') or {}).get('versioned_id') or entry.get('versioned_id') or entry.get('transcript_id') or label


def reference_fingerprint(label, entry):
    value = {'reference': reference_id(label, entry), 'chrom': entry.get('chrom'),
             'strand': entry.get('strand'), 'exons': entry.get('exons_tx')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compile_scope(runtime, goal, program, anchor, base, constraints):
    catalog = goal.get('isoform_transcripts') or {}
    required = []
    for reference in program.get('transcript_ids') or []:
        label = resolve_catalog_label(reference, catalog, catalog)
        if label is None:
            raise ValueError('mature_binding_cohort_reference_unresolved: ' + str(reference))
        if label not in required:
            required.append(label)
    if required and anchor not in required:
        raise ValueError('mature_binding_anchor_outside_required_cohort')
    labels = list(dict.fromkeys([anchor, *required]))
    filters = []
    for constraint in constraints:
        space = ('cdna' if constraint.get('coordinate_space') == 'cdna'
                 or constraint.get('source_type') == 'transcript_feature'
                 or constraint.get('materialized_cdna_rows') else 'genomic')
        rows = runtime._constraint_rows_in_output_space(structured_goal=goal, constraint=constraint,
            output_space=space, base_rows=base, isoform_transcripts=catalog, iso_label_primary=anchor)
        normalized = []
        for row in rows:
            if space == 'cdna':
                label = resolve_catalog_label(row[0], catalog, catalog)
                if label is None:
                    raise ValueError('mature_binding_constraint_reference_unresolved: ' + str(row[0]))
                if label not in labels:
                    labels.append(label)
                normalized.append([reference_id(label, catalog[label]), *row[1:]])
            else:
                normalized.append(deepcopy(row))
        rule = {'rows': normalized, 'coordinate_space': space,
                'operator': constraint.get('operator') or 'union',
                'aso_overlap_requirement': constraint.get('aso_overlap_requirement') or 'any_overlap',
                'minimum_overlap_bp': constraint.get('minimum_overlap_bp') or 1,
                'constraint_id': constraint.get('constraint_id')}
        if space == 'genomic':
            groups = genomic_sources(goal, constraint)
            rule['region_groups'] = [item['genomic_rows'] for item in groups] or [[row] for row in normalized]
        filters.append(rule)
    if len(labels) < 2:
        return None
    anchor_rows = [[reference_id(anchor, catalog[anchor]), *row[1:]] for row in base
                   if resolve_catalog_label(row[0], catalog, catalog) == anchor]
    if len(anchor_rows) != len(base):
        raise ValueError('mature_binding_search_reference_unresolved')
    return {'anchor_label': anchor, 'required_labels': list(dict.fromkeys([anchor, *required])),
            'reference_labels': labels, 'anchor_search_rows': anchor_rows, 'filters': filters,
            'references': {label: reference_id(label, catalog[label]) for label in labels},
            'reference_fingerprints': {label: reference_fingerprint(label, catalog[label]) for label in labels},
            'placement_rule': 'One coherent exact target placement per required reference; ordered Boolean feature predicates.'}


class BindingContext:
    """Index each reference once per length and deduplicate predicate states."""
    def __init__(self, runtime, goal, scope):
        self.runtime, self.scope = runtime, scope
        self.catalog = {label: goal['isoform_transcripts'][label] for label in scope['reference_labels']}
        for label, entry in self.catalog.items():
            if reference_fingerprint(label, entry) != scope['reference_fingerprints'][label]:
                raise ValueError('mature_binding_reference_changed: ' + str(label))
        self.sequences = runtime._build_isoform_cdna_sequences(self.catalog, goal.get('species') or 'human',
            reference_fasta=runtime._resolve_patient_reference_fasta(goal))
        for label, entry in self.catalog.items():
            sequence = self.sequences.get(label)
            expected = sum(exon['end']-exon['start']+1 for exon in entry.get('exons_tx') or [])
            if not sequence or len(sequence) != expected or set(sequence) - set('ACGTN'):
                raise ValueError('mature_binding_reference_sequence_unresolved: ' + str(label))
        self.indices, self.signatures = {}, {}
        self.sequence_hashes = {scope['references'][label]: hashlib.sha256(seq.encode()).hexdigest()
                                for label, seq in self.sequences.items()}

    def positions(self, target):
        length = len(target)
        if length not in self.indices:
            indices = {}
            for label, sequence in self.sequences.items():
                index = {}
                for start in range(len(sequence)-length+1):
                    window = sequence[start:start+length]
                    if 'N' not in window:
                        index.setdefault(window, []).append(start)
                indices[label] = index
            self.indices[length] = indices
        return {label: index.get(target, []) for label, index in self.indices[length].items()}

    def segments(self, label, start, length):
        segments = self.runtime._map_cdna_range_to_genomic_segments(start+1, start+length, self.catalog[label])
        if sum(b-a+1 for _, a, b in segments) != length:
            raise ValueError('mature_binding_genomic_projection_incomplete')
        return [{'chrom': c, 'start': a, 'end': b} for c, a, b in segments]

    def signature(self, label, start, length):
        key = (label, start, length)
        if key in self.signatures:
            return self.signatures[key]
        cdna = [(self.scope['references'][label], start, start+length)]
        genomic = None
        mask = 0
        for i, rule in enumerate(self.scope['filters']):
            requirement, minimum = rule['aso_overlap_requirement'], rule['minimum_overlap_bp']
            if rule['coordinate_space'] == 'cdna':
                matches = candidate_matches_regions(cdna, rule['rows'], requirement, minimum)
            else:
                if genomic is None:
                    genomic = [(s['chrom'], s['start']-1, s['end']) for s in self.segments(label, start, length)]
                matches = candidate_matches_region_groups(genomic, rule['region_groups'], requirement, minimum)
            if matches:
                mask |= 1 << i
        self.signatures[key] = mask
        return mask

    def evaluate(self, target, anchor_start):
        if not target or set(target) - set('ACGT'):
            return None
        positions = self.positions(target)
        anchor = self.scope['anchor_label']
        if anchor_start not in positions[anchor] or not candidate_matches_regions(
                [(self.scope['references'][anchor], anchor_start, anchor_start+len(target))],
                self.scope['anchor_search_rows'], 'full_containment'):
            return None
        if any(not positions[label] for label in self.scope['required_labels']):
            return None
        states = {self.signature(anchor, anchor_start, len(target)): {anchor: anchor_start}}
        for label in self.scope['reference_labels']:
            if label == anchor:
                continue
            alternatives = {}
            for start in positions[label]:
                alternatives.setdefault(self.signature(label, start, len(target)), start)
            if not alternatives:
                alternatives = {0: None}  # An optional constraint reference has no exact site.
            next_states = {}
            for mask, witness in states.items():
                for local_mask, start in alternatives.items():
                    next_states.setdefault(mask | local_mask, {**witness, label: start})
            states = next_states
        witness = next((w for mask, w in states.items() if combine_constraint_matches(
            [bool(mask & (1 << i)) for i in range(len(self.scope['filters']))], self.scope['filters'])), None)
        if witness is None:
            return None
        matches = []
        for label, starts in positions.items():
            for start in starts:
                matches.append({'transcript_id': self.scope['references'][label],
                    'start': start+1, 'end': start+len(target), 'coordinate_convention': 'one_based_inclusive',
                    'genomic_segments': self.segments(label, start, len(target)),
                    'strand': '-' if str(self.catalog[label].get('strand')) in {'-', '-1'} else '+',
                    'constraint_witness': witness.get(label) == start})
        return {'status': 'verified_exact_shared_binding', 'target_sequence': target,
                'required_transcript_ids': [self.scope['references'][label] for label in self.scope['required_labels']],
                'matches': matches, 'reference_sequence_sha256': dict(self.sequence_hashes),
                'placement_rule': self.scope['placement_rule']}

    def verify_row(self, row):
        target = generated_sequence_fields(row).get('target_binding_sequence')
        if not target:
            return None
        segments = row.get('Genomic_segments') or row.get('genomic_segments')
        if isinstance(segments, str):
            try:
                segments = json.loads(segments)
            except ValueError:
                return None
        if not isinstance(segments, list) or not segments:
            return None
        def signature(items):
            return sorted((str(x['chrom']).removeprefix('chr'), int(x['start']), int(x['end'])) for x in items)
        try:
            observed = signature(segments)
        except (ValueError, TypeError, KeyError):
            return None
        anchor = self.scope['anchor_label']
        strand = '-' if str(self.catalog[anchor].get('strand')) in {'-', '-1'} else '+'
        if str(row.get('Strand') or row.get('strand') or '') not in {strand, '1' if strand == '+' else '-1'}:
            return None
        for start in self.positions(target)[anchor]:
            if signature(self.segments(anchor, start, len(target))) == observed:
                return self.evaluate(target, start)
        return None
