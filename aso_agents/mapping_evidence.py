"""Alignment geometry for supplied molecules, without efficacy thresholds."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def mismatch_annotations(reference_window: str, aligned_query: str, orientation: str) -> dict:
    """Describe an ungapped comparison in both target and supplied orientations.

    Both arguments run along the reference transcript's 5′ to 3′ direction.
    Genome strand does not reverse target-relative positions a second time.
    The aligned query is the supplied sequence or its reverse complement, as
    identified by orientation. This reports sequence differences, not potency.
    """
    if orientation not in {'input_sequence', 'reverse_complement'}:
        raise ValueError('Mismatch annotation requires a known query orientation')
    reference, query = reference_window.upper(), aligned_query.upper()
    if len(reference) != len(query):
        raise ValueError('Ungapped mismatch annotation requires equal-length sequences')
    differences = []
    for position, (reference_base, query_base) in enumerate(zip(reference, query), start=1):
        if reference_base != query_base:
            differences.append({
                'target_position_1based': position,
                'reference_base': reference_base,
                'aligned_query_base': query_base,
                'supplied_sequence_position_1based': (
                    len(query) - position + 1 if orientation == 'reverse_complement' else position),
            })
    return {
        'mismatch_positions_in_supplied_sequence_1based': sorted(
            item['supplied_sequence_position_1based'] for item in differences),
        'mismatch_positions_in_target_sequence_1based': [
            item['target_position_1based'] for item in differences],
        'mismatch_details_in_target_orientation': differences,
        'mismatch_annotation_coordinate_system': (
            '1based positions in the full aligned window; target and aligned-query bases '
            'are in reference-transcript 5prime_to_3prime orientation; '
            'supplied-sequence positions are in input 5prime_to_3prime orientation'),
    }


def supplied_molecule_sequence_metrics(sequence: str) -> dict:
    """Measure physical sequence properties without an efficacy inference."""
    seq = str(sequence or "").upper()
    longest = current = 0
    previous = ""
    for base in seq:
        current = current + 1 if base == previous else 1
        longest = max(longest, current)
        previous = base
    return {"maximum_homopolymer_run_nt": longest if seq else None}


def principal_reference_exon_membership(row: dict, reference_transcript_id: str) -> dict:
    """Locate the principal antisense alignment in its attested exon catalog."""
    reference_base = str(reference_transcript_id or "").split(".")[0]
    hits = [hit for hit in (row.get("mapping_hits") or []) + (row.get("near_match_candidates") or [])
            if isinstance(hit, dict) and hit.get("match_orientation") == "reverse_complement"
            and str(hit.get("transcript_id") or "").split(".")[0] == reference_base
            and hit.get("mapping_scope") == "spliced_cdna"]
    hits.sort(key=lambda hit: (int(hit.get("mismatch_count") or 0), int(hit.get("cdna_start") or 10**12)))
    if not hits:
        return {"status": "unresolved", "reason": "No antisense alignment on the selected mature transcript."}
    hit = hits[0]
    identity = hit.get("reference_identity") if isinstance(hit.get("reference_identity"), dict) else {}
    sequence_evidence = identity.get("sequence_evidence") if isinstance(identity.get("sequence_evidence"), dict) else {}
    exons = sequence_evidence.get("exons_tx") if isinstance(sequence_evidence.get("exons_tx"), list) else []
    segments = hit.get("genomic_segments") if isinstance(hit.get("genomic_segments"), list) else []
    if not exons or not segments:
        return {"status": "unresolved", "reason": "Reference exon geometry or aligned genomic blocks unavailable."}
    membership = []
    for ordinal, exon in enumerate(exons, start=1):
        if not isinstance(exon, dict) or not isinstance(exon.get("start"), int) or not isinstance(exon.get("end"), int):
            continue
        overlap = sum(max(0, min(block.get("end", -1), exon["end"]) - max(block.get("start", 10**12), exon["start"]) + 1)
                      for block in segments if isinstance(block, dict)
                      and isinstance(block.get("start"), int) and isinstance(block.get("end"), int))
        if overlap:
            membership.append({"transcript_exon_ordinal": ordinal, "exon_start_1based": exon["start"],
                               "exon_end_1based": exon["end"], "overlap_nt": overlap})
    if not membership:
        return {"status": "unresolved", "reason": "Aligned blocks do not overlap annotated exons."}
    aligned_length = sum(block["end"] - block["start"] + 1 for block in segments
                         if isinstance(block, dict) and isinstance(block.get("start"), int)
                         and isinstance(block.get("end"), int))
    return {"status": "resolved", "transcript_id": identity.get("transcript_id") or reference_transcript_id,
            "coordinate_system": "1based_inclusive", "alignment_mismatches": int(hit.get("mismatch_count") or 0),
            "fully_exonic": sum(item["overlap_nt"] for item in membership) == aligned_length,
            "exons": membership, "scope": "Principal antisense alignment geometry; not a biological exon-name or efficacy assertion."}


def annotate_reference_junctions(row, catalog):
    """Expose actual exon adjacency independently of request-facing exon names.

    This is geometry, not a nomenclature resolver. In particular, an exon ordinal
    in one isoform is never silently assigned to a different isoform's label.
    """
    def tid(entry, label):
        return str((entry.get('metadata') or {}).get('versioned_id') or entry.get('transcript_id') or entry.get('id') or label)
    for match in (row.get('mapping_hits') or []) + (row.get('near_match_candidates') or []):
        match['reference_junctions_spanned'] = []
        if match.get('mapping_scope') != 'spliced_cdna':
            continue
        match_id = str(match.get('transcript_id') or '')
        candidates = [(label,entry) for label,entry in catalog.items() if isinstance(entry,dict)
            and tid(entry,label).split('.')[0] == match_id.split('.')[0]
            and ('.' not in match_id or tid(entry,label) == match_id)]
        if len(candidates) != 1:
            continue
        label, transcript = candidates[0]
        blocks = match.get('genomic_segments') or []
        if len(blocks) < 2:
            continue
        strand = '-' if str(transcript.get('strand')) in {'-','-1'} else '+'
        chrom = transcript.get('chrom') or (transcript.get('metadata') or {}).get('seq_region_name')
        exons = transcript.get('exons_tx') or []
        for left,right in zip(exons,exons[1:]):
            donor = left.get('start' if strand == '-' else 'end')
            acceptor = right.get('end' if strand == '-' else 'start')
            if not isinstance(donor,int) or not isinstance(acceptor,int):
                continue
            def containing(position):
                return {i for i,b in enumerate(blocks) if
                    str(b.get('chrom')).removeprefix('chr') == str(chrom).removeprefix('chr')
                    and isinstance(b.get('start'),int) and isinstance(b.get('end'),int)
                    and b['start'] <= position <= b['end']}
            if not any(i != j for i in containing(donor) for j in containing(acceptor)):
                continue
            context = []
            for other_label, other in catalog.items():
                if not isinstance(other,dict):
                    continue
                other_chrom = other.get('chrom') or (other.get('metadata') or {}).get('seq_region_name')
                other_strand = '-' if str(other.get('strand')) in {'-','-1'} else '+'
                if str(other_chrom).removeprefix('chr') != str(chrom).removeprefix('chr') or strand != other_strand:
                    continue
                between = [deepcopy(exon) for exon in other.get('exons_tx') or []
                           if isinstance(exon.get('start'),int) and isinstance(exon.get('end'),int)
                           and min(donor,acceptor) < exon['start'] <= exon['end'] < max(donor,acceptor)]
                if between:
                    context.append({'transcript_id':tid(other,other_label),'intervening_annotated_exons':between})
            match['reference_junctions_spanned'].append({'transcript_id':tid(transcript,label),
                'chrom':chrom, 'strand':strand, 'donor':donor, 'acceptor':acceptor,
                'upstream_transcript_ordinal':left.get('index'), 'downstream_transcript_ordinal':right.get('index'),
                'coordinate_system':'1based_inclusive', 'catalog_intervening_exon_context':context,
                'scope':'Verified alignment across adjacent reference exons. Ordinals are annotation indices, not assumed biological exon labels; no activity or selectivity is inferred.'})


def nearest_windows(reference: str, query: str, limit: int = 5):
    """Best ungapped windows, retaining tied locations up to a declared bound."""
    if not query or len(reference) < len(query):
        return []
    source = np.frombuffer(reference.upper().encode('ascii'), dtype=np.uint8)
    target = np.frombuffer(query.upper().encode('ascii'), dtype=np.uint8)
    count = len(source) - len(target) + 1
    distances = np.zeros(count, dtype=np.int32)
    for offset, base in enumerate(target):
        distances += source[offset:offset + count] != base
    best = int(distances.min())
    return [(int(pos), best) for pos in np.flatnonzero(distances == best)[:limit]]


def cdna_near_matches(entry, catalog, sequences, map_range, preferred_transcript=None):
    """Search mature RNA separately so exon-junction mismatch hypotheses survive."""
    results = []
    query = str(entry.get('sequence') or '').upper()
    terms = [('input_sequence', query), ('reverse_complement', str(entry.get('reverse_complement_sequence') or '').upper())]
    for label, reference in sequences.items():
        transcript = catalog.get(label) or {}
        tid = str(transcript.get('transcript_id') or transcript.get('id') or label)
        for orientation, term in terms:
            for pos, mismatches in nearest_windows(reference, term):
                window = reference[pos:pos + len(term)].upper()
                blocks = map_range(pos + 1, pos + len(term), transcript) if callable(map_range) else []
                results.append({'transcript_id':tid, 'isoform_label':label,
                    'mapping_scope':'spliced_cdna', 'rna_context':'mature_spliced_cdna',
                    'match_orientation':orientation, 'matched_window_sequence':window,
                    'matched_target_sequence':window, 'mismatch_count':mismatches,
                    **mismatch_annotations(window, term, orientation),
                    'cdna_start':pos+1, 'cdna_end':pos+len(term),
                    'coordinate_system':'1based_inclusive',
                    'genomic_segments':[{'chrom':chrom,'start':start,'end':end} for chrom,start,end in blocks],
                    'strand':'-' if str(transcript.get('strand')) in {'-','-1'} else '+',
                    'binding_interpretation':'antisense_binding_hypothesis' if orientation == 'reverse_complement' else 'same_strand_similarity_only; requires confirmed target-site input',
                    'search_scope':'best ungapped windows per transcript/orientation; at most five tied positions; no indels or activity threshold'})
    return sorted(results, key=lambda x:(x['mismatch_count'],
        0 if x['transcript_id'].split('.')[0] == str(preferred_transcript).split('.')[0] else 1,
        0 if x['match_orientation']=='reverse_complement' else 1, x['transcript_id'], x['cdna_start']))


def requested_feature_membership(row, targets, transcript, transcript_id, *, catalog=None):
    """Report requested exon/interval membership for every exact or near mapping."""
    if not isinstance(transcript, dict):
        return []
    from .evaluation_feature_identity import (
        resolve_evaluation_target_identities, transcript_identifier, transcript_matches,
        is_resolved_feature_identity,
    )
    catalog = catalog or {transcript_id: transcript}
    features = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        target_tid = str(target.get('transcript_id') or transcript_id or '')
        if isinstance(target.get('splice_relative_window'), dict):
            resolved_target = target if 'relative_window_resolution' in target else resolve_evaluation_target_identities(
                [target], catalog, transcript_id)[0]
            identity = resolved_target.get('relative_window_resolution') or {}
            resolved = identity.get('status') == 'resolved'
            feature = {'feature': 'interval' if resolved else 'unresolved_relative_window',
                'label': target.get('label'), 'transcript_id': identity.get('reference_transcript_id') or target_tid,
                'evaluated_transcript_id': target_tid, 'requested_feature_identity_resolved': resolved,
                'identity_resolution': deepcopy(identity)}
            if resolved:
                feature.update({key: identity[key] for key in ('chrom', 'start', 'end', 'strand', 'coordinate_system')})
            features.append(feature)
            continue
        start, end = target.get('start'), target.get('end')
        if isinstance(start, int) and isinstance(end, int) and target.get('chrom'):
            convention = target.get('coordinate_system')
            if convention in {'0based_halfopen', '0based_half_open', 'bed'}:
                start += 1
            elif convention not in {'1based_inclusive', '1based_closed'}:
                continue  # Unspecified coordinate conventions require clarification.
            if start <= end:
                features.append({'feature':'interval', 'label':target.get('label'), 'transcript_id':target_tid,
                    'evaluated_transcript_id':target_tid, 'requested_feature_identity_resolved':True,
                    'chrom':target['chrom'], 'start':start, 'end':end, 'coordinate_system':'1based_inclusive'})
            continue  # Explicit coordinates take precedence over label-only hypotheses.
        resolved_target = target if 'exon_identity_resolution' in target else resolve_evaluation_target_identities(
            [target], catalog, transcript_id)[0]
        identity = resolved_target.get('exon_identity_resolution')
        if not isinstance(identity, dict):
            continue
        resolved = is_resolved_feature_identity(identity)
        feature = {'feature':'exon' if resolved else 'unresolved_exon_label',
            'label':target.get('label'), 'requested_exon_label':target.get('exon_label'),
            'transcript_id':identity.get('reference_transcript_id') or identity.get('requested_reference_transcript_id'),
            'evaluated_transcript_id':target_tid, 'requested_feature_identity_resolved':resolved,
            'identity_resolution':deepcopy(identity)}
        if resolved and all(identity.get(k) is not None for k in ('chrom','start','end')):
            feature.update({k:identity[k] for k in ('chrom','start','end','coordinate_system')})
            feature['exon_index'] = identity['resolved_transcript_ordinal']
        features.append(feature)
    observations = []
    for feature in features:
        evidence = []
        if not all(feature.get(k) is not None for k in ('chrom','start','end')):
            observations.append({**feature, 'observations': [],
                'interpretation':'Requested feature identity or coordinate frame remains unresolved; a hypothesis cannot establish membership or non-membership. Inspect reference-junction geometry independently.'})
            continue
        for match in (row.get('mapping_hits') or []) + (row.get('near_match_candidates') or []):
            match_tid = str(match.get('transcript_id') or '')
            if '.' not in match_tid:
                known = {transcript_identifier(e,k) for k,e in catalog.items() if isinstance(e,dict)
                         and transcript_matches(match_tid,transcript_identifier(e,k))}
                if len(known) == 1:
                    match_tid = next(iter(known))
            if not transcript_matches(feature['evaluated_transcript_id'], match_tid):
                continue
            blocks = match.get('genomic_segments') or [match.get('genomic_span') or {}]
            blocks = [b for b in blocks if isinstance(b.get('start'),int) and isinstance(b.get('end'),int)]
            if not blocks:
                continue
            overlap = sum(max(0, min(b['end'],feature['end'])-max(b['start'],feature['start'])+1)
                          for b in blocks if str(b.get('chrom')).removeprefix('chr') == str(feature['chrom']).removeprefix('chr'))
            length = sum(b['end']-b['start']+1 for b in blocks)
            evidence.append({'mapping_scope':match.get('mapping_scope'), 'match_orientation':match.get('match_orientation'),
                'mapping_transcript_id':match_tid,
                'mismatch_count':match.get('mismatch_count',0), 'overlap_bp':overlap,
                'fully_contained':overlap == length, 'overlaps':overlap > 0,
                'genomic_segments':deepcopy(blocks), 'cdna_start':match.get('cdna_start'), 'cdna_end':match.get('cdna_end')})
        observations.append({**feature, 'observations':evidence,
            'interpretation':'Geometric membership of exact or conditional mismatched hypotheses; same-strand observations do not establish binding of an unmodified input role. Membership is not efficacy.'})
    return observations
