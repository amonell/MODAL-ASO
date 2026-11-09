"""Export-only views of saved candidate ranks, geometry and route provenance."""
from collections import defaultdict
import json
from .report_explorer_data import coordinates, number, source_row


def named_rank_components(row):
    saved = row.get('recommendation_rank_key')
    if not isinstance(saved,(list,tuple)):
        return row.get('selection_score_components') or {}
    fields = {4:('objective_evidence_tier','objective_directional_mean','site_evidence_tier','site_directional_mean'),
              6:('objective_evidence_tier','safeguard_evidence_tier','negative_safeguard_bound_excess',
                 'objective_directional_mean','site_evidence_tier','site_directional_mean')}.get(len(saved))
    return dict(zip(fields,saved)) if fields else {'uninterpreted_saved_values':list(saved)}


def candidate_context_key(row):
    from .candidate_presentation import molecule_key
    return (molecule_key(row),row.get('program_id'),row.get('job_id'),row.get('route_label'),
            json.dumps(coordinates(row),sort_keys=True))


def with_native_ranks(rows):
    """Rank saved recommendation keys within their producing route only.

    route_rank orders strategies, not ASOs. Original selection_rank values take
    precedence. This export projection never changes the selected molecules.
    """
    result=[dict(row) for row in rows]
    groups=defaultdict(list)
    for row in result:
        if number(row.get('selection_rank')) is not None:
            row.update(native_rank=row['selection_rank'],native_rank_basis='selection_rank')
        elif number(row.get('native_rank')) is not None:
            row.setdefault('native_rank_basis','native_rank')
        else:
            key=row.get('recommendation_rank_key')
            if isinstance(key,(list,tuple)) and key and all(number(v) is not None for v in key):
                groups[(row.get('program_id'),row.get('job_id'),row.get('route_label'))].append(row)
    from .splice_recommendations import recommendation_sort_key
    for members in groups.values():
        for rank,row in enumerate(sorted(members,key=recommendation_sort_key,reverse=True),1):
            row.update(native_rank=rank,native_rank_basis='recommendation_key_descending_within_route')
    return result


def export_context(row):
    source = source_row(row)
    intervals = coordinates(row)
    effects = row.get('selected_effect_evidence') or []
    primary = next((item for item in effects if item.get('label')==row.get('route_label')),effects[0] if len(effects)==1 else {})
    prediction = primary.get('prediction') or {}
    values = prediction.get('values') or []
    rank_field = next((key for key in ('selection_rank','native_rank') if number(row.get(key)) is not None),None)
    contiguous = intervals[0] if len(intervals)==1 else {}
    return {'chrom':contiguous.get('chrom'),'start':contiguous.get('start'),'end':contiguous.get('end'),
            'strand':source.get('strand') or source.get('Strand') or source.get('Genomic_strand'),
            'coordinate_system':'genomic_1based_inclusive', 'binding_intervals_1based':intervals,
            'program_id':row.get('program_id'),'job_id':row.get('job_id'),'route_label':row.get('route_label'),
            'native_rank':row.get(rank_field) if rank_field else None,
            'native_rank_basis':row.get('native_rank_basis') or rank_field or 'unavailable',
            'native_rank_scope':{'program_id':row.get('program_id'),'job_id':row.get('job_id'),'route_label':row.get('route_label')},
            'recommendation_rank_components':named_rank_components(row),
            'primary_objective_site':primary.get('label'),'primary_objective_direction':prediction.get('direction'),
            'primary_objective_delta_deletion':values[0] if values else None,
            'primary_objective_delta_replacement':values[1] if len(values)>1 else None,
            'primary_objective_directional_mean':prediction.get('directional_mean')}
