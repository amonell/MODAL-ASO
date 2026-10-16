"""Lossless navigation of recorded splice predictions, without efficacy inference."""
from copy import deepcopy
from math import isfinite


METRICS = (
    'delta_acceptor_deletion', 'delta_acceptor_replacement',
    'delta_donor_deletion', 'delta_donor_replacement',
    'delta_inclusion_deletion', 'delta_inclusion_replacement',
    'delta_deletion', 'delta_replacement',
    'primary_delta_deletion', 'primary_delta_replacement',
    'selected_delta_deletion', 'selected_delta_replacement',
)
IDENTITY = (
    'label', 'canonical_label', 'site_kind', 'exon_index', 'isoform_label',
    'reference_transcript_id', 'reference_binding_status', 'acceptor', 'donor',
    'primary_label', 'primary_direction', 'selected_label', 'selected_site_kind',
    'selected_site_position', 'evaluation_status', 'splice_eval_status',
    'splice_eval_skip_reason',
)


def numeric_prediction(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def splice_prediction_records(row):
    """Keep summary and each site record distinct, including unscored records.

    Only explicit numeric fields are measurements. Legacy comma-separated
    arrays remain in the source evidence; never guess their site attribution.
    Repeated labels or coordinates do not justify merging observations.
    """
    summary = row.get('splice_eval_summary')
    if not isinstance(summary, dict):
        return []
    source_records = [('splice_eval_summary', summary)]
    sites = summary.get('splice_site_deltas')
    if isinstance(sites, list):
        source_records.extend((f'splice_eval_summary.splice_site_deltas[{index}]', site)
                              for index, site in enumerate(sites) if isinstance(site, dict))
    return [{
        'source_field': field,
        'recorded_identity': {key: deepcopy(record[key]) for key in IDENTITY if key in record},
        'numeric_measurements': {key: record[key] for key in METRICS
                                 if numeric_prediction(record.get(key))},
        'unrecorded_metric_fields': [key for key in METRICS if key in record and record[key] is None],
    } for field, record in source_records]
