"""Prediction transport must not change the quantity being reported."""
from copy import deepcopy
import json

import pytest

from aso_agents import pipeline_tools as pipeline
from aso_agents.evaluation_evidence import attach_measurement_references


@pytest.mark.parametrize('primary,selected,inclusion', [
    (0.0, -0.7, None), (-0.8, 0.0, 0.0), (None, 0.4, -0.2),
])
def test_selected_site_is_never_substituted_for_inclusion(primary, selected, inclusion):
    row = {'primary_label': 'acceptor@410', 'primary_delta_deletion': primary,
           'selected_delta_deletion': selected, 'selected_site_kind': 'acceptor',
           'delta_inclusion_deletion': inclusion}
    summary = pipeline._splice_summary_from_scored_row(row)
    assert summary['delta_inclusion_deletion'] == inclusion
    assert summary['delta_inclusion_replacement'] is None
    if primary is not None:
        assert summary['primary_delta_deletion'] == primary
    if selected is not None:
        assert summary['selected_delta_deletion'] == selected
    assert summary['selected_site_kind'] == 'acceptor'


@pytest.mark.parametrize('serialized', [False, True])
def test_per_site_measurements_survive_compaction_and_citation(serialized):
    sites = [{'site_kind': 'acceptor', 'reference_transcript_id': 'TX_NEUTRAL.2',
              'acceptor': {'chrom': 'chrN', 'position': 410, 'strand': '-'},
              'delta_acceptor_deletion': -0.73, 'delta_acceptor_replacement': 0.0,
              'delta_donor_deletion': None, 'delta_inclusion_deletion': None,
              'evaluation_status': 'evaluated'}]
    scored = {'splice_site_deltas': json.dumps(sites) if serialized else sites,
              'delta_inclusion_deletion': None, 'delta_deletion': '-0.7300',
              'delta_names': 'acceptor@410'}
    before = deepcopy(scored)
    summary = pipeline._splice_summary_from_scored_row(scored)
    compact = pipeline._compact_existing_aso_evaluation_for_response(
        {'provided_asos': [{'name': 'neutral', 'input_row_id': 'row-1',
                            'splice_eval_summary': summary}]})
    attach_measurement_references(compact)
    retained = compact['provided_asos'][0]['splice_eval_summary']
    assert retained['splice_site_deltas'] == scored['splice_site_deltas']
    assert retained['delta_inclusion_deletion'] is None
    reference = compact['measurement_evidence']['M001_splice_site_deltas']
    assert reference['value'] == scored['splice_site_deltas']
    assert reference['field'] == 'splice_eval_summary.splice_site_deltas'
    if not serialized:
        retained['splice_site_deltas'][0]['acceptor']['position'] = 999
    assert scored == before


def test_summary_names_the_metric_it_actually_selected():
    row = {'name': 'neutral', 'splice_eval_summary': {
        'delta_inclusion_deletion': -0.1, 'delta_replacement': -0.6,
        'delta_names': 'acceptor@410'}}
    text = '\n'.join(pipeline._provided_aso_knockdown_answer_paragraphs(
        [], gene='NEUTRAL', species='human', target_context='', splice_rows=[row]))
    assert '`delta_replacement` = -0.6000' in text
    assert '`delta_inclusion_deletion` = -0.6000' not in text
    assert pipeline._summary_row_skip_delta(row) == -0.6


def test_absent_metrics_stay_unknown_and_zero_stays_recorded():
    assert pipeline._summary_row_skip_measurement({}) == (None, None)
    row = {'splice_eval_summary': {'delta_inclusion_deletion': None, 'delta_deletion': 0.0}}
    assert pipeline._summary_row_skip_measurement(row) == ('delta_deletion', 0.0)
