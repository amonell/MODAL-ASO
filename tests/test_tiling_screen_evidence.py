from copy import deepcopy
import json

from aso_agents.candidate_assessment import assess_candidates, off_target_evidence, rank_candidates
from aso_agents.tiling_screen_evidence import bind_tiling_screen, index_identity


def configured_output(tmp_path):
    output = tmp_path/'tiles'; output.mkdir()
    prefix = tmp_path/'reference'
    for part in ('1','2','3','4','rev.1','rev.2'):
        (tmp_path/f'reference.{part}.ebwt').write_text(part)
    runtime = tmp_path/'tiles.runtime'; runtime.mkdir()
    config = {'OUTPUT_FOLDER':str(output), 'BOWTIE_INDEX':str(prefix),
              'SECONDARY_INDEX':str(prefix), 'UCSC_SPECIES':'test_build', 'NUM_MISMATCHES':2}
    (runtime/'config.generated.yaml').write_text(json.dumps(config))
    return output, prefix


def native_row(sequence, count):
    return {'synthesized_antisense_sequence':sequence, 'program_id':'one',
            'off_target_search_status':'complete', 'off_target_exact_count_unit':'loci',
            'Off_target_mismatch_budget':2, 'Off_target_exact_count':count,
            'Off_target_1mm_count':2, 'Off_target_2mm_count':3}


def test_actual_index_identity_and_native_status_enable_comparable_triage(tmp_path):
    output, prefix = configured_output(tmp_path)
    payload = {'rows':[native_row('ACGTTGCA',4),native_row('AACCTTGG',0)]}
    bind_tiling_screen(payload, output, 'human')
    rows = assess_candidates(payload['rows'], {})
    assert all(row['candidate_assessment']['off_target']['status'] == 'complete_within_declared_scope' for row in rows)
    assert all(row['candidate_assessment']['off_target']['references_by_species'] for row in rows)
    assert rank_candidates(rows)[0]['synthesized_antisense_sequence'] == 'AACCTTGG'
    original = index_identity(prefix)['sha256']
    (tmp_path/'reference.1.ebwt').write_text('changed content')
    assert index_identity(prefix)['sha256'] != original


def test_missing_reference_and_budget_conflicts_do_not_gain_provenance(tmp_path):
    output, prefix = configured_output(tmp_path)
    row = native_row('ACGTTGCA',0)
    payload = {'rows':[{**row,'Off_target_mismatch_budget':1}]}
    bind_tiling_screen(payload,output,'human')
    assert 'off_target_search_provenance_by_species' not in payload['rows'][0]
    (tmp_path/'reference.4.ebwt').unlink()
    payload = {'rows':[deepcopy(row)]}
    bind_tiling_screen(payload,output,'human')
    assert 'off_target_search_provenance_by_species' not in payload['rows'][0]


def test_censored_conflicting_or_missing_counts_remain_incomplete():
    good = native_row('ACGTTGCA',0)
    for change in ({'off_target_search_status':'incomplete_or_censored'},
                   {'Off_target_2mm_count':None}, {'off_target_mapping_capped':True},
                   {'off_target_count_status':'incomplete'},
                   {'off_target_count_status':'complete_within_declared_scope',
                    'off_target_search_status':'incomplete_or_censored'}):
        assert off_target_evidence({**good, **change})['status'] == 'incomplete_or_unavailable'
