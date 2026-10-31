"""Requested accession aliases and executed jobs determine reported targets."""
from copy import deepcopy

import run_goal_intake as runtime
from aso_agents.execution_references import executed_reference, merged_executed_references


def reference_goal():
    def entry(identifier, accession, canonical=False):
        return {'transcript_id': identifier, 'chrom': 'chr8', 'strand': '+',
            'exons_tx': [{'index': 1, 'start': 100, 'end': 130},
                         {'index': 2, 'start': 300, 'end': 330}],
            'metadata': {'id': identifier, 'versioned_id': identifier + '.4',
                'is_canonical': canonical,
                'reference_accession_resolution': {'requested_accession': accession,
                    'exact_accession_verified': True, 'transcript_id': identifier}}}
    return {'intent': 'knockdown', 'transcript_lock': {'kind': 'explicit_id', 'value': 'NM_123456.7'},
        'isoform_transcripts': {'protected': entry('TXPROTECTED', 'NM_999999.4', True),
                               'target': entry('TXTARGET', 'NM_123456.7')},
        'design_programs': [{'program_id': 'knockdown', 'modality': 'gapmer', 'intent': 'knockdown',
                            'transcript_id': 'NM_123456.7', 'isoform_label': 'target'}]}


def test_accession_lock_selects_its_attested_target_before_canonical_protected_transcript():
    goal = reference_goal()
    before = deepcopy(goal)
    assert runtime._select_primary_transcript(goal) == ('TXTARGET', 'target')
    assert goal == before


def test_single_program_reference_selects_target_without_global_lock():
    goal = reference_goal()
    goal.pop('transcript_lock')
    assert runtime._select_primary_transcript(goal) == ('TXTARGET', 'target')


def test_unattested_external_accession_is_not_promoted_to_a_reference_match():
    goal = reference_goal()
    goal['isoform_transcripts']['target']['metadata']['reference_accession_resolution']['exact_accession_verified'] = False
    # An unresolved explicit accession must not silently become the canonical transcript.
    assert runtime._select_primary_transcript(goal) == ('NM_123456.7', None)


def test_merged_reference_is_executed_target_not_canonical_anchor():
    goal = reference_goal()
    job = {'job_id': 'j', 'program_id': 'knockdown'}
    result = {'j': {'executed_reference': executed_reference(job, 'target', goal['isoform_transcripts']['target'])}}
    merged = merged_executed_references([job], result)
    assert merged['transcript_id'] == 'TXTARGET'
    assert merged['primary_isoform'] == 'target'
    assert merged['executed_references'][0]['full_rna_sequence_equivalence'] == 'not_established_by_this_receipt'


def test_multiple_executed_references_have_no_false_global_template():
    goal = reference_goal()
    jobs = [{'job_id': key} for key in goal['isoform_transcripts']]
    results = {job['job_id']: {'executed_reference': executed_reference(job, job['job_id'], goal['isoform_transcripts'][job['job_id']])} for job in jobs}
    merged = merged_executed_references(jobs, results)
    assert merged['reference_scope'] == 'multiple_references'
    assert merged['transcript_id'] is merged['primary_isoform'] is None
    results['target'].pop('executed_reference')
    assert merged_executed_references(jobs, results)['reference_scope'] == 'unresolved'


def test_feature_selection_honors_each_verified_alias_without_replacing_earlier_receipts():
    goal = reference_goal()
    goal['transcript_policy'] = 'user_locked'
    entry = goal['isoform_transcripts']['target']
    receipt = entry['metadata'].pop('reference_accession_resolution')
    entry['metadata']['reference_accession_resolution'] = {**receipt, 'requested_accession': 'NM_999123.4'}
    entry['metadata']['reference_accession_resolutions'] = [receipt]
    assert runtime._select_isoform_for_utr(goal['isoform_transcripts'], goal) == ('target', entry)
    receipt['exact_accession_verified'] = False
    assert runtime._select_isoform_for_utr(goal['isoform_transcripts'], goal) == (None, None)
