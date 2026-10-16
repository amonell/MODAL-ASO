from copy import deepcopy
import json
from pathlib import Path

import pytest
import run_goal_intake as native
from aso_agents.mature_steric import compile_regions, build_tiles
from aso_agents.mature_binding import BindingContext
from aso_agents.candidate_region_gate import filter_final_rows_by_regions
from aso_agents.pipeline_tools import _filter_tiles_json_to_output, _normalized_generated_candidate_row


def setup(monkeypatch, secondary='TTTTTATGCAACCGG', strand='-'):
    sequences = {'a': 'CCCATGCAAGGGTTT', 'b': secondary}
    entries = {}
    for label, identity, lo, direction in [('a', 'TX_A.2', 1001, '+'), ('b', 'TX_B.7', 2001, strand)]:
        length = len(sequences[label])
        entries[label] = {'transcript_id': identity, 'metadata': {'versioned_id': identity},
                         'chrom': 'chr1', 'strand': direction,
                         'exons_tx': [{'index': 1, 'start': lo, 'end': lo+length-1}]}
    rules = [{'constraint_id': label, 'source_type': 'explicit_rows', 'coordinate_space': 'cdna',
              'materialized_rows': [[entries[label]['transcript_id'], start, start+3, label, 0, '+']],
              'operator': 'intersect', 'aso_overlap_requirement': 'contains_feature'}
             for label, start in [('a', 3), ('b', 5)]]
    program = {'program_id': 'panel', 'transcript_ids': ['TX_A.2', 'TX_B.7'], 'transcript_id': 'TX_A.2',
               'coordinate_space': 'cdna', 'modality': 'steric', 'intent': 'binding', 'length': 6,
               'region_constraints': [{**r, 'rows': r['materialized_rows']} for r in rules]}
    goal = {'species': 'human', 'isoform_transcripts': entries, 'design_programs': [program],
            'materialized_region_constraints': {'global': [], 'by_program': {'panel': rules}}}
    monkeypatch.setattr(native, '_build_isoform_cdna_sequences',
                        lambda catalog, *a, **kw: {label: sequences[label] for label in catalog})
    monkeypatch.setattr(native, '_annotate_tile_rows_with_rnafold', lambda rows: None)
    monkeypatch.setattr(native, '_annotate_gapmer_off_targets', lambda *a, **kw: {'status': 'complete'})
    return goal, program, sequences


def context(goal, program):
    _, compiled = compile_regions(native, goal, program, [])
    return BindingContext(native, goal, compiled['mature_rna']['shared_binding']), compiled


def row_for(binding, target='ATGCAA', start=3):
    idt, synthesis, _ = native._format_idt_and_synthesis(target, 'steric-hindrance')
    return {'Sequence': target, 'Synthesis': synthesis, 'IDT': idt, 'Chrom': 'TX_A.2',
            'Start': start+1, 'End': start+len(target), 'Strand': '+', 'program_id': 'panel',
            'source': 'steric_cdna', 'mature_rna_reference': 'TX_A.2',
            'Genomic_segments': json.dumps(binding.segments('a', start, len(target)))}


@pytest.mark.parametrize('strand', ['+', '-'])
def test_shared_target_uses_each_transcripts_own_mature_and_genomic_coordinates(monkeypatch, strand):
    goal, program, _ = setup(monkeypatch, strand=strand)
    binding, _ = context(goal, program)
    evidence = binding.evaluate('ATGCAA', 3)
    assert evidence['required_transcript_ids'] == ['TX_A.2', 'TX_B.7']
    assert [(m['transcript_id'], m['start'], m['end']) for m in evidence['matches']] == [
        ('TX_A.2', 4, 9), ('TX_B.7', 6, 11)]
    second = evidence['matches'][1]
    assert second['strand'] == strand
    assert second['genomic_segments'][0]['start'] == (2006 if strand == '+' else 2005)
    assert all(m['constraint_witness'] for m in evidence['matches'])


def test_all_repeat_sites_are_retained_but_one_coherent_site_must_satisfy_each_reference(monkeypatch):
    goal, program, _ = setup(monkeypatch, secondary='TTTTTATGCAAGGATGCAACC')
    binding, _ = context(goal, program)
    evidence = binding.evaluate('ATGCAA', 3)
    matches = [m for m in evidence['matches'] if m['transcript_id'] == 'TX_B.7']
    assert [m['start'] for m in matches] == [6, 14]
    assert [m['constraint_witness'] for m in matches] == [True, False]
    extra = deepcopy(goal['materialized_region_constraints']['by_program']['panel'][1])
    extra['materialized_rows'][0][1:3] = [13, 16]
    goal['materialized_region_constraints']['by_program']['panel'].append(extra)
    binding, _ = context(goal, program)
    assert binding.evaluate('ATGCAA', 3) is None  # Different repeats cannot jointly cover both features.


def test_exclusion_and_union_keep_ordered_candidate_predicate_semantics(monkeypatch):
    goal, program, _ = setup(monkeypatch, secondary='TTTTTATGCAAGGATGCAACC')
    rules = goal['materialized_region_constraints']['by_program']['panel']
    rules[1]['operator'] = 'exclude'
    binding, _ = context(goal, program)
    evidence = binding.evaluate('ATGCAA', 3)
    assert [m['start'] for m in evidence['matches'] if m['transcript_id'] == 'TX_B.7' and m['constraint_witness']] == [14]
    rules[0]['materialized_rows'][0][1:3] = [12, 15]
    rules[1]['operator'] = 'union'
    binding, _ = context(goal, program)
    assert binding.evaluate('ATGCAA', 3)


def test_shared_binding_requires_actual_sequence_in_every_required_reference(monkeypatch):
    goal, program, _ = setup(monkeypatch, secondary='TTTTTCCCCCCCCCC')
    binding, _ = context(goal, program)
    assert binding.evaluate('ATGCAA', 3) is None


def test_reference_versions_are_never_collapsed(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    program['transcript_ids'][1] = 'TX_B.8'
    with pytest.raises(ValueError, match='reference_unresolved'):
        context(goal, program)


def test_missing_reference_sequence_is_unresolved_not_infeasible(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    monkeypatch.setattr(native, '_build_isoform_cdna_sequences', lambda *a, **kw: {'a': 'CCCATGCAAGGGTTT'})
    with pytest.raises(ValueError, match='sequence_unresolved'):
        context(goal, program)


def test_changed_reference_geometry_invalidates_compiled_scope(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    _, compiled = compile_regions(native, goal, program, [])
    goal['isoform_transcripts']['b']['exons_tx'][0]['end'] += 1
    with pytest.raises(ValueError, match='reference_changed'):
        BindingContext(native, goal, compiled['mature_rna']['shared_binding'])


def test_reference_index_is_reused_and_ambiguous_bases_never_claim_exact_identity(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    binding, _ = context(goal, program)
    binding.positions('ATGCAA')
    first = binding.indices[6]
    binding.positions('CCCCCC')
    assert binding.indices[6] is first
    assert binding.evaluate('ATGCAN', 3) is None


def test_builder_emits_verified_shared_rows_with_real_sequence_orientation(tmp_path, monkeypatch):
    goal, program, _ = setup(monkeypatch)
    _, compiled = context(goal, program)
    job = {**program, 'job_id': 'job1', 'selection_meta': {'region_constraints': compiled}}
    result = build_tiles(native, goal, job, {'recommendation': {'CHEMISTRY': 'steric-hindrance', 'length': 6}}, str(tmp_path))
    rows = json.loads(Path(result['json_path']).read_text())['rows']
    assert len(rows) == 1
    assert rows[0]['Sequence'] == 'ATGCAA' and rows[0]['Synthesis'] == 'TTGCAT'
    assert len(rows[0]['mature_rna_binding']['matches']) == 2


@pytest.mark.parametrize('field,value', [('Sequence', 'CCCCCC'), ('Strand', '-'), ('Genomic_segments', '[]')])
def test_row_revalidation_does_not_trust_a_stale_or_fabricated_binding_receipt(monkeypatch, field, value):
    goal, program, _ = setup(monkeypatch)
    binding, _ = context(goal, program)
    row = row_for(binding)
    row['mature_rna_binding'] = binding.evaluate('ATGCAA', 3)
    row[field] = value
    assert binding.verify_row(row) is None


def test_job_subset_and_final_panel_gate_recompute_shared_binding(tmp_path, monkeypatch):
    goal, program, _ = setup(monkeypatch)
    binding, compiled = context(goal, program)
    good = row_for(binding)
    bad = row_for(binding, target='CCCATG', start=0)
    source = tmp_path/'tiles.json'; source.write_text(json.dumps({'rows': [good, bad]}))
    result = _filter_tiles_json_to_output(source_json_path=str(source), region_rows=compiled['tile_filter_rows'],
        output_path=str(tmp_path/'filtered.json'), tile_overlap_requirement='full_containment',
        mature_binding_scope=compiled['mature_rna']['shared_binding'], structured_goal=goal)
    assert result['row_count'] == 1
    normalized = [_normalized_generated_candidate_row(row) for row in (good, bad)]
    kept, receipt = filter_final_rows_by_regions(normalized, goal, native._tile_row_overlap_intervals)
    assert len(kept) == 1 and receipt['excluded_count'] == 1
    assert kept[0]['mature_rna_binding']['status'] == 'verified_exact_shared_binding'


def test_global_and_program_constraints_both_apply_to_shared_binding(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    global_rule = deepcopy(goal['materialized_region_constraints']['by_program']['panel'][0])
    global_rule.update(operator='intersect', aso_overlap_requirement='full_containment')
    global_rule['materialized_rows'][0][1:3] = [0, 6]
    goal['materialized_region_constraints']['global'] = [global_rule]
    binding, _ = context(goal, program)
    assert binding.evaluate('ATGCAA', 3) is None


def test_shared_exon_junction_keeps_disjoint_blocks_and_rejects_intron_spanning_claim(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    goal['isoform_transcripts']['a']['exons_tx'] = [
        {'index': 1, 'start': 1001, 'end': 1006}, {'index': 2, 'start': 1501, 'end': 1509}]
    binding, _ = context(goal, program)
    row = row_for(binding)
    evidence = binding.verify_row(row)
    anchor = next(m for m in evidence['matches'] if m['transcript_id'] == 'TX_A.2')
    assert anchor['genomic_segments'] == [{'chrom': 'chr1', 'start': 1004, 'end': 1006},
                                          {'chrom': 'chr1', 'start': 1501, 'end': 1503}]
    row['Genomic_segments'] = json.dumps([{'chrom': 'chr1', 'start': 1004, 'end': 1503}])
    assert binding.verify_row(row) is None


def test_final_gate_excludes_unverified_shared_reference_and_marks_uncertainty(monkeypatch):
    goal, program, _ = setup(monkeypatch)
    binding, _ = context(goal, program)
    row = _normalized_generated_candidate_row(row_for(binding))
    monkeypatch.setattr(native, '_build_isoform_cdna_sequences', lambda *a, **kw: {})
    kept, receipt = filter_final_rows_by_regions([row], goal, native._tile_row_overlap_intervals)
    assert kept == [] and receipt['status'] == 'unresolved'
    assert receipt['unresolved_rows'][0]['reason'] == 'mature_binding_unverified'
