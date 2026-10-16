from aso_agents.pipeline_tools import _apply_generated_candidate_limit


def rows():
    return [dict(program_id=program, candidate_group_label=program, selection_rank=rank,
                 sequence=sequence, synthesized_antisense_sequence=sequence[::-1],
                 chemistry_label="5-10-5", binding_intervals_bed0_half_open=[["chrTest", start, start + 20]])
            for program in ("program_a", "program_b")
            for rank, (sequence, start) in enumerate([
                ("ACGTACGTCAGTACCTTGAC", 100),
                ("CGTACGTCAGTACCTTGACA", 101),
                ("GACTCCATTGACGTCATACC", 150),
            ], 1)]


def test_grouped_candidate_caps_preserve_diversity_and_distinct_molecules():
    selected, summary = _apply_generated_candidate_limit(rows(), candidate_limit=None,
        per_site=True, group_scope="program", group_limits={"program_a": 2, "program_b": 2},
        global_candidate_limit=2, candidate_minimum=2, diversify=True)
    assert len(selected) == 2
    assert len({row['synthesized_antisense_sequence'] for row in selected}) == 2
    assert [row['binding_intervals_bed0_half_open'][0][1] for row in selected] == [100, 150]
    assert summary['global_selection']['method'] == 'ranked_nonoverlapping_targets_then_minimum_fill'
    assert sum(group['selected_count'] for group in summary['groups']) == 2
    assert summary['eligible_count'] == 6


def test_per_group_selection_keeps_disjoint_alternatives_before_global_stage():
    selected, _ = _apply_generated_candidate_limit(rows(), candidate_limit=2,
        per_site=True, group_scope="program", diversify=True)
    for group in ('program_a', 'program_b'):
        assert [row['binding_intervals_bed0_half_open'][0][1] for row in selected
                if row['program_id'] == group] == [100, 150]


def test_ranked_policy_is_unchanged_when_diversity_was_not_requested():
    selected, summary = _apply_generated_candidate_limit(rows(), candidate_limit=2,
        per_site=True, group_scope="program", diversify=False)
    for group in ('program_a', 'program_b'):
        assert [row['binding_intervals_bed0_half_open'][0][1] for row in selected
                if row['program_id'] == group] == [100, 101]
    assert summary['method'] == 'top_ranked_after_required_evidence_and_constraints_per_site'


def test_empty_requested_group_remains_an_explicit_shortfall():
    selected, summary = _apply_generated_candidate_limit(rows(), candidate_limit=2,
        per_site=True, group_scope="program", expected_group_labels=['missing'], diversify=True)
    missing = next(group for group in summary['groups'] if group['group_label'] == 'missing')
    assert missing['selected_count'] == 0 and missing['shortfall_count'] == 2
    assert len(selected) == 4
