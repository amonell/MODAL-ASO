from copy import deepcopy

from aso_agents.translation_start_contract import canonicalize_translation_start_program_windows


def _goal():
    window = {"anchor_site": "acceptor", "exon_label": 1,
              "feature_name": "start_codon_region", "reference_transcript_id": "ENST_TEST.1",
              "relative_start_nt": -30, "relative_end_nt": 15}
    constraint = {"source_type": "splice_relative_window",
                  "reference_transcript_id": "ENST_TEST.1",
                  "aso_overlap_requirement": "full_containment",
                  "splice_relative_window": window}
    return {"design_programs": [
        {"program_id": "gapmer", "modality": "gapmer", "intent": "knockdown",
         "transcript_id": "ENST_TEST.1", "region_constraints": []},
        {"program_id": "steric_a", "modality": "steric", "intent": "binding",
         "transcript_id": "ENST_TEST.1", "region_constraints": [deepcopy(constraint)]},
        {"program_id": "steric_b", "modality": "steric", "intent": "binding",
         "transcript_id": "ENST_TEST.1", "region_constraints": [deepcopy(constraint)]},
    ], "evaluation_targets": [{"target_type": "region", "transcript_id": "ENST_TEST.1",
                               "splice_relative_window": deepcopy(window)}]}


def test_composite_translation_start_windows_are_not_splice_acceptors():
    goal = _goal()
    assert canonicalize_translation_start_program_windows(goal)
    assert "target_region" not in goal
    assert goal["design_programs"][0]["region_constraints"] == []
    for program in goal["design_programs"][1:]:
        assert program["target_region"] == "translation_start"
        assert program["flanks"] == {"up": 30, "down": 15}
        assert program["region_constraints"] == []
    assert "splice_relative_window" not in goal["evaluation_targets"][0]
    assert not canonicalize_translation_start_program_windows(goal)


def test_other_splice_and_explicit_regions_are_untouched():
    for mutate in (
        lambda g: g["design_programs"][1].update(intent="splice_modulation"),
        lambda g: g["design_programs"][1]["region_constraints"][0]["splice_relative_window"].update(feature_name="splice acceptor"),
        lambda g: g["design_programs"][1]["region_constraints"][0].update(rows=[["1", 10, 20]]),
        lambda g: g["design_programs"][1].update(flanks={"up": 10, "down": 10}),
    ):
        goal = _goal()
        mutate(goal)
        original = deepcopy(goal["design_programs"][1])
        canonicalize_translation_start_program_windows(goal)
        assert goal["design_programs"][1] == original


def test_model_inferred_genomic_row_yields_to_user_requested_start_window():
    goal = _goal()
    goal['design_programs'] = [goal['design_programs'][1]]
    goal['design_programs'][0]['region_constraints'] = [{
        'source_type': 'explicit_rows', 'source': 'user_request',
        'aso_overlap_requirement': 'full_containment',
        'intervals_bed0_half_open': [{'chrom': '17', 'start': 42317580,
                                      'end': 42317650, 'strand': '-'}],
    }]
    request = 'Target 30 nt before through 15 nt after the annotated translation start.'
    assert canonicalize_translation_start_program_windows(goal, user_evidence_text=request)
    assert goal['design_programs'][0]['target_region'] == 'translation_start'
    assert goal['design_programs'][0]['flanks'] == {'up': 30, 'down': 15}
    assert goal['design_programs'][0]['region_constraints'] == []


def test_user_supplied_genomic_row_is_not_discarded():
    goal = _goal()
    goal['design_programs'] = [goal['design_programs'][1]]
    constraint = {'source_type': 'explicit_rows', 'source': 'user_request',
                  'aso_overlap_requirement': 'full_containment',
                  'intervals_bed0_half_open': [{'chrom': '17', 'start': 42317580,
                                                'end': 42317650, 'strand': '-'}]}
    goal['design_programs'][0]['region_constraints'] = [deepcopy(constraint)]
    request = 'Use genomic interval chr17:42,317,581–42,317,650 as the explicit binding region.'
    assert not canonicalize_translation_start_program_windows(goal, user_evidence_text=request)
    assert goal['design_programs'][0]['region_constraints'] == [constraint]
