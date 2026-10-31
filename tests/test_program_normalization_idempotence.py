from copy import deepcopy

from aso_agents import pipeline_tools as pipeline


def program():
    return {"program_id":"joint_event", "modality":"steric", "intent":"splice_modulation",
        "label":"Joint event", "objective":"Alter the requested event", "site_kind":"both",
        "region_constraints":[{"exon_label":1,"source_type":"splice_relative_window"},
                              {"exon_label":2,"source_type":"splice_relative_window"}]}


def test_intron_flank_references_never_expand_into_exon_outcomes():
    initial = program()
    initial["splice_event"] = {"event_type":"intron_retention","intron_number":1,
                                "reference_transcript_id":"TX_NEUTRAL.1"}
    goal = {"exon_labels":[1,2], "design_programs":[initial]}
    rows = [deepcopy(initial)]
    for _ in range(6):
        rows = pipeline._request_contract_split_steric_programs_by_exon(rows,goal,{})
        assert rows == [initial]


def test_repeated_exon_normalization_is_idempotent_with_shared_binding_constraints():
    initial = program()
    initial["exon_labels"] = [1,2]
    goal = {"exon_labels":[1,2], "design_programs":[initial]}
    expected = pipeline._request_contract_split_steric_programs_by_exon([initial],goal,{})
    assert len(expected) == 2
    assert {row["exon_label"] for row in expected} == {1,2}
    actual = deepcopy(expected)
    for _ in range(6):
        actual = pipeline._request_contract_split_steric_programs_by_exon(actual,goal,{})
        assert actual == expected
    assert all(row["region_constraints"] == initial["region_constraints"] for row in actual)
