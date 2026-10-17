from copy import deepcopy

import pytest

from aso_agents.feature_identity import exon_number
from aso_agents import pipeline_tools as p


@pytest.mark.parametrize("label,expected", [
    (7, 7), (7.0, 7), ("7", 7), ("#7", 7), ("exon 7", 7),
    ("Exon #7", 7), ("exon_7", 7), ("exon-7", 7), ("7th exon", 7),
    ("7 (reference numbering)", 7), ("29-nt exon 7", 7),
    ("exon 7 (29 nt)", 7), ("TX000123.4 exon 7", 7),
    ("29-nt alternative exon", None), ("exon 29 nt long", None),
    ("29bp cassette", None), ("TX000123.4", None), ("chr8:20-40", None),
    ("8x", None), ("exon 8x", None), ("exon 8.1", None),
    ("exon 7 or exon 9", None), ("exon 7-9", None),
    ("exon 7/9", None), ("exon 7 and 9", None),
    (0, None), (-7, None), (True, None), (7.2, None),
    (float("nan"), None), ("-7", None), ("exon -7", None),
])
def test_explicit_identity_is_distinct_from_length_reference_and_ambiguity(label, expected):
    assert exon_number(label) == expected
    assert p._coerce_exon_label_int(label) == expected
    assert p._coerce_standard_exon_constraint_label(label) == expected


def test_evaluation_normalization_preserves_named_features_without_invented_ordinals():
    original = [
        {"objective_type": "monitoring_safeguard", "target_type": "exon",
         "exon_label": "29-nt alternative exon", "label": "preserved cassette",
         "transcript_id": "TX000123.4", "chrom": "8", "start": 120, "end": 148},
        {"target_type": "exon", "exon_label": "8x", "label": "patient feature"},
        {"target_type": "exon", "exon_label": "exon 7 (29 nt)", "label": "numbered exon"},
    ]
    before = deepcopy(original)
    normalized = p._normalize_evaluation_targets(original)
    assert original == before
    assert normalized[0] == original[0]
    assert normalized[1] == original[1]
    assert normalized[2]["exon_label"] == 7
    assert p._normalize_evaluation_targets(normalized) == normalized
    targets = p._collect_structured_exon_targets({"evaluation_targets": normalized})
    assert [row["exon_label"] for row in targets] == [7]
    assert p._first_standard_exon_label_from_text_or_goal(
        "Preserve the named cassette.",
        {"splice_isoform_objectives": [{"exon_label": "29-nt alternative exon"}]},
    ) is None
