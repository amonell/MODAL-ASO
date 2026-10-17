"""A public file preview must not create extra physical observations."""

from aso_agents.input_rows import reconcile_file_expansions


def test_prime_notation_in_file_preview_is_one_observation() -> None:
    file_row = {"name": "P01", "sequence": "TGCTCAGACAGTGTCTGCAC",
                "chemistry": "full-PS 5-10-5 2-MOE/DNA gapmer",
                "source_file": "/input/panel.csv", "source_row": 2}
    preview = {"name": "P01", "sequence": "TGCTCAGACAGTGTCTGCAC",
               "chemistry": "full-PS 5-10-5 2′-MOE/DNA gapmer"}
    rows = reconcile_file_expansions([file_row, preview], 1, "Evaluate /input/panel.csv")
    assert len(rows) == 1
    assert rows[0]["source_file"] == "/input/panel.csv"
    assert rows[0]["input_reconciliation"][0]["kind"] == "parsed_file_row_alias"


def test_distinct_chemistry_and_explicit_inline_source_remain_distinct() -> None:
    file_row = {"name": "P01", "sequence": "TGCTCAGACAGTGTCTGCAC",
                "chemistry": "full-PS 5-10-5 2-MOE/DNA gapmer",
                "source_file": "/input/panel.csv", "source_row": 2}
    distinct = {"name": "P01", "sequence": "TGCTCAGACAGTGTCTGCAC",
                "chemistry": "full-PS 5-10-5 LNA/DNA gapmer"}
    inline = {"name": "P01", "sequence": "TGCTCAGACAGTGTCTGCAC",
              "chemistry": "full-PS 5-10-5 2′-MOE/DNA gapmer",
              "input_origin": "inline_user"}
    rows = reconcile_file_expansions([file_row, distinct, inline], 1, "Evaluate /input/panel.csv")
    assert len(rows) == 3
