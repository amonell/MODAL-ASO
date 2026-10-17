import math

import run_goal_intake as runtime


class Track:
    def values(self, chrom, start, end):
        assert chrom == "chrFixture"
        return [1.0, 2.0, float("nan"), 8.0][start:end]


def test_single_base_exon_piece_contributes_to_the_target_mean():
    assert runtime._avg_bigwig_over_segments(Track(), [("chrFixture", 1, 1), ("chrFixture", 4, 4)]) == 4.5


def test_missing_bases_and_invalid_ranges_do_not_replace_real_single_base_values():
    observed = runtime._avg_bigwig_over_segments(Track(), [("chrFixture", 4, 2), ("chrFixture", 2, 3)])
    assert observed == 2.0
    assert runtime._avg_bigwig_over_segments(Track(), [("chrFixture", 3, 3)]) is None
