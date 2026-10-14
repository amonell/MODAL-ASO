from aso_agents.pipeline_tools import (_annotate_requested_exon_proximity,
                                       _apply_generated_candidate_limit)


def _goal():
    return {'design_programs': [{'program_id': 'program-a', 'focus_exons_only': True,
             'transcript_id': 'ENST000001.1', 'exon_label': 6,
             'anchor_site_records': [{'chrom': 'chr1', 'position': 500, 'site_kind': 'donor'}]}],
            'isoform_transcripts': {'ENST000001.1': {'exons_tx': [
                {'index': 6, 'start': 500, 'end': 570, 'seq_region_name': 'chr1'},
                {'index': 7, 'start': 800, 'end': 870, 'seq_region_name': 'chr1'}]}}}


def _row(start, rank):
    return {'sequence': 'ACGTACGTACGTACGTACGT', 'chrom': 'chr1', 'start': start,
            'end': start + 20, 'strand': '+', 'selection_rank': rank,
            'program_id': 'program-a'}


def test_exon_focused_selection_uses_binding_location_before_route_rank():
    rows = [_row(820, 1), _row(510, 2), _row(480, 3)]
    _annotate_requested_exon_proximity(rows, _goal())
    selected, receipt = _apply_generated_candidate_limit(rows, candidate_limit=2, per_site=False)
    assert [row['start'] for row in selected] == [510, 480]
    assert [row['requested_exon_proximity_tier'] for row in selected] == [0, 0]
    assert receipt['selected_count'] == 2


def test_distal_candidate_is_available_for_shortfall_and_unspecified_scope_is_unchanged():
    rows = [_row(820, 1), _row(510, 2)]
    _annotate_requested_exon_proximity(rows, _goal())
    selected, _ = _apply_generated_candidate_limit(rows, candidate_limit=2, per_site=False)
    assert len(selected) == 2
    unscoped = [_row(820, 1), _row(510, 2)]
    _annotate_requested_exon_proximity(unscoped, {'design_programs': []})
    selected, _ = _apply_generated_candidate_limit(unscoped, candidate_limit=1, per_site=False)
    assert selected[0]['start'] == 820


def test_unranked_route_rows_still_prioritize_physical_objective():
    rows = [_row(820, 1), _row(510, 2)]
    for row in rows:
        del row['selection_rank']
    _annotate_requested_exon_proximity(rows, _goal())
    selected, _ = _apply_generated_candidate_limit(rows, candidate_limit=1, per_site=False)
    assert selected[0]['start'] == 510


def test_exon_overlap_and_anchor_accept_reference_contig_aliases():
    goal = _goal()
    goal['isoform_transcripts']['ENST000001.1']['exons_tx'][0]['seq_region_name'] = '1'
    goal['design_programs'][0]['anchor_site_records'][0]['chrom'] = '1'
    rows = [_row(498, 1)]
    _annotate_requested_exon_proximity(rows, goal)
    assert rows[0]['requested_exon_overlap_bp'] == 19
    assert rows[0]['requested_anchor_distance_bp'] == 0


def test_one_based_inclusive_native_segment_counts_all_twenty_bases():
    rows = [_row(500, 1)]
    rows[0]['end'] = 519
    rows[0]['Genomic_segments'] = '[{"chrom":"chr1","start":500,"end":519}]'
    _annotate_requested_exon_proximity(rows, _goal())
    assert rows[0]['requested_exon_overlap_bp'] == 20


def test_bed0_binding_interval_takes_precedence_over_ambiguous_display_span():
    rows = [_row(500, 1)]
    rows[0]['end'] = 519
    rows[0]['Genomic_span_start'] = 500
    rows[0]['binding_intervals_bed0_half_open'] = [['chr1', 499, 519]]
    _annotate_requested_exon_proximity(rows, _goal())
    assert rows[0]['requested_exon_overlap_bp'] == 20


def test_multiple_genomic_segments_are_counted_without_intronic_gap():
    rows = [_row(500, 1)]
    rows[0]['Genomic_segments'] = '[{"chrom":"chr1","start":500,"end":509},{"chrom":"chr1","start":800,"end":809}]'
    _annotate_requested_exon_proximity(rows, _goal())
    assert rows[0]['requested_exon_overlap_bp'] == 10
