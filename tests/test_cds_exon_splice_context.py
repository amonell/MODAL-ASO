"""Numbered-exon splice design keeps transcript resolution after re-intake."""


def test_numbered_exon_splice_objective_seeds_context_despite_broad_cds_label():
    import run_goal_intake as runtime

    goal = {
        "gene_symbol": "GENE", "species": "human", "intent": "splice_modulation",
        "target_region": "cds", "focus_exons_only": True,
        "run_splice_eval": True,
        "splice_isoform_objectives": [{
            "objective_scope": "default_exon_context", "exon_label": 7,
            "desired_effect": "skipping", "desired_abundance": "decrease",
        }],
    }
    assert runtime._should_seed_default_splice_context(goal)
    assert not runtime._should_seed_default_splice_context({
        **goal, "splice_isoform_objectives": [], "focus_exons_only": False,
    })


def test_saved_postclarification_exon_request_exposes_same_general_guard():
    # A synthetic version of the observed post-clarification goal shape;
    # no case, gene, transcript, coordinate or sequence is hardcoded.
    import run_goal_intake as runtime

    goal = {
        "gene_symbol": "GENE", "species": "human", "intent": "splice_modulation",
        "target_region": "cds", "transcript_id": "TX.1",
        "focus_exons_only": True, "run_splice_eval": True,
        "splice_isoform_objectives": [{
            "objective_scope": "default_exon_context", "exon_label": 12,
            "desired_effect": "skipping", "desired_abundance": "decrease",
            "role": "context",
        }],
    }
    assert runtime._should_seed_default_splice_context(goal)


def test_postclarification_cds_exon_request_materializes_transcript_context(monkeypatch):
    import run_goal_intake as runtime

    transcript = {"id": "TX.1", "is_canonical": True, "length": 150}
    structure = {
        "id": "TX.1", "seq_region_name": "1", "strand": 1,
        "exons_tx": [
            {"index": 11, "start": 101, "end": 150, "contains_coding": True},
            {"index": 12, "start": 201, "end": 250, "contains_coding": True},
            {"index": 13, "start": 301, "end": 350, "contains_coding": True},
        ],
    }
    monkeypatch.setattr(runtime, "get_gene_transcript_structures", lambda **kwargs: {
        "transcripts": [transcript], "structures": {"TX.1": structure},
    })
    monkeypatch.setattr(runtime, "_generate_splice_strategy_analysis", lambda *args, **kwargs: {})
    goal = {
        "gene_symbol": "GENE", "species": "human", "intent": "splice_modulation",
        "target_region": "cds", "transcript_id": "TX.1", "focus_exons_only": True,
        "run_splice_eval": True,
        "splice_isoform_objectives": [{
            "objective_scope": "default_exon_context", "exon_label": 12,
            "desired_effect": "skipping", "desired_abundance": "decrease", "role": "context",
        }],
    }

    result = runtime._ensure_default_splice_context(goal, {}, user_request="Design exon-skipping ASOs")

    assert result["resolved_transcript_strategy"]["transcript_ids"] == ["TX.1"]
    assert goal["isoform_transcripts"]["TX.1"]["exons_tx"] == structure["exons_tx"]


def test_cross_species_cdna_hit_maps_through_both_negative_strand_exons():
    import run_goal_intake as runtime

    placement = {"reference_id": "TX", "start_1based": 5, "end_1based": 9,
                 "coordinate_scope": "transcript_cdna"}
    transcript = {
        "transcript_id": "TX", "chrom": "2", "strand": "-",
        "metadata": {"version": 3, "assembly_name": "assembly-A", "reference_source": "annotated-reference"},
        "sequence_evidence": {"method": "reference_genome_exon_assembly"},
        "exons_tx": [
            {"index": 1, "start": 201, "end": 206, "cdna_start": 1, "cdna_end": 6},
            {"index": 2, "start": 101, "end": 106, "cdna_start": 7, "cdna_end": 12},
        ],
    }
    runtime._annotate_cdna_placement_genomic_context(
        [placement], {"TX": transcript}, {"TX": "ACGTACGTACGT"})

    assert placement["genomic_mapping_status"] == "resolved_from_searched_transcript_exons"
    assert placement["transcript_versioned_id"] == "TX.3"
    assert placement["assembly"] == "assembly-A"
    assert placement["target_strand"] == "-"
    assert placement["exon_ordinals"] == [1, 2]
    assert [(part["start"], part["end"]) for part in placement["genomic_segments_1based_inclusive"]] == [
        (201, 202), (104, 106)]
