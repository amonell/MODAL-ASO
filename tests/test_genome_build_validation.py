from copy import deepcopy

from aso_agents import pipeline_tools


def _validation(goal, request="Design splice-switching ASOs for human GENE1."):
    context = {"structured_goal": deepcopy(goal)}
    task = {"task_type": "design_new_asos", "context": context}
    envelope = {"request_text": request, "structured_goal": deepcopy(goal), "tasks": [task]}
    return pipeline_tools._mixed_genome_build_validation_entry(envelope, task, context)


def test_archive_citation_does_not_create_a_second_coordinate_frame():
    goal = {
        "patient_variant": {"genome_build": "GRCh38", "resolution": {"assembly_name": "GRCh38"}},
        "pending_isoform_transcript_confirmations": [{
            "rationale": "The GRCh37 archive confirms the accession identity and lists chr1:100-200; the selected design uses GRCh38.",
        }],
    }
    assert _validation(goal) is None


def test_clarification_prose_does_not_reopen_the_resolved_build_choice():
    goal = {"coordinate_build": "hg38", "clarification_history": [{
        "question": "Which single coordinate build should I use: GRCh37 or GRCh38?",
        "answer": "Use GRCh38/hg38. Convert any GRCh37/hg19 intervals before tiling.",
    }]}
    assert _validation(goal) is None


def test_actual_conflicting_coordinate_metadata_still_blocks():
    goal = {"genome_build": "GRCh38", "region_constraints": [{
        "coordinate_space": "genomic", "intervals_bed0_half_open": [{
            "chrom": "chr1", "start": 100, "end": 200, "genome_build": "GRCh37",
        }],
    }]}
    assert _validation(goal)["issue"] == "mixed_genome_build_coordinates"


def test_a_build_preference_does_not_relabel_unconverted_intervals():
    goal = {"coordinate_build": "GRCh38", "notes": "Use GRCh38 and convert GRCh37 intervals.",
        "region_constraints": [{"genome_build": "GRCh37", "chrom": "chr1", "start": 100, "end": 200}]}
    assert _validation(goal)["issue"] == "mixed_genome_build_coordinates"


def test_resolved_assembly_metadata_is_validated():
    goal = {"genome_build": "GRCh38", "patient_variant": {"resolution": {"assembly_name": "GRCh37"}}}
    assert _validation(goal)["issue"] == "mixed_genome_build_coordinates"


def test_free_text_target_coordinates_remain_subject_to_validation():
    goal = {"genome_build": "GRCh38", "target_locus": "hg19 chr1:100-200"}
    assert _validation(goal)["issue"] == "mixed_genome_build_coordinates"


def test_aliases_for_one_assembly_are_not_a_conflict():
    assert _validation({"genome_build": "GRCh38/hg38", "assembly_name": "hg38"}) is None


def test_mixed_coordinates_in_original_request_still_block():
    assert _validation({}, "Use hg19 chr1:100-200 and hg38 chr1:110-210.")["issue"] == "mixed_genome_build_coordinates"
