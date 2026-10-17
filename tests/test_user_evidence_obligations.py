from copy import deepcopy
import asyncio
import json

import run_goal_intake as intake
from aso_agents.design_delivery import write_design_panel
from aso_agents.pipeline_tools import _normalized_generated_candidate_row


def test_accessibility_is_mandatory_only_with_exact_user_provenance():
    request = "Design three ASOs and report sequence-based target RNA accessibility predictions."
    goal = {"target_accessibility": True,
            "target_accessibility_request_quote": "target RNA accessibility predictions"}
    intake._normalize_requested_accessibility_provenance(goal, request)
    assert goal["target_accessibility"] is True

    for quote in (None, "accessibility of the genome", "predictions not in the request"):
        unsupported = {"target_accessibility": True}
        if quote is not None:
            unsupported["target_accessibility_request_quote"] = quote
        intake._normalize_requested_accessibility_provenance(unsupported, request)
        assert "target_accessibility" not in unsupported
        assert "target_accessibility_request_quote" not in unsupported


def test_inferred_accessibility_does_not_block_an_unrelated_splice_request():
    request = "Design three distinct steric ASOs for exon inclusion with candidate-specific evidence."
    payload = {"status": "ready", "structured_goal": {
        "gene_symbol": "GENE", "species": "human", "target_context": "cultured cells",
        "intent": "splice_modulation", "target_accessibility": True,
    }}
    normalized = intake._normalize_intake_output(deepcopy(payload), request_text=request)
    assert "target_accessibility" not in normalized["structured_goal"]


def test_coding_feature_requires_complete_site_containment(monkeypatch):
    monkeypatch.setattr(intake, "_build_cds_bed_rows",
                        lambda goal: [["chr1", 100, 120, "coding", 0, "+"]])
    monkeypatch.setattr(intake, "_map_genomic_bed_rows_to_cdna",
                        lambda rows, transcript, label: [[label, 200, 220, "coding", 0, "+"]])
    goal = {"isoform_transcripts": {"TX": {"transcript_id": "TX", "strand": "+"}}}
    rows, meta = intake._build_program_target_region_rows(
        goal, goal["isoform_transcripts"],
        {"program_id": "coding", "target_region": "cds", "transcript_id": "TX"},
        output_level="cdna", iso_label_primary="TX")
    assert rows
    assert meta["tile_overlap_requirement"] == "full_containment"


def test_independent_quoted_audit_repairs_omitted_panel_and_feature(monkeypatch):
    request = ("Design three distinct gapmers. Each complete binding site must lie in an "
               "annotated coding exon. Calculate the conditional frame consequence of skipping exon 9.")
    programs = [{"program_id": f"slot_{i}", "label": f"gapmer {i}",
                 "modality": "gapmer", "intent": "knockdown", "objective": "same panel"}
                for i in range(1, 4)]
    draft = {"status": "ready", "target_accessibility": True,
             "design_programs": deepcopy(programs), "structured_goal": {
        "candidate_limit_per_site": 3, "min_asos_per_site": 3,
        "target_accessibility": True, "design_programs": programs,
    }, "request_envelope": {"tasks": [{"task_type": "design_new_asos",
        "context": {"structured_goal": {"candidate_limit_per_site": 3,
                                         "target_accessibility": True,
                                         "design_programs": deepcopy(programs)}}}]}}
    audit = {
        "candidate_count": {"count": 3, "scope": "global", "kind": "exact",
                            "quote": "Design three distinct gapmers"},
        "target_feature": {"feature": "cds", "applies_to_all_candidates": True,
                           "quote": "Each complete binding site must lie in an annotated coding exon"},
        "target_accessibility": None,
        "conditional_exon_skip_events": [{"exon_label": 9, "transcript_id": None,
                                         "quote": "conditional frame consequence of skipping exon 9"}],
    }
    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()
    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    goal = result["structured_goal"]
    assert (goal["candidate_minimum"], goal["candidate_limit"]) == (3, 3)
    assert "candidate_limit_per_site" not in goal
    assert "target_accessibility" not in goal
    assert goal["target_region"] == "cds"
    assert len(goal["design_programs"]) == 1
    assert goal["design_programs"][0]["target_region"] == "cds"
    assert goal["conditional_exon_skip_events"][0]["exon_label"] == 9
    nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert nested["candidate_limit"] == 3
    assert nested["target_region"] == "cds"
    assert len(nested["design_programs"]) == 1
    assert "target_accessibility" not in nested
    assert "target_accessibility" not in result
    assert len(result["design_programs"]) == 1


def test_quoted_global_upper_bound_preserves_lower_bound_and_limits_panel(monkeypatch):
    request = "Design an exploratory panel of 4–6 distinct steric oligos."
    goal = {"candidate_minimum": 4, "design_programs": [{
        "program_id": "panel", "modality": "steric", "intent": "binding",
        "objective": "Design an exploratory panel",
    }]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"tasks": [{"task_type": "design_new_asos",
                "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"candidate_count": {"count": 6, "scope": "global", "kind": "maximum",
                                 "quote": "4–6 distinct"},
             "target_feature": None, "target_window": None,
             "target_accessibility": None, "conditional_exon_skip_events": []}

    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    assert result["structured_goal"]["candidate_minimum"] == 4
    assert result["structured_goal"]["candidate_limit"] == 6
    nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert nested["candidate_minimum"] == 4
    assert nested["candidate_limit"] == 6


def test_quoted_separated_site_requirement_selects_diverse_physical_panel(monkeypatch):
    request = "Design six steric oligos. Include exon-internal and splice-flank hypotheses where physically feasible."
    goal = {"candidate_minimum": 4, "candidate_limit": 6,
            "design_programs": [{"program_id": "panel", "modality": "steric",
                                 "objective": "Explore splice blocking sites"}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"tasks": [{"task_type": "design_new_asos",
                "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"candidate_count": None,
             "candidate_site_diversity": {"required": True,
                "quote": "Include exon-internal and splice-flank hypotheses where physically feasible"},
             "target_feature": None, "target_window": None,
             "target_accessibility": None, "conditional_exon_skip_events": []}

    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    assert result["structured_goal"]["candidate_selection_policy"] == "diverse_target_sites"
    nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert nested["candidate_selection_policy"] == "diverse_target_sites"


def test_quoted_start_window_replaces_unattested_model_coordinates(monkeypatch):
    request = ("Design steric ASOs around the annotated start codon, from 30 nt "
               "upstream through 15 nt downstream, on the named transcript.")
    inferred = {"source_type": "explicit_rows", "coordinate_space": "genomic",
                "intervals_bed0_half_open": [{"chrom": "chr1", "start": 900001, "end": 900047}]}
    program = {"program_id": "start", "label": "start", "modality": "steric",
               "intent": "binding", "objective": "block initiation", "target_region": "translation_start",
               "region_constraints": [deepcopy(inferred)]}
    goal = {"target_region": "translation_start", "design_programs": [program],
            "region_constraints": [deepcopy(inferred)],
            "evaluation_targets": [{"target_type": "region", "chrom": "chr1",
                                    "start": 900002, "end": 900047, "strand": "+"}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"tasks": [{"task_type": "design_new_asos",
                "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"candidate_count": None,
             "target_feature": {"feature": "translation_start", "applies_to_all_candidates": True,
                                "quote": "annotated start codon"},
             "target_window": {"feature": "translation_start", "upstream_nt": 30,
                               "downstream_nt": 15,
                               "quote": "from 30 nt upstream through 15 nt downstream"},
             "target_accessibility": None, "conditional_exon_skip_events": []}
    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()
    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    for checked in (result["structured_goal"],
                    result["request_envelope"]["tasks"][0]["context"]["structured_goal"]):
        assert checked["flanks"] == {"up": 30, "down": 15}
        assert "region_constraints" not in checked
        assert "region_constraints" not in checked["design_programs"][0]
        assert checked["design_programs"][0]["flanks"] == checked["flanks"]
        assert "start" not in checked["evaluation_targets"][0]
    assert result["request_obligation_audit"]["target_window"]["request_quote"] in request


def test_composite_report_with_one_design_task_audits_quoted_start_window(monkeypatch):
    request = ("Design steric ASOs around the annotated start codon, from 60 nt "
               "upstream through 30 nt downstream, and export a report.")
    goal = {"target_region": "cds", "flanks": {"up": 60, "down": 30},
            "design_programs": [{"program_id": "panel", "modality": "steric",
                                 "intent": "binding", "objective": "block initiation"}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"workflow_name": "build_composite_report",
                 "tasks": [{"task_type": "design_new_asos",
                            "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"target_feature": {"feature": "translation_start",
                "applies_to_all_candidates": True, "quote": "annotated start codon"},
             "target_window": {"feature": "translation_start", "upstream_nt": 60,
                "downstream_nt": 30,
                "quote": "from 60 nt upstream through 30 nt downstream"}}

    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    assert result["request_obligation_audit"]["status"] == "accepted"
    for checked in (result["structured_goal"],
                    result["request_envelope"]["tasks"][0]["context"]["structured_goal"]):
        assert checked["target_region"] == "translation_start"
        assert checked["design_programs"][0]["target_region"] == "translation_start"
        assert checked["flanks"] == {"up": 60, "down": 30}


def test_quoted_start_window_discards_unattested_splice_anchor(monkeypatch):
    request = ("Design steric ASOs at the annotated translation-initiation AUG, "
               "between 60 nt upstream and 30 nt downstream of it.")
    inferred = {"source_type": "splice_relative_window", "exon_label": 1,
                "splice_relative_window": {"exon_label": 1, "anchor_site": "acceptor",
                    "relative_start_nt": -60, "relative_end_nt": 30}}
    explicit = {**deepcopy(inferred), "request_quote": "at the exon 1 acceptor"}
    goal = {"target_region": "cds", "region_constraints": [deepcopy(inferred)],
            "design_programs": [{"program_id": "p1", "modality": "steric", "intent": "binding",
                                 "region_constraints": [deepcopy(inferred)]}]}
    draft = {"status": "ready", "structured_goal": deepcopy(goal),
             "request_envelope": {"workflow_name": "build_composite_report",
                 "tasks": [{"task_type": "design_new_asos",
                            "context": {"structured_goal": deepcopy(goal)}}]}}
    audit = {"target_feature": {"feature": "translation_start",
                "applies_to_all_candidates": True, "quote": "annotated translation-initiation AUG"},
             "target_window": {"feature": "translation_start", "upstream_nt": 60,
                "downstream_nt": 30,
                "quote": "between 60 nt upstream and 30 nt downstream of it"}}

    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()

    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    resolved = result["structured_goal"]
    assert resolved["target_region"] == "translation_start"
    assert "region_constraints" not in resolved
    assert "region_constraints" not in resolved["design_programs"][0]
    nested = result["request_envelope"]["tasks"][0]["context"]["structured_goal"]
    assert "region_constraints" not in nested
    assert "region_constraints" not in nested["design_programs"][0]

    quoted_request = request + " Also include a site at the exon 1 acceptor."
    preserved = {"target_region": "translation_start", "region_constraints": [explicit]}
    assert not intake._remove_unattested_splice_windows_for_start(preserved, quoted_request)
    assert preserved["region_constraints"] == [explicit]


def test_materialization_rechecks_later_unattested_start_constraints(tmp_path):
    request = ("Design steric ASOs at the annotated translation-initiation AUG, "
               "between 60 nt upstream and 30 nt downstream of it.")
    (tmp_path / "request.txt").write_text(request)
    inferred_splice = {"source_type": "splice_relative_window",
                       "splice_relative_window": {"exon_label": 1, "anchor_site": "donor",
                           "relative_start_nt": -60, "relative_end_nt": 30}}
    inferred_absolute = {"source_type": "explicit_rows", "coordinate_space": "genomic",
                         "intervals_bed0_half_open": [{"chrom": "chr12", "start": 0, "end": 91}]}
    goal = {"target_region": "translation_start", "artifact_refs": [
                {"artifact_type": "reference_annotation", "path": "public_reference"}],
            "region_constraints": [inferred_absolute], "design_programs": [{
                "program_id": "p1", "modality": "steric", "intent": "binding",
                "target_region": "translation_start", "region_constraints": [inferred_splice]}]}
    payload = {"request_obligation_audit": {"status": "accepted",
               "target_feature": {"feature": "translation_start"},
               "target_window": {"up": 60, "down": 30}}}
    result = intake._materialize_region_constraints(goal, payload, str(tmp_path))
    assert result == {"constraint_count": 0, "status": "skipped"}
    assert "region_constraints" not in goal
    assert "region_constraints" not in goal["design_programs"][0]
    assert "region_constraints" not in payload


def test_user_written_absolute_interval_survives_relative_window_repair():
    goal = {"artifact_refs": [], "region_constraints": [{"source_type": "explicit_rows",
            "coordinate_space": "genomic", "intervals_bed0_half_open": [
                {"chrom": "chr1", "start": 99, "end": 200}]}]}
    assert not intake._remove_unattested_absolute_start_windows(goal, "chr1:100-200")
    assert goal["region_constraints"][0]["intervals_bed0_half_open"][0]["start"] == 99

    supplied = deepcopy(goal)
    supplied["artifact_refs"] = [{"artifact_type": "user_region", "path": "input.bed"}]
    assert not intake._remove_unattested_absolute_start_windows(supplied, "No coordinates typed here")
    assert supplied["region_constraints"] == goal["region_constraints"]


def test_design_audit_does_not_reinterpret_a_submitted_oligo_evaluation(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("No design obligation audit for submitted-molecule evaluation")
    monkeypatch.setattr(intake.Runner, "run", forbidden)
    original = {"status": "ready", "structured_goal": {"provided_asos": [{"sequence": "ACGT"}]},
                "request_envelope": {"workflow_name": "evaluate_existing_aso_set",
                                     "tasks": [{"task_type": "evaluate_existing_aso_set"}]}}
    result = asyncio.run(intake._audit_request_obligations(deepcopy(original),
                          "Evaluate exactly two submitted ASOs as written", []))
    assert result == original


def test_quoted_supplied_count_reconciles_only_redundant_inline_parser_row(monkeypatch):
    request = ("Evaluate these two submitted oligos: alpha = AACCGGTTAACCGGTTAACC; "
               "beta = TTGGAACCTTGGAACCTTGG. Do not redesign.")
    alpha = {"name": "alpha", "sequence": "AACCGGTTAACCGGTTAACC", "chemistry": "MOE"}
    beta = {"name": "beta", "sequence": "TTGGAACCTTGGAACCTTGG", "chemistry": "MOE"}
    draft = {"status": "ready", "structured_goal": {}, "request_envelope": {
        "workflow_name": "evaluate_existing_aso_set", "tasks": [{
            "task_type": "evaluate_existing_aso_set", "context": {
                "provided_asos": [deepcopy(alpha), deepcopy(beta), deepcopy(beta)]}}]}}
    audit = {"supplied_observation_count": {
        "count": 2, "quote": "these two submitted oligos"}}
    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps(audit)})()
    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    rows = result["request_envelope"]["tasks"][0]["context"]["provided_asos"]
    assert [(row["name"], row["sequence"]) for row in rows] == [
        (alpha["name"], alpha["sequence"]), (beta["name"], beta["sequence"])]
    assert rows[1]["input_reconciliation"][0]["kind"] == "parser_duplicate_inline_observation"
    assert result["supplied_observation_audit"]["status"] == "reconciled"


def test_supplied_count_does_not_erase_attested_duplicate_or_file_rows(monkeypatch):
    sequence = "AACCGGTTAACCGGTTAACC"
    request = f"Evaluate two observations of the same oligo: {sequence}; {sequence}."
    row = {"name": "same", "sequence": sequence}
    draft = {"status": "ready", "structured_goal": {}, "request_envelope": {
        "workflow_name": "evaluate_existing_aso_set", "tasks": [{
            "task_type": "evaluate_existing_aso_set", "context": {
                "provided_asos": [deepcopy(row), deepcopy(row)]}}]}}
    async def fake_run(*args, **kwargs):
        return type("Response", (), {"final_output": json.dumps({
            "supplied_observation_count": {"count": 1, "quote": "same oligo"}})})()
    monkeypatch.setattr(intake.Runner, "run", fake_run)
    result = asyncio.run(intake._audit_request_obligations(draft, request, []))
    assert len(result["request_envelope"]["tasks"][0]["context"]["provided_asos"]) == 2
    assert result["supplied_observation_audit"]["status"] == "unresolved"
    assert intake._reconcile_redundant_inline_observations([
        {**row, "source_file": "input.csv", "source_row": 1},
        {**row, "source_file": "input.csv", "source_row": 2}], 1, request) is None


def test_gapmer_target_strand_survives_cdna_to_genomic_export(tmp_path):
    tile = tmp_path / "tiles.json"
    tile.write_text(json.dumps({"schema": {"columns": {}}, "rows": [{
        "Chrom": "TX", "Start": 1, "End": 20, "Strand": "-",
        "Sequence": "ACGTACGTACGTACGTACGT",
        "Synthesis": "ACGTACGTACGTACGTACGT", "selection_rank": 1,
    }]}))
    catalog = {"TX": {"transcript_id": "TX", "strand": "-", "chrom": "chr1",
                      "exons_tx": [{"index": 1, "seq_region_name": "chr1", "start": 101,
                                    "end": 120, "cdna_start": 1, "cdna_end": 20}]}}
    assert intake._annotate_gapmer_genomic_coords(str(tile), catalog, "TX") == {"annotated": 1}
    intake._strip_cdna_location_columns(str(tile))
    raw = json.loads(tile.read_text())["rows"][0]
    assert "Strand" not in raw and raw["Genomic_strand"] == "-"
    row = _normalized_generated_candidate_row(raw)
    assert row["strand"] == "-"
    exports = write_design_panel({"rows": [row]}, {}, tmp_path)
    assert "target_genomic_strand" in (tmp_path / "design_panel.csv").read_text()
    assert "|- |" not in (tmp_path / "design_panel.md").read_text()
    assert "| - |" in (tmp_path / "design_panel.md").read_text()
