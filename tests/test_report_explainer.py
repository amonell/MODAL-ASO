from __future__ import annotations

import asyncio
import json

from aso_agents import pipeline_tools
from aso_agents import report_explainer


def test_selected_panel_search_budget_preserves_zero_and_unknown():
    exact_only = {"rows": [
        {"selection_rank": 1, "Off_target_mismatch_budget": 0},
        {"selection_rank": 2, "Off_target_mismatch_budget": 0},
    ]}
    text = report_explainer._selected_off_target_budget_markdown(exact_only)
    assert "0 mismatch(es)" in text
    assert "one- and higher-mismatch bins were not searched and are unknown" in text
    mixed = {"rows": [
        {"selection_rank": 1, "Off_target_mismatch_budget": 0},
        {"selection_rank": 2, "Off_target_mismatch_budget": None},
    ]}
    text = report_explainer._selected_off_target_budget_markdown(mixed)
    assert "rank 1: 0" in text and "rank 2: unavailable" in text
    assert "unknown, not zero" in text


def test_transcript_context_keeps_gapmer_primary_before_truncation() -> None:
    primary = "ENSMUST00000465880"
    isoforms = {
        f"ENSMUST00000{i:05d}": {
            "transcript_id": f"ENSMUST00000{i:05d}",
            "match_info": {"method": "catalog"},
        }
        for i in range(20)
    }
    isoforms[primary] = {
        "transcript_id": primary,
        "match_info": {"method": "primary"},
    }

    context = report_explainer.build_report_explainer_context(
        run_dir="/tmp/run",
        payload={"request": "Design mouse Mapt gapmers."},
        structured_goal={"gene_symbol": "Mapt", "species": "mouse"},
        final_payload={
            "isoform_transcripts": isoforms,
            "gapmer_primary_transcript_id": primary,
        },
        report_payload={
            "structured_goal": {"gene_symbol": "Mapt", "species": "mouse"},
        },
    )

    items = context["transcript_context"]["items"]
    assert len(items) <= 12
    assert items[0]["label"] == "gapmer_primary"
    assert items[0]["transcript_id"] == primary


def _minimal_splice_report_context(tmp_path):
    structured_goal = {
        "gene_symbol": "SMN2",
        "intent": "splice_modulation",
        "desired_effect": "increase exon 7 inclusion",
        "species": "human",
        "target_context": "motor neurons",
    }
    report_payload = {
        "structured_goal": structured_goal,
        "branch_outcomes": {
            "steric": {
                "status": "outputs_present",
                "reason": "Steric splice-modulation outputs were produced.",
            }
        },
        "chemistry": {
            "recommendation": {
                "platform": "steric-hindrance",
                "length": 15,
            }
        },
        "tiling": {
            "initial_filtering": {
                "total_rows": 100,
                "passed": {"count": 45},
            }
        },
        "splicing_eval": {
            "site_count": 2,
            "evaluated": 45,
        },
        "strategy_routing": {
            "goal_count": 2,
            "goals": [
                {
                    "rank": 1,
                    "goal_label": "decrease acceptor@70077019",
                    "count": 16,
                    "rationale": "Best primary route.",
                }
            ],
        },
        "steric_jobs": [
            {
                "job_id": "steric_smn2",
                "run_splice_eval": True,
                "run_splice_router": True,
                "region_count": 2,
            }
        ],
        "gapmer_jobs": [],
    }
    payload = {"request": "Design SMN2 steric ASOs to increase exon 7 inclusion."}
    context = report_explainer.build_report_explainer_context(
        run_dir=str(tmp_path),
        payload=payload,
        structured_goal=structured_goal,
        final_payload={},
        report_payload=report_payload,
    )
    return payload, structured_goal, report_payload, context


def test_fallback_report_explainer_is_valid(tmp_path) -> None:
    _, _, _, context = _minimal_splice_report_context(tmp_path)

    fallback = report_explainer.normalize_report_explainer(
        report_explainer.build_fallback_report_explainer(context),
        context,
    )

    assert report_explainer.report_explainer_is_valid(fallback)
    assert fallback["workflow_graph"]["nodes"]
    assert fallback["summary"]["headline"]
    assert any("acceptor@70077019" in item["decision"] for item in fallback["key_decisions"])


def test_report_explainer_timeout_uses_fallback(tmp_path, monkeypatch) -> None:
    payload, structured_goal, report_payload, _ = _minimal_splice_report_context(tmp_path)

    async def slow_run(*_args, **_kwargs):
        await asyncio.sleep(1)
        return None

    monkeypatch.setenv("AGENTIC_ASO_REPORT_EXPLAINER_TIMEOUT_SEC", "0.01")
    monkeypatch.setattr(pipeline_tools.Runner, "run", slow_run)

    result = pipeline_tools._generate_report_explainer(
        run_dir=str(tmp_path),
        payload=payload,
        structured_goal=structured_goal,
        final_payload={},
        report_payload=report_payload,
    )

    assert result["fallback_used"] is True
    assert "timeout" in result["generation_error"]
    assert report_explainer.report_explainer_is_valid(result["explainer"])


def test_gapmer_exon_region_evidence_is_preserved_in_report_manifest() -> None:
    structured_goal = {
        "gene_symbol": "PCSK9",
        "species": "human",
        "intent": "knockdown",
        "exon_label": 4,
        "focus_exons_only": True,
        "region_constraints": [
            {
                "constraint_id": "constraint_01",
                "source_type": "exon",
                "exon_label": 4,
                "transcript_id": "ENST00000302118",
                "aso_overlap_requirement": "full_containment",
            }
        ],
        "design_programs": [
            {
                "program_id": "gapmer_pcsk9_01",
                "modality": "gapmer",
                "exon_label": 4,
                "focus_exons_only": True,
            }
        ],
    }
    final_payload = {
        "gapmer_region_bed": {
            "rows": [["ENST00000302118", 813, 947, "constraint_01|gapmer_pcsk9_01", 0, "+"]]
        },
        "gapmer_region_filter": {"sources": ["program_exon"]},
        "gapmer_design_jobs": [
            {
                "job_id": "gapmer_job_gapmer_pcsk9_01",
                "program_id": "gapmer_pcsk9_01",
                "modality": "gapmer",
                "exon_label": 4,
                "focus_exons_only": True,
                "tile_overlap_requirement": "full_containment",
                "region_rows": [["ENST00000302118", 813, 947, "constraint_01|gapmer_pcsk9_01", 0, "+"]],
            }
        ],
    }

    manifest = pipeline_tools._build_report_payload_manifest(
        structured_goal=structured_goal,
        final_payload=final_payload,
    )

    assert manifest["gapmer_region_bed"]["rows"][0][1:3] == [813, 947]
    assert manifest["structured_goal"]["region_constraints"][0]["aso_overlap_requirement"] == "full_containment"
    assert manifest["structured_goal"]["design_programs"][0]["exon_label"] == 4
    assert manifest["gapmer_jobs"][0]["tile_overlap_requirement"] == "full_containment"


def test_render_report_markdown_builds_chooser_summary_for_gapmer_outputs(tmp_path) -> None:
    run_dir = tmp_path / "goal_chooser"
    candidates = run_dir / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    scored_path = candidates / "gapmer_scored_uniqueness.json"
    csv_path = candidates / "gapmer_uniqueness_ranked.csv"
    xlsx_path = candidates / "gapmer_scored_uniqueness.xlsx"
    (run_dir / "request.txt").write_text("Design human OTUD4 gapmers.", encoding="utf-8")
    rows = [
        {
            "Name": "candidate_b",
            "Sequence": "CCCCCCCCCCCCCCCCCCCC",
            "selection_rank": 2,
            "isoform_uniqueness_score": 0.5,
            "isoform_uniqueness_class": "gene_wide",
            "Off_target_exact_count": 1,
        },
        {
            "Name": "candidate_a",
            "Sequence": "ACGTACGTACGTACGTACGT",
            "selection_rank": 1,
            "isoform_uniqueness_score": 1.0,
            "isoform_uniqueness_class": "gene_wide",
            "Off_target_exact_count": 0,
        },
    ]
    scored_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    csv_path.write_text("Name,Sequence\ncandidate_a,ACGTACGTACGTACGTACGT\n", encoding="utf-8")
    xlsx_path.write_text("placeholder", encoding="utf-8")
    (candidates / "generated_aso_set.json").write_text(
        json.dumps(
            {
                "row_count": 2,
                "unique_sequence_count": 2,
                "excluded_row_count": 0,
                "eligibility_filters": [
                    {
                        "name": "cross_species_exact_match",
                        "required": False,
                        "requested_species": ["human"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (candidates / "generated_filter_trace.json").write_text(json.dumps({"row_count": 4}), encoding="utf-8")

    markdown = report_explainer.render_report_markdown(
        {"summary": {"headline": "Older explainer headline"}},
        {
            "structured_goal": {
                "gene_symbol": "OTUD4",
                "species": "human",
                "intent": "knockdown",
                "target_context": "neurons",
                "snapgene_source_path": "/mnt/data/external_reference.dna",
            },
            "gapmer_tiling": {
                "scored_tiles_path": str(scored_path),
                "uniqueness_csv": str(csv_path),
                "row_count": 2,
                "initial_filtering": {"total_rows": 8, "passed": {"count": 2}},
            },
        },
    )

    assert markdown.startswith("# Run Output Summary")
    assert "Original request" in markdown
    assert "Design human OTUD4 gapmers." in markdown
    assert "## Where Users Should Look To Choose ASOs" in markdown
    assert "artifacts/candidates/gapmer_scored_uniqueness.xlsx" in markdown
    assert "artifacts/candidates/gapmer_uniqueness_ranked.csv" in markdown
    assert "artifacts/candidates/generated_aso_set.json" in markdown
    assert "gapmer_tiling/gapmer_uniqueness_ranked.csv" not in markdown
    assert "Final generated ASO set: 2 retained row(s), 2 unique sequence(s), 0 excluded row(s)." in markdown
    assert "Cross-species exact matching was not required for this human run." in markdown
    assert "## Practical Interpretation" in markdown
    assert "| 1 | candidate_a | ACGTACGTACGTACGTACGT (unverified_source_sequence); synthesized ASO: unverified | gene_wide | 1.0000 | NA / NA | NA | NA | NA / NA |" in markdown
    assert "| rank | candidate | source sequence / role (5′–3′) | isoform score class | isoform score | matched / catalog isoforms | coverage fraction | off-target genes |" in markdown
    assert str(run_dir) not in markdown
    assert "goal_chooser/gapmer_tiling" not in markdown


def test_render_report_markdown_explains_allele_specific_ranking_and_limitations(
    tmp_path,
) -> None:
    run_dir = tmp_path / "goal_allele_specific"
    candidates = run_dir / "artifacts" / "candidates"
    candidates.mkdir(parents=True)
    scored_path = candidates / "gapmer_scored_uniqueness.json"
    scored_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "Name": "candidate-central",
                        "Sequence": "A" * 20,
                        "mutant_aso_sequence": "T" * 20,
                        "selection_rank": 1,
                        "allele_specific": True,
                        "wildtype_mismatch_count": 1,
                        "variant_position_in_aso": 10,
                        "variant_in_gap": True,
                        "off_target_gene_count": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    markdown = report_explainer.render_report_markdown(
        {},
        {
            "structured_goal": {
                "gene_symbol": "GENE",
                "species": "human",
                "intent": "knockdown",
            },
            "gapmer_tiling": {
                "scored_tiles_path": str(scored_path),
                "row_count": 1,
            },
        },
    )

    assert "## Top Allele-Specific Candidates" in markdown
    assert "mutant-matched ASO (5′→3′)" in markdown
    assert "central DNA-gap placement" in markdown
    assert "does not establish zero wild-type knockdown" in markdown
    assert "| 1 | candidate-central | TTTTTTTTTTTTTTTTTTTT | 10 | yes | 1 | 0 |" in markdown

    scored_payload = json.loads(scored_path.read_text(encoding="utf-8"))
    scored_payload["rows"][0]["variant_position_in_aso"] = 18
    scored_payload["rows"][0]["variant_in_gap"] = False
    scored_path.write_text(json.dumps(scored_payload), encoding="utf-8")
    wing_only_markdown = report_explainer.render_report_markdown(
        {},
        {
            "structured_goal": {
                "gene_symbol": "GENE",
                "species": "human",
                "intent": "knockdown",
            },
            "gapmer_tiling": {
                "scored_tiles_path": str(scored_path),
                "row_count": 1,
            },
        },
    )

    assert "No retained candidate places the discriminating base in the DNA gap" in wing_only_markdown
    assert "Treat them as lower-confidence fallback candidates" in wing_only_markdown
    assert "consider a different oligo architecture, length, or chemistry" in wing_only_markdown


def test_render_report_markdown_still_reads_legacy_gapmer_layout(tmp_path) -> None:
    run_dir = tmp_path / "goal_legacy_chooser"
    gapmer_dir = run_dir / "gapmer_tiling"
    gapmer_dir.mkdir(parents=True)
    scored_path = gapmer_dir / "gapmer_scored_uniqueness.json"
    csv_path = gapmer_dir / "gapmer_uniqueness_ranked.csv"
    xlsx_path = gapmer_dir / "gapmer_scored_uniqueness.xlsx"
    (run_dir / "request.txt").write_text("Design human OTUD4 gapmers.", encoding="utf-8")
    rows = [
        {
            "Name": "candidate_a",
            "Sequence": "ACGTACGTACGTACGTACGT",
            "selection_rank": 1,
            "isoform_uniqueness_score": 1.0,
            "isoform_uniqueness_class": "gene_wide",
            "Off_target_exact_count": 0,
        }
    ]
    scored_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    csv_path.write_text("Name,Sequence\ncandidate_a,ACGTACGTACGTACGTACGT\n", encoding="utf-8")
    xlsx_path.write_text("placeholder", encoding="utf-8")
    (run_dir / "generated_aso_set.json").write_text(
        json.dumps({"row_count": 1, "unique_sequence_count": 1, "excluded_row_count": 0}),
        encoding="utf-8",
    )

    markdown = report_explainer.render_report_markdown(
        {},
        {
            "structured_goal": {"gene_symbol": "OTUD4", "species": "human", "intent": "knockdown"},
            "gapmer_tiling": {
                "scored_tiles_path": str(scored_path),
                "uniqueness_csv": str(csv_path),
                "row_count": 1,
            },
        },
    )

    assert "gapmer_tiling/gapmer_scored_uniqueness.xlsx" in markdown
    assert "gapmer_tiling/gapmer_uniqueness_ranked.csv" in markdown
    assert "generated_aso_set.json" in markdown


def test_gapmer_report_does_not_treat_isoform_hits_as_offtargets(tmp_path) -> None:
    run_dir = tmp_path / "goal_mapt"
    gapmer_dir = run_dir / "gapmer_tiling"
    gapmer_dir.mkdir(parents=True)
    scored_path = gapmer_dir / "gapmer_scored_uniqueness.json"
    csv_path = gapmer_dir / "gapmer_uniqueness_ranked.csv"
    xlsx_path = gapmer_dir / "gapmer_scored_uniqueness.xlsx"
    (run_dir / "request.txt").write_text("Design a gapmer ASO for MAPT knockdown.", encoding="utf-8")
    rows = [
        {
            "Name": "ENST00000262410_cdna238-257",
            "Sequence": "ACCATGCACCAAGACCAAGA",
            "selection_rank": 1,
            "isoform_uniqueness_score": 0.8909,
            "isoform_uniqueness_class": "gene_wide",
            "isoform_uniqueness_count": 49,
            "isoform_hit_count": 49,
            "Times_mapped": 49,
            "repeat_hit_count_exact": 0,
        }
    ]
    scored_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    csv_path.write_text("Name,Sequence\nENST00000262410_cdna238-257,ACCATGCACCAAGACCAAGA\n", encoding="utf-8")
    xlsx_path.write_text("placeholder", encoding="utf-8")

    markdown = report_explainer.render_report_markdown(
        {},
        {
            "structured_goal": {
                "gene_symbol": "MAPT",
                "species": "human",
                "intent": "knockdown",
                "target_context": "HEK293 cells",
            },
            "gapmer_tiling": {
                "scored_tiles_path": str(scored_path),
                "uniqueness_csv": str(csv_path),
                "row_count": 1,
            },
        },
    )

    assert "| 1 | ENST00000262410_cdna238-257 | ACCATGCACCAAGACCAAGA (unverified_source_sequence); synthesized ASO: unverified | gene_wide | 0.8909 | 49 / NA | NA | NA | NA / NA |" in markdown
    assert "| exact off-targets |" not in markdown
    assert "1/1 scored rows have 0 off-target gene hits" not in markdown
    assert "Do not filter `Times_mapped`" in markdown
    assert "legacy alias for that same target-isoform hit count" in markdown
