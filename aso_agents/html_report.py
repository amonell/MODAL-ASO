from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import quote
from aso_agents.reporting_contracts import ISOFORM_SCORE_DEFINITION
from aso_agents.ranking_contracts import recorded_ranking_explanation


_COLLECTION_ORDER = [
    "strategy_routing",
    "splice_strategy_analysis",
    "splice_site_plan",
    "splice_site_comparisons",
    "splice_isoform_objectives",
    "steric_design_jobs",
    "tiling_results",
    "steric_tiling_results_by_job",
    "gapmer_design_jobs",
    "gapmer_tiling_results",
    "gapmer_tiling_results_by_job",
    "rbp_binding_result",
    "mirna_binding_result",
    "motif_tiling_result",
    "patient_reference",
    "snapgene_annotation",
    "snapgene_annotations",
    "parameter_constraints",
    "goal_schema_validation",
    "design_programs",
]

_COLLECTION_TITLES = {
    "strategy_routing": "Strategy Routing",
    "splice_strategy_analysis": "Splice Strategy Analysis",
    "splice_site_plan": "Splice Site Plan",
    "splice_site_comparisons": "Splice Site Comparisons",
    "splice_isoform_objectives": "Isoform Objectives",
    "steric_design_jobs": "Steric Design Jobs",
    "tiling_results": "Tiling Results",
    "steric_tiling_results_by_job": "Steric Results by Job",
    "gapmer_design_jobs": "Gapmer Design Jobs",
    "gapmer_tiling_results": "Gapmer Tiling Results",
    "gapmer_tiling_results_by_job": "Gapmer Results by Job",
    "rbp_binding_result": "RBP Dataset Selection",
    "mirna_binding_result": "miRNA Target-Site Selection",
    "motif_tiling_result": "Motif Targeting",
    "patient_reference": "Patient Reference",
    "snapgene_annotation": "SnapGene Annotation",
    "snapgene_annotations": "SnapGene Isoform Annotations",
    "parameter_constraints": "Parameter Constraints",
    "goal_schema_validation": "Goal Schema Validation",
    "design_programs": "Design Programs",
}

_ARTIFACT_GROUP_ORDER = {
    "run root": 0,
    "candidates": 1,
    "design": 2,
    "intake": 3,
    "tiling": 4,
    "gapmer_tiling": 5,
    "strategy_routes": 6,
    "snapgene": 7,
    "patient_reference": 8,
}

_PRIMARY_ARTIFACT_PATHS = [
    "report.md",
    "report.html",
    "request.txt",
    "artifacts/intake/report_explainer.json",
    "artifacts/intake/goal_intake.json",
    "artifacts/intake/structured_goal.json",
    "artifacts/candidates/generated_aso_set.json",
    "artifacts/candidates/gapmer_scored_uniqueness.json",
    "artifacts/candidates/gapmer_uniqueness_ranked.csv",
    "artifacts/candidates/scored_tiles.json",
    "artifacts/design/strategy_router_summary.json",
    "artifacts/design/tiling/tiles.json",
    "artifacts/design/gapmer_tiling/tiles.json",
    "artifacts/design/steric_regions.bed",
    "artifacts/design/snapgene/aso_export.gb",
    "report_explainer.json",
    "goal_intake.json",
    "structured_goal.json",
    "goal_intake_report.html",
    "strategy_router_summary.json",
    "tiling/tiles.json",
    "tiling/scored_tiles.json",
    "gapmer_tiling/gapmer_scored_uniqueness.json",
    "steric_regions.bed",
    "snapgene/aso_export.gb",
]

_FALLBACK_SKIP_KEYS = {
    "all_transcripts_full",
    "excel_outputs",
    "final_payload",
    "gene_id",
    "gene_id_resolution",
    "gene_symbol",
    "goal_intake",
    "homopolymers",
    "isoform_resolution_locked",
    "qa_history",
    "report_aux_exports",
    "report_explainer_path",
    "report_html_path",
    "report_md_path",
    "report_payload",
    "report_payload_built_at",
    "request",
    "run_splice_eval",
    "run_splice_router",
    "species",
    "status",
    "structured_goal",
    "transcript_policy",
}


def build_goal_intake_html_payload(
    *,
    run_dir: str,
    core_payload: Dict[str, Any],
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    qa_history: List[Dict[str, Any]],
    report_markdown: str,
    report_explainer: Dict[str, Any],
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    meta = core_payload.get("meta") if isinstance(core_payload.get("meta"), dict) else {}
    targets = core_payload.get("targets") if isinstance(core_payload.get("targets"), list) else []
    strategy_routing = (
        core_payload.get("strategy_routing")
        if isinstance(core_payload.get("strategy_routing"), dict)
        else {}
    )
    llm_rationale = (
        core_payload.get("llm_rationale")
        if isinstance(core_payload.get("llm_rationale"), dict)
        else {}
    )
    pipeline_decisions = (
        core_payload.get("pipeline_decisions")
        if isinstance(core_payload.get("pipeline_decisions"), list)
        else []
    )
    gene = core_payload.get("gene") if isinstance(core_payload.get("gene"), dict) else {}
    artifact_inventory = _collect_artifacts(run_path)
    branch_outcomes = _collect_branch_outcomes(report_manifest, final_payload, structured_goal)
    alerts = _collect_alerts(meta, report_manifest, structured_goal, final_payload)
    target_summary = _build_target_summary(targets)
    modalities = _collect_modalities(
        targets=targets,
        report_manifest=report_manifest,
        final_payload=final_payload,
        structured_goal=structured_goal,
    )
    output_collections = _build_output_collections(
        run_path=run_path,
        core_payload=core_payload,
        report_manifest=report_manifest,
        structured_goal=structured_goal,
        final_payload=final_payload,
    )
    modality_explanations = _build_modality_explanations(
        run_path=run_path,
        report_manifest=report_manifest,
        structured_goal=structured_goal,
        final_payload=final_payload,
    )
    branch_outcomes = _apply_explainer_branch_text(branch_outcomes, report_explainer)
    output_collections = _apply_explainer_collection_text(output_collections, report_explainer)
    help_text = _merge_help_texts(
        _build_help_texts(
            meta=meta,
            gene=gene,
            targets=targets,
            branch_outcomes=branch_outcomes,
            alerts=alerts,
            artifact_inventory=artifact_inventory,
            strategy_routing=strategy_routing,
            output_collections=output_collections,
            qa_history=qa_history,
            modalities=modalities,
            report_markdown=report_markdown,
        ),
        report_explainer,
    )

    return {
        "meta": {
            "gene_symbol": meta.get("gene_symbol") or structured_goal.get("gene_symbol"),
            "gene_id": meta.get("gene_id") or structured_goal.get("gene_id"),
            "run_dir": meta.get("run_dir") or run_dir,
            "run_name": run_path.name,
            "generated_at": meta.get("generated_at"),
            "user_request": meta.get("user_request"),
            "ranked_aso_count": meta.get("ranked_aso_count") or 0,
            "target_count": len(targets),
            "artifact_count": artifact_inventory.get("total_files") or 0,
            "modalities": modalities,
        },
        "help": help_text,
        "guide": _build_guide_payload(report_explainer),
        "overview": {
            "key_metrics": _build_key_metrics(
                artifact_inventory=artifact_inventory,
                branch_outcomes=branch_outcomes,
                core_payload=core_payload,
                qa_history=qa_history,
                strategy_routing=strategy_routing,
                target_summary=target_summary,
            ),
            "branch_outcomes": branch_outcomes,
            "alerts": alerts,
            "primary_artifacts": _select_primary_artifacts(artifact_inventory),
            "report_markdown": report_markdown or "",
        },
        "decisions": {
            "pipeline_decisions": pipeline_decisions,
            "timeline": _build_decision_timeline(
                core_payload=core_payload,
                structured_goal=structured_goal,
                final_payload=final_payload,
                report_manifest=report_manifest,
                qa_history=qa_history,
                artifact_inventory=artifact_inventory,
            ),
            "strategy_routing": strategy_routing,
            "llm_rationale": {
                "strategy_analysis": llm_rationale.get("strategy_analysis") or "",
                "agent_trace": llm_rationale.get("agent_trace") or [],
            },
        },
        "outputs": {
            "collections": output_collections,
            "artifacts": artifact_inventory,
        },
        "modality_explanations": modality_explanations,
        "targets": {
            "summary": target_summary,
            "gene": gene,
            "items": targets,
        },
        "data": {
            "views": _build_data_views(
                core_payload=core_payload,
                report_manifest=report_manifest,
                structured_goal=structured_goal,
                final_payload=final_payload,
                qa_history=qa_history,
                report_explainer=report_explainer,
            ),
        },
    }


def render_goal_intake_html(report_payload: Dict[str, Any]) -> str:
    data_json = json.dumps(report_payload, ensure_ascii=True, indent=2).replace("</", "<\\/")
    html = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Goal Intake Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <style>
      :root {
        --bg: #f7fbff;
        --bg-accent: #edf4ff;
        --panel: rgba(255, 255, 255, 0.94);
        --panel-strong: rgba(255, 255, 255, 0.99);
        --line: #d7e4f5;
        --line-strong: #b9cde8;
        --ink: #153a64;
        --ink-soft: #4c6c92;
        --muted: #728bac;
        --teal: #225fb7;
        --teal-soft: #e9f1ff;
        --rust: #3b77c7;
        --rust-soft: #edf4ff;
        --rose: #b91c1c;
        --rose-soft: #fee9e9;
        --sage: #2f5d8a;
        --shadow: 0 18px 42px rgba(34, 95, 183, 0.08);
        --radius: 22px;
        --radius-sm: 14px;
        --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      }

      * {
        box-sizing: border-box;
      }

      html {
        scroll-behavior: smooth;
      }

      body {
        margin: 0;
        color: var(--ink);
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(34, 95, 183, 0.08), transparent 26%),
          linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
        min-height: 100vh;
      }

      a {
        color: var(--teal);
        text-decoration: none;
      }

      a:hover {
        text-decoration: underline;
      }

      .shell {
        width: min(1500px, calc(100vw - 40px));
        margin: 28px auto 48px;
      }

      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.7fr) minmax(320px, 1fr);
        gap: 24px;
        padding: 34px;
        border-radius: calc(var(--radius) + 4px);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(245, 250, 255, 0.97)),
          linear-gradient(90deg, rgba(34, 95, 183, 0.04), rgba(102, 152, 226, 0.03));
        border: 1px solid rgba(185, 205, 232, 0.88);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
      }

      .hero::after {
        content: "";
        position: absolute;
        inset: auto -80px -80px auto;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(34, 95, 183, 0.08), transparent 70%);
        pointer-events: none;
      }

      .eyebrow {
        margin: 0 0 12px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 12px;
        font-weight: 700;
        color: var(--teal);
      }

      .hero h1 {
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-weight: 700;
        font-size: clamp(2rem, 4vw, 3.5rem);
        line-height: 1.08;
        letter-spacing: -0.04em;
      }

      .hero-gene-id {
        margin: 8px 0 0;
        font-family: var(--mono);
        font-size: 0.96rem;
        line-height: 1.4;
        color: var(--muted);
        letter-spacing: 0.02em;
      }

      .hero-request {
        margin: 16px 0 0;
        font-size: 1.05rem;
        line-height: 1.65;
        color: var(--ink-soft);
        max-width: 70ch;
      }

      .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
      }

      .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid transparent;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.03em;
      }

      .chip--teal {
        background: var(--teal-soft);
        color: var(--teal);
        border-color: rgba(34, 95, 183, 0.16);
      }

      .chip--rust {
        background: var(--rust-soft);
        color: var(--rust);
        border-color: rgba(59, 119, 199, 0.16);
      }

      .chip--rose {
        background: var(--rose-soft);
        color: var(--rose);
        border-color: rgba(185, 28, 28, 0.16);
      }

      .chip--neutral {
        background: rgba(76, 108, 146, 0.08);
        color: var(--ink-soft);
        border-color: rgba(76, 108, 146, 0.12);
      }

      .hero-side {
        display: grid;
        gap: 14px;
      }

      .hero-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(185, 205, 232, 0.72);
        border-radius: var(--radius);
        padding: 18px;
      }

      .hero-card-title {
        margin: 0 0 12px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }

      .hero-meta-grid {
        display: grid;
        gap: 10px;
      }

      .meta-row {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        padding-bottom: 10px;
        border-bottom: 1px dashed rgba(185, 205, 232, 0.82);
      }

      .meta-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
      }

      .meta-row span:first-child {
        color: var(--muted);
        font-weight: 600;
      }

      .meta-row span:last-child {
        text-align: right;
        word-break: break-word;
      }

      .artifact-links {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .artifact-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid rgba(34, 95, 183, 0.18);
        background: rgba(34, 95, 183, 0.06);
        color: var(--teal);
        font-size: 13px;
        font-weight: 700;
      }

      .artifact-link:hover {
        background: rgba(34, 95, 183, 0.1);
        text-decoration: none;
      }

      .tabs {
        position: sticky;
        top: 12px;
        z-index: 30;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 22px 0 20px;
        padding: 12px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(185, 205, 232, 0.76);
        box-shadow: 0 10px 24px rgba(34, 95, 183, 0.08);
      }

      .tab-btn {
        border: none;
        background: transparent;
        color: var(--ink-soft);
        font: inherit;
        font-weight: 700;
        padding: 10px 16px;
        border-radius: 999px;
        cursor: pointer;
      }

      .tab-btn.active {
        background: var(--teal);
        color: white;
        box-shadow: 0 8px 18px rgba(34, 95, 183, 0.22);
      }

      .tab-panel {
        display: none;
        animation: fade-up 0.24s ease;
      }

      .tab-panel.active {
        display: block;
      }

      .tab-panel.active.workflow-section {
        display: grid;
        gap: 20px;
      }

      @keyframes fade-up {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .stack {
        display: grid;
        gap: 20px;
      }

      .grid-2,
      .grid-3 {
        display: grid;
        gap: 20px;
      }

      .grid-2 {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .grid-3 {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .panel {
        background: var(--panel);
        border: 1px solid rgba(185, 205, 232, 0.82);
        border-radius: var(--radius);
        padding: 22px;
        box-shadow: var(--shadow);
      }

      .panel--strong {
        background: var(--panel-strong);
      }

      .panel h2 {
        margin: 0 0 16px;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-weight: 700;
        font-size: 1.45rem;
        letter-spacing: -0.02em;
      }

      .panel-title-row,
      .section-title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 16px;
      }

      .panel-title-row h2,
      .section-title-row h2 {
        margin: 0;
      }

      .panel h3 {
        margin: 0 0 14px;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }

      .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 14px;
      }

      .metric-card {
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(248, 251, 255, 0.96);
        border: 1px solid rgba(185, 205, 232, 0.72);
      }

      .metric-label {
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
      }

      .metric-value {
        margin-top: 8px;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-weight: 700;
        font-size: 2rem;
        line-height: 1;
      }

      .metric-detail {
        margin-top: 10px;
        font-size: 13px;
        color: var(--ink-soft);
        line-height: 1.55;
      }

      .empty {
        padding: 18px;
        border-radius: 16px;
        background: rgba(233, 241, 255, 0.7);
        color: var(--ink-soft);
        border: 1px dashed rgba(114, 139, 172, 0.24);
      }

      .branch-grid,
      .collection-grid,
      .data-grid,
      .timeline {
        display: grid;
        gap: 14px;
      }

      .branch-card,
      .collection-card,
      .timeline-card,
      .data-card,
      .trace-card,
      .target-card {
        border: 1px solid rgba(185, 205, 232, 0.72);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.96);
        padding: 16px 18px;
      }

      .branch-card {
        display: grid;
        gap: 10px;
      }

      .branch-head,
      .collection-head,
      .timeline-head,
      .target-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 14px;
      }

      .branch-title,
      .collection-title,
      .timeline-title,
      .target-title {
        font-size: 1rem;
        font-weight: 700;
      }

      .subtle {
        color: var(--ink-soft);
        line-height: 1.6;
      }

      .info-icon {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 1px solid rgba(34, 95, 183, 0.2);
        background: rgba(34, 95, 183, 0.08);
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        cursor: help;
        flex: 0 0 auto;
      }

      .info-icon[data-tooltip]:hover::after,
      .info-icon[data-tooltip]:focus-visible::after {
        content: attr(data-tooltip);
        position: absolute;
        top: calc(100% + 10px);
        left: 50%;
        transform: translateX(-50%);
        width: min(360px, 80vw);
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(21, 58, 100, 0.96);
        color: #fff;
        box-shadow: 0 16px 28px rgba(21, 58, 100, 0.22);
        font-size: 12px;
        line-height: 1.55;
        letter-spacing: 0;
        text-transform: none;
        z-index: 40;
        white-space: normal;
      }

      .info-icon[data-tooltip]:hover::before,
      .info-icon[data-tooltip]:focus-visible::before {
        content: "";
        position: absolute;
        top: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-bottom: 6px solid rgba(21, 58, 100, 0.96);
        z-index: 41;
      }

      .info-icon.is-hidden {
        display: none;
      }

      .is-hidden {
        display: none !important;
      }

      .section-kicker {
        margin: 0 0 10px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 700;
      }

      .fact-list {
        display: grid;
        gap: 8px;
      }

      .fact-row {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding-bottom: 8px;
        border-bottom: 1px dashed rgba(185, 205, 232, 0.82);
      }

      .fact-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
      }

      .fact-row span:first-child {
        color: var(--muted);
        font-weight: 600;
      }

      .fact-row span:last-child {
        text-align: right;
        word-break: break-word;
      }

      .bullet-list {
        margin: 0;
        padding-left: 18px;
        display: grid;
        gap: 8px;
        color: var(--ink-soft);
        line-height: 1.55;
      }

      .summary-text {
        margin: 0;
        color: var(--ink-soft);
        line-height: 1.7;
      }

      .markdown {
        color: var(--ink);
        line-height: 1.75;
      }

      .markdown h1,
      .markdown h2,
      .markdown h3 {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
      }

      .markdown h1 {
        font-size: 1.7rem;
      }

      .markdown h2 {
        font-size: 1.35rem;
        margin-top: 1.5rem;
      }

      .markdown h3 {
        font-size: 1.15rem;
        margin-top: 1.25rem;
      }

      .markdown p,
      .markdown ul,
      .markdown ol,
      .markdown pre {
        margin: 0 0 14px;
      }

      .markdown code,
      .json-block {
        font-family: var(--mono);
      }

      .markdown code {
        background: rgba(34, 95, 183, 0.08);
        padding: 1px 5px;
        border-radius: 6px;
      }

      .markdown pre,
      .json-block {
        padding: 14px 16px;
        border-radius: 16px;
        background: #f5f9ff;
        border: 1px solid rgba(185, 205, 232, 0.8);
        overflow-x: auto;
        font-size: 12px;
        line-height: 1.6;
      }

      .timeline-card {
        display: grid;
        gap: 12px;
      }

      .timeline-index {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--teal-soft);
        color: var(--teal);
        font-weight: 700;
        flex: 0 0 auto;
      }

      .timeline-title-wrap {
        display: flex;
        gap: 12px;
      }

      details {
        border-radius: 16px;
      }

      details > summary {
        cursor: pointer;
        list-style: none;
      }

      details > summary::-webkit-details-marker {
        display: none;
      }

      .record-list {
        display: grid;
        gap: 12px;
      }

      .record {
        border: 1px solid rgba(185, 205, 232, 0.78);
        border-radius: 16px;
        padding: 12px 14px;
        background: rgba(247, 251, 255, 0.96);
      }

      .record summary {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
      }

      .record-title {
        font-weight: 700;
      }

      .record-subtitle {
        margin-top: 4px;
        color: var(--ink-soft);
        line-height: 1.55;
      }

      .record-body {
        margin-top: 12px;
        display: grid;
        gap: 10px;
      }

      .artifact-toolbar,
      .target-toolbar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
      }

      .search-input {
        width: min(420px, 100%);
        padding: 11px 14px;
        border-radius: 12px;
        border: 1px solid rgba(185, 205, 232, 0.92);
        background: rgba(255, 255, 255, 0.98);
        font: inherit;
        color: var(--ink);
      }

      .artifact-group {
        border: 1px solid rgba(185, 205, 232, 0.76);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.97);
        padding: 16px 18px;
      }

      .artifact-group summary {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        font-weight: 700;
      }

      .artifact-list {
        display: grid;
        gap: 10px;
        margin-top: 14px;
      }

      .artifact-item {
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(120px, auto) minmax(90px, auto);
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px dashed rgba(185, 205, 232, 0.82);
      }

      .artifact-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
      }

      .artifact-path {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--ink-soft);
        margin-top: 4px;
        word-break: break-all;
      }

      .artifact-meta {
        text-align: right;
        color: var(--ink-soft);
        font-size: 13px;
      }

      .target-summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
      }

      .map-detail-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
        gap: 18px;
        align-items: start;
      }

      .map-row--stacked {
        grid-template-columns: 1fr;
      }

      #workflow-step-details {
        min-height: 120px;
      }

      #target-details {
        min-height: 140px;
      }

      #safeguard-details {
        min-height: 140px;
      }

      .panel-stack {
        display: grid;
        gap: 16px;
      }

      .map-row {
        display: grid;
        grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
        gap: 16px;
        align-items: start;
      }

      .map-list {
        min-height: 180px;
      }

      .list {
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-height: 420px;
        overflow-y: auto;
        padding-right: 6px;
      }

      .list-item {
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(185, 205, 232, 0.76);
        background: rgba(255, 255, 255, 0.98);
        cursor: pointer;
        transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
      }

      .list-item:hover,
      .list-item.active {
        transform: translateX(4px);
        border-color: rgba(34, 95, 183, 0.35);
        box-shadow: 0 10px 18px rgba(34, 95, 183, 0.1);
      }

      .list-title {
        font-weight: 700;
      }

      .list-sub {
        margin-top: 6px;
        color: var(--ink-soft);
        font-size: 13px;
      }

      .gene-structure {
        width: 100%;
        overflow-x: auto;
        padding: 12px;
        background: #f7fbff;
        border: 1px solid rgba(185, 205, 232, 0.82);
        border-radius: 18px;
      }

      .gene-structure svg {
        display: block;
        cursor: grab;
        touch-action: none;
      }

      .gene-structure svg:active {
        cursor: grabbing;
      }

      .workflow-lane-label {
        font-size: 16px;
        fill: #58779d;
        font-family: "IBM Plex Sans", sans-serif;
        font-weight: 700;
      }

      .workflow-lane-line {
        stroke: rgba(185, 205, 232, 0.92);
        stroke-width: 1.5;
        stroke-dasharray: 8 8;
      }

      .workflow-edge {
        fill: none;
        stroke: rgba(76, 108, 146, 0.72);
        stroke-width: 3;
      }

      .workflow-node {
        cursor: pointer;
      }

      .workflow-node rect {
        rx: 20;
        ry: 20;
        stroke-width: 2;
        filter: drop-shadow(0 10px 18px rgba(34, 95, 183, 0.08));
      }

      .workflow-node--shared rect {
        fill: #ffffff;
        stroke: #bed2eb;
      }

      .workflow-node--steric rect {
        fill: #edf4ff;
        stroke: rgba(34, 95, 183, 0.38);
      }

      .workflow-node--gapmer rect {
        fill: #e4efff;
        stroke: rgba(59, 119, 199, 0.38);
      }

      .workflow-node--report rect {
        fill: #dce9ff;
        stroke: rgba(34, 95, 183, 0.34);
      }

      .workflow-node--branch rect {
        fill: #f5f9ff;
        stroke: rgba(76, 108, 146, 0.26);
      }

      .workflow-node.active rect {
        stroke-width: 3;
        stroke: #225fb7;
      }

      .workflow-node-title {
        font-size: 14px;
        fill: #153a64;
        font-family: "IBM Plex Sans", sans-serif;
        font-weight: 700;
      }

      .workflow-node-subtitle {
        font-size: 11px;
        fill: #4c6c92;
        font-family: "IBM Plex Sans", sans-serif;
      }

      .gene-help {
        margin-top: 10px;
        color: var(--ink-soft);
        font-size: 12px;
      }

      .legend {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 12px;
        color: var(--ink-soft);
        font-size: 12px;
      }

      .legend span {
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }

      .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
      }

      .legend-dot.target {
        background: var(--teal);
      }

      .legend-dot.increase {
        background: #15803d;
      }

      .legend-dot.decrease {
        background: #b91c1c;
      }

      .legend-dot.neutral {
        background: #5e7fa7;
      }

      .gene-label {
        font-size: 22px;
        fill: #58779d;
        font-family: "IBM Plex Sans", sans-serif;
        pointer-events: none;
      }

      .gene-position-label {
        font-size: 18px;
        fill: #7e95b5;
      }

      .gene-line {
        stroke: #bfd4ec;
        stroke-width: 2;
      }

      .exon {
        fill: #bfd1e8;
      }

      .exon.coding {
        fill: var(--teal);
      }

      .exon.noncoding {
        fill: #d2deef;
      }

      .marker {
        stroke: #ffffff;
        stroke-width: 2;
        fill: #6c88ab;
        transition: transform 0.16s ease;
      }

      .marker:hover {
        transform: scale(1.2);
      }

      .marker.target,
      .marker.safeguard-target {
        fill: var(--teal);
      }

      .marker.active {
        stroke: #fff;
        stroke-width: 3;
      }

      .marker.highlight-increase {
        fill: #15803d;
      }

      .marker.highlight-decrease {
        fill: #b91c1c;
      }

      .marker.highlight-neutral {
        fill: #5e7fa7;
      }

      .marker-label {
        font-size: 32px;
        fill: #5a7597;
        font-family: "IBM Plex Sans", sans-serif;
        font-weight: 600;
        pointer-events: none;
      }

      .marker-label.target,
      .marker-label.safeguard-target {
        fill: var(--teal);
      }

      .marker-label.highlight-increase {
        fill: #15803d;
      }

      .marker-label.highlight-decrease {
        fill: #b91c1c;
      }

      .marker-label.highlight-neutral {
        fill: #5e7fa7;
      }

      .marker-connector {
        stroke: rgba(108, 136, 171, 0.55);
        stroke-width: 2;
      }

      .marker-connector.target,
      .marker-connector.safeguard-target {
        stroke: rgba(34, 95, 183, 0.55);
      }

      .marker-connector.highlight-increase {
        stroke: #15803d;
      }

      .marker-connector.highlight-decrease {
        stroke: #b91c1c;
      }

      .marker-connector.highlight-neutral {
        stroke: #5e7fa7;
      }

      .detail-row {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        padding: 10px 0;
        border-bottom: 1px dashed rgba(185, 205, 232, 0.82);
        font-size: 13px;
      }

      .detail-row:last-child {
        border-bottom: none;
      }

      .detail-row span:first-child {
        color: var(--muted);
        font-weight: 600;
      }

      .detail-row span:last-child {
        text-align: right;
        word-break: break-word;
      }

      .detail-note {
        display: block;
        margin-top: 4px;
        color: var(--ink-soft);
        font-size: 12px;
      }

      .detail-panel {
        margin-top: 14px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(185, 205, 232, 0.76);
        background: rgba(247, 251, 255, 0.98);
      }

      .detail-panel-title {
        font-weight: 700;
        margin-bottom: 10px;
      }

      .detail-subtitle {
        color: var(--ink-soft);
        font-size: 12px;
        margin-bottom: 8px;
      }

      .table-scroll {
        overflow-x: auto;
      }

      .aso-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }

      .aso-table th,
      .aso-table td {
        padding: 10px 8px;
        border-bottom: 1px solid rgba(185, 205, 232, 0.72);
        text-align: left;
      }

      .aso-table th {
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .aso-table td:first-child {
        font-family: var(--mono);
        font-size: 12px;
      }

      .copy-btn {
        border: 1px solid rgba(34, 95, 183, 0.18);
        background: rgba(34, 95, 183, 0.06);
        color: var(--teal);
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 700;
        cursor: pointer;
      }

      .copy-btn.copied {
        background: rgba(21, 128, 61, 0.1);
        border-color: rgba(21, 128, 61, 0.2);
        color: #15803d;
      }

      .target-summary-card {
        padding: 16px;
        border-radius: 18px;
        border: 1px solid rgba(185, 205, 232, 0.72);
        background: rgba(255, 255, 255, 0.98);
      }

      .target-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 13px;
      }

      .target-table th,
      .target-table td {
        padding: 10px 8px;
        border-bottom: 1px solid rgba(185, 205, 232, 0.72);
        text-align: left;
        vertical-align: top;
      }

      .target-table th {
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .target-table td.sequence {
        font-family: var(--mono);
        font-size: 12px;
        max-width: 24ch;
        overflow-wrap: anywhere;
      }

      .muted {
        color: var(--muted);
      }

      .spacer {
        height: 4px;
      }

      @media (max-width: 1100px) {
        .hero,
        .grid-2,
        .grid-3 {
          grid-template-columns: 1fr;
        }

        .map-detail-grid,
        .map-row {
          grid-template-columns: 1fr;
        }

        .artifact-item {
          grid-template-columns: 1fr;
        }

        .artifact-meta {
          text-align: left;
        }
      }

      @media (max-width: 720px) {
        .shell {
          width: min(100vw - 20px, 1500px);
          margin-top: 10px;
        }

        .hero {
          padding: 20px;
        }

        .tabs {
          top: 8px;
        }

        .panel,
        .hero-card {
          padding: 18px;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="hero">
        <div>
          <p class="eyebrow">Goal Intake Report</p>
          <h1 id="hero-title"></h1>
          <p class="hero-gene-id" id="hero-gene-id"></p>
          <p class="hero-request" id="hero-request"></p>
          <div class="chip-row" id="hero-chips"></div>
        </div>
        <div class="hero-side">
          <section class="hero-card">
            <p class="hero-card-title">Run metadata</p>
            <div class="hero-meta-grid" id="hero-meta-grid"></div>
          </section>
        </div>
      </header>

      <section class="panel panel--strong" style="margin-top:20px">
        <div class="panel-title-row"><h2>Quick Overview</h2><span class="info-icon" data-help-key="overview_quick" tabindex="0">i</span></div>
        <div class="grid-2">
          <div id="quick-overview-copy"></div>
          <div>
            <div class="metric-grid" id="quick-overview-metrics"></div>
            <div class="spacer"></div>
            <div class="artifact-links" id="quick-overview-artifacts"></div>
          </div>
        </div>
        <div class="spacer"></div>
        <div class="chip-row" id="quick-overview-branches"></div>
        <div class="spacer"></div>
        <div id="quick-overview-alerts"></div>
      </section>

      <nav class="tabs">
        <button class="tab-btn active" data-tab="workflow">Decisions / Workflow</button>
        <button class="tab-btn is-hidden" data-tab="gapmer">Gapmer</button>
        <button class="tab-btn is-hidden" data-tab="steric-binding">Steric Hindrance</button>
        <button class="tab-btn" data-tab="targets">Maps / Sites</button>
        <button class="tab-btn" data-tab="safeguards">Safeguards</button>
      </nav>

      <main class="stack">
        <section class="tab-panel active workflow-section" id="workflow">
          <div class="panel panel--strong">
            <div class="panel-title-row"><h2>Workflow Graph</h2><span class="info-icon" data-help-key="workflow_graph" tabindex="0">i</span></div>
            <div class="gene-help">Drag to pan, use the mouse wheel to zoom, and hover a step for a plain-English explanation.</div>
            <div class="gene-structure" id="workflow-graph"></div>
          </div>
          <div class="panel panel--strong">
            <div class="panel-title-row"><h2>Step Details</h2><span class="info-icon" data-help-key="workflow_details" tabindex="0">i</span></div>
            <div id="workflow-step-details" class="empty">Hover a workflow step to see what it did and why it mattered.</div>
          </div>
          <div class="grid-2">
            <div class="panel">
              <div class="panel-title-row"><h2>Key Decisions</h2><span class="info-icon" data-help-key="guide_decisions" tabindex="0">i</span></div>
              <div class="record-list" id="workflow-decision-cards"></div>
            </div>
            <div class="panel">
              <div class="panel-title-row"><h2>Branch Outcomes</h2><span class="info-icon" data-help-key="overview_branches" tabindex="0">i</span></div>
              <div class="record-list" id="workflow-branch-cards"></div>
            </div>
          </div>
          <div class="panel">
            <details class="record">
              <summary>
                <div class="panel-title-row" style="margin-bottom:0"><h2>Detailed Step Notes</h2><span class="info-icon" data-help-key="guide_walkthrough" tabindex="0">i</span></div>
              </summary>
              <div class="record-body">
                <div class="record-list" id="workflow-notes"></div>
              </div>
            </details>
          </div>
        </section>

        <section class="tab-panel is-hidden" id="gapmer">
          <div id="gapmer-content"></div>
        </section>

        <section class="tab-panel is-hidden" id="steric-binding">
          <div id="steric-binding-content"></div>
        </section>

        <section class="tab-panel" id="targets">
          <div class="panel">
            <div class="panel-title-row"><h2>Target Summary</h2><span class="info-icon" data-help-key="targets_summary" tabindex="0">i</span></div>
            <div class="target-summary-grid" id="target-summary-grid"></div>
          </div>
          <div class="panel">
            <div class="panel-title-row"><h2>Transcript Map</h2><span class="info-icon" data-help-key="maps_targets" tabindex="0">i</span></div>
            <div class="panel-stack" id="target-map-groups"></div>
          </div>
          <div class="panel panel--strong">
            <div class="panel-title-row"><h2>Target Details</h2><span class="info-icon" data-help-key="maps_target_details" tabindex="0">i</span></div>
            <div id="target-details" class="empty">Click a target site to see details.</div>
          </div>
          <div class="panel">
            <details class="record">
              <summary>
                <div class="panel-title-row" style="margin-bottom:0"><h2>Expanded Target Records</h2><span class="info-icon" data-help-key="maps_target_records" tabindex="0">i</span></div>
              </summary>
              <div class="record-body">
                <div class="target-toolbar">
                  <input class="search-input" id="target-search" type="search" placeholder="Filter by label, direction, site kind, isoform, or output type" />
                </div>
                <div class="spacer"></div>
                <div class="collection-grid" id="target-grid"></div>
              </div>
            </details>
          </div>
        </section>

        <section class="tab-panel" id="safeguards">
          <div class="panel">
            <div class="panel-title-row"><h2>Safeguard Map</h2><span class="info-icon" data-help-key="maps_safeguards" tabindex="0">i</span></div>
            <div class="panel-stack" id="safeguard-map-groups"></div>
          </div>
          <div class="panel panel--strong">
            <div class="panel-title-row"><h2>Safeguard Details</h2><span class="info-icon" data-help-key="maps_safeguard_details" tabindex="0">i</span></div>
            <div id="safeguard-details" class="empty">Click a target site to see safeguards.</div>
          </div>
        </section>
      </main>
    </div>

    <script id="report-data" type="application/json">__REPORT_DATA__</script>
    <script>
      const reportData = JSON.parse(document.getElementById("report-data").textContent);
      const helpText = reportData.help || {};

      const tabs = Array.from(document.querySelectorAll(".tab-btn"));
      const panels = Array.from(document.querySelectorAll(".tab-panel"));
      tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          tabs.forEach((node) => node.classList.toggle("active", node === tab));
          panels.forEach((panel) => panel.classList.toggle("active", panel.id === tab.dataset.tab));
        });
      });

      function escapeHtml(value) {
        if (value === null || value === undefined) {
          return "";
        }
        return String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#039;");
      }

      function humanizeKey(value) {
        const labels = {
          times_mapped: "Mappings (source reference)",
          isoform_hit_count: "Target isoforms",
          isoform_uniqueness_count: "Target isoforms",
          off_target_exact_count: "Exact off-target hits (see count unit)",
          off_target_gene_count: "Off-target genes",
          times_mapped_off_target_genes: "Off-target genes",
          repeat_hit_count_exact: "Alternate exact repeat loci",
          mean_splice_delta: "Selected-site mean delta",
          recommendation_note: "Prediction evidence",
          exon_objective_evidence: "Exon-pair deltas (deletion / replacement)",
        };
        if (Object.prototype.hasOwnProperty.call(labels, value)) {
          return labels[value];
        }
        return String(value || "")
          .replace(/_/g, " ")
          .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
          .replace(/\s+/g, " ")
          .trim()
          .replace(/^\w/, (char) => char.toUpperCase());
      }

      function formatValue(value) {
        if (value === null || value === undefined || value === "") {
          return "N/A";
        }
        if (Array.isArray(value)) {
          return value.join(", ");
        }
        if (typeof value === "number") {
          return value.toLocaleString();
        }
        return String(value);
      }

      function formatExonObjectiveEvidence(items) {
        const number = (value) => typeof value === "number" && Number.isFinite(value)
          ? String(Number(value.toPrecision(6))) : "N/A";
        return (Array.isArray(items) ? items : []).map((item) => {
          const predictions = (item.measurements || []).map((measurement) =>
            `${measurement.window_bp ?? "unknown"} bp window: ${(measurement.values || []).map(number).join(" / ")}`
          ).join("; ");
          return `${item.reference_transcript_id || "Unresolved reference"}, exon ordinal ${item.exon_index ?? "unknown"}: ${item.desired_effect}; ${predictions || item.status}`;
        }).join("; ") || "No explicit exon objective recorded.";
      }

      function helpIconHtml(text) {
        if (!text) {
          return "";
        }
        return `<span class="info-icon" tabindex="0" data-tooltip="${escapeHtml(text)}" aria-label="${escapeHtml(text)}">i</span>`;
      }

      function hydrateStaticHelpIcons() {
        document.querySelectorAll(".info-icon[data-help-key]").forEach((node) => {
          const text = helpText[node.dataset.helpKey] || "";
          if (!text) {
            node.classList.add("is-hidden");
            return;
          }
          node.setAttribute("data-tooltip", text);
          node.setAttribute("aria-label", text);
          node.removeAttribute("title");
        });
      }

      function badge(text, tone) {
        if (!text) {
          return "";
        }
        const normalized = tone || "neutral";
        return `<span class="chip chip--${escapeHtml(normalized)}">${escapeHtml(text)}</span>`;
      }

      function simpleMarkdownToHtml(markdown) {
        if (!markdown) {
          return '<div class="empty">No markdown summary available.</div>';
        }
        let html = escapeHtml(markdown);
        html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
        html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
        html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
        html = html.replace(/```([\\s\\S]*?)```/g, "<pre><code>$1</code></pre>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
        html = html.replace(/^(\\d+)\\. (.+)$/gm, "<li>$2</li>");
        html = html.replace(/(<li>.*<\\/li>\\n?)+/g, (match) => `<ul>${match}</ul>`);
        const lines = html.split("\\n").map((line) => {
          const trimmed = line.trim();
          if (!trimmed) {
            return "";
          }
          if (trimmed.startsWith("<h") || trimmed.startsWith("<ul>") || trimmed.startsWith("<pre>")) {
            return line;
          }
          if (trimmed.startsWith("</ul>") || trimmed.startsWith("</pre>")) {
            return line;
          }
          return `<p>${line}</p>`;
        });
        return lines.join("\\n");
      }

      function renderFacts(facts) {
        if (!Array.isArray(facts) || facts.length === 0) {
          return '<div class="empty">No details captured.</div>';
        }
        return `<div class="fact-list">${facts.map((fact) => {
          const value = fact.href
            ? `<a href="${escapeHtml(fact.href)}">${escapeHtml(formatValue(fact.value))}</a>`
            : escapeHtml(formatValue(fact.value));
          return `<div class="fact-row"><span>${escapeHtml(fact.label)}</span><span>${value}</span></div>`;
        }).join("")}</div>`;
      }

      function renderBullets(items) {
        if (!Array.isArray(items) || items.length === 0) {
          return "";
        }
        return `<ul class="bullet-list">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
      }

      function renderEvidenceRefs(items) {
        if (!Array.isArray(items) || !items.length) {
          return "";
        }
        return `<p class="subtle" style="margin-top:10px">Evidence: ${escapeHtml(items.join(", "))}</p>`;
      }

      function stripInlineEvidenceRefs(value) {
        const text = String(value || "");
        if (!text) {
          return "";
        }
        return text
          .replace(/\s*\(evidence_refs:\s*\[[\s\S]*?\]\)\s*/gi, "")
          .replace(/\s+/g, " ")
          .trim();
      }

      function renderGuide() {
        const guide = reportData.guide || {};
        const summary = guide.summary || {};
        const summaryRoot = document.getElementById("guide-summary");
        const summaryBits = [
          summary.headline,
          summary.plain_language_goal,
          summary.what_happened,
          summary.result,
        ].filter((item, index, arr) => item && arr.indexOf(item) === index);
        summaryRoot.innerHTML = summaryBits.length
          ? `
              ${summaryBits.map((item) => `<p class="summary-text">${escapeHtml(item)}</p>`).join("")}
              ${renderBullets(summary.bullets || [])}
              ${(guide.start_here || []).length ? `<div class="spacer"></div><p class="subtle">Start here</p>${renderBullets(guide.start_here || [])}` : ""}
            `
          : '<div class="empty">No beginner-facing summary available.</div>';

        const walkthrough = Array.isArray(guide.walkthrough) ? guide.walkthrough : [];
        document.getElementById("guide-walkthrough").innerHTML = walkthrough.length
          ? walkthrough.map((item, index) => `
              <article class="timeline-card">
                <div class="timeline-title-wrap">
                  <div class="timeline-index">${index + 1}</div>
                  <div>
                    <div class="timeline-title">${escapeHtml(item.title || "Step")}</div>
                    ${item.summary ? `<div class="record-subtitle">${escapeHtml(item.summary)}</div>` : ""}
                  </div>
                </div>
                ${item.why_it_matters ? `<p class="summary-text">${escapeHtml(item.why_it_matters)}</p>` : ""}
                ${renderEvidenceRefs(item.evidence_refs || [])}
              </article>
            `).join("")
          : '<div class="empty">No walkthrough steps recorded.</div>';

        const decisions = Array.isArray(guide.key_decisions) ? guide.key_decisions : [];
        document.getElementById("guide-decisions").innerHTML = decisions.length
          ? decisions.map((item) => `
              <article class="record" open>
                <div class="record-title">${escapeHtml(item.title || "Decision")}</div>
                ${item.decision ? `<div class="record-subtitle">${escapeHtml(item.decision)}</div>` : ""}
                <div class="record-body">
                  ${item.why ? `<p class="summary-text">${escapeHtml(item.why)}</p>` : ""}
                  ${renderEvidenceRefs(item.evidence_refs || [])}
                </div>
              </article>
            `).join("")
          : '<div class="empty">No key decisions recorded.</div>';

        const branches = Array.isArray(guide.branch_explanations) ? guide.branch_explanations : [];
        document.getElementById("guide-branches").innerHTML = branches.length
          ? branches.map((item) => `
              <article class="record" open>
                <div class="record-title">${escapeHtml(item.title || "Branch")}</div>
                <div class="record-subtitle">${escapeHtml(item.status || "unknown")}</div>
                <div class="record-body">
                  ${item.summary ? `<p class="summary-text">${escapeHtml(item.summary)}</p>` : ""}
                  ${item.why ? `<p class="subtle">${escapeHtml(item.why)}</p>` : ""}
                  ${renderEvidenceRefs(item.evidence_refs || [])}
                </div>
              </article>
            `).join("")
          : '<div class="empty">No branch explanations recorded.</div>';

        const caveats = Array.isArray(guide.caveats) ? guide.caveats : [];
        document.getElementById("guide-caveats").innerHTML = caveats.length
          ? renderBullets(caveats)
          : '<div class="empty">No caveats recorded.</div>';

        const outputGuides = Array.isArray(guide.output_guides) ? guide.output_guides : [];
        document.getElementById("guide-outputs").innerHTML = outputGuides.length
          ? outputGuides.map((item) => `
              <article class="collection-card">
                <div class="collection-head">
                  <div>
                    <div class="collection-title">${escapeHtml(item.title || "Output")}</div>
                    ${item.what_it_is ? `<p class="summary-text">${escapeHtml(item.what_it_is)}</p>` : ""}
                  </div>
                </div>
                ${item.how_to_read ? `<p class="summary-text">${escapeHtml(item.how_to_read)}</p>` : ""}
                ${item.why_it_matters ? `<p class="subtle">${escapeHtml(item.why_it_matters)}</p>` : ""}
                ${renderEvidenceRefs(item.evidence_refs || [])}
              </article>
            `).join("")
          : '<div class="empty">No output guides recorded.</div>';

        const glossary = Array.isArray(guide.glossary) ? guide.glossary : [];
        document.getElementById("guide-glossary").innerHTML = glossary.length
          ? glossary.map((item) => `
              <article class="collection-card">
                <div class="collection-title">${escapeHtml(item.term || "Term")}</div>
                <p class="summary-text">${escapeHtml(item.definition || "")}</p>
              </article>
            `).join("")
          : '<div class="empty">No glossary terms recorded.</div>';
      }

      function renderQuickOverview() {
        const guide = reportData.guide || {};
        const overview = reportData.overview || {};
        const summary = guide.summary || {};
        const copyRoot = document.getElementById("quick-overview-copy");
        if (copyRoot) {
          const paragraphs = [
            summary.headline,
            summary.plain_language_goal,
            summary.result,
          ]
            .map((item) => stripInlineEvidenceRefs(item))
            .filter((item, index, arr) => item && arr.indexOf(item) === index);
          const bullets = (summary.bullets || [])
            .map((item) => stripInlineEvidenceRefs(item))
            .filter((item) => item);
          copyRoot.innerHTML = paragraphs.length
            ? `
                ${paragraphs.map((item) => `<p class="summary-text">${escapeHtml(item)}</p>`).join("")}
                ${renderBullets(bullets.slice(0, 4))}
              `
            : '<div class="empty">No overview summary available.</div>';
        }

        const preferredMetricLabels = new Set(["Branches", "Targets", "Routed goals"]);
        const metrics = (overview.key_metrics || []).filter((item) => preferredMetricLabels.has(item.label)).slice(0, 4);
        const metricsRoot = document.getElementById("quick-overview-metrics");
        if (metricsRoot) {
          metricsRoot.innerHTML = metrics.length
            ? metrics.map((item) => `
                <article class="metric-card">
                  <div class="metric-label">${escapeHtml(item.label)}</div>
                  <div class="metric-value">${escapeHtml(formatValue(item.value))}</div>
                  <div class="metric-detail">${escapeHtml(item.detail || "")}</div>
                </article>
              `).join("")
            : '<div class="empty">No overview metrics available.</div>';
        }

        const artifactsRoot = document.getElementById("quick-overview-artifacts");
        const artifacts = (overview.primary_artifacts || []).slice(0, 6);
        if (artifactsRoot) {
          artifactsRoot.innerHTML = artifacts.length
            ? artifacts.map((item) => `<a class="artifact-link" href="${escapeHtml(item.href)}">${escapeHtml(item.label)}</a>`).join("")
            : '<div class="empty">No primary artifacts linked.</div>';
        }

        const branchesRoot = document.getElementById("quick-overview-branches");
        const branches = overview.branch_outcomes || [];
        if (branchesRoot) {
          branchesRoot.innerHTML = branches.length
            ? branches.map((item) => badge(`${item.title}: ${item.status || "unknown"}`, item.tone || "neutral")).join("")
            : "";
        }

        const alertsRoot = document.getElementById("quick-overview-alerts");
        const alerts = (overview.alerts || []).map((item) => stripInlineEvidenceRefs(item)).filter((item) => item);
        if (alertsRoot) {
          alertsRoot.innerHTML = alerts.length
            ? `<div class="panel" style="padding:16px;margin-top:6px"><div class="section-kicker">Caveats</div>${renderBullets(alerts.slice(0, 5))}</div>`
            : "";
        }
      }

      function renderArtifactLinks(items) {
        const artifacts = Array.isArray(items)
          ? items.filter((item) => item && item.href)
          : [];
        if (!artifacts.length) {
          return '<div class="empty">No linked artifacts captured for this branch.</div>';
        }
        return `<div class="artifact-links">${artifacts.map((item) => `<a class="artifact-link" href="${escapeHtml(item.href)}">${escapeHtml(item.label || "Artifact")}</a>`).join("")}</div>`;
      }

      function renderModalityExplanation(containerId, payload) {
        const root = document.getElementById(containerId);
        if (!root) return;
        if (!payload) {
          root.innerHTML = '<div class="empty">No modality explanation captured.</div>';
          return;
        }
        const facts = Array.isArray(payload.facts) ? payload.facts : [];
        const sections = Array.isArray(payload.sections) ? payload.sections : [];
        const artifacts = Array.isArray(payload.artifacts) ? payload.artifacts.filter((item) => item && item.href) : [];
        root.innerHTML = `
          <div class="panel panel--strong">
            <div class="panel-title-row"><h2>${escapeHtml(payload.title || "Branch explanation")}</h2></div>
            ${payload.summary ? `<p class="summary-text">${escapeHtml(payload.summary)}</p>` : ""}
            ${payload.note ? `<p class="subtle" style="margin-top:10px">${escapeHtml(payload.note)}</p>` : ""}
            ${facts.length ? `<div style="margin-top:16px">${renderFacts(facts)}</div>` : ""}
            ${artifacts.length ? `<div class="spacer"></div><p class="section-kicker" style="margin-top:18px">Key artifacts</p>${renderArtifactLinks(artifacts)}` : ""}
          </div>
          <div class="grid-2">
            ${sections.length ? sections.map((section) => `
              <article class="collection-card">
                <div class="collection-title">${escapeHtml(section.title || "Section")}</div>
                ${section.summary ? `<p class="summary-text" style="margin-top:10px">${escapeHtml(section.summary)}</p>` : ""}
                ${section.note ? `<p class="subtle" style="margin-top:10px">${escapeHtml(section.note)}</p>` : ""}
                ${renderBullets(section.bullets || [])}
              </article>
            `).join("") : '<div class="empty">No explanation sections captured.</div>'}
          </div>
        `;
      }

      function setupModalityTab(tabId, payload) {
        const button = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
        const panel = document.getElementById(tabId);
        if (!button || !panel) return;
        const visible = Boolean(payload);
        button.classList.toggle("is-hidden", !visible);
        panel.classList.toggle("is-hidden", !visible);
        if (visible) {
          renderModalityExplanation(`${tabId}-content`, payload);
        }
      }

      function findGuideWalkthrough(patterns) {
        const items = (((reportData.guide || {}).walkthrough) || []);
        const normalized = patterns.map((item) => String(item || "").toLowerCase());
        return items.find((item) => {
          const haystack = `${item.title || ""} ${item.summary || ""} ${item.why_it_matters || ""}`.toLowerCase();
          return normalized.some((pattern) => haystack.includes(pattern));
        }) || null;
      }

      function findGuideDecision(patterns) {
        const items = (((reportData.guide || {}).key_decisions) || []);
        const normalized = patterns.map((item) => String(item || "").toLowerCase());
        return items.find((item) => {
          const haystack = `${item.title || ""} ${item.decision || ""} ${item.why || ""}`.toLowerCase();
          return normalized.some((pattern) => haystack.includes(pattern));
        }) || null;
      }

      function findOutputGuide(patterns) {
        const items = (((reportData.guide || {}).output_guides) || []);
        const normalized = patterns.map((item) => String(item || "").toLowerCase());
        return items.find((item) => {
          const haystack = `${item.key || ""} ${item.title || ""} ${item.what_it_is || ""} ${item.how_to_read || ""} ${item.why_it_matters || ""}`.toLowerCase();
          return normalized.some((pattern) => haystack.includes(pattern));
        }) || null;
      }

      function findBranchGuide(branchKey) {
        const items = (((reportData.guide || {}).branch_explanations) || []);
        const key = String(branchKey || "").toLowerCase();
        return items.find((item) => String(item.branch || "").toLowerCase() === key) || null;
      }

      function pickDetailText(...parts) {
        return parts.find((part) => part && String(part).trim()) || "";
      }

      function buildWorkflowModel() {
        const guide = reportData.guide || {};
        const summary = guide.summary || {};
        const rawGraph = guide.workflow_graph || {};
        const lanes = Array.isArray(rawGraph.lanes) ? rawGraph.lanes : [];
        const rawNodes = Array.isArray(rawGraph.nodes) ? rawGraph.nodes : [];
        const laneMap = {};
        const graph = {
          lanes: [],
          nodes: [],
          edges: [],
        };

        lanes.forEach((lane) => {
          const laneId = String((lane || {}).id || "").trim() || "shared";
          if (laneMap[laneId]) return;
          const normalized = {
            id: laneId,
            label: String((lane || {}).label || "").trim() || humanizeKey(laneId),
          };
          laneMap[laneId] = normalized;
          graph.lanes.push(normalized);
        });

        if (!laneMap.shared) {
          laneMap.shared = { id: "shared", label: "Shared planning" };
          graph.lanes.unshift(laneMap.shared);
        }

        const nodeIds = new Set();
        rawNodes.forEach((node, index) => {
          const nodeId = String((node || {}).id || "").trim();
          if (!nodeId || nodeIds.has(nodeId)) return;
          const laneId = String((node || {}).lane || "").trim() || "shared";
          if (!laneMap[laneId]) {
            laneMap[laneId] = { id: laneId, label: String((node || {}).lane_label || "").trim() || humanizeKey(laneId) };
            graph.lanes.push(laneMap[laneId]);
          }
          graph.nodes.push({
            id: nodeId,
            title: String((node || {}).title || "").trim() || `Step ${index + 1}`,
            lane: laneId,
            lane_label: String((node || {}).lane_label || "").trim() || laneMap[laneId].label,
            column: Number.isFinite(Number((node || {}).column)) ? Number((node || {}).column) : index,
            tone: String((node || {}).tone || "").trim() || (laneId === "shared" ? "shared" : "branch"),
            subtitle: String((node || {}).subtitle || "").trim(),
            description: String((node || {}).description || "").trim(),
            why: String((node || {}).why || "").trim(),
            facts: [],
            evidence_refs: Array.isArray((node || {}).evidence_refs) ? node.evidence_refs : [],
          });
          nodeIds.add(nodeId);
        });

        const edgeKeys = new Set();
        const rawEdges = Array.isArray(rawGraph.edges) ? rawGraph.edges : [];
        rawEdges.forEach((edge) => {
          const source = String((edge || {}).source || "").trim();
          const target = String((edge || {}).target || "").trim();
          if (!source || !target || !nodeIds.has(source) || !nodeIds.has(target) || source === target) {
            return;
          }
          const edgeKey = `${source}->${target}`;
          if (edgeKeys.has(edgeKey)) return;
          graph.edges.push({ source, target });
          edgeKeys.add(edgeKey);
        });

        if (!graph.nodes.length) {
          graph.nodes.push({
            id: "workflow_unavailable",
            title: "Workflow unavailable",
            lane: "shared",
            lane_label: laneMap.shared.label,
            column: 0,
            tone: "shared",
            subtitle: "workflow graph missing",
            description: pickDetailText(summary.what_happened, summary.result, "No workflow graph was available for this run."),
            why: "The report needs a valid explainer workflow graph to render this panel.",
            facts: [],
            evidence_refs: [],
          });
        }

        if (!graph.edges.length && graph.nodes.length > 1) {
          const sortedNodes = [...graph.nodes].sort((left, right) => {
            const leftColumn = Number(left.column || 0);
            const rightColumn = Number(right.column || 0);
            if (leftColumn !== rightColumn) {
              return leftColumn - rightColumn;
            }
            return String(left.id).localeCompare(String(right.id));
          });
          for (let index = 0; index < sortedNodes.length - 1; index += 1) {
            graph.edges.push({ source: sortedNodes[index].id, target: sortedNodes[index + 1].id });
          }
        }

        graph.nodes.sort((left, right) => {
          const leftColumn = Number(left.column || 0);
          const rightColumn = Number(right.column || 0);
          if (leftColumn !== rightColumn) {
            return leftColumn - rightColumn;
          }
          const leftLane = graph.lanes.findIndex((lane) => lane.id === left.lane);
          const rightLane = graph.lanes.findIndex((lane) => lane.id === right.lane);
          if (leftLane !== rightLane) {
            return leftLane - rightLane;
          }
          return String(left.id).localeCompare(String(right.id));
        });

        if (!graph.lanes.length) {
          graph.lanes = [{ id: "shared", label: "Shared planning" }];
        }

        return graph;
      }

      function workflowTextLines(text, maxChars, maxLines) {
        const clean = String(text || "").replace(/\s+/g, " ").trim();
        if (!clean) return [];
        const words = clean.split(" ");
        const lines = [];
        let current = "";
        words.forEach((word) => {
          const candidate = current ? `${current} ${word}` : word;
          if (candidate.length <= maxChars) {
            current = candidate;
          } else {
            if (current) lines.push(current);
            current = word;
          }
        });
        if (current) lines.push(current);
        if (lines.length > maxLines) {
          const limited = lines.slice(0, maxLines);
          limited[maxLines - 1] = `${limited[maxLines - 1].slice(0, Math.max(0, maxChars - 3)).trim()}...`;
          return limited;
        }
        return lines;
      }

      function renderWorkflowDetails(node) {
        const root = document.getElementById("workflow-step-details");
        if (!root || !node) return;
        const facts = [];
        if (node.subtitle) facts.push({ label: "Stage", value: node.subtitle });
        if (node.lane_label) facts.push({ label: "Lane", value: node.lane_label });
        root.innerHTML = `
          <div class="record-title">${escapeHtml(node.title || "Workflow step")}</div>
          ${node.description ? `<p class="summary-text" style="margin-top:10px">${escapeHtml(node.description)}</p>` : ""}
          ${node.why ? `<p class="subtle" style="margin-top:10px">${escapeHtml(node.why)}</p>` : ""}
          ${facts.length ? `<div style="margin-top:14px">${renderFacts(facts)}</div>` : ""}
          ${renderEvidenceRefs(node.evidence_refs || [])}
        `;
      }

      function renderWorkflowGraph() {
        const model = buildWorkflowModel();
        const container = document.getElementById("workflow-graph");
        if (!container) return;
        if (!model.nodes.length) {
          container.innerHTML = '<div class="empty">No workflow steps available.</div>';
          return;
        }

        const laneOrder = model.lanes;
        const laneY = {};
        laneOrder.forEach((lane, index) => {
          laneY[lane.id] = 80 + index * 170;
        });
        const nodeWidth = 210;
        const nodeHeight = 92;
        const columnWidth = 250;
        const leftPad = 190;
        const rightPad = 110;
        const topPad = 34;
        const maxColumn = Math.max(...model.nodes.map((node) => node.column || 0));
        const width = leftPad + (maxColumn + 1) * columnWidth + rightPad;
        const height = topPad + laneOrder.length * 170 + 80;
        const laneLineStart = leftPad - 24;
        const nodeMap = {};
        model.nodes.forEach((node) => {
          nodeMap[node.id] = node;
        });

        function nodePosition(node) {
          return {
            x: leftPad + (node.column || 0) * columnWidth,
            y: laneY[node.lane] || laneY.shared || 80,
          };
        }

        const laneHtml = laneOrder.map((lane) => {
          const y = laneY[lane.id];
          return `
            <text class="workflow-lane-label" x="18" y="${y + 34}">${escapeHtml(lane.label)}</text>
            <line class="workflow-lane-line" x1="${laneLineStart}" y1="${y + 46}" x2="${width - 30}" y2="${y + 46}"></line>
          `;
        }).join("");

        const edgeHtml = model.edges.map((edge) => {
          const source = nodeMap[edge.source];
          const target = nodeMap[edge.target];
          if (!source || !target) return "";
          const sourcePos = nodePosition(source);
          const targetPos = nodePosition(target);
          const startX = sourcePos.x + nodeWidth;
          const startY = sourcePos.y + nodeHeight / 2;
          const endX = targetPos.x;
          const endY = targetPos.y + nodeHeight / 2;
          const midX = startX + (endX - startX) / 2;
          const path = `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;
          return `<path class="workflow-edge" d="${path}" marker-end="url(#workflow-arrow)"></path>`;
        }).join("");

        const nodeHtml = model.nodes.map((node) => {
          const pos = nodePosition(node);
          const titleLines = workflowTextLines(node.title || "", 24, 2);
          const subtitleLines = workflowTextLines(node.subtitle || "", 28, 1);
          const tone = node.tone || "shared";
          const nodeTooltip = [node.title, node.description, node.why].filter(Boolean).join(" \u2022 ");
          return `
            <g class="workflow-node workflow-node--${escapeHtml(tone)}" data-node-id="${escapeHtml(node.id)}" tabindex="0" aria-label="${escapeHtml(nodeTooltip || node.title || "Workflow step")}">
              <rect x="${pos.x}" y="${pos.y}" width="${nodeWidth}" height="${nodeHeight}"></rect>
              ${titleLines.map((line, index) => `<text class="workflow-node-title" x="${pos.x + 14}" y="${pos.y + 26 + index * 18}">${escapeHtml(line)}</text>`).join("")}
              ${subtitleLines.map((line, index) => `<text class="workflow-node-subtitle" x="${pos.x + 14}" y="${pos.y + 70 + index * 14}">${escapeHtml(line)}</text>`).join("")}
            </g>
          `;
        }).join("");

        container.innerHTML = `
          <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMinYMin meet">
            <defs>
              <marker id="workflow-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
                <path d="M 0 0 L 12 6 L 0 12 z" fill="rgba(76, 108, 146, 0.82)"></path>
              </marker>
            </defs>
            ${laneHtml}
            ${edgeHtml}
            ${nodeHtml}
          </svg>
        `;

        const svg = container.querySelector("svg");
        if (!svg) return;
        const containerWidth = Math.max(container.clientWidth - 24, 760);
        const renderedHeight = Math.max(360, Math.round((containerWidth / width) * height));
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", String(renderedHeight));
        enablePanZoom(svg);

        const detailNodes = Array.from(container.querySelectorAll(".workflow-node"));
        detailNodes.forEach((element) => {
          const node = nodeMap[element.dataset.nodeId];
          element.addEventListener("mouseenter", () => {
            detailNodes.forEach((item) => item.classList.remove("active"));
            element.classList.add("active");
            renderWorkflowDetails(node);
          });
          element.addEventListener("focus", () => {
            detailNodes.forEach((item) => item.classList.remove("active"));
            element.classList.add("active");
            renderWorkflowDetails(node);
          });
          element.addEventListener("click", () => {
            detailNodes.forEach((item) => item.classList.remove("active"));
            element.classList.add("active");
            renderWorkflowDetails(node);
          });
        });

        const firstNode = model.nodes[0];
        const firstElement = container.querySelector(`.workflow-node[data-node-id="${firstNode.id}"]`);
        if (firstElement) firstElement.classList.add("active");
        renderWorkflowDetails(firstNode);
      }

      function renderWorkflowPanels() {
        const guide = reportData.guide || {};
        const decisionsRoot = document.getElementById("workflow-decision-cards");
        const decisions = Array.isArray(guide.key_decisions) ? guide.key_decisions : [];
        if (decisionsRoot) {
          decisionsRoot.innerHTML = decisions.length
            ? decisions.slice(0, 6).map((item) => `
                <article class="record" open>
                  <div class="record-title">${escapeHtml(item.title || "Decision")}</div>
                  ${item.decision ? `<div class="record-subtitle">${escapeHtml(item.decision)}</div>` : ""}
                  <div class="record-body">
                    ${item.why ? `<p class="summary-text">${escapeHtml(item.why)}</p>` : ""}
                  </div>
                </article>
              `).join("")
            : '<div class="empty">No key decisions recorded.</div>';
        }

        const branchRoot = document.getElementById("workflow-branch-cards");
        const branches = Array.isArray(guide.branch_explanations) ? guide.branch_explanations : [];
        if (branchRoot) {
          branchRoot.innerHTML = branches.length
            ? branches.map((item) => `
                <article class="record" open>
                  <div class="record-title">${escapeHtml(item.title || "Branch")}</div>
                  <div class="record-subtitle">${escapeHtml(item.status || "unknown")}</div>
                  <div class="record-body">
                    ${item.summary ? `<p class="summary-text">${escapeHtml(item.summary)}</p>` : ""}
                    ${item.why ? `<p class="subtle">${escapeHtml(item.why)}</p>` : ""}
                  </div>
                </article>
              `).join("")
            : '<div class="empty">No branch outcomes recorded.</div>';
        }

        const notesRoot = document.getElementById("workflow-notes");
        const notes = Array.isArray(guide.walkthrough) ? guide.walkthrough : [];
        if (notesRoot) {
          notesRoot.innerHTML = notes.length
            ? notes.map((item, index) => `
                <article class="timeline-card">
                  <div class="timeline-title-wrap">
                    <div class="timeline-index">${index + 1}</div>
                    <div>
                      <div class="timeline-title">${escapeHtml(item.title || "Step")}</div>
                      ${item.summary ? `<div class="record-subtitle">${escapeHtml(item.summary)}</div>` : ""}
                    </div>
                  </div>
                  ${item.why_it_matters ? `<p class="summary-text">${escapeHtml(item.why_it_matters)}</p>` : ""}
                </article>
              `).join("")
            : '<div class="empty">No workflow notes recorded.</div>';
        }
      }

      function renderHero() {
        const meta = reportData.meta || {};
        document.getElementById("hero-title").textContent = meta.gene_symbol || meta.gene_id || "Unknown gene";
        const geneIdEl = document.getElementById("hero-gene-id");
        if (geneIdEl) {
          if (meta.gene_symbol && meta.gene_id) {
            geneIdEl.textContent = meta.gene_id;
            geneIdEl.style.display = "block";
          } else {
            geneIdEl.textContent = "";
            geneIdEl.style.display = "none";
          }
        }
        document.getElementById("hero-request").textContent = meta.user_request || "No user request captured.";

        const chipBits = [];
        (meta.modalities || []).forEach((entry) => chipBits.push(badge(entry, "teal")));
        if (meta.target_count) {
          chipBits.push(badge(`${meta.target_count} target${meta.target_count === 1 ? "" : "s"}`, "rust"));
        }
        if (meta.ranked_aso_count) {
          chipBits.push(badge(`${meta.ranked_aso_count} ranked ASOs`, "neutral"));
        }
        document.getElementById("hero-chips").innerHTML = chipBits.join("");

        const metaRows = [
          ["Run", meta.run_name],
          ["Generated", meta.generated_at],
          ["Run directory", meta.run_dir],
        ].filter((row) => row[1]);
        document.getElementById("hero-meta-grid").innerHTML = metaRows
          .map((row) => `<div class="meta-row"><span>${escapeHtml(row[0])}</span><span>${escapeHtml(formatValue(row[1]))}</span></div>`)
          .join("");
      }

      function renderOverview() {
        const overview = reportData.overview || {};
        const metrics = overview.key_metrics || [];
        document.getElementById("metric-grid").innerHTML = metrics.length
          ? metrics.map((item) => `
              <article class="metric-card">
                <div class="metric-label">${escapeHtml(item.label)}</div>
                <div class="metric-value">${escapeHtml(formatValue(item.value))}</div>
                <div class="metric-detail">${escapeHtml(item.detail || "")}</div>
              </article>
            `).join("")
          : '<div class="empty">No key metrics captured.</div>';

        const branches = overview.branch_outcomes || [];
        document.getElementById("branch-grid").innerHTML = branches.length
          ? branches.map((entry) => `
              <article class="branch-card">
                <div class="branch-head">
                  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                    <div class="branch-title">${escapeHtml(entry.title)}</div>
                    ${helpIconHtml(entry.help_text || "")}
                  </div>
                  ${badge(entry.status || "unknown", entry.tone)}
                </div>
                <p class="summary-text">${escapeHtml(entry.summary || "No branch summary recorded.")}</p>
                ${renderFacts(entry.facts || [])}
              </article>
            `).join("")
          : '<div class="empty">No branch outcomes recorded.</div>';

        const alerts = overview.alerts || [];
        document.getElementById("alerts-panel").innerHTML = alerts.length
          ? renderBullets(alerts)
          : '<div class="empty">No pipeline alerts recorded.</div>';

        document.getElementById("markdown-report").innerHTML = simpleMarkdownToHtml(overview.report_markdown || "");
      }

      function renderPipelineDecisions() {
        const stages = ((reportData.decisions || {}).pipeline_decisions || []);
        const root = document.getElementById("pipeline-decisions");
        if (!stages.length) {
          root.innerHTML = '<div class="empty">No pipeline decisions recorded.</div>';
          return;
        }
        root.innerHTML = stages.map((stage) => `
          <article class="timeline-card">
            <div class="timeline-head">
              <div class="timeline-title">${escapeHtml(stage.title || "Stage")}</div>
            </div>
            ${stage.summary ? `<p class="summary-text">${escapeHtml(stage.summary)}</p>` : ""}
            ${(stage.items || []).length ? renderFacts(stage.items) : ""}
            ${(stage.bullets || []).length ? renderBullets(stage.bullets) : ""}
          </article>
        `).join("");
      }

      function renderTimeline() {
        const timeline = ((reportData.decisions || {}).timeline || []);
        const root = document.getElementById("decision-timeline");
        if (!timeline.length) {
          root.innerHTML = '<div class="empty">No decision timeline recorded.</div>';
          return;
        }
        root.innerHTML = timeline.map((event, index) => `
          <article class="timeline-card">
            <div class="timeline-title-wrap">
              <div class="timeline-index">${index + 1}</div>
              <div>
                <div class="timeline-title">${escapeHtml(event.title || "Decision")}</div>
                <div class="subtle">${escapeHtml(event.summary || "")}</div>
              </div>
            </div>
            ${(event.badges || []).map((item) => badge(item.text, item.tone)).join("")}
            ${(event.facts || []).length ? renderFacts(event.facts) : ""}
            ${(event.bullets || []).length ? renderBullets(event.bullets) : ""}
          </article>
        `).join("");
      }

      function renderStrategyRouting() {
        const routing = ((reportData.decisions || {}).strategy_routing || {});
        const root = document.getElementById("strategy-routing");
        const goals = Array.isArray(routing.goals) ? routing.goals : [];
        if (!goals.length) {
          root.innerHTML = '<div class="empty">No ranked strategy routes recorded.</div>';
          return;
        }
        const planner = routing.planner_plan || {};
        const headerFacts = [];
        if (planner.plan_summary) headerFacts.push({ label: "Plan", value: planner.plan_summary });
        if (planner.job_count !== undefined) headerFacts.push({ label: "Jobs", value: planner.job_count });
        if (planner.goal_count !== undefined) headerFacts.push({ label: "Goals", value: planner.goal_count });
        if (planner.ranking && planner.ranking.model) headerFacts.push({ label: "Model", value: planner.ranking.model });
        const goalCards = goals.slice(0, 18).map((goal) => {
          const warningText = goal.warning || "";
          const facts = [
            { label: "Rank", value: goal.rank },
            { label: "Confidence", value: goal.ranking_confidence },
            { label: "Candidates", value: goal.count },
            { label: "Threshold", value: goal.threshold_used },
            { label: "Met threshold", value: goal.threshold_pass_count !== goal.count ? goal.threshold_pass_count : null },
            { label: "Primary targets", value: Array.isArray(goal.primary_targets) ? goal.primary_targets.join(", ") : goal.primary_targets },
            { label: "Support", value: goal.observed_support },
          ].filter((item) => item.value !== null && item.value !== undefined && item.value !== "");
          const warningPanel = warningText
            ? `<div class="panel" style="padding:14px;margin-top:12px;border-color:rgba(245, 158, 11, 0.35);background:rgba(245, 158, 11, 0.08)"><div class="section-kicker">Warning</div>${renderBullets([warningText])}</div>`
            : "";
          return `
            <article class="record" open>
              <div class="record-title">${escapeHtml(goal.goal_label || "Unnamed route")}</div>
              <div class="record-subtitle">${escapeHtml(goal.rationale || goal.primary_effect || "No rationale recorded.")}</div>
              ${renderFacts(facts)}
              ${warningPanel}
            </article>
          `;
        }).join("");
        root.innerHTML = `
          ${headerFacts.length ? `<div style="margin-bottom:14px">${renderFacts(headerFacts)}</div>` : ""}
          <div class="record-list">${goalCards}</div>
          ${goals.length > 18 ? `<p class="subtle" style="margin-top:12px">Showing 18 of ${goals.length} ranked routes.</p>` : ""}
        `;
      }

      function renderAgentTrace() {
        const trace = (((reportData.decisions || {}).llm_rationale || {}).agent_trace || []);
        const strategyAnalysis = (((reportData.decisions || {}).llm_rationale || {}).strategy_analysis || "");
        document.getElementById("strategy-analysis").innerHTML = simpleMarkdownToHtml(strategyAnalysis);

        const root = document.getElementById("agent-trace");
        if (!trace.length) {
          root.innerHTML = '<div class="empty">No agent trace recorded.</div>';
          return;
        }
        root.innerHTML = trace.map((step, index) => {
          const title = step.agent_name || step.agent || step.name || `Trace step ${index + 1}`;
          const facts = [];
          if (step.model) facts.push({ label: "Model", value: step.model });
          if (step.tool_name) facts.push({ label: "Tool", value: step.tool_name });
          if (step.status) facts.push({ label: "Status", value: step.status });
          if (step.final_output && typeof step.final_output === "object") {
            facts.push({ label: "Output keys", value: Object.keys(step.final_output).join(", ") });
          }
          const summary = step.summary || step.message || step.reasoning_summary || "";
          return `
            <article class="trace-card">
              <div class="trace-title">
                <div class="record-title">${escapeHtml(title)}</div>
                ${summary ? `<div class="record-subtitle">${escapeHtml(summary)}</div>` : ""}
              </div>
              ${facts.length ? renderFacts(facts) : ""}
              <details class="record" style="margin-top:12px">
                <summary class="record-title">Raw step JSON</summary>
                <div class="record-body">
                  <pre class="json-block">${escapeHtml(JSON.stringify(step, null, 2))}</pre>
                </div>
              </details>
            </article>
          `;
        }).join("");
      }

      function renderCollectionRecord(record) {
        const title = record.title || "Record";
        const subtitle = record.subtitle || "";
        return `
          <details class="record">
            <summary>
              <div>
                <div class="record-title">${escapeHtml(title)}</div>
                ${subtitle ? `<div class="record-subtitle">${escapeHtml(subtitle)}</div>` : ""}
              </div>
            </summary>
            <div class="record-body">
              ${(record.fields || []).length ? renderFacts(record.fields) : ""}
              ${(record.bullets || []).length ? renderBullets(record.bullets) : ""}
              ${record.raw ? `<pre class="json-block">${escapeHtml(JSON.stringify(record.raw, null, 2))}</pre>` : ""}
            </div>
          </details>
        `;
      }

      function renderCollections() {
        const collections = (((reportData.outputs || {}).collections) || []);
        const root = document.getElementById("collection-grid");
        if (!collections.length) {
          root.innerHTML = '<div class="empty">No output collections recorded.</div>';
          return;
        }
        root.innerHTML = collections.map((collection) => `
          <article class="collection-card">
            <div class="collection-head">
              <div>
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                  <div class="collection-title">${escapeHtml(collection.title || "Collection")}</div>
                  ${helpIconHtml(collection.help_text || "")}
                </div>
                ${collection.summary ? `<p class="summary-text">${escapeHtml(collection.summary)}</p>` : ""}
              </div>
              ${collection.status ? badge(collection.status, collection.tone) : ""}
            </div>
            ${(collection.stats || []).length ? `<div style="margin-top:12px">${renderFacts(collection.stats)}</div>` : ""}
            ${(collection.facts || []).length ? `<div style="margin-top:12px">${renderFacts(collection.facts)}</div>` : ""}
            ${(collection.bullets || []).length ? `<div style="margin-top:12px">${renderBullets(collection.bullets)}</div>` : ""}
            ${(collection.records || []).length ? `<div class="record-list" style="margin-top:12px">${collection.records.map(renderCollectionRecord).join("")}</div>` : ""}
            ${collection.raw ? `
              <details class="record" style="margin-top:12px">
                <summary class="record-title">Compact JSON</summary>
                <div class="record-body">
                  <pre class="json-block">${escapeHtml(JSON.stringify(collection.raw, null, 2))}</pre>
                </div>
              </details>` : ""}
          </article>
        `).join("");
      }

      function renderArtifacts(filterValue) {
        const artifacts = (((reportData.outputs || {}).artifacts) || {});
        const groups = Array.isArray(artifacts.groups) ? artifacts.groups : [];
        const query = String(filterValue || "").trim().toLowerCase();
        let visibleCount = 0;
        const html = groups.map((group) => {
          const items = (group.items || []).filter((item) => {
            const haystack = [
              item.name,
              item.relative_path,
              item.kind,
              item.group,
              item.ext,
            ].join(" ").toLowerCase();
            return !query || haystack.includes(query);
          });
          if (!items.length) {
            return "";
          }
          visibleCount += items.length;
          return `
            <details class="artifact-group" open>
              <summary>
                <span>${escapeHtml(group.name)} (${items.length})</span>
                <span class="muted">${escapeHtml(group.summary || "")}</span>
              </summary>
              <div class="artifact-list">
                ${items.map((item) => `
                  <div class="artifact-item">
                    <div>
                      <div><a href="${escapeHtml(item.href)}">${escapeHtml(item.name)}</a></div>
                      <div class="artifact-path">${escapeHtml(item.relative_path)}</div>
                    </div>
                    <div class="artifact-meta">
                      <div>${escapeHtml(item.kind || "file")}</div>
                      <div>${escapeHtml(item.size_label || "")}</div>
                    </div>
                    <div class="artifact-meta">
                      <div>${escapeHtml(item.group || "")}</div>
                      <div>${escapeHtml(item.ext ? `.${item.ext}` : "")}</div>
                    </div>
                  </div>
                `).join("")}
              </div>
            </details>
          `;
        }).join("");
        document.getElementById("artifact-groups").innerHTML = html || '<div class="empty">No artifacts matched this filter.</div>';
        document.getElementById("artifact-count").textContent = `${visibleCount} artifact${visibleCount === 1 ? "" : "s"} shown`;
      }

      function copyToClipboard(text, btn) {
        navigator.clipboard.writeText(text).then(() => {
          btn.classList.add("copied");
          btn.textContent = "Copied";
          setTimeout(() => {
            btn.classList.remove("copied");
            btn.textContent = "Copy";
          }, 1200);
        }).catch(() => {});
      }

      function sortTable(tableEl, colIndex, type) {
        const tbody = tableEl.querySelector("tbody");
        if (!tbody) return;
        const rows = Array.from(tbody.querySelectorAll("tr"));
        const headers = tableEl.querySelectorAll("th");
        const header = headers[colIndex];
        const isAsc = header.classList.contains("sorted-asc");
        headers.forEach((node) => node.classList.remove("sorted-asc", "sorted-desc"));
        const dir = isAsc ? -1 : 1;
        header.classList.add(isAsc ? "sorted-desc" : "sorted-asc");
        rows.sort((a, b) => {
          const cellA = a.cells[colIndex]?.textContent.trim() || "";
          const cellB = b.cells[colIndex]?.textContent.trim() || "";
          if (type === "number") {
            const numA = parseFloat(cellA.replace(/[^0-9.-]/g, "")) || 0;
            const numB = parseFloat(cellB.replace(/[^0-9.-]/g, "")) || 0;
            return (numA - numB) * dir;
          }
          return cellA.localeCompare(cellB) * dir;
        });
        rows.forEach((row) => tbody.appendChild(row));
      }

      const outputTypeLabels = {
        splice_mod: "Splice modulation",
        steric_tiling: "Steric tiling",
        gapmer_tiling: "Gapmer tiling",
        other: "Other",
      };
      const outputTypeOrder = ["splice_mod", "steric_tiling", "gapmer_tiling", "other"];

      function getOutputTypeLabel(outputType) {
        return outputTypeLabels[outputType] || outputType || "Other";
      }

      function getTargetOutputTypes(target) {
        if (Array.isArray(target.output_types) && target.output_types.length) {
          return target.output_types;
        }
        if (target.output_type) {
          return [target.output_type];
        }
        return ["other"];
      }

      function orderOutputTypes(types) {
        const ordered = outputTypeOrder.filter((entry) => types.includes(entry));
        const extras = types.filter((entry) => !outputTypeOrder.includes(entry));
        return ordered.concat(extras);
      }

      function getTargetRank(target) {
        if (!target || typeof target !== "object") return null;
        const rankKeys = ["route_rank", "synthetic_target_rank", "target_rank", "rank"];
        for (const key of rankKeys) {
          const value = Number(target[key]);
          if (Number.isFinite(value)) return value;
        }
        return null;
      }

      function compareTargetsByRank(left, right) {
        const rankLeft = getTargetRank(left);
        const rankRight = getTargetRank(right);
        if (Number.isFinite(rankLeft) && Number.isFinite(rankRight) && rankLeft !== rankRight) {
          return rankLeft - rankRight;
        }
        if (Number.isFinite(rankLeft)) return -1;
        if (Number.isFinite(rankRight)) return 1;
        const posLeft = Number(left && left.position);
        const posRight = Number(right && right.position);
        if (Number.isFinite(posLeft) && Number.isFinite(posRight) && posLeft !== posRight) {
          return posLeft - posRight;
        }
        if (Number.isFinite(posLeft)) return -1;
        if (Number.isFinite(posRight)) return 1;
        return String((left && left.label) || "").localeCompare(String((right && right.label) || ""));
      }

      function parseLabel(label) {
        const match = String(label || "").match(/([A-Za-z_]+)@(\\d+)/);
        if (match) {
          return { kind: match[1], position: parseInt(match[2], 10) };
        }
        return { kind: null, position: null };
      }

      const geneModel = (((reportData.targets || {}).gene) || {});
      const reportTargets = (((reportData.targets || {}).items) || []);
      const targetsByLabel = {};
      const targetsByOutputType = {};
      const safeguardTargetsByType = {};
      const targetMarkerMaps = {};
      const safeguardMarkerMaps = {};
      const safeguardGuardsByType = {};
      reportTargets.forEach((target) => {
        targetsByLabel[target.label] = target;
        getTargetOutputTypes(target).forEach((outputType) => {
          if (!targetsByOutputType[outputType]) {
            targetsByOutputType[outputType] = [];
          }
          targetsByOutputType[outputType].push(target);
        });
      });
      Object.keys(targetsByOutputType).forEach((outputType) => {
        const typedTargets = (targetsByOutputType[outputType] || []).filter((target) => {
          const guards = ((target.safeguards_by_type || {})[outputType]) || target.safeguards || [];
          return Array.isArray(guards) && guards.length > 0;
        });
        if (typedTargets.length) {
          safeguardTargetsByType[outputType] = typedTargets;
        }
      });

      function enablePanZoom(svg) {
        const initialViewBox = svg.getAttribute("viewBox") || "";
        const parts = initialViewBox.split(/\\s+/).map(parseFloat).filter((v) => !Number.isNaN(v));
        if (parts.length !== 4) {
          return;
        }
        let state = { x: parts[0], y: parts[1], w: parts[2], h: parts[3], baseW: parts[2], baseH: parts[3] };
        let isPanning = false;
        let lastPoint = null;
        const minScale = 0.2;
        const maxScale = 16;
        const zoomIntensity = 0.0015;
        let rafId = null;

        function updateZoomScale() {
          const scale = state.baseW / state.w;
          svg.querySelectorAll(".marker").forEach((marker) => {
            const baseRadius = parseFloat(marker.dataset.baseRadius || marker.getAttribute("r") || "8");
            const baseStroke = parseFloat(marker.dataset.baseStrokeWidth || marker.getAttribute("stroke-width") || "3");
            const ratio = baseRadius > 0 ? baseStroke / baseRadius : 0.4;
            const minRadius = baseRadius * 0.7;
            const radius = Math.max(baseRadius / scale, minRadius);
            const strokeWidth = Math.max(radius * ratio, 0.8);
            marker.setAttribute("r", radius.toFixed(2));
            marker.setAttribute("stroke-width", strokeWidth.toFixed(2));
          });
          svg.querySelectorAll(".marker-label").forEach((label) => {
            const baseFont = parseFloat(label.dataset.baseFontSize || "10");
            label.style.fontSize = `${baseFont / scale}px`;
          });
        }

        function applyViewBoxImmediate() {
          svg.setAttribute("viewBox", `${state.x} ${state.y} ${state.w} ${state.h}`);
          updateZoomScale();
        }

        function applyViewBox() {
          if (rafId !== null) return;
          rafId = requestAnimationFrame(() => {
            rafId = null;
            applyViewBoxImmediate();
          });
        }

        function screenToSvg(event) {
          const rect = svg.getBoundingClientRect();
          const px = event.clientX - rect.left;
          const py = event.clientY - rect.top;
          return {
            x: state.x + (px / rect.width) * state.w,
            y: state.y + (py / rect.height) * state.h,
            rect,
          };
        }

        svg.addEventListener("wheel", (event) => {
          event.preventDefault();
          const rawDelta = event.deltaY || event.wheelDelta || 0;
          const deltaScale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1;
          const delta = rawDelta * deltaScale;
          const zoomFactor = Math.exp(delta * zoomIntensity);
          const nextW = state.w * zoomFactor;
          const nextH = state.h * zoomFactor;
          const scale = state.baseW / nextW;
          if (scale < minScale || scale > maxScale) return;
          const point = screenToSvg(event);
          const ratioX = (point.x - state.x) / state.w;
          const ratioY = (point.y - state.y) / state.h;
          state.w = nextW;
          state.h = nextH;
          state.x = point.x - ratioX * state.w;
          state.y = point.y - ratioY * state.h;
          applyViewBox();
        }, { passive: false });

        svg.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          if (event.target && event.target.classList && event.target.classList.contains("marker")) return;
          event.preventDefault();
          isPanning = true;
          lastPoint = { px: event.clientX, py: event.clientY };
          svg.setPointerCapture(event.pointerId);
        });

        svg.addEventListener("pointermove", (event) => {
          if (!isPanning || !lastPoint) return;
          if (event.buttons === 0) {
            isPanning = false;
            lastPoint = null;
            return;
          }
          const rect = svg.getBoundingClientRect();
          const dxPx = typeof event.movementX === "number" ? event.movementX : event.clientX - lastPoint.px;
          const dyPx = typeof event.movementY === "number" ? event.movementY : event.clientY - lastPoint.py;
          state.x -= dxPx * (state.w / rect.width);
          state.y -= dyPx * (state.h / rect.height);
          lastPoint = { px: event.clientX, py: event.clientY };
          applyViewBox();
        });

        svg.addEventListener("pointerup", () => {
          isPanning = false;
          lastPoint = null;
        });

        svg.addEventListener("dblclick", () => {
          state = { x: parts[0], y: parts[1], w: parts[2], h: parts[3], baseW: parts[2], baseH: parts[3] };
          applyViewBoxImmediate();
        });

        updateZoomScale();
      }

      function renderGene(containerId, gene, markers, onSelect, options) {
        const container = document.getElementById(containerId);
        if (!container) return {};
        container.innerHTML = "";
        if (!gene || !Array.isArray(gene.transcripts) || !gene.transcripts.length) {
          container.innerHTML = '<div class="empty">No transcript data available.</div>';
          return {};
        }
        const settings = options || {};
        const guardBelow = Boolean(settings.guardBelow);
        const hoverSelect = settings.hoverSelect !== false;
        const clickSelect = settings.clickSelect !== false;
        const width = 1200;
        const transcriptCount = gene.transcripts.length;
        const densityScale = Math.max(0.25, Math.min(1, 12 / transcriptCount));
        const markerRadius = Math.max(4, Math.round(8 * densityScale));
        const markerY = Math.max(markerRadius + 2, Math.round(24 * densityScale));
        const markerLabelFont = Math.max(14, Math.round(32 * densityScale));
        const guardLabelFont = Math.max(13, Math.round(18 * densityScale));
        const transcriptLabelFont = Math.max(16, Math.round(26 * densityScale));
        const exonHeight = Math.max(8, Math.round(18 * densityScale));
        const transcriptRowHeight = 2 * Math.max(exonHeight + 12, Math.round(48 * densityScale));
        const leftPad = 200;
        const rightPad = 60;
        const labelOffsetUp = -8;
        const guardRowOffset = 30;
        const topPad = Math.max(markerY + markerRadius + Math.ceil(exonHeight / 2) + 2, 30);
        const lastTranscriptY = topPad + Math.max(0, transcriptCount - 1) * transcriptRowHeight;
        const guardMarkerY = lastTranscriptY + guardRowOffset;
        const isGuardMarker = (marker) =>
          marker
          && Number.isFinite(marker.position)
          && (marker.placement === "below" || (guardBelow && marker.role === "other"));
        const guardMarkers = (markers || []).filter((marker) => isGuardMarker(marker));
        const guardLabelRowHeight = guardLabelFont + 14;
        const bottomPad = Math.max(
          Math.ceil(exonHeight / 2) + 1,
          guardBelow ? markerRadius + 24 + guardMarkers.length * guardLabelRowHeight : 14
        );
        const height = Math.max(lastTranscriptY + bottomPad, 200);
        const viewPadX = 100;
        const viewPadY = 16;
        const plotWidth = width - leftPad - rightPad;
        const geneStart = Number.isFinite(parseInt(gene.start, 10)) ? parseInt(gene.start, 10) : 0;
        const geneEnd = Number.isFinite(parseInt(gene.end, 10)) ? parseInt(gene.end, 10) : geneStart;
        const extentPositions = [geneStart, geneEnd];
        (gene.transcripts || []).forEach((tx) => {
          (tx.exons || []).forEach((exon) => {
            if (Number.isFinite(exon.start)) extentPositions.push(exon.start);
            if (Number.isFinite(exon.end)) extentPositions.push(exon.end);
          });
        });
        (markers || []).forEach((marker) => {
          if (Number.isFinite(marker.position)) extentPositions.push(marker.position);
        });
        const overallStart = Math.min(...extentPositions);
        const overallEnd = Math.max(...extentPositions);
        const displayStart = Math.max(0, overallStart - 200);
        const displayEnd = overallEnd + 200;
        const span = Math.max(1, displayEnd - displayStart);
        const xScale = (pos) => leftPad + ((pos - displayStart) / span) * plotWidth;
        const guardLayouts = {};
        guardMarkers
          .map((marker) => ({
            key: marker.key || marker.label,
            label: String(marker.label || ""),
            x: xScale(marker.position),
          }))
          .sort((left, right) => {
            if (left.x !== right.x) return left.x - right.x;
            return left.label.localeCompare(right.label);
          })
          .forEach((marker, index) => {
            const labelBaselineY = guardMarkerY + markerRadius + 18 + (index + 1) * guardLabelRowHeight;
            const nearRightEdge = marker.x > leftPad + plotWidth - 180;
            guardLayouts[marker.key] = {
              labelBaselineY,
              labelX: nearRightEdge ? marker.x - 10 : marker.x + 10,
              textAnchor: nearRightEdge ? "end" : "start",
              connectorEndY: labelBaselineY - guardLabelFont + 4,
            };
          });

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", `${-viewPadX} ${-viewPadY} ${width + viewPadX * 2} ${height + viewPadY * 2}`);
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", Math.max(height, 700));

        gene.transcripts.forEach((tx, idx) => {
          const y = topPad + idx * transcriptRowHeight;
          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", xScale(overallStart));
          line.setAttribute("x2", xScale(overallEnd));
          line.setAttribute("y1", y);
          line.setAttribute("y2", y);
          line.setAttribute("class", "gene-line");
          line.dataset.baseStrokeWidth = "3";
          svg.appendChild(line);

          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.textContent = tx.display_name || tx.id;
          label.setAttribute("x", leftPad - 8);
          label.setAttribute("y", y + 4);
          label.setAttribute("text-anchor", "end");
          label.setAttribute("class", "gene-label");
          label.dataset.baseFontSize = String(transcriptLabelFont);
          svg.appendChild(label);

          (tx.exons || []).forEach((exon) => {
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            const x = xScale(exon.start);
            const w = Math.max(2, xScale(exon.end) - x);
            rect.setAttribute("x", x);
            rect.setAttribute("y", y - exonHeight / 2);
            rect.setAttribute("width", w);
            rect.setAttribute("height", exonHeight);
            rect.setAttribute("class", `exon ${exon.contains_coding ? "coding" : "noncoding"}`);
            svg.appendChild(rect);
          });
        });

        const markerMap = {};
        (markers || []).forEach((marker) => {
          if (!marker.position) return;
          const isGuard = isGuardMarker(marker);
          const currentMarkerY = isGuard ? guardMarkerY : markerY;
          const markerKey = marker.key || marker.label;
          const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
          const roleClass = marker.role === "target" ? (guardBelow ? "safeguard-target" : "target") : "other";
          const guardLayout = guardLayouts[markerKey] || null;
          const markerX = xScale(marker.position);

          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", markerX);
          circle.setAttribute("cy", currentMarkerY);
          circle.setAttribute("r", markerRadius);
          circle.dataset.baseRadius = String(markerRadius);
          circle.dataset.baseStrokeWidth = "3";
          circle.setAttribute("class", `marker ${roleClass}`);
          group.appendChild(circle);

          if (isGuard && guardLayout) {
            const connector = document.createElementNS("http://www.w3.org/2000/svg", "line");
            connector.setAttribute("x1", markerX);
            connector.setAttribute("x2", markerX);
            connector.setAttribute("y1", currentMarkerY + markerRadius + 2);
            connector.setAttribute("y2", Math.max(currentMarkerY + markerRadius + 8, guardLayout.connectorEndY));
            connector.setAttribute("class", `marker-connector ${roleClass}`);
            group.appendChild(connector);
          }

          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          const labelX = isGuard && guardLayout ? guardLayout.labelX : markerX;
          const labelY = isGuard && guardLayout ? guardLayout.labelBaselineY : currentMarkerY + labelOffsetUp;
          label.textContent = marker.label;
          label.setAttribute("x", labelX);
          label.setAttribute("y", labelY);
          label.setAttribute("class", `marker-label ${roleClass}`);
          if (isGuard && guardLayout) {
            label.setAttribute("text-anchor", guardLayout.textAnchor);
          } else {
            label.setAttribute("transform", `rotate(-35 ${labelX} ${labelY})`);
          }
          label.dataset.baseFontSize = String(isGuard ? guardLabelFont : markerLabelFont);
          group.appendChild(label);

          group.dataset.label = markerKey;
          if (onSelect && marker.selectable !== false) {
            if (hoverSelect) group.addEventListener("mouseenter", () => onSelect(marker.label));
            if (clickSelect) group.addEventListener("click", () => onSelect(marker.label));
          }
          markerMap[markerKey] = group;
          svg.appendChild(group);
        });

        container.appendChild(svg);
        enablePanZoom(svg);
        return markerMap;
      }

      function setActiveListItem(label) {
        document.querySelectorAll(".list-item").forEach((item) => {
          item.classList.toggle("active", item.dataset.label === label);
        });
      }

      function formatExonIndices(exonIndices) {
        if (!exonIndices || typeof exonIndices !== "object") return "N/A";
        const exons = [...new Set(Object.values(exonIndices).filter((value) => value !== null && value !== undefined))];
        if (!exons.length) return "N/A";
        exons.sort((a, b) => a - b);
        if (exons.length === 1) return `Exon ${exons[0]}`;
        const isConsecutive = exons.every((value, index) => index === 0 || value === exons[index - 1] + 1);
        if (isConsecutive) return `Exons ${exons[0]}-${exons[exons.length - 1]}`;
        return `Exons ${exons.join(", ")}`;
      }

      function buildOutputTypeMapHelp(targetsForType, outputType, mode) {
        const count = Array.isArray(targetsForType) ? targetsForType.length : 0;
        if (mode === "safeguards") {
          const guards = (targetsForType || []).reduce((sum, target) => sum + getSafeguardsForType(target, outputType).length, 0);
          return `${getOutputTypeLabel(outputType)} safeguards: ${count} target site(s), ${guards} safeguard marker(s). Click a target in the list to highlight its attached safeguards.`;
        }
        return `${getOutputTypeLabel(outputType)} map: ${count} target site(s) projected onto transcript models. Click a target in the list to inspect details and recommended ASOs.`;
      }

      function buildTargetHelpText(target) {
        if (!target) return "";
        const isoforms = []
          .concat(target.present_in || [])
          .concat(target.absent_in || []);
        const uniqueIsoforms = [...new Set(isoforms.filter(Boolean))];
        const outputTypes = orderOutputTypes(getTargetOutputTypes(target)).map(getOutputTypeLabel).join(", ");
        const recommendationCount = Array.isArray(target.recommended_asos) ? target.recommended_asos.length : 0;
        return `${target.display_label || target.label || "Target"} is a ${target.direction || "direction-unknown"} ${target.site_kind || "site"}${target.position ? ` at ${target.position}` : ""}. It appears in ${uniqueIsoforms.length} annotated isoform context(s) and has ${recommendationCount} recommended ASO(s) across ${outputTypes || "no recorded output types"}.`;
      }

      function renderTargetDetails(target) {
        const container = document.getElementById("target-details");
        if (!container) return;
        if (!target) {
          container.innerHTML = '<div class="empty">Click a target site to see details.</div>';
          return;
        }
        const regionText = target.region_start !== undefined && target.region_end !== undefined
          ? `${target.chrom || "chr?"}:${target.region_start}-${target.region_end}`
          : "N/A";
        const exonText = formatExonIndices(target.exon_indices);
        const infoRows = [
          ["Label", target.display_label || target.label],
          ["Route rank", target.route_rank || "N/A"],
          ["Route goal", target.route_goal_label || "N/A"],
          ["Exon", exonText],
          ["Direction", target.direction || "unknown"],
          ["Site", target.site_kind || "unknown"],
          ["Position", target.position || "unknown"],
          ["Region", regionText],
          ["Uniqueness", target.uniqueness || "unknown"],
          ["Coding", target.all_coding === undefined ? "unknown" : target.all_coding ? "yes" : "no"],
          ["Covers all isoforms", target.covers_all_isoforms === undefined ? "unknown" : target.covers_all_isoforms ? "yes" : "no"],
          ["Present in", (target.present_in || []).join(", ") || "N/A"],
          ["Absent in", (target.absent_in || []).join(", ") || "N/A"],
          ["Route rationale", target.route_rationale || "N/A"],
        ];
        let html = `<div class="record-subtitle">${escapeHtml(buildTargetHelpText(target))}</div>`;
        html += infoRows.map(([label, value]) => `<div class="detail-row"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>`).join("");

        const recommendationsByType = target.recommended_asos_by_type || {};
        const outputTypes = orderOutputTypes(getTargetOutputTypes(target));
        const fallbackAsos = target.recommended_asos || [];
        const hasTypedRecommendations = Object.keys(recommendationsByType).length > 0;

        function wrapDetailPanel(title, bodyHtml) {
          return `<div class="detail-panel"><div class="detail-panel-title">${escapeHtml(title)}</div><div class="detail-panel-body">${bodyHtml}</div></div>`;
        }

        let tableIndex = 0;
        function renderAsosTable(asos, outputType) {
          if (!asos.length) {
            return '<div class="empty">No ASOs found.</div>';
          }
          const isTiling = outputType === "gapmer_tiling" || outputType === "steric_tiling";
          function toNumber(value) {
            if (value === null || value === undefined || value === "") return null;
            const num = Number(value);
            return Number.isFinite(num) ? num : null;
          }
          function compareDesc(a, b) {
            const hasA = Number.isFinite(a);
            const hasB = Number.isFinite(b);
            if (hasA && hasB) return b - a;
            if (hasA) return -1;
            if (hasB) return 1;
            return 0;
          }
          function getUniquenessScore(aso) {
            const score = toNumber(aso.isoform_uniqueness_score ?? aso.uniqueness_score ?? aso.uniqueness);
            if (Number.isFinite(score)) return score;
            const count = toNumber(aso.isoform_uniqueness_count);
            if (Number.isFinite(count) && count > 0) return 1 / count;
            return null;
          }
          const sortedAsos = [...asos].sort((a, b) => {
            if (isTiling) {
              const uniqCmp = compareDesc(getUniquenessScore(a), getUniquenessScore(b));
              if (uniqCmp !== 0) return uniqCmp;
              const phastCmp = compareDesc(toNumber(a.phastcon), toNumber(b.phastcon));
              if (phastCmp !== 0) return phastCmp;
              const phylopCmp = compareDesc(toNumber(a.phylop ?? a.phyloP), toNumber(b.phylop ?? b.phyloP));
              if (phylopCmp !== 0) return phylopCmp;
              const scoreCmp = compareDesc(toNumber(a.score), toNumber(b.score));
              if (scoreCmp !== 0) return scoreCmp;
              return (toNumber(a.start) || 0) - (toNumber(b.start) || 0);
            }
            // Preserve the backend's scoped, signed recommendation order.
            return 0;
          });
          const columns = isTiling
            ? [
                { label: "Antisense ASO (5′–3′)", sort: "text", value: (aso) => aso.synthesized_antisense_sequence || "Orientation unverified" },
                { label: "Source sequence (see role)", sort: "text", value: (aso) => aso.sequence || "" },
                { label: "Source sequence role", sort: "text", value: (aso) => aso.sequence_role || "Unverified" },
                { label: "Uniqueness", sort: "number", value: (aso) => formatValue(getUniquenessScore(aso)) },
                { label: "Phastcon", sort: "number", value: (aso) => formatValue(aso.phastcon) },
                { label: "PhyloP", sort: "number", value: (aso) => formatValue(aso.phylop ?? aso.phyloP) },
                { label: "Score", sort: "number", value: (aso) => formatValue(aso.score) },
                { label: "Start", sort: "number", value: (aso) => formatValue(aso.start) },
                { label: "End", sort: "number", value: (aso) => formatValue(aso.end) },
                { label: "GC", sort: "number", value: (aso) => formatValue(aso.gc) },
              ]
            : [
                { label: "Antisense ASO (5′–3′)", sort: "text", value: (aso) => aso.synthesized_antisense_sequence || "Orientation unverified" },
                { label: "Source sequence (see role)", sort: "text", value: (aso) => aso.sequence || "" },
                { label: "Source sequence role", sort: "text", value: (aso) => aso.sequence_role || "Unverified" },
                { label: "Selected-site mean delta", sort: "number", value: (aso) => formatValue(aso.mean_splice_delta) },
                { label: "Prediction evidence", sort: "text", value: (aso) => aso.recommendation_note || "Inspect the recorded site scores and route rationale." },
                { label: "Exon-pair deltas (deletion / replacement)", sort: "text", value: (aso) => formatExonObjectiveEvidence(aso.exon_objective_evidence) },
                { label: "Score", sort: "number", value: (aso) => formatValue(aso.score) },
                { label: "Start", sort: "number", value: (aso) => formatValue(aso.start) },
                { label: "End", sort: "number", value: (aso) => formatValue(aso.end) },
                { label: "GC", sort: "number", value: (aso) => formatValue(aso.gc) },
              ];
          tableIndex += 1;
          const tableId = `detail-table-${Date.now()}-${tableIndex}`;
          const headerCells = columns.map((col, idx) => `<th class="sortable" onclick="sortTable(document.getElementById('${tableId}'), ${idx}, '${col.sort}')">${escapeHtml(col.label)}</th>`).join("");
          const bodyRows = sortedAsos.map((aso) => `
            <tr>
              ${columns.map((col) => `<td>${escapeHtml(col.value(aso))}</td>`).join("")}
              <td><button class="copy-btn" onclick="copyToClipboard('${escapeHtml(aso.synthesized_antisense_sequence || "")}', this)">Copy</button></td>
            </tr>
          `).join("");
          return `
            <div class="detail-subtitle">Top ${sortedAsos.length} recommendation(s)</div>
            <div class="table-scroll">
              <table class="aso-table" id="${tableId}">
                <thead><tr>${headerCells}<th>Copy</th></tr></thead>
                <tbody>${bodyRows}</tbody>
              </table>
            </div>
          `;
        }

        if (!outputTypes.length || !hasTypedRecommendations) {
          html += wrapDetailPanel("Recommended ASOs", renderAsosTable(fallbackAsos, outputTypes[0]));
          container.innerHTML = html;
          return;
        }
        outputTypes.forEach((outputType) => {
          html += wrapDetailPanel(`${getOutputTypeLabel(outputType)} ASOs`, renderAsosTable(recommendationsByType[outputType] || [], outputType));
        });
        container.innerHTML = html;
      }

      function getSafeguardsForType(target, outputType) {
        if (!target || typeof target !== "object") return [];
        const safeguardsByType = target.safeguards_by_type || {};
        const typedSafeguards = safeguardsByType[outputType];
        if (Array.isArray(typedSafeguards) && typedSafeguards.length) return typedSafeguards;
        return Array.isArray(target.safeguards) ? target.safeguards : [];
      }

      function renderSafeguardDetails(target) {
        const container = document.getElementById("safeguard-details");
        if (!container) return;
        if (!target) {
          container.innerHTML = '<div class="empty">Click a target site to see safeguards.</div>';
          return;
        }
        const safeguardsByType = target.safeguards_by_type || {};
        const outputTypes = orderOutputTypes(getTargetOutputTypes(target));
        const fallbackSafeguards = target.safeguards || [];
        const hasTypedSafeguards = Object.keys(safeguardsByType).length > 0;

        function renderSafeguardList(safeguards) {
          if (!safeguards.length) {
            return '<div class="empty">No safeguards found.</div>';
          }
          return safeguards.map((guard) => {
            const limits = [
              guard.max_abs_delta ? `abs <= ${guard.max_abs_delta}` : null,
              guard.max_increase ? `inc <= ${guard.max_increase}` : null,
              guard.max_decrease ? `dec <= ${guard.max_decrease}` : null,
            ].filter(Boolean).join(", ");
            const direction = guard.direction || "neutral";
            const note = guard.note ? `<span class="detail-note">${escapeHtml(guard.note)}</span>` : "";
            return `<div class="detail-row"><span>${escapeHtml(guard.label)}</span><span>${escapeHtml(direction)} ${escapeHtml(limits || "no limits")}${note}</span></div>`;
          }).join("");
        }

        function wrapSafeguardPanel(title, bodyHtml) {
          return `<div class="detail-panel"><div class="detail-panel-title">${escapeHtml(title)}</div><div class="detail-panel-body">${bodyHtml}</div></div>`;
        }

        let html = `<div class="record-subtitle">${escapeHtml(helpText.maps_safeguards || "")}</div>`;
        if (!outputTypes.length || !hasTypedSafeguards) {
          container.innerHTML = fallbackSafeguards.length
            ? html + wrapSafeguardPanel("Safeguards", renderSafeguardList(fallbackSafeguards))
            : '<div class="empty">No safeguards recorded for this target.</div>';
          return;
        }
        outputTypes.forEach((outputType) => {
          html += wrapSafeguardPanel(`${getOutputTypeLabel(outputType)} Safeguards`, renderSafeguardList(safeguardsByType[outputType] || []));
        });
        container.innerHTML = html || '<div class="empty">No safeguards recorded for this target.</div>';
      }

      function hideMarkerMap(markerMap) {
        Object.values(markerMap || {}).forEach((marker) => {
          marker.style.display = "none";
          marker.querySelectorAll("circle, text, line").forEach((node) => {
            node.classList.remove("active", "highlight-increase", "highlight-decrease", "highlight-neutral");
          });
        });
      }

      function hideAllMarkerMaps(markerMaps) {
        Object.values(markerMaps || {}).forEach((markerMap) => hideMarkerMap(markerMap));
      }

      function showMarker(markerMap, label, classes) {
        if (!markerMap || !label) return false;
        const marker = markerMap[label];
        if (!marker) return false;
        marker.style.display = "";
        if (classes) {
          marker.querySelectorAll("circle, text, line").forEach((node) => {
            classes.forEach((name) => node.classList.add(name));
          });
        }
        return true;
      }

      function showAllSafeguardMarkers(outputType) {
        const markerMap = safeguardMarkerMaps[outputType];
        hideMarkerMap(markerMap);
        Object.values(markerMap || {}).forEach((marker) => {
          const circle = marker.querySelector("circle");
          if (!circle || !circle.classList.contains("other")) return;
          marker.style.display = "";
        });
      }

      function highlightSafeguardMarker(markerMap, guardLabel, direction) {
        if (!markerMap || !guardLabel) return;
        const highlightClass = `highlight-${direction || "neutral"}`;
        const canonicalKey = `guard:${guardLabel}`;
        if (showMarker(markerMap, canonicalKey, [highlightClass])) return;
      }

      function showTargetMarkers(outputType, label) {
        hideAllMarkerMaps(targetMarkerMaps);
        if (!label) return;
        showMarker(targetMarkerMaps[outputType], label, ["active"]);
      }

      function showSafeguardMarkers(outputType, targetLabel) {
        hideAllMarkerMaps(safeguardMarkerMaps);
        if (!targetLabel) return;
        const markerMap = safeguardMarkerMaps[outputType];
        const guardLookup = safeguardGuardsByType[outputType] || {};
        let guards = guardLookup[targetLabel] || [];
        if (!guards.length) {
          const fallbackTarget = targetsByLabel[targetLabel];
          guards = getSafeguardsForType(fallbackTarget, outputType).map((guard) => ({ ...guard, label: String(guard.label || "").trim(), map_key: `guard:${String(guard.label || "").trim()}` }));
        }
        showMarker(markerMap, targetLabel, ["active"]);
        guards.forEach((guard) => {
          const direction = guard.direction || "neutral";
          const label = String(guard.label || "").trim();
          showMarker(markerMap, `guard:${label}`, [`highlight-${direction}`]);
          if (label) highlightSafeguardMarker(markerMap, label, direction);
        });
      }

      function buildTargetMapGroups() {
        const container = document.getElementById("target-map-groups");
        if (!container) return;
        container.innerHTML = "";
        const groupKeys = Object.keys(targetsByOutputType || {});
        if (!groupKeys.length) {
          container.innerHTML = '<div class="empty">No target sites available.</div>';
          return;
        }
        orderOutputTypes(groupKeys).forEach((outputType) => {
          const mapId = `target-map-${outputType}`;
          const listId = `target-list-${outputType}`;
          const targetsForType = targetsByOutputType[outputType] || [];
          const sortedTargets = [...targetsForType].sort(compareTargetsByRank);
          const card = document.createElement("article");
          card.className = "target-card";
          card.innerHTML = `
            <div class="target-head">
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <div class="target-title">${escapeHtml(getOutputTypeLabel(outputType))} targets</div>
                ${helpIconHtml(buildOutputTypeMapHelp(targetsForType, outputType, "targets"))}
              </div>
            </div>
            <div class="map-row map-row--stacked">
              <div class="map-list">
                <input type="text" class="search-input" placeholder="Search targets..." oninput="filterMapList('${listId}', this.value)" />
                <div class="spacer"></div>
                <div class="list" id="${listId}"></div>
              </div>
              <div>
                <div class="gene-structure" id="${mapId}"></div>
                <div class="gene-help">Click a target in the list to place it on the map. Scroll to zoom, drag to pan, double-click to reset.</div>
              </div>
            </div>
          `;
          container.appendChild(card);
          const listContainer = card.querySelector(`#${listId}`);
          if (!targetsForType.length) {
            listContainer.innerHTML = '<div class="empty">No target sites available.</div>';
          } else {
            listContainer.innerHTML = sortedTargets.map((target) => `
              <div class="list-item" data-label="${escapeHtml(target.label)}">
                <div class="list-title">${escapeHtml(target.display_label || target.label)}</div>
                <div class="list-sub">${escapeHtml(target.direction || "direction unknown")}</div>
              </div>
            `).join("");
            listContainer.querySelectorAll(".list-item").forEach((item) => {
              item.addEventListener("click", () => {
                const label = item.dataset.label;
                const target = targetsByLabel[label];
                if (target) {
                  renderTargetDetails(target);
                  setActiveListItem(label);
                }
                showTargetMarkers(outputType, label);
              });
            });
          }
          const markers = sortedTargets.map((target) => ({
            label: target.label,
            position: target.genomic_map_position,
            direction: target.direction,
            role: "target",
          }));
          const markerMap = renderGene(mapId, geneModel, markers, (label) => {
            const target = targetsByLabel[label];
            if (target) {
              renderTargetDetails(target);
              setActiveListItem(label);
            }
            showTargetMarkers(outputType, label);
          }, { hoverSelect: false });
          targetMarkerMaps[outputType] = markerMap || {};
          hideMarkerMap(targetMarkerMaps[outputType]);
        });
      }

      function buildSafeguardMapGroups() {
        const container = document.getElementById("safeguard-map-groups");
        if (!container) return;
        container.innerHTML = "";
        const groupKeys = Object.keys(safeguardTargetsByType || {});
        if (!groupKeys.length) {
          container.innerHTML = '<div class="empty">No safeguard targets available.</div>';
          return;
        }
        orderOutputTypes(groupKeys).forEach((outputType) => {
          const mapId = `safeguard-map-${outputType}`;
          const listId = `safeguard-list-${outputType}`;
          const targetsForType = safeguardTargetsByType[outputType] || [];
          const sortedTargets = [...targetsForType].sort(compareTargetsByRank);
          const card = document.createElement("article");
          card.className = "target-card";
          card.innerHTML = `
            <div class="target-head">
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <div class="target-title">${escapeHtml(getOutputTypeLabel(outputType))} safeguards</div>
                ${helpIconHtml(buildOutputTypeMapHelp(targetsForType, outputType, "safeguards"))}
              </div>
            </div>
            <div class="map-row map-row--stacked">
              <div class="map-list">
                <input type="text" class="search-input" placeholder="Search safeguards..." oninput="filterMapList('${listId}', this.value)" />
                <div class="spacer"></div>
                <div class="list" id="${listId}"></div>
              </div>
              <div>
                <div class="gene-structure" id="${mapId}"></div>
                <div class="gene-help">Click a target in the list to place it on the map. Scroll to zoom, drag to pan, double-click to reset.</div>
                <div class="legend">
                  <span><span class="legend-dot target"></span>Selected target</span>
                  <span><span class="legend-dot increase"></span>Safeguard increase</span>
                  <span><span class="legend-dot decrease"></span>Safeguard decrease</span>
                  <span><span class="legend-dot neutral"></span>Safeguard neutral</span>
                </div>
              </div>
            </div>
          `;
          container.appendChild(card);
          const listContainer = card.querySelector(`#${listId}`);
          listContainer.innerHTML = sortedTargets.map((target) => `
            <div class="list-item" data-label="${escapeHtml(target.label)}">
              <div class="list-title">${escapeHtml(target.display_label || target.label)}</div>
              <div class="list-sub">${escapeHtml(target.direction || "direction unknown")}</div>
            </div>
          `).join("");
          listContainer.querySelectorAll(".list-item").forEach((item) => {
            item.addEventListener("click", () => {
              const label = item.dataset.label;
              const target = targetsByLabel[label];
              if (target) {
                renderSafeguardDetails(target);
                setActiveListItem(label);
              }
              showSafeguardMarkers(outputType, label);
            });
          });
          const guardLookup = {};
          const markers = [];
          const seenTargets = new Set();
          const seenGuards = new Set();
          targetsForType.forEach((target) => {
            const position = target.genomic_map_position;
            if (!seenTargets.has(target.label)) {
              markers.push({ label: target.label, position, direction: target.direction, role: "target" });
              seenTargets.add(target.label);
            }
            const safeguardsForType = getSafeguardsForType(target, outputType);
            guardLookup[target.label] = safeguardsForType.map((guard) => {
              const guardLabel = String(guard.label || "").trim();
              return { ...guard, label: guardLabel, map_key: `guard:${guardLabel}` };
            });
          });
          targetsForType.forEach((target) => {
            getSafeguardsForType(target, outputType).forEach((guard) => {
              const guardLabel = String(guard.label || "").trim();
              if (!guardLabel) return;
              const parsed = parseLabel(guardLabel);
              const mapKey = `guard:${guardLabel}`;
              if (!parsed.position || seenGuards.has(mapKey)) return;
              markers.push({
                key: mapKey,
                label: guardLabel,
                position: parsed.position,
                direction: guard.direction,
                role: "other",
                placement: "below",
                selectable: false,
              });
              seenGuards.add(mapKey);
            });
          });
          safeguardGuardsByType[outputType] = guardLookup;
          const markerMap = renderGene(mapId, geneModel, markers, (label) => {
            if (!guardLookup[label]) return;
            const target = targetsByLabel[label];
            if (target) {
              renderSafeguardDetails(target);
              setActiveListItem(label);
            }
            showSafeguardMarkers(outputType, label);
          }, { guardBelow: true, hoverSelect: false });
          safeguardMarkerMaps[outputType] = markerMap || {};
          hideMarkerMap(safeguardMarkerMaps[outputType]);
        });
      }

      function filterMapList(listId, query) {
        const container = document.getElementById(listId);
        if (!container) return;
        const lowerQuery = query.toLowerCase();
        container.querySelectorAll(".list-item").forEach((item) => {
          const label = (item.dataset.label || "").toLowerCase();
          const direction = (item.querySelector(".list-sub")?.textContent || "").toLowerCase();
          const visible = label.includes(lowerQuery) || direction.includes(lowerQuery);
          item.style.display = visible ? "" : "none";
        });
      }

      function getAsoColumns(groups) {
        const preferred = [
          "sequence",
          "mean_splice_delta",
          "recommendation_note",
          "exon_objective_evidence",
          "score",
          "isoform_uniqueness_score",
          "isoform_uniqueness_count",
          "isoform_hit_count",
          "phastcon",
          "phylop",
          "start",
          "end",
          "strand",
          "gc",
          "off_target_gene_count",
          "off_target_exact_count",
          "repeat_hit_count_exact",
          "times_mapped",
          "off_target_1mm_count",
          "off_target_2mm_count",
          "off_target_3mm_count",
          "off_target_expression_weighted_risk",
        ];
        const seen = new Set();
        (groups || []).forEach((group) => {
          (group.asos || []).forEach((aso) => {
            Object.keys(aso || {}).forEach((key) => seen.add(key));
          });
        });
        const ordered = preferred.filter((key) => {
          if (!seen.has(key)) return false;
          if (key === "isoform_hit_count" && seen.has("isoform_uniqueness_count")) {
            return false;
          }
          return true;
        });
        const internalEvidence = new Set(["recommendation_rank_key", "recommendation_basis", "selected_effect_evidence", "splice_site_deltas"]);
        if (seen.has("recommendation_note")) internalEvidence.add("recommendation_status");
        Array.from(seen).sort().forEach((key) => {
          if (!internalEvidence.has(key) && !ordered.includes(key) && ordered.length < 12) {
            ordered.push(key);
          }
        });
        return ordered;
      }

      function renderAsoTables(target) {
        const groups = [];
        const byType = target.recommended_asos_by_type || {};
        Object.keys(byType).forEach((key) => {
          groups.push({
            label: humanizeKey(key),
            outputType: key,
            asos: Array.isArray(byType[key]) ? byType[key] : [],
            safeguards: ((target.safeguards_by_type || {})[key] || []),
          });
        });
        if (!groups.length) {
          groups.push({
            label: "All recommendations",
            outputType: "all",
            asos: Array.isArray(target.recommended_asos) ? target.recommended_asos : [],
            safeguards: Array.isArray(target.safeguards) ? target.safeguards : [],
          });
        }
        return groups.map((group) => {
          const columns = getAsoColumns([group]);
          const guardMarkup = group.safeguards.length
            ? `<div class="chip-row" style="margin-top:10px">${group.safeguards.map((guard) => badge(guard.label || "safeguard", (guard.direction || "").toLowerCase() === "increase" ? "teal" : (guard.direction || "").toLowerCase() === "decrease" ? "rose" : "neutral")).join("")}</div>`
            : "";
          if (!group.asos.length) {
            return `
              <div class="record">
                <div class="record-title">${escapeHtml(group.label)}</div>
                <div class="record-subtitle">No ASOs were attached to this target for this output type.</div>
                ${guardMarkup}
              </div>
            `;
          }
          return `
            <div class="record">
              <div class="record-title">${escapeHtml(group.label)}</div>
              <table class="target-table">
                <thead>
                  <tr>${columns.map((column) => `<th>${escapeHtml(humanizeKey(column))}</th>`).join("")}</tr>
                </thead>
                <tbody>
                  ${group.asos.slice(0, 14).map((aso) => `
                    <tr>
                      ${columns.map((column) => `<td class="${column === "sequence" ? "sequence" : ""}">${escapeHtml(column === "exon_objective_evidence" ? formatExonObjectiveEvidence(aso[column]) : formatValue(aso[column]))}</td>`).join("")}
                    </tr>
                  `).join("")}
                </tbody>
              </table>
              ${group.asos.length > 14 ? `<div class="subtle" style="margin-top:8px">Showing 14 of ${group.asos.length} ASOs.</div>` : ""}
              ${guardMarkup}
            </div>
          `;
        }).join("");
      }

      function renderTargetSummary() {
        const summary = ((reportData.targets || {}).summary || {});
        const metrics = Array.isArray(summary.metrics) ? summary.metrics : [];
        document.getElementById("target-summary-grid").innerHTML = metrics.length
          ? metrics.map((item) => `
              <div class="target-summary-card">
                <div class="metric-label">${escapeHtml(item.label)}</div>
                <div class="metric-value">${escapeHtml(formatValue(item.value))}</div>
                <div class="metric-detail">${escapeHtml(item.detail || "")}</div>
              </div>
            `).join("")
          : '<div class="empty">No target summary available.</div>';
      }

      function renderTargets(filterValue) {
        const targets = (((reportData.targets || {}).items) || []);
        const query = String(filterValue || "").trim().toLowerCase();
        const visible = targets.filter((target) => {
          const haystack = [
            target.label,
            target.display_label,
            target.direction,
            target.site_kind,
            (target.output_types || []).join(" "),
            (target.present_in || []).join(" "),
            (target.absent_in || []).join(" "),
            target.route_goal_label,
          ].join(" ").toLowerCase();
          return !query || haystack.includes(query);
        });
        const root = document.getElementById("target-grid");
        if (!visible.length) {
          root.innerHTML = '<div class="empty">No targets matched this filter.</div>';
          return;
        }
        root.innerHTML = visible.map((target) => {
          const targetHelp = buildTargetHelpText(target);
          const facts = [
            { label: "Direction", value: target.direction },
            { label: "Site kind", value: target.site_kind },
            { label: "Position", value: target.position },
            { label: "Route rank", value: target.route_rank },
            { label: "Route goal", value: target.route_goal_label },
            { label: "Region", value: target.region_start && target.region_end ? `${target.chrom || "chr?"}:${target.region_start}-${target.region_end}` : null },
            { label: "Present in", value: (target.present_in || []).join(", ") },
            { label: "Absent in", value: (target.absent_in || []).join(", ") },
          ].filter((row) => row.value !== null && row.value !== undefined && row.value !== "");
          const chips = []
            .concat((target.output_types || []).map((entry) => badge(humanizeKey(entry), "teal")))
            .concat(target.direction ? [badge(target.direction, target.direction === "decrease" ? "rose" : target.direction === "increase" ? "teal" : "neutral")] : []);
          return `
            <details class="target-card">
              <summary class="target-head">
                <div>
                  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                    <div class="target-title">${escapeHtml(target.display_label || target.label || "Unnamed target")}</div>
                    ${helpIconHtml(targetHelp)}
                  </div>
                  <div class="record-subtitle">${escapeHtml(target.route_rationale || target.rationale || "")}</div>
                </div>
                <div>${chips.join("")}</div>
              </summary>
              <div class="record-body">
                ${facts.length ? renderFacts(facts) : ""}
                ${renderAsoTables(target)}
                <details class="record">
                  <summary class="record-title">Target JSON</summary>
                  <div class="record-body">
                    <pre class="json-block">${escapeHtml(JSON.stringify(target, null, 2))}</pre>
                  </div>
                </details>
              </div>
            </details>
          `;
        }).join("");
      }

      function renderDataViews() {
        const views = (((reportData.data || {}).views) || []);
        const root = document.getElementById("data-grid");
        if (!views.length) {
          root.innerHTML = '<div class="empty">No data snapshots recorded.</div>';
          return;
        }
        root.innerHTML = views.map((view) => `
          <article class="data-card">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
              <div class="collection-title">${escapeHtml(view.title || "View")}</div>
              ${helpIconHtml(view.summary || "")}
            </div>
            ${view.summary ? `<p class="summary-text">${escapeHtml(view.summary)}</p>` : ""}
            ${(view.facts || []).length ? `<div style="margin-top:12px">${renderFacts(view.facts)}</div>` : ""}
            ${Array.isArray(view.rows) && view.rows.length ? `
              <div class="record-list" style="margin-top:12px">
                ${view.rows.map((row) => `
                  <details class="record">
                    <summary>
                      <div>
                        <div class="record-title">${escapeHtml(row.title || "Row")}</div>
                        ${row.subtitle ? `<div class="record-subtitle">${escapeHtml(row.subtitle)}</div>` : ""}
                      </div>
                    </summary>
                    <div class="record-body">
                      ${(row.fields || []).length ? renderFacts(row.fields) : ""}
                    </div>
                  </details>
                `).join("")}
              </div>` : ""}
            ${view.raw ? `
              <details class="record" style="margin-top:12px">
                <summary class="record-title">Compact JSON</summary>
                <div class="record-body">
                  <pre class="json-block">${escapeHtml(JSON.stringify(view.raw, null, 2))}</pre>
                </div>
              </details>` : ""}
          </article>
        `).join("");
      }

      renderHero();
      renderQuickOverview();
      renderWorkflowGraph();
      renderWorkflowPanels();
      setupModalityTab("gapmer", ((reportData.modality_explanations || {}).gapmer) || null);
      setupModalityTab("steric-binding", ((reportData.modality_explanations || {}).steric_binding) || null);
      renderTargetSummary();
      buildTargetMapGroups();
      buildSafeguardMapGroups();
      renderTargets("");
      hydrateStaticHelpIcons();

      const targetSearch = document.getElementById("target-search");
      if (targetSearch) {
        targetSearch.addEventListener("input", (event) => {
          renderTargets(event.target.value);
        });
      }
    </script>
  </body>
</html>
"""
    return html.replace("__REPORT_DATA__", data_json)


def _collect_branch_outcomes(
    report_manifest: Dict[str, Any],
    final_payload: Dict[str, Any],
    structured_goal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_outcomes = report_manifest.get("branch_outcomes")
    outcomes = raw_outcomes if isinstance(raw_outcomes, dict) else {}
    if not outcomes:
        inferred = {}
        if final_payload.get("tiling_results") or structured_goal.get("tiling_results"):
            inferred["steric"] = {"status": "outputs_present", "reason": "Tiling outputs are present."}
        if final_payload.get("gapmer_tiling_results") or structured_goal.get("gapmer_tiling_results"):
            inferred["gapmer"] = {"status": "outputs_present", "reason": "Gapmer outputs are present."}
        outcomes = inferred

    cards: List[Dict[str, Any]] = []
    for branch, payload in sorted(outcomes.items()):
        entry = payload if isinstance(payload, dict) else {}
        status = str(entry.get("status") or "unknown").strip()
        reason = str(entry.get("reason") or "").strip()
        facts = []
        base_path = Path(final_payload.get("report_html_path") or final_payload.get("report_md_path") or ".")
        if base_path.suffix:
            base_path = base_path.parent
        if entry.get("job_count") not in (None, "", [], {}):
            facts.append({"label": "Jobs", "value": entry.get("job_count")})
        if entry.get("output_dir"):
            facts.append(
                {
                    "label": "Output dir",
                    "value": entry.get("output_dir"),
                    "href": _href_for_path(entry.get("output_dir"), base_path),
                }
            )
        cards.append(
            {
                "title": _humanize_key(branch),
                "status": status,
                "tone": _tone_for_status(status),
                "summary": reason or "No branch rationale recorded.",
                "facts": facts,
                "help_text": _build_branch_help_text(branch, entry),
            }
        )
    return cards


def _collect_alerts(
    meta: Dict[str, Any],
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> List[str]:
    alerts: List[str] = []
    for container in (meta, report_manifest, structured_goal, final_payload):
        if not isinstance(container, dict):
            continue
        values = container.get("pipeline_alerts")
        if not isinstance(values, list):
            continue
        for value in values:
            text = _stringify(value)
            if text and text not in alerts:
                alerts.append(text)
    return alerts


def _collect_modalities(
    *,
    targets: Sequence[Dict[str, Any]],
    report_manifest: Dict[str, Any],
    final_payload: Dict[str, Any],
    structured_goal: Dict[str, Any],
) -> List[str]:
    modalities: List[str] = []

    def add(value: Optional[str]) -> None:
        text = _stringify(value)
        if text and text not in modalities:
            modalities.append(text)

    for target in targets:
        for output_type in target.get("output_types") or []:
            add(output_type.replace("_", " "))
    for container in (report_manifest, final_payload, structured_goal):
        if not isinstance(container, dict):
            continue
        if container.get("tiling_results"):
            add("steric tiling")
        if container.get("gapmer_tiling_results"):
            add("gapmer tiling")
        if container.get("strategy_routing"):
            add("splice routing")
        if container.get("rbp_binding_result"):
            add("rbp dataset")
        if container.get("motif_tiling_result"):
            add("motif targeting")
        if container.get("patient_reference"):
            add("patient reference")
        if container.get("snapgene_annotation"):
            add("snapgene context")
    return modalities


def _coerce_bool_for_report(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_target_region_for_report(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    allowed = {"utr3", "utr5", "utrs", "translation_start", "cds"}
    return text if text in allowed else None


def _format_target_region_label(value: Any) -> Optional[str]:
    region = _normalize_target_region_for_report(value)
    mapping = {
        "utr3": "3' UTR",
        "utr5": "5' UTR",
        "utrs": "UTRs",
        "translation_start": "translation start",
        "cds": "CDS",
    }
    if region in mapping:
        return mapping[region]
    return _stringify(value)


def _collect_design_jobs_for_report(
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    modality: str,
) -> List[Dict[str, Any]]:
    key = f"{modality}_design_jobs"
    for container in (structured_goal, final_payload):
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, list) and value:
            return [job for job in value if isinstance(job, dict)]
    return []


def _collect_result_block_for_report(
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    modality: str,
) -> Dict[str, Any]:
    key = "gapmer_tiling_results" if modality == "gapmer" else "tiling_results"
    for container in (structured_goal, final_payload):
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _collect_result_map_for_report(
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    modality: str,
) -> Dict[str, Dict[str, Any]]:
    key = "gapmer_tiling_results_by_job" if modality == "gapmer" else "steric_tiling_results_by_job"
    for container in (structured_goal, final_payload):
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, dict) and value:
            return {str(job_id): result for job_id, result in value.items() if isinstance(result, dict)}
    return {}


def _first_job_result_for_report(
    results_by_job: Dict[str, Dict[str, Any]],
    jobs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    for job in jobs:
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            continue
        result = results_by_job.get(job_id)
        if isinstance(result, dict) and result:
            return result
    for result in results_by_job.values():
        if isinstance(result, dict) and result:
            return result
    return {}


def _resolve_uniqueness_target_label_for_report(
    structured_goal: Dict[str, Any],
    iso_label_primary: Optional[str],
) -> Optional[str]:
    scope = structured_goal.get("target_scope") or structured_goal.get("isoform_scope")
    if isinstance(scope, str):
        scope_norm = scope.strip().lower()
        if scope_norm in {"isoform_specific", "isoform", "specific"}:
            return iso_label_primary
        if scope_norm in {"gene_wide", "all_isoforms", "all"}:
            return None

    lock_requested = bool(structured_goal.get("transcript_lock"))
    requested_labels = structured_goal.get("requested_isoform_labels")
    single_requested = isinstance(requested_labels, list) and len(requested_labels) == 1
    if lock_requested or single_requested:
        return iso_label_primary
    return None


def _resolve_gapmer_uniqueness_context_for_report(
    structured_goal: Dict[str, Any],
    iso_label_primary: Optional[str],
) -> Dict[str, Any]:
    target_label = _resolve_uniqueness_target_label_for_report(structured_goal, iso_label_primary)
    beneficial_labels: List[str] = []
    pathogenic_labels: List[str] = []
    objectives = structured_goal.get("splice_isoform_objectives")
    if isinstance(objectives, list):
        for objective in objectives:
            if not isinstance(objective, dict):
                continue
            isoform_label = str(
                objective.get("isoform_label")
                or objective.get("transcript_label")
                or objective.get("transcript_id")
                or ""
            ).strip()
            if not isoform_label:
                continue
            direction = str(
                objective.get("desired_abundance")
                or objective.get("direction")
                or objective.get("goal")
                or ""
            ).strip().lower()
            role = str(objective.get("role") or "").strip().lower()
            if direction == "increase" or role == "beneficial":
                beneficial_labels.append(isoform_label)
            if direction == "decrease" or role == "pathogenic":
                pathogenic_labels.append(isoform_label)

    if not target_label and iso_label_primary and iso_label_primary in pathogenic_labels:
        target_label = iso_label_primary
    elif not target_label and iso_label_primary and not beneficial_labels and not pathogenic_labels:
        target_label = iso_label_primary

    beneficial_labels = list(dict.fromkeys(label for label in beneficial_labels if label))
    pathogenic_labels = list(dict.fromkeys(label for label in pathogenic_labels if label))
    target_labels = [target_label] if target_label else []
    mode = "gene_wide"
    if target_labels and beneficial_labels:
        mode = "pathogenic_preferred"
    elif target_labels:
        mode = "target_isoform_specific"
    return {
        "mode": mode,
        "target_labels": target_labels,
        "beneficial_labels": beneficial_labels,
        "pathogenic_labels": pathogenic_labels,
        "target_label": target_label,
    }


def _artifact_entry_for_report(label: str, path_value: Any, run_path: Path) -> Optional[Dict[str, Any]]:
    href = _href_for_path(path_value, run_path)
    if not href:
        return None
    return {"label": label, "href": href}


def _dedupe_artifact_entries(entries: Sequence[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        href = str(entry.get("href") or "").strip()
        if not href or href in seen:
            continue
        seen.add(href)
        deduped.append(entry)
    return deduped


def _build_gapmer_explanation(
    run_path: Path,
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    gapmer_jobs = _collect_design_jobs_for_report(structured_goal, final_payload, "gapmer")
    gapmer_results = _collect_result_block_for_report(structured_goal, final_payload, "gapmer")
    if not gapmer_jobs and not gapmer_results:
        return None

    raw_branch_outcomes = report_manifest.get("branch_outcomes") if isinstance(report_manifest.get("branch_outcomes"), dict) else {}
    branch_entry = raw_branch_outcomes.get("gapmer")
    if not isinstance(branch_entry, dict):
        branch_entry = {}
    first_job = gapmer_jobs[0] if gapmer_jobs else {}
    job_results = _collect_result_map_for_report(structured_goal, final_payload, "gapmer")
    first_job_result = _first_job_result_for_report(job_results, gapmer_jobs)
    ranking = gapmer_results.get("selection_ranking") or first_job_result.get("selection_ranking")
    ranking_bullets = recorded_ranking_explanation(ranking)
    primary_target = _stringify(
        first_job.get("isoform_label")
        or gapmer_results.get("primary_isoform")
        or gapmer_results.get("target_isoform")
        or first_job.get("target_label"),
        max_len=1000,
    )
    uniqueness_context = _resolve_gapmer_uniqueness_context_for_report(structured_goal, primary_target)
    uniqueness_mode = str(
        (first_job.get("selection_meta") or {}).get("gapmer_uniqueness_mode")
        or gapmer_results.get("mode")
        or uniqueness_context.get("mode")
        or "gene_wide"
    ).strip().lower()
    beneficial_labels = uniqueness_context.get("beneficial_labels") or []
    branch_status = _humanize_key(branch_entry.get("status") or ("outputs_present" if gapmer_results else "requested"))
    branch_reason = _stringify(branch_entry.get("reason"), max_len=2000)
    job_objective = _stringify(first_job.get("objective"), max_len=2000)
    job_note = _first_nonempty_lines(first_job.get("notes"), limit=3)

    if uniqueness_mode == "pathogenic_preferred" and primary_target:
        summary = (
            f"Gapmer was run because {primary_target} was explicitly framed as the isoform to reduce. "
            "That gives the pipeline a direct transcript-lowering branch alongside any steric splice-redirection branch."
        )
    elif str(first_job.get("intent") or structured_goal.get("intent") or "").strip().lower() == "knockdown" and primary_target:
        summary = (
            f"Gapmer was run because the request included direct knockdown of {primary_target}, "
            "not only splice redirection."
        )
    elif primary_target:
        summary = f"Gapmer was run because the pipeline had a transcript-level knockdown branch targeting {primary_target}."
    else:
        summary = "Gapmer was run because this request needed a transcript-lowering branch rather than only steric splice modulation."

    why_bullets: List[str] = []
    if job_objective:
        why_bullets.append(job_objective)
    if beneficial_labels:
        why_bullets.append(
            "Beneficial isoforms were explicitly protected in uniqueness scoring: "
            + ", ".join(str(label) for label in beneficial_labels[:4])
            + ("." if len(beneficial_labels) <= 4 else ", plus additional protected isoforms.")
        )
    if branch_reason:
        why_bullets.append(branch_reason)

    uniqueness_summary = ISOFORM_SCORE_DEFINITION

    uniqueness_bullets = [
        "The isoform class and score describe sharing within the recorded transcript catalog.",
        "These annotations are components of selection, not a complete description of the final candidate order.",
        "Read selection_rank and the recorded final selection criteria for the delivered recommendation order.",
    ]
    if uniqueness_mode == "gene_wide":
        uniqueness_bullets = [
            "For gene-wide knockdown, higher coverage of the recorded target-gene transcript catalog is preferred.",
            "Target-gene isoform hit counts measure breadth within that catalog; they are not other-gene off-target counts.",
            "Coverage is a sequence property, not a measurement of knockdown or potency.",
        ]

    conservation_summary = (
        "Conservation is calculated by projecting each tile's cDNA span onto genomic exon segments of the primary transcript, "
        "then averaging PhastCons and PhyloP values across the covered genomic bases."
    )
    conservation_bullets = [
        "The row-level fields written back to the tile table are Phastcon and PhyloP.",
        "Conservation supplies sequence context; its position in the final ordering is specified by the recorded selection criteria below.",
    ]
    conservation = gapmer_results.get("conservation_annotation") if isinstance(gapmer_results.get("conservation_annotation"), dict) else {}
    if not conservation:
        conservation = first_job_result.get("conservation_annotation") if isinstance(first_job_result.get("conservation_annotation"), dict) else {}
    if conservation.get("annotated"):
        conservation_bullets.insert(0, f"This run annotated conservation for {conservation.get('annotated')} tile(s).")
    else:
        conservation_bullets.insert(0, "This run did not retain an aggregate conservation summary, but the method above is what the gapmer branch uses when conservation annotation is available.")
    phast_path = conservation.get("phastcons")
    if phast_path:
        conservation_bullets.append(f"PhastCons source: {Path(phast_path).name}.")
    phylop_path = conservation.get("phylop")
    if phylop_path:
        conservation_bullets.append(f"PhyloP source: {Path(phylop_path).name}.")

    initial_filtering = gapmer_results.get("initial_filtering") if isinstance(gapmer_results.get("initial_filtering"), dict) else {}
    if not initial_filtering:
        initial_filtering = first_job_result.get("initial_filtering") if isinstance(first_job_result.get("initial_filtering"), dict) else {}
    passed = initial_filtering.get("passed") if isinstance(initial_filtering.get("passed"), dict) else {}
    constraint_filtering = gapmer_results.get("constraint_filtering") if isinstance(gapmer_results.get("constraint_filtering"), dict) else {}
    if not constraint_filtering:
        constraint_filtering = first_job_result.get("constraint_filtering") if isinstance(first_job_result.get("constraint_filtering"), dict) else {}

    prioritize_bullets: List[str] = []
    passed_count = passed.get("count") if isinstance(passed, dict) else None
    total_rows = initial_filtering.get("total_rows")
    if passed_count not in (None, "") and total_rows not in (None, ""):
        prioritize_bullets.append(f"Start with the {passed_count} of {total_rows} tiles that survived the initial GC and homopolymer filters in this run.")
    prioritize_bullets.extend([
        "Interpret gene-wide isoform scores as target-gene catalog coverage; sharing across the intended gene is desired."
        if uniqueness_mode == "gene_wide" else
        "Interpret target-specific isoform scores as selectivity within the recorded catalog, not as whole-gene coverage or potency.",
        "Use conservation as supporting sequence context, not as evidence of potency.",
        "Inspect other-gene and genomic-site matches using the recorded count units, mismatch budgets and expression context. Missing or unsearched bins are unknown, and a zero in one count unit does not establish transcriptome-wide specificity.",
        "Before synthesis, still inspect off-target annotations, chemistry feasibility, and whether the tile is shared with any isoform you are trying to preserve.",
    ])
    if isinstance(constraint_filtering, dict) and constraint_filtering.get("kept_count") not in (None, ""):
        prioritize_bullets.append(
            f"Constraint filtering kept {constraint_filtering.get('kept_count')} row(s) and removed {constraint_filtering.get('removed_count') or 0} row(s) after uniqueness scoring."
        )

    facts = [
        {"label": "Branch status", "value": branch_status},
        {"label": "Gapmer jobs", "value": len(gapmer_jobs) or None},
        {"label": "Primary knockdown target", "value": primary_target},
        {"label": "Uniqueness mode", "value": _humanize_key(uniqueness_mode)},
        {"label": "Protected isoforms", "value": ", ".join(str(label) for label in beneficial_labels[:4]) if beneficial_labels else None},
        {"label": "Ranked candidates", "value": gapmer_results.get("row_count") or passed_count},
        {"label": "Conservation annotated", "value": conservation.get("annotated")},
    ]
    facts = [fact for fact in facts if _is_nonempty(fact.get("value"))]

    artifacts = _dedupe_artifact_entries([
        _artifact_entry_for_report("Ranked CSV", gapmer_results.get("uniqueness_csv") or first_job_result.get("uniqueness_csv"), run_path),
        _artifact_entry_for_report("Scored tiles JSON", gapmer_results.get("scored_tiles_path") or first_job_result.get("scored_tiles_path"), run_path),
        _artifact_entry_for_report("Raw tiles JSON", gapmer_results.get("json_path") or first_job_result.get("json_path"), run_path),
        _artifact_entry_for_report("Initial-filter tiles", passed.get("path") or (first_job_result.get("initial_filtering") or {}).get("passed", {}).get("path"), run_path),
    ])

    return {
        "title": "Why gapmer ran",
        "summary": summary,
        "note": job_note or branch_reason or "",
        "facts": facts,
        "artifacts": artifacts,
        "sections": [
            {
                "title": "Why this branch ran",
                "summary": summary,
                "bullets": why_bullets,
            },
            {
                "title": "How uniqueness was calculated",
                "summary": uniqueness_summary,
                "bullets": uniqueness_bullets,
            },
            {
                "title": "How conservation was calculated",
                "summary": conservation_summary,
                "bullets": conservation_bullets,
            },
            {
                "title": "How the final candidates were ranked",
                "summary": (
                    "The recorded criteria below are applied in order: each later criterion only breaks ties left by earlier criteria. Read selection_rank for the resulting order. These computational proxies do not measure efficacy."
                    if ranking_bullets else
                    "A complete final ranking policy was not recorded in this result. Consult the delivered rank fields and component values; an ordering cannot be reconstructed from the uniqueness score alone."
                ),
                "bullets": (ranking_bullets or []) + (
                    [str(ranking["tie_policy"])] if ranking_bullets and ranking.get("tie_policy") else []
                ),
            },
            {
                "title": "How to prioritize ASOs",
                "summary": "Use the recorded final selection order together with its component values, eligibility filters and scientific limitations.",
                "bullets": prioritize_bullets,
            },
        ],
    }


def _build_binding_steric_explanation(
    run_path: Path,
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    steric_jobs = _collect_design_jobs_for_report(structured_goal, final_payload, "steric")
    binding_jobs = [
        job
        for job in steric_jobs
        if str(job.get("intent") or structured_goal.get("intent") or "").strip().lower() == "binding"
        and _coerce_bool_for_report(job.get("run_splice_eval")) is False
        and _coerce_bool_for_report(job.get("run_splice_router")) is False
    ]
    steric_results = _collect_result_block_for_report(structured_goal, final_payload, "steric")
    if not binding_jobs:
        intent = str(structured_goal.get("intent") or "").strip().lower()
        if intent != "binding":
            return None
        if _coerce_bool_for_report(structured_goal.get("run_splice_eval")) not in {False, None}:
            return None
        if _coerce_bool_for_report(structured_goal.get("run_splice_router")) not in {False, None}:
            return None
    if not binding_jobs and not steric_results and str(structured_goal.get("intent") or "").strip().lower() != "binding":
        return None

    raw_branch_outcomes = report_manifest.get("branch_outcomes") if isinstance(report_manifest.get("branch_outcomes"), dict) else {}
    branch_entry = raw_branch_outcomes.get("steric") if isinstance(raw_branch_outcomes, dict) else {}
    first_job = binding_jobs[0] if binding_jobs else (steric_jobs[0] if steric_jobs else {})
    job_results = _collect_result_map_for_report(structured_goal, final_payload, "steric")
    first_job_result = _first_job_result_for_report(job_results, binding_jobs or steric_jobs)
    target_regions = [
        label
        for label in (_format_target_region_label(job.get("target_region")) for job in (binding_jobs or steric_jobs))
        if label
    ]
    target_regions = list(dict.fromkeys(target_regions))
    primary_transcript = _resolve_uniqueness_target_label_for_report(
        structured_goal,
        _stringify(first_job.get("isoform_label") or first_job.get("transcript_id") or structured_goal.get("transcript_id"), max_len=1000),
    )
    branch_status = _humanize_key(branch_entry.get("status") or ("outputs_present" if steric_results else "requested"))
    branch_reason = _stringify(branch_entry.get("reason"), max_len=2000)
    job_objective = _stringify(first_job.get("objective"), max_len=2000)
    job_note = _first_nonempty_lines(first_job.get("notes"), limit=3)

    region_text = ", ".join(target_regions[:3]) if target_regions else "the requested binding region"
    summary = (
        f"Steric hindrance was run as a binding-only branch because this request targeted {region_text} "
        "without intended splice modulation. The pipeline kept steric tiling but disabled splice evaluation and splice routing for these jobs."
    )

    uniqueness_summary = ISOFORM_SCORE_DEFINITION

    why_bullets: List[str] = []
    if job_objective:
        why_bullets.append(job_objective)
    if target_regions:
        why_bullets.append("Binding-only steric jobs were requested for: " + ", ".join(target_regions[:4]) + ".")
    if branch_reason:
        why_bullets.append(branch_reason)

    uniqueness_bullets = [
        "Binding-only steric scoring reuses the isoform uniqueness sorter used for transcript-tiling branches.",
        "Read the recorded selection rank alongside its sequence, structure, specificity and expression evidence; missing measurements are not measured zeros.",
        "Because splice evaluation is disabled here, there is no route-rank layer on top of the uniqueness ranking."
    ]

    conservation_summary = (
        "When conservation tracks are configured, steric tiling annotates genomic candidate intervals with "
        "PhastCons and PhyloP values. These annotations are retained in the tile and candidate tables; "
        "they are distinct from transcript-isoform coverage and from measured ASO activity."
    )
    conservation_bullets = [
        "Phastcon and PhyloP are the row-level annotation fields. The filter summary can display their configured threshold tests even when conservation is not a hard candidate-retention constraint.",
        "Track averages use the available measured bases. The pc_covered_bases and ph_covered_bases fields, compared with conservation_interval_bases, identify partial or absent coverage. Missing annotations cannot establish a threshold pass; availability is shown separately in the filter summary.",
        "The candidate selection sorter can use conservation as a late tie-break after its other sequence and specificity criteria. Conservation alone does not establish regulatory-site occupancy or functional blockade.",
        "Use the saved candidate values and filtering configuration to determine what was measured and required in this run; a missing annotation must not be interpreted as measured lack of conservation."
    ]

    initial_filtering = steric_results.get("initial_filtering") if isinstance(steric_results.get("initial_filtering"), dict) else {}
    if not initial_filtering:
        initial_filtering = first_job_result.get("initial_filtering") if isinstance(first_job_result.get("initial_filtering"), dict) else {}
    uniqueness_filtering = steric_results.get("uniqueness_constraint_filtering") if isinstance(steric_results.get("uniqueness_constraint_filtering"), dict) else {}
    if not uniqueness_filtering:
        uniqueness_filtering = first_job_result.get("uniqueness_constraint_filtering") if isinstance(first_job_result.get("uniqueness_constraint_filtering"), dict) else {}
    passed = initial_filtering.get("passed") if isinstance(initial_filtering.get("passed"), dict) else {}

    prioritize_bullets: List[str] = []
    passed_count = passed.get("count") if isinstance(passed, dict) else None
    total_rows = initial_filtering.get("total_rows")
    if passed_count not in (None, "") and total_rows not in (None, ""):
        prioritize_bullets.append(f"Start with the {passed_count} of {total_rows} tiles that survived the initial sequence filters in this run.")
    prioritize_bullets.extend([
        "Prefer candidates that overlap the requested binding region cleanly and remain highly unique in the ranked CSV.",
        "If the run is transcript-specific, prefer rows unique to that transcript; otherwise prefer the rows with the strongest gene-wide coverage score that still fit the requested region.",
        "Use shared-hit counts, GC, exact/near-match off-target bins, and any target-feature overlap annotations as secondary triage signals after uniqueness.",
        "Compare off-target counts within their recorded search scope and count units. Unsearched or censored bins and incomplete expression aggregates remain unknown.",
        "Because there is no splice-routing layer here, the ranked CSV itself is the primary prioritization view for this branch.",
    ])
    if isinstance(uniqueness_filtering, dict) and uniqueness_filtering.get("kept_count") not in (None, ""):
        prioritize_bullets.append(
            f"Post-uniqueness constraint filtering kept {uniqueness_filtering.get('kept_count')} row(s) and removed {uniqueness_filtering.get('removed_count') or 0} row(s)."
        )

    facts = [
        {"label": "Branch status", "value": branch_status},
        {"label": "Binding-only steric jobs", "value": len(binding_jobs) or len(steric_jobs) or None},
        {"label": "Requested region(s)", "value": region_text if target_regions else None},
        {"label": "Splice evaluation", "value": "disabled"},
        {"label": "Splice routing", "value": "disabled"},
        {"label": "Transcript-specific scoring", "value": primary_transcript or None},
        {"label": "Ranked candidates", "value": steric_results.get("row_count") or passed_count},
    ]
    facts = [fact for fact in facts if _is_nonempty(fact.get("value"))]

    artifacts = _dedupe_artifact_entries([
        _artifact_entry_for_report("Ranked CSV", steric_results.get("uniqueness_csv") or first_job_result.get("uniqueness_csv"), run_path),
        _artifact_entry_for_report("Scored uniqueness JSON", steric_results.get("uniqueness_scored_path") or first_job_result.get("uniqueness_scored_path"), run_path),
        _artifact_entry_for_report("Raw tiles JSON", steric_results.get("json_path") or first_job_result.get("json_path"), run_path),
        _artifact_entry_for_report("Initial-filter tiles", passed.get("path") or (first_job_result.get("initial_filtering") or {}).get("passed", {}).get("path"), run_path),
    ])

    return {
        "title": "Why binding-only steric hindrance ran",
        "summary": summary,
        "note": job_note or branch_reason or "",
        "facts": facts,
        "artifacts": artifacts,
        "sections": [
            {
                "title": "Why this branch ran",
                "summary": summary,
                "bullets": why_bullets,
            },
            {
                "title": "How uniqueness was calculated",
                "summary": uniqueness_summary,
                "bullets": uniqueness_bullets,
            },
            {
                "title": "How conservation is handled",
                "summary": conservation_summary,
                "bullets": conservation_bullets,
            },
            {
                "title": "How to prioritize ASOs",
                "summary": "Read the binding-only steric outputs as ranked binding hypotheses; cellular occupancy has not been measured.",
                "bullets": prioritize_bullets,
            },
        ],
    }


def _build_modality_explanations(
    *,
    run_path: Path,
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> Dict[str, Any]:
    explanations: Dict[str, Any] = {}
    gapmer = _build_gapmer_explanation(run_path, report_manifest, structured_goal, final_payload)
    if gapmer:
        explanations["gapmer"] = gapmer
    steric_binding = _build_binding_steric_explanation(run_path, report_manifest, structured_goal, final_payload)
    if steric_binding:
        explanations["steric_binding"] = steric_binding
    return explanations


def _build_key_metrics(
    *,
    artifact_inventory: Dict[str, Any],
    branch_outcomes: Sequence[Dict[str, Any]],
    core_payload: Dict[str, Any],
    qa_history: Sequence[Dict[str, Any]],
    strategy_routing: Dict[str, Any],
    target_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    completed_branches = sum(
        1
        for branch in branch_outcomes
        if str(branch.get("status") or "").lower() in {"outputs_present", "ready", "completed"}
    )
    trace = (((core_payload.get("llm_rationale") or {}).get("agent_trace")) or [])
    goals = strategy_routing.get("goals") or []
    metrics = [
        {
            "label": "Artifacts",
            "value": artifact_inventory.get("total_files") or 0,
            "detail": "Files discovered under this run directory.",
        },
        {
            "label": "Branches",
            "value": completed_branches,
            "detail": f"{len(branch_outcomes)} branch outcome(s) summarized.",
        },
        {
            "label": "Targets",
            "value": target_summary.get("target_count") or 0,
            "detail": "Unique target sites surfaced in the HTML payload.",
        },
        {
            "label": "Clarifications",
            "value": len(qa_history or []),
            "detail": "Follow-up questions answered during goal intake.",
        },
        {
            "label": "Trace steps",
            "value": len(trace),
            "detail": "Captured LLM reasoning steps from splice strategy analysis.",
        },
        {
            "label": "Routed goals",
            "value": len(goals),
            "detail": "Ranked strategy routes retained from the router.",
        },
    ]
    return metrics


def _build_target_summary(targets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output_type_counts: Dict[str, int] = {}
    direction_counts: Dict[str, int] = {}
    safeguards = 0
    recommended = 0
    for target in targets:
        for output_type in target.get("output_types") or []:
            output_type_counts[output_type] = output_type_counts.get(output_type, 0) + 1
        direction = _stringify(target.get("direction"))
        if direction:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        safeguards += len(target.get("safeguards") or [])
        recommended += len(target.get("recommended_asos") or [])

    metrics = [
        {
            "label": "Target sites",
            "value": len(targets),
            "detail": "Unique labels after merging splice, steric, and gapmer views.",
        },
        {
            "label": "Output types",
            "value": len(output_type_counts),
            "detail": ", ".join(
                f"{_humanize_key(key)} ({value})" for key, value in sorted(output_type_counts.items())
            ) or "None",
        },
        {
            "label": "Directions",
            "value": len(direction_counts),
            "detail": ", ".join(f"{key} ({value})" for key, value in sorted(direction_counts.items())) or "None",
        },
        {
            "label": "Safeguards",
            "value": safeguards,
            "detail": "Unique safeguard sites attached to target labels.",
        },
        {
            "label": "Recommendations",
            "value": recommended,
            "detail": "Total recommended ASOs across target views.",
        },
    ]
    return {
        "target_count": len(targets),
        "metrics": metrics,
    }


def _build_decision_timeline(
    *,
    core_payload: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    report_manifest: Dict[str, Any],
    qa_history: Sequence[Dict[str, Any]],
    artifact_inventory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    meta = core_payload.get("meta") if isinstance(core_payload.get("meta"), dict) else {}

    def add(
        title: str,
        summary: str,
        *,
        facts: Optional[List[Dict[str, Any]]] = None,
        bullets: Optional[List[str]] = None,
        badges: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        if not summary and not facts and not bullets:
            return
        timeline.append(
            {
                "title": title,
                "summary": summary,
                "facts": facts or [],
                "bullets": bullets or [],
                "badges": badges or [],
            }
        )

    add(
        "Goal intake",
        _stringify(meta.get("user_request")) or "No user request captured.",
        facts=[
            {"label": "Gene", "value": structured_goal.get("gene_symbol")},
            {"label": "Intent", "value": structured_goal.get("intent")},
            {"label": "Species", "value": structured_goal.get("species")},
            {"label": "Target context", "value": structured_goal.get("target_context")},
        ],
        badges=[{"text": "parsed", "tone": "teal"}],
    )

    if qa_history:
        for index, pair in enumerate(qa_history, start=1):
            question = _stringify(pair.get("question"))
            answer = _stringify(pair.get("answer"))
            add(
                f"Clarification {index}",
                question or "Clarification captured without a question.",
                facts=[{"label": "Answer", "value": answer}],
                badges=[{"text": "qa", "tone": "rust"}],
            )

    gene_resolution = structured_goal.get("gene_id_resolution") or final_payload.get("gene_id_resolution")
    if isinstance(gene_resolution, dict):
        add(
            "Gene resolution",
            _stringify(gene_resolution.get("description"))
            or "Gene ID was resolved for downstream agents.",
            facts=[
                {"label": "Gene ID", "value": gene_resolution.get("gene_id") or structured_goal.get("gene_id")},
                {"label": "Method", "value": gene_resolution.get("method")},
                {"label": "Display name", "value": gene_resolution.get("display_name")},
            ],
            badges=[{"text": "ensembl", "tone": "teal"}],
        )

    isoforms = structured_goal.get("isoform_transcripts")
    if isinstance(isoforms, dict) and isoforms:
        bullets = []
        for label, payload in sorted(isoforms.items()):
            if not isinstance(payload, dict):
                continue
            transcript_id = _stringify(payload.get("transcript_id"))
            if not transcript_id:
                continue
            match_info = payload.get("match_info") if isinstance(payload.get("match_info"), dict) else {}
            method = _stringify(match_info.get("method"))
            confidence = _stringify(match_info.get("confidence"))
            line = f"{label}: {transcript_id}"
            details = ", ".join(part for part in [method, confidence] if part)
            if details:
                line += f" ({details})"
            bullets.append(line)
        add(
            "Transcript resolution",
            "Isoform labels were mapped to transcripts for downstream targeting.",
            bullets=bullets[:10],
            badges=[{"text": "transcripts", "tone": "teal"}],
        )

    strategy = structured_goal.get("splice_strategy_analysis") or final_payload.get("splice_strategy_analysis")
    if isinstance(strategy, dict):
        add(
            "Splice strategy",
            _first_nonempty_lines(strategy.get("analysis"), limit=3)
            or "Splice strategy analysis was generated.",
            facts=[
                {"label": "Model", "value": strategy.get("model")},
                {"label": "Unique splice sites", "value": strategy.get("unique_splice_site_count")},
                {"label": "Fallback sites", "value": strategy.get("additional_site_count")},
            ],
            badges=[{"text": "llm", "tone": "teal"}],
        )

    routing = core_payload.get("strategy_routing") if isinstance(core_payload.get("strategy_routing"), dict) else {}
    goals = routing.get("goals") if isinstance(routing.get("goals"), list) else []
    if goals:
        top_goals = []
        for goal in goals[:5]:
            goal_label = _stringify(goal.get("goal_label"))
            rationale = _stringify(goal.get("rationale"))
            if goal_label:
                top_goals.append(f"{goal_label}: {rationale}" if rationale else goal_label)
        planner = routing.get("planner_plan") if isinstance(routing.get("planner_plan"), dict) else {}
        ranking = planner.get("ranking") if isinstance(planner.get("ranking"), dict) else {}
        add(
            "Route ranking",
            "Strategy routes were ranked for candidate selection.",
            facts=[
                {"label": "Goal count", "value": len(goals)},
                {"label": "Model", "value": ranking.get("model")},
                {"label": "Summary", "value": ranking.get("summary")},
            ],
            bullets=top_goals,
            badges=[{"text": "router", "tone": "rust"}],
        )

    chemistry_profiles = report_manifest.get("chemistry_profiles") or final_payload.get("chemistry_profiles") or structured_goal.get("chemistry_profiles")
    by_modality = chemistry_profiles.get("by_modality") if isinstance(chemistry_profiles, dict) and isinstance(chemistry_profiles.get("by_modality"), dict) else {}
    if by_modality:
        bullets = []
        mixed_modalities = 0
        for modality_name, summary in by_modality.items():
            if not isinstance(summary, dict):
                continue
            recommendation = summary.get("recommendation") if isinstance(summary.get("recommendation"), dict) else None
            if isinstance(recommendation, dict):
                chem_label = recommendation.get("CHEMISTRY") or recommendation.get("chemistry")
                platform = recommendation.get("platform")
                length = recommendation.get("length")
                bullet = f"{modality_name}: {chem_label}" if chem_label else str(modality_name)
                if length not in (None, ""):
                    bullet += f" ({length} nt)"
                if platform:
                    bullet += f" [{platform}]"
                bullets.append(bullet)
            else:
                mixed_modalities += 1
                program_count = summary.get("program_count")
                bullets.append(f"{modality_name}: mixed chemistry across {program_count or '?'} program(s)")
        add(
            "Chemistry selection",
            "Chemistry was resolved per request type.",
            facts=[
                {"label": "Request types", "value": len(by_modality)},
                {"label": "Mixed types", "value": mixed_modalities or None},
            ],
            bullets=bullets[:8],
            badges=[{"text": "chemistry", "tone": "teal"}],
        )
    else:
        chemistry = structured_goal.get("chemistry") or final_payload.get("chemistry")
        recommendation = chemistry.get("recommendation") if isinstance(chemistry, dict) and isinstance(chemistry.get("recommendation"), dict) else chemistry
        if isinstance(recommendation, dict):
            add(
                "Chemistry selection",
                _stringify(recommendation.get("rationale")) or "Chemistry was selected for the output branch.",
                facts=[
                    {"label": "Chemistry", "value": recommendation.get("CHEMISTRY") or recommendation.get("chemistry")},
                    {"label": "Platform", "value": recommendation.get("platform")},
                    {"label": "Length", "value": recommendation.get("length")},
                ],
                badges=[{"text": "chemistry", "tone": "teal"}],
            )

    for key, title in (
        ("rbp_binding_result", "RBP dataset"),
        ("mirna_binding_result", "miRNA target-site selection"),
        ("motif_tiling_result", "Motif targeting"),
        ("patient_reference", "Patient reference"),
        ("snapgene_annotation", "SnapGene context"),
    ):
        payload = final_payload.get(key) or structured_goal.get(key)
        if not isinstance(payload, dict):
            continue
        facts = _extract_simple_facts(payload, Path(core_payload.get("meta", {}).get("run_dir") or "."), limit=4)
        add(
            title,
            _guess_collection_summary(key, payload),
            facts=facts,
            badges=[{"text": "context", "tone": "neutral"}],
        )

    branch_outcomes = _collect_branch_outcomes(report_manifest, final_payload, structured_goal)
    if branch_outcomes:
        add(
            "Execution outcomes",
            "Branch execution was summarized into report outputs.",
            bullets=[
                f"{entry['title']}: {entry['status']}"
                for entry in branch_outcomes
                if entry.get("title") and entry.get("status")
            ],
            facts=[{"label": "Artifacts", "value": artifact_inventory.get("total_files")}],
            badges=[{"text": "outputs", "tone": "rust"}],
        )

    return timeline


def _build_output_collections(
    *,
    run_path: Path,
    core_payload: Dict[str, Any],
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    collections: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    sources = [
        ("strategy_routing", core_payload.get("strategy_routing")),
        ("splice_strategy_analysis", final_payload.get("splice_strategy_analysis") or structured_goal.get("splice_strategy_analysis")),
        ("splice_site_plan", final_payload.get("splice_site_plan") or structured_goal.get("splice_site_plan")),
        ("splice_site_comparisons", final_payload.get("splice_site_comparisons") or structured_goal.get("splice_site_comparisons")),
        ("splice_isoform_objectives", final_payload.get("splice_isoform_objectives") or structured_goal.get("splice_isoform_objectives")),
        ("steric_design_jobs", report_manifest.get("steric_jobs") or final_payload.get("steric_design_jobs") or structured_goal.get("steric_design_jobs")),
        ("tiling_results", final_payload.get("tiling_results") or structured_goal.get("tiling_results") or report_manifest.get("tiling")),
        ("steric_tiling_results_by_job", final_payload.get("steric_tiling_results_by_job") or structured_goal.get("steric_tiling_results_by_job")),
        ("gapmer_design_jobs", report_manifest.get("gapmer_jobs") or final_payload.get("gapmer_design_jobs") or structured_goal.get("gapmer_design_jobs")),
        ("gapmer_tiling_results", final_payload.get("gapmer_tiling_results") or structured_goal.get("gapmer_tiling_results") or report_manifest.get("gapmer_tiling")),
        ("gapmer_tiling_results_by_job", final_payload.get("gapmer_tiling_results_by_job") or structured_goal.get("gapmer_tiling_results_by_job")),
        ("rbp_binding_result", final_payload.get("rbp_binding_result") or structured_goal.get("rbp_binding_result")),
        ("mirna_binding_result", final_payload.get("mirna_binding_result") or structured_goal.get("mirna_binding_result")),
        ("motif_tiling_result", final_payload.get("motif_tiling_result") or structured_goal.get("motif_tiling_result")),
        ("patient_reference", final_payload.get("patient_reference") or structured_goal.get("patient_reference")),
        ("snapgene_annotation", final_payload.get("snapgene_annotation") or structured_goal.get("snapgene_annotation")),
        ("snapgene_annotations", final_payload.get("snapgene_annotations") or structured_goal.get("snapgene_annotations")),
        ("parameter_constraints", final_payload.get("parameter_constraints") or structured_goal.get("parameter_constraints")),
        ("goal_schema_validation", final_payload.get("goal_schema_validation") or structured_goal.get("goal_schema_validation")),
        ("design_programs", final_payload.get("design_programs") or structured_goal.get("design_programs")),
        ("chemistry_profiles", report_manifest.get("chemistry_profiles") or final_payload.get("chemistry_profiles") or structured_goal.get("chemistry_profiles")),
    ]

    for key, value in sources:
        collection = _make_collection(key=key, value=value, run_path=run_path)
        if collection:
            collections.append(collection)
            seen.add(key)

    for container in (final_payload, structured_goal):
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            if key in seen or key in _FALLBACK_SKIP_KEYS or not _is_nonempty(value):
                continue
            collection = _make_collection(key=key, value=value, run_path=run_path)
            if collection:
                collections.append(collection)
                seen.add(key)

    order = {key: index for index, key in enumerate(_COLLECTION_ORDER)}
    collections.sort(key=lambda item: (order.get(item["key"], 999), item["title"]))
    return collections


def _make_collection(*, key: str, value: Any, run_path: Path) -> Optional[Dict[str, Any]]:
    if not _is_nonempty(value):
        return None
    title = _COLLECTION_TITLES.get(key) or _humanize_key(key)
    summary = _guess_collection_summary(key, value)
    facts = _extract_simple_facts(value, run_path, limit=10)
    stats = _extract_count_facts(key, value, run_path)
    bullets = _extract_list_bullets(value)
    records = _extract_records(value, run_path, limit=18)
    status = _guess_status(value)
    return {
        "key": key,
        "title": title,
        "summary": summary,
        "status": status,
        "tone": _tone_for_status(status),
        "stats": stats,
        "facts": facts,
        "bullets": bullets,
        "records": records,
        "raw": _compact_nested(value),
        "help_text": _build_collection_help_text(key, value, summary),
    }


def _build_data_views(
    *,
    core_payload: Dict[str, Any],
    report_manifest: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    qa_history: Sequence[Dict[str, Any]],
    report_explainer: Dict[str, Any],
) -> List[Dict[str, Any]]:
    views = [
        {
            "title": "Structured Goal Outline",
            "summary": "Top-level keys present in the structured goal.",
            "facts": [
                {"label": "Key count", "value": len(structured_goal or {})},
                {"label": "Intent", "value": structured_goal.get("intent")},
                {"label": "Species", "value": structured_goal.get("species")},
            ],
            "rows": _mapping_outline_rows(structured_goal),
            "raw": _compact_nested(report_manifest.get("structured_goal") or structured_goal),
        },
        {
            "title": "Final Payload Outline",
            "summary": "Top-level keys persisted in goal_intake.json final_payload.",
            "facts": [{"label": "Key count", "value": len(final_payload or {})}],
            "rows": _mapping_outline_rows(final_payload),
            "raw": _compact_nested(_outline_mapping(final_payload)),
        },
        {
            "title": "Report Manifest",
            "summary": "Compact canonical payload produced by the report stage.",
            "facts": [
                {"label": "Payload kind", "value": report_manifest.get("payload_kind")},
                {"label": "Payload version", "value": report_manifest.get("payload_version")},
            ],
            "rows": _mapping_outline_rows(report_manifest),
            "raw": _compact_nested(report_manifest),
        },
        {
            "title": "Report Explainer",
            "summary": "Structured beginner-facing explanation used to render the Guide tab and markdown report.",
            "facts": [
                {"label": "Schema version", "value": report_explainer.get("schema_version")},
                {"label": "Audience", "value": report_explainer.get("audience")},
            ],
            "rows": _mapping_outline_rows(report_explainer),
            "raw": _compact_nested(report_explainer),
        },
    ]

    if qa_history:
        views.append(
            {
                "title": "Clarification History",
                "summary": "Questions and answers recorded during intake.",
                "facts": [{"label": "Clarifications", "value": len(qa_history)}],
                "rows": [
                    {
                        "title": f"Question {index}",
                        "subtitle": _stringify(item.get("question")),
                        "fields": [{"label": "Answer", "value": item.get("answer")}],
                    }
                    for index, item in enumerate(qa_history, start=1)
                ],
                "raw": _compact_nested(list(qa_history)),
            }
        )

    llm_rationale = core_payload.get("llm_rationale") if isinstance(core_payload.get("llm_rationale"), dict) else {}
    trace = llm_rationale.get("agent_trace") if isinstance(llm_rationale.get("agent_trace"), list) else []
    if trace:
        views.append(
            {
                "title": "Agent Trace Snapshot",
                "summary": "Compact trace used to populate the Decisions tab.",
                "facts": [{"label": "Trace steps", "value": len(trace)}],
                "rows": [
                    {
                        "title": _stringify(step.get("agent_name") or step.get("agent") or step.get("name") or f"Step {index + 1}"),
                        "subtitle": _stringify(step.get("summary") or step.get("message")),
                        "fields": [
                            {"label": "Model", "value": step.get("model")},
                            {"label": "Status", "value": step.get("status")},
                        ],
                    }
                    for index, step in enumerate(trace[:12])
                    if isinstance(step, dict)
                ],
                "raw": _compact_nested(trace),
            }
        )

    return [view for view in views if view]


def _build_help_texts(
    *,
    meta: Dict[str, Any],
    gene: Dict[str, Any],
    targets: Sequence[Dict[str, Any]],
    branch_outcomes: Sequence[Dict[str, Any]],
    alerts: Sequence[str],
    artifact_inventory: Dict[str, Any],
    strategy_routing: Dict[str, Any],
    output_collections: Sequence[Dict[str, Any]],
    qa_history: Sequence[Dict[str, Any]],
    modalities: Sequence[str],
    report_markdown: str,
) -> Dict[str, str]:
    gene_symbol = _stringify(meta.get("gene_symbol")) or "this gene"
    transcript_count = len(gene.get("transcripts") or []) if isinstance(gene, dict) else 0
    target_count = len(targets)
    safeguard_count = sum(len(target.get("safeguards") or []) for target in targets if isinstance(target, dict))
    safeguard_target_count = sum(1 for target in targets if target.get("safeguards"))
    route_count = len(strategy_routing.get("goals") or []) if isinstance(strategy_routing, dict) else 0
    collection_titles = ", ".join(collection.get("title") for collection in output_collections[:5] if collection.get("title"))
    branch_summary = ", ".join(
        f"{branch.get('title')}: {branch.get('status')}" for branch in branch_outcomes if branch.get("title")
    )
    modality_text = ", ".join(modalities) if modalities else "no explicit modality tags"
    markdown_sections = sum(1 for line in (report_markdown or "").splitlines() if line.strip().startswith("#"))

    return {
        "overview_quick": (
            f"This is the fastest human-readable summary of the {gene_symbol} run. "
            "It focuses on what the user asked for, which branches actually ran, and which files matter first."
        ),
        "workflow_graph": (
            "This graph shows the actual run flow from intake through branch-specific workstreams to the final reports. "
            "Hover a node for a plain-English explanation, drag to pan, and use the mouse wheel to zoom."
        ),
        "workflow_details": (
            "This detail panel explains the currently highlighted workflow node in plain language, including why that step mattered in this run."
        ),
        "guide_summary": (
            f"This guide turns the {gene_symbol} run into a beginner-friendly narrative before you inspect raw artifacts."
        ),
        "guide_walkthrough": (
            "The walkthrough explains what the run actually did, in order, using only the preserved run context."
        ),
        "guide_decisions": (
            "Key decisions explain why the pipeline chose particular transcripts, branches, chemistries, or routing outcomes."
        ),
        "guide_outputs": (
            "Output guides explain what the major outputs are, how to read them, and why they matter for this run."
        ),
        "guide_caveats": (
            "Caveats collect warnings, timeouts, and uncertainty so new users do not over-interpret the outputs."
        ),
        "guide_glossary": (
            "The glossary defines technical terms that appear in this run and may not be obvious to a new teammate."
        ),
        "overview_snapshot": (
            f"This run for {gene_symbol} is summarized from {artifact_inventory.get('total_files') or 0} tracked files. "
            f"It surfaced {target_count} target site(s), {meta.get('ranked_aso_count') or 0} ranked ASO candidate(s), "
            f"and modality tags: {modality_text}."
        ),
        "overview_branches": (
            f"Each branch card describes whether a design branch produced usable outputs. "
            f"On this run: {branch_summary or 'no branch outcomes were recorded'}."
        ),
        "overview_alerts": (
            f"Alerts collect pipeline warnings and blockers carried through the run payload. "
            f"This run has {len(alerts)} alert(s)."
        ),
        "overview_markdown": (
            f"This is the persisted markdown narrative from report.md. "
            f"It has {markdown_sections} heading section(s) and is rendered from the structured explainer for this run."
        ),
        "maps_targets": (
            f"This transcript map overlays {target_count} target site(s) across {transcript_count} transcript model(s) for {gene_symbol}. "
            "Click a target in the list to inspect route rank, isoform presence, and ASO recommendations."
        ),
        "maps_target_details": (
            "This panel resolves the currently selected target into interpretable fields. "
            "ASO tables are grouped by output type when the run produced multiple modalities."
        ),
        "maps_target_records": (
            "Use these expanded target records when the transcript map is not enough. "
            "They retain per-target metadata, grouped ASO tables, and raw target JSON."
        ),
        "maps_safeguards": (
            f"This safeguard map is only meaningful when routing attached safeguard sites. "
            f"Here it covers {safeguard_target_count} target(s) with {safeguard_count} safeguard marker(s). "
            "Click a target in the list to inspect its safeguard set. Blue markers are selected targets; colored markers below the transcript track show safeguards."
        ),
        "maps_safeguard_details": (
            "Safeguard details explain what nearby or competing sites should remain neutral, increase, or decrease while targeting the selected route."
        ),
        "decisions_pipeline": (
            "Pipeline decisions are the stage-by-stage structured notes assembled from goal intake, transcript resolution, chemistry selection, and downstream branch metadata."
        ),
        "decisions_timeline": (
            f"This timeline turns the run into a readable sequence of decisions. "
            f"It includes intake, clarification, routing, chemistry, and context-specific steps for {gene_symbol}."
        ),
        "decisions_routing": (
            f"Strategy routing summarizes ranked splice route goals. "
            f"This run retained {route_count} route goal(s) after routing."
        ),
        "decisions_strategy_analysis": (
            "This panel shows the LLM-written splice strategy analysis that motivated junction choice, tradeoffs, and fallback routes."
        ),
        "decisions_trace": (
            f"The agent trace is the most detailed decision log available from the splice strategy stage. "
            f"It is useful when you need the raw intermediate outputs instead of the condensed summary."
        ),
        "outputs_collections": (
            f"Output collections are grouped semantic payloads rather than raw files. "
            f"This run exposed {len(output_collections)} collection(s): {collection_titles or 'none'}."
        ),
        "outputs_artifacts": (
            f"Artifact inventory groups all files under the run directory by subfolder so you can audit exactly what was produced for {gene_symbol}."
        ),
        "targets_summary": (
            f"Target summary aggregates all merged target labels in this run, including splice, steric, and gapmer-derived entries when present."
        ),
        "data_views": (
            "Data views are compact payload outlines meant for debugging structure without forcing you to read the full JSON files."
        ),
    }


def _build_guide_payload(report_explainer: Dict[str, Any]) -> Dict[str, Any]:
    explainer = report_explainer if isinstance(report_explainer, dict) else {}
    summary = explainer.get("summary") if isinstance(explainer.get("summary"), dict) else {}
    return {
        "summary": {
            "headline": _stringify(summary.get("headline"), max_len=12000),
            "plain_language_goal": _stringify(summary.get("plain_language_goal"), max_len=12000),
            "what_happened": _stringify(summary.get("what_happened"), max_len=24000),
            "result": _stringify(summary.get("result"), max_len=12000),
            "bullets": [item for item in (_stringify(entry, max_len=4000) for entry in summary.get("bullets") or []) if item][:6],
        },
        "start_here": [item for item in (_stringify(entry, max_len=8000) for entry in explainer.get("start_here") or []) if item][:6],
        "walkthrough": _normalize_guide_steps(explainer.get("walkthrough")),
        "workflow_graph": _normalize_workflow_graph_payload(explainer.get("workflow_graph")),
        "key_decisions": _normalize_guide_decisions(explainer.get("key_decisions")),
        "branch_explanations": _normalize_guide_branches(explainer.get("branch_explanations")),
        "output_guides": _normalize_output_guides_payload(explainer.get("output_guides")),
        "caveats": [item for item in (_stringify(entry, max_len=8000) for entry in explainer.get("caveats") or []) if item][:8],
        "glossary": _normalize_glossary_payload(explainer.get("glossary")),
    }


def _merge_help_texts(base: Dict[str, str], report_explainer: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(base or {})
    tooltips = report_explainer.get("tooltips") if isinstance(report_explainer, dict) else {}
    if isinstance(tooltips, dict):
        for key, value in tooltips.items():
            text = _stringify(value)
            if text:
                merged[str(key)] = text
    return merged


def _apply_explainer_branch_text(branch_outcomes: Sequence[Dict[str, Any]], report_explainer: Dict[str, Any]) -> List[Dict[str, Any]]:
    explanations = report_explainer.get("branch_explanations") if isinstance(report_explainer, dict) else []
    explanation_map: Dict[str, Dict[str, Any]] = {}
    for item in explanations if isinstance(explanations, list) else []:
        if not isinstance(item, dict):
            continue
        branch = _stringify(item.get("branch")).lower()
        if branch:
            explanation_map[branch] = item

    cards: List[Dict[str, Any]] = []
    for entry in branch_outcomes:
        card = dict(entry)
        branch_key = _stringify(entry.get("title")).lower().replace(" ", "_")
        explanation = explanation_map.get(branch_key)
        if explanation:
            summary = _stringify(explanation.get("summary"))
            why = _stringify(explanation.get("why"))
            if summary:
                card["summary"] = summary
            if summary or why:
                card["help_text"] = " ".join(piece for piece in [summary, why] if piece)
        cards.append(card)
    return cards


def _apply_explainer_collection_text(collections: Sequence[Dict[str, Any]], report_explainer: Dict[str, Any]) -> List[Dict[str, Any]]:
    guides = report_explainer.get("output_guides") if isinstance(report_explainer, dict) else []
    guide_map: Dict[str, Dict[str, Any]] = {}
    for item in guides if isinstance(guides, list) else []:
        if not isinstance(item, dict):
            continue
        key = _stringify(item.get("key"))
        if key:
            guide_map[key] = item

    items: List[Dict[str, Any]] = []
    for collection in collections:
        record = dict(collection)
        guide = guide_map.get(_stringify(collection.get("key")))
        if guide:
            parts = [
                _stringify(guide.get("what_it_is")),
                _stringify(guide.get("how_to_read")),
                _stringify(guide.get("why_it_matters")),
            ]
            text = " ".join(part for part in parts if part)
            if text:
                record["help_text"] = text
        items.append(record)
    return items


def _normalize_guide_steps(value: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = _stringify(item.get("title"), max_len=4000)
        lane = _stringify(item.get("lane"), max_len=256).lower() or "shared"
        summary = _stringify(item.get("summary"), max_len=20000)
        why = _stringify(item.get("why_it_matters"), max_len=12000)
        if not (title or summary or why):
            continue
        items.append(
            {
                "title": title or "Step",
                "lane": lane,
                "summary": summary,
                "why_it_matters": why,
                "evidence_refs": [_stringify(ref, max_len=4000) for ref in item.get("evidence_refs") or [] if _stringify(ref, max_len=4000)][:5],
            }
        )
    return items[:12]


def _normalize_workflow_graph_payload(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    raw_lanes = value.get("lanes") if isinstance(value.get("lanes"), list) else []
    raw_nodes = value.get("nodes") if isinstance(value.get("nodes"), list) else []
    raw_edges = value.get("edges") if isinstance(value.get("edges"), list) else []

    lanes: List[Dict[str, Any]] = []
    lane_ids: Set[str] = set()
    for item in raw_lanes:
        if not isinstance(item, dict):
            continue
        lane_id = _stringify(item.get("id"), max_len=128)
        if not lane_id or lane_id in lane_ids:
            continue
        lanes.append(
            {
                "id": lane_id,
                "label": _stringify(item.get("label"), max_len=1000) or _humanize_key(lane_id),
            }
        )
        lane_ids.add(lane_id)
    if "shared" not in lane_ids:
        lanes.insert(0, {"id": "shared", "label": "Shared planning"})
        lane_ids.add("shared")

    nodes: List[Dict[str, Any]] = []
    node_ids: Set[str] = set()
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            continue
        node_id = _stringify(item.get("id"), max_len=256)
        if not node_id or node_id in node_ids:
            continue
        lane = _stringify(item.get("lane"), max_len=128).lower() or "shared"
        if lane not in lane_ids:
            lanes.append({"id": lane, "label": _stringify(item.get("lane_label"), max_len=1000) or _humanize_key(lane)})
            lane_ids.add(lane)
        try:
            column = int(item.get("column"))
        except Exception:
            column = index
        nodes.append(
            {
                "id": node_id,
                "title": _stringify(item.get("title"), max_len=4000) or f"Step {index + 1}",
                "lane": lane,
                "lane_label": _stringify(item.get("lane_label"), max_len=1000) or _humanize_key(lane),
                "column": max(0, column),
                "tone": _stringify(item.get("tone"), max_len=128) or ("shared" if lane == "shared" else "branch"),
                "subtitle": _stringify(item.get("subtitle"), max_len=4000),
                "description": _stringify(item.get("description"), max_len=20000),
                "why": _stringify(item.get("why"), max_len=12000),
                "evidence_refs": [_stringify(ref, max_len=4000) for ref in item.get("evidence_refs") or [] if _stringify(ref, max_len=4000)][:6],
            }
        )
        node_ids.add(node_id)

    edges: List[Dict[str, Any]] = []
    seen_edges: Set[str] = set()
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        source = _stringify(item.get("source"), max_len=256)
        target = _stringify(item.get("target"), max_len=256)
        if not source or not target or source == target or source not in node_ids or target not in node_ids:
            continue
        edge_key = f"{source}->{target}"
        if edge_key in seen_edges:
            continue
        edges.append({"source": source, "target": target})
        seen_edges.add(edge_key)

    if not nodes:
        return {}
    return {"lanes": lanes, "nodes": nodes, "edges": edges}


def _normalize_guide_decisions(value: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = _stringify(item.get("title"), max_len=4000)
        decision = _stringify(item.get("decision"), max_len=16000)
        why = _stringify(item.get("why"), max_len=12000)
        if not (title or decision or why):
            continue
        items.append(
            {
                "title": title or "Decision",
                "decision": decision,
                "why": why,
                "confidence": _stringify(item.get("confidence"), max_len=64) or "unknown",
                "evidence_refs": [_stringify(ref, max_len=4000) for ref in item.get("evidence_refs") or [] if _stringify(ref, max_len=4000)][:5],
            }
        )
    return items[:10]


def _normalize_guide_branches(value: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = _stringify(item.get("title"), max_len=4000)
        status = _stringify(item.get("status"), max_len=128)
        summary = _stringify(item.get("summary"), max_len=12000)
        why = _stringify(item.get("why"), max_len=12000)
        if not (title or summary or why):
            continue
        items.append(
            {
                "branch": _stringify(item.get("branch"), max_len=256),
                "title": title or _humanize_key(item.get("branch")) or "Branch",
                "status": status or "unknown",
                "summary": summary,
                "why": why,
                "evidence_refs": [_stringify(ref, max_len=4000) for ref in item.get("evidence_refs") or [] if _stringify(ref, max_len=4000)][:4],
            }
        )
    return items[:10]


def _normalize_output_guides_payload(value: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = _stringify(item.get("title"), max_len=4000)
        what = _stringify(item.get("what_it_is"), max_len=12000)
        how = _stringify(item.get("how_to_read"), max_len=16000)
        why = _stringify(item.get("why_it_matters"), max_len=12000)
        if not (title or what or how or why):
            continue
        items.append(
            {
                "key": _stringify(item.get("key"), max_len=256),
                "title": title or "Output",
                "what_it_is": what,
                "how_to_read": how,
                "why_it_matters": why,
                "evidence_refs": [_stringify(ref, max_len=4000) for ref in item.get("evidence_refs") or [] if _stringify(ref, max_len=4000)][:4],
            }
        )
    return items[:12]


def _normalize_glossary_payload(value: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        term = _stringify(item.get("term"), max_len=1000)
        definition = _stringify(item.get("definition"), max_len=12000)
        if term and definition:
            items.append({"term": term, "definition": definition})
    return items[:12]


def _build_branch_help_text(branch: str, entry: Dict[str, Any]) -> str:
    status = _stringify(entry.get("status")) or "unknown"
    reason = _stringify(entry.get("reason")) or "No reason was recorded."
    return f"{_humanize_key(branch)} branch status: {status}. {reason}"


def _build_collection_help_text(key: str, value: Any, summary: str) -> str:
    title = _COLLECTION_TITLES.get(key) or _humanize_key(key)
    if isinstance(value, list):
        return f"{title} contains {len(value)} item(s). {summary}"
    if isinstance(value, dict):
        return f"{title} exposes {len(value)} top-level field(s). {summary}"
    return f"{title}: {summary}"


def _collect_artifacts(run_path: Path) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    total_size = 0
    for path in sorted(run_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_path)
        rel_parts = relative.parts
        if "controller" in rel_parts or "state_snapshots" in rel_parts or rel_parts[0].startswith("agentic_"):
            continue
        stat = path.stat()
        total_size += stat.st_size
        if rel_parts[0] == "artifacts" and len(rel_parts) > 1:
            group = rel_parts[1]
        elif len(rel_parts) > 1:
            group = rel_parts[0]
        else:
            group = "run root"
        item = {
            "name": path.name,
            "relative_path": relative.as_posix(),
            "href": _quote_path(relative.parts),
            "group": group,
            "ext": path.suffix.lower().lstrip("."),
            "kind": _guess_file_kind(path),
            "size_bytes": stat.st_size,
            "size_label": _format_bytes(stat.st_size),
        }
        items.append(item)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["group"], []).append(item)

    groups = [
        {
            "name": group,
            "summary": f"{len(entries)} file(s)",
            "items": entries,
        }
        for group, entries in grouped.items()
    ]
    groups.sort(key=lambda item: (_ARTIFACT_GROUP_ORDER.get(item["name"], 999), item["name"]))
    return {
        "total_files": len(items),
        "total_size_bytes": total_size,
        "total_size_label": _format_bytes(total_size),
        "groups": groups,
        "items": items,
    }


def _select_primary_artifacts(artifact_inventory: Dict[str, Any]) -> List[Dict[str, str]]:
    items = artifact_inventory.get("items") if isinstance(artifact_inventory, dict) else []
    if not isinstance(items, list):
        return []
    by_path = {item.get("relative_path"): item for item in items if isinstance(item, dict)}
    selected: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for relative_path in _PRIMARY_ARTIFACT_PATHS:
        item = by_path.get(relative_path)
        if not item:
            continue
        label = relative_path.replace("/", " / ")
        selected.append({"label": label, "href": item.get("href")})
        seen.add(relative_path)
    for item in items:
        path = item.get("relative_path")
        if not path or path in seen or item.get("kind") not in {"excel", "genbank"}:
            continue
        selected.append({"label": path.replace("/", " / "), "href": item.get("href")})
        if len(selected) >= 10:
            break
    return selected[:10]


def _design_job_collection_summary(value: Any, modality: str) -> str:
    jobs = [job for job in value if isinstance(job, dict)] if isinstance(value, list) else []
    if not jobs:
        return f"No {modality.lower()} design jobs were recorded."
    skipped = [job for job in jobs if (_stringify(job.get("status")) or "").lower() == "skipped" or _stringify(job.get("reason"))]
    skipped_count = len(skipped)
    if skipped_count <= 0:
        return f"{len(jobs)} {modality.lower()} design job(s) were planned or executed."
    active_count = max(0, len(jobs) - skipped_count)
    if active_count > 0:
        summary = f"{active_count} {modality.lower()} design job(s) remained active and {skipped_count} were skipped."
    else:
        summary = f"All {len(jobs)} {modality.lower()} design job(s) were skipped."
    first_reason = _stringify(skipped[0].get("reason")) if skipped else ""
    if first_reason:
        summary += f" Example rationale: {first_reason}"
    return summary


def _guess_collection_summary(key: str, value: Any) -> str:
    if key in {"tiling_results", "tiling"} and isinstance(value, dict):
        initial = value.get("initial_filtering") if isinstance(value.get("initial_filtering"), dict) else {}
        passed = initial.get("passed") if isinstance(initial.get("passed"), dict) else {}
        total_rows = initial.get("total_rows")
        passed_count = passed.get("count")
        scored_path = value.get("scored_tiles_path") or value.get("uniqueness_scored_path")
        parts = []
        if total_rows not in (None, "", [], {}):
            parts.append(f"{_format_count(total_rows)} tiles generated")
        if passed_count not in (None, "", [], {}):
            parts.append(f"{_format_count(passed_count)} passed filters")
        if scored_path:
            parts.append("scored output available")
        return ". ".join(parts) or "Tiling results were written for this run."

    if key in {"gapmer_tiling_results", "gapmer_tiling"} and isinstance(value, dict):
        parts = []
        if value.get("row_count") not in (None, "", [], {}):
            parts.append(f"{_format_count(value['row_count'])} gapmer candidates")
        if value.get("mode"):
            parts.append(f"mode={value['mode']}")
        if value.get("primary_isoform"):
            parts.append(f"primary isoform={value['primary_isoform']}")
        return ". ".join(parts) or "Gapmer tiling results were written for this run."

    if key == "strategy_routing" and isinstance(value, dict):
        goals = value.get("goals") if isinstance(value.get("goals"), list) else []
        ranking = (
            (value.get("planner_plan") or {}).get("ranking")
            if isinstance(value.get("planner_plan"), dict)
            else {}
        )
        model = ranking.get("model") if isinstance(ranking, dict) else None
        parts = []
        if goals:
            parts.append(f"{len(goals)} ranked goals")
        if model:
            parts.append(f"ranked by {model}")
        return ". ".join(parts) or "Strategy routing output is available."

    if key == "splice_strategy_analysis" and isinstance(value, dict):
        parts = []
        if value.get("unique_splice_site_count") not in (None, "", [], {}):
            parts.append(f"{value['unique_splice_site_count']} unique splice sites")
        if value.get("additional_site_count") not in (None, "", [], {}):
            parts.append(f"{value['additional_site_count']} fallback sites")
        if value.get("model"):
            parts.append(f"model={value['model']}")
        return ". ".join(parts) or "Splice strategy analysis is available."

    if key.endswith("_design_jobs") and isinstance(value, list):
        modality = "Gapmer" if key.startswith("gapmer") else "Steric"
        return _design_job_collection_summary(value, modality)

    if key == "rbp_binding_result" and isinstance(value, dict):
        parts = []
        if value.get("rbp"):
            parts.append(str(value["rbp"]))
        if value.get("assay"):
            parts.append(str(value["assay"]))
        if value.get("peak_count") not in (None, "", [], {}):
            parts.append(f"{value['peak_count']} peak(s)")
        return " ".join(parts) or "RBP dataset selection is available."

    if key == "mirna_binding_result" and isinstance(value, dict):
        requested = value.get("requested_mirnas") if isinstance(value.get("requested_mirnas"), list) else []
        families = value.get("matched_families") if isinstance(value.get("matched_families"), list) else []
        if not families:
            families = value.get("resolved_families") if isinstance(value.get("resolved_families"), list) else []
        parts = []
        if requested:
            parts.append(", ".join(str(item) for item in requested[:4]))
        if families:
            parts.append(f"{len(families)} TargetScan family/families")
        if value.get("hit_count") not in (None, "", [], {}):
            parts.append(f"{value['hit_count']} site(s)")
        return " | ".join(parts) or "miRNA target-site selection is available."

    if key == "motif_tiling_result" and isinstance(value, dict):
        motifs = value.get("motifs") if isinstance(value.get("motifs"), list) else []
        status = _stringify(value.get("status"))
        parts = []
        if status:
            parts.append(status)
        if motifs:
            parts.append(f"{len(motifs)} motif(s)")
        if value.get("target_level"):
            parts.append(str(value["target_level"]))
        return " | ".join(parts) or "Motif targeting output is available."

    if key == "patient_reference" and isinstance(value, dict):
        parts = []
        if value.get("variant_count") not in (None, "", [], {}):
            parts.append(f"{value['variant_count']} variant(s)")
        if value.get("matches_reference") is not None:
            parts.append("matches reference" if value.get("matches_reference") else "differs from reference")
        return ". ".join(parts) or "Patient reference output is available."

    if key in {"snapgene_annotation", "snapgene_annotations"}:
        return "SnapGene-derived reference context is available for this run."

    if key == "splice_site_plan" and isinstance(value, list):
        return f"{len(value)} splice-site candidate row(s) captured for routing."

    if key == "splice_site_comparisons" and isinstance(value, list):
        return f"{len(value)} splice-site comparison row(s) persisted for inspection."

    if key == "splice_isoform_objectives" and isinstance(value, list):
        return f"{len(value)} isoform objective(s) were derived from the request."

    if isinstance(value, list):
        return f"{len(value)} item(s) captured in this section."
    if isinstance(value, dict):
        status = _stringify(value.get("status"))
        reason = _stringify(value.get("reason"))
        if status and reason:
            return f"{status}: {reason}"
        if status:
            return status
        return f"{len(value)} top-level field(s) captured in this section."
    return _stringify(value)


def _guess_status(value: Any) -> str:
    if isinstance(value, dict):
        status = _stringify(value.get("status"))
        if status:
            return status
    if isinstance(value, list):
        return "available" if value else "empty"
    return "available"


def _extract_count_facts(key: str, value: Any, run_path: Path) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    if isinstance(value, list):
        stats.append({"label": "Count", "value": len(value)})
        return stats
    if not isinstance(value, dict):
        return stats

    initial = value.get("initial_filtering") if isinstance(value.get("initial_filtering"), dict) else {}
    passed = initial.get("passed") if isinstance(initial.get("passed"), dict) else {}
    pairs = [
        ("job_count", "Jobs"),
        ("goal_count", "Goals"),
        ("row_count", "Rows"),
        ("region_count", "Regions"),
        ("peak_count", "Peaks"),
        ("variant_count", "Variants"),
        ("evaluated", "Evaluated"),
        ("actual_evaluated", "Actual evaluated"),
        ("unique_splice_site_count", "Unique splice sites"),
        ("additional_site_count", "Fallback sites"),
        ("derived_program_count", "Derived programs"),
    ]
    if initial:
        if initial.get("total_rows") not in (None, "", [], {}):
            stats.append({"label": "Initial rows", "value": initial.get("total_rows")})
        if passed.get("count") not in (None, "", [], {}):
            stats.append({"label": "Passed filters", "value": passed.get("count"), "href": _href_for_path(passed.get("path"), run_path)})
    for key_name, label in pairs:
        raw = value.get(key_name)
        if raw in (None, "", [], {}):
            continue
        stats.append({"label": label, "value": raw, "href": _href_for_path(raw, run_path) if isinstance(raw, str) else None})
        if len(stats) >= 8:
            break
    if key == "strategy_routing":
        goals = value.get("goals") if isinstance(value.get("goals"), list) else []
        if goals:
            stats.append({"label": "Ranked goals", "value": len(goals)})
    return stats[:8]


def _extract_simple_facts(value: Any, run_path: Path, limit: int = 8) -> List[Dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    facts: List[Dict[str, Any]] = []
    skip_keys = {"analysis", "context_prompt", "full_sequence"}
    for key, raw in value.items():
        if len(facts) >= limit:
            break
        if key in skip_keys:
            continue
        if isinstance(raw, dict):
            continue
        if isinstance(raw, list):
            if not raw or any(isinstance(item, (dict, list)) for item in raw):
                continue
            text = _stringify(raw)
            if not text:
                continue
            facts.append({"label": _humanize_key(key), "value": text})
            continue
        if isinstance(raw, str) and len(raw.strip()) > 240 and not _looks_like_path(raw):
            continue
        text = _stringify(raw)
        if not text:
            continue
        facts.append(
            {
                "label": _humanize_key(key),
                "value": text,
                "href": _href_for_path(raw, run_path),
            }
        )
    return facts


def _extract_list_bullets(value: Any) -> List[str]:
    bullets: List[str] = []
    if isinstance(value, dict):
        for key, raw in value.items():
            if isinstance(raw, list) and raw and all(not isinstance(item, (dict, list)) for item in raw):
                text = _stringify(raw, max_len=280)
                if text:
                    bullets.append(f"{_humanize_key(key)}: {text}")
    elif isinstance(value, list) and value and all(not isinstance(item, (dict, list)) for item in value):
        bullets.extend(_stringify(item) for item in value[:10] if _stringify(item))
    return bullets[:10]


def _extract_records(value: Any, run_path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(value, list):
        for index, item in enumerate(value[:limit]):
            if isinstance(item, dict):
                record = _mapping_to_record(item, run_path, fallback_title=f"Item {index + 1}")
                if record:
                    records.append(record)
            else:
                text = _stringify(item)
                if text:
                    records.append({"title": f"Item {index + 1}", "subtitle": text, "fields": [], "bullets": [], "raw": None})
        return records
    if isinstance(value, dict):
        for key, item in value.items():
            if len(records) >= limit:
                break
            if isinstance(item, dict):
                record = _mapping_to_record(item, run_path, fallback_title=_humanize_key(key))
                if record:
                    records.append(record)
            elif isinstance(item, list) and item and all(isinstance(entry, dict) for entry in item):
                for nested in item[: max(1, limit - len(records))]:
                    record = _mapping_to_record(nested, run_path, fallback_title=_humanize_key(key))
                    if record:
                        records.append(record)
                        if len(records) >= limit:
                            break
    return records[:limit]


def _mapping_to_record(
    value: Dict[str, Any],
    run_path: Path,
    *,
    fallback_title: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    title = (
        _stringify(value.get("label"))
        or _stringify(value.get("goal_label"))
        or _stringify(value.get("job_id"))
        or _stringify(value.get("program_id"))
        or _stringify(value.get("isoform_label"))
        or _stringify(value.get("id"))
        or fallback_title
    )
    status_text = (_stringify(value.get("status")) or "").lower()
    reason_text = _stringify(value.get("reason"), max_len=240)
    if status_text == "skipped" and reason_text:
        subtitle = reason_text
    else:
        subtitle = (
            _stringify(value.get("objective"), max_len=240)
            or _stringify(value.get("rationale"), max_len=240)
            or _stringify(value.get("analysis"), max_len=240)
            or _stringify(value.get("summary"), max_len=240)
            or _stringify(value.get("notes"), max_len=240)
            or reason_text
        )
    base_fields = _extract_simple_facts(value, run_path, limit=8)
    prioritized_fields: List[Dict[str, Any]] = []
    if _stringify(value.get("status")):
        prioritized_fields.append({"label": "Status", "value": _stringify(value.get("status"))})
    if reason_text:
        prioritized_fields.append({"label": "Reason", "value": reason_text})
    remaining_fields = [
        field
        for field in base_fields
        if str(field.get("label") or "") not in {"Status", "Reason"}
    ]
    fields = (prioritized_fields + remaining_fields)[:8]
    bullets = _extract_list_bullets(value)
    return {
        "title": title,
        "subtitle": subtitle,
        "fields": fields,
        "bullets": bullets,
        "raw": _compact_nested(value),
    }


def _mapping_outline_rows(value: Dict[str, Any]) -> List[Dict[str, Any]]:
    outline = _outline_mapping(value)
    return [
        {
            "title": row["key"],
            "subtitle": row.get("preview"),
            "fields": [
                {"label": "Type", "value": row.get("type")},
                {"label": "Size", "value": row.get("size")},
            ],
        }
        for row in outline[:24]
    ]


def _outline_mapping(value: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows = []
    for key, raw in sorted(value.items()):
        entry = {
            "key": key,
            "type": type(raw).__name__,
            "size": len(raw) if isinstance(raw, (list, dict, set, tuple)) else None,
            "preview": None,
        }
        if not isinstance(raw, (list, dict)):
            entry["preview"] = _stringify(raw, max_len=180)
        rows.append(entry)
    return rows


def _compact_nested(
    value: Any,
    depth: int = 0,
    *,
    max_depth: int = 4,
    max_items: int = 10,
    max_string: int = 360,
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _stringify(value, max_len=max_string)
    if depth >= max_depth:
        if isinstance(value, dict):
            return f"{len(value)} key(s)"
        if isinstance(value, list):
            return f"{len(value)} item(s)"
        return _stringify(value, max_len=max_string)
    if isinstance(value, list):
        compact = [
            _compact_nested(item, depth + 1, max_depth=max_depth, max_items=max_items, max_string=max_string)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            compact.append(f"... {len(value) - max_items} more item(s)")
        return compact
    if isinstance(value, dict):
        compact_dict = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            compact_dict[str(key)] = _compact_nested(
                item,
                depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
        if len(items) > max_items:
            compact_dict["__truncated__"] = f"{len(items) - max_items} more key(s)"
        return compact_dict
    return _stringify(value, max_len=max_string)


def _is_nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _stringify(value: Any, max_len: int = 220) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if item not in (None, "")]
        text = ", ".join(parts)
    else:
        text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _first_nonempty_lines(value: Any, limit: int = 3) -> str:
    text = _stringify(value, max_len=10000)
    if not text:
        return ""
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    return " | ".join(lines[:limit])


def _href_for_path(value: Any, run_path: Path) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if not _looks_like_path(text):
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (run_path / candidate).resolve()
    if not candidate.exists():
        return None
    try:
        relative = os.path.relpath(candidate, run_path)
    except Exception:
        return None
    return _quote_path(Path(relative).parts)


def _quote_path(parts: Iterable[str]) -> str:
    return "/".join(quote(str(part)) for part in parts)


def _looks_like_path(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > 4096:
        return False
    lowered = text.lower()
    if text.startswith("/") or text.startswith("./") or text.startswith("../") or text.startswith("~"):
        return True
    if "/" in text:
        if " " in text:
            return False
        tail = text.rsplit("/", 1)[-1]
        return "." in tail
    return lowered.endswith(
        (
            ".json",
            ".csv",
            ".tsv",
            ".xlsx",
            ".xls",
            ".pdf",
            ".md",
            ".html",
            ".bed",
            ".bed.gz",
            ".gb",
            ".gbk",
            ".fa",
            ".fasta",
            ".fai",
            ".vcf",
            ".txt",
            ".log",
        )
    )


def _humanize_key(value: str) -> str:
    text = str(value or "").replace("_", " ").strip()
    if not text:
        return ""
    return " ".join(word.capitalize() for word in text.split())


def _tone_for_status(status: Optional[str]) -> str:
    text = str(status or "").strip().lower()
    if text in {"outputs_present", "ready", "completed", "available", "true", "applied"}:
        return "teal"
    if text in {"warning", "deferred", "partial", "skipped"}:
        return "rust"
    if text in {"error", "failed", "missing", "false", "timeout"}:
        return "rose"
    return "neutral"


def _guess_file_kind(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix == ".bed" or name.endswith(".bed.gz"):
        return "bed"
    if suffix in {".pdf"}:
        return "pdf"
    if suffix in {".md"}:
        return "markdown"
    if suffix in {".html"}:
        return "html"
    if suffix in {".gb", ".gbk"}:
        return "genbank"
    if suffix in {".fa", ".fasta", ".fai"}:
        return "fasta"
    if suffix in {".vcf"}:
        return "vcf"
    return suffix.lstrip(".") or "file"


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _format_count(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)
