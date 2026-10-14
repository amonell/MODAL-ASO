from __future__ import annotations

import json
import os
import re
from pathlib import Path
from aso_agents.reporting_contracts import generated_sequence_fields, exact_gene_count, ISOFORM_SCORE_DEFINITION

from typing import Any, Dict, Iterable, List, Optional, Sequence

from aso_agents.run_layout import resolve_run_path


REPORT_TOOLTIP_TARGETS: Dict[str, str] = {
    "overview_quick": "Fast plain-English overview of the completed run.",
    "workflow_graph": "Interactive graph of the run workflow and branch structure.",
    "workflow_details": "Plain-English explanation of the currently selected workflow step.",
    "guide_summary": "Top-level plain-English summary of the run.",
    "guide_walkthrough": "Step-by-step explanation of what the run did.",
    "guide_decisions": "The most important decisions and why they were made.",
    "guide_outputs": "How to interpret the major outputs produced by this run.",
    "guide_caveats": "Warnings, uncertainties, and things the user should treat carefully.",
    "guide_glossary": "Definitions for run-specific technical terms that may be unfamiliar.",
    "overview_snapshot": "Compact snapshot of the run outcome and scale.",
    "overview_branches": "Branch-level execution outcomes.",
    "overview_alerts": "Warnings and caveats carried through the run.",
    "overview_markdown": "Persisted markdown narrative for the run.",
    "targets_summary": "Aggregate summary of targeting outputs.",
    "maps_targets": "Transcript map of selected targets.",
    "maps_target_details": "Detailed interpretation of the selected target marker.",
    "maps_target_records": "Expanded target records and associated ASO recommendations.",
    "maps_safeguards": "Safeguard markers tied to targeted splice routes.",
    "maps_safeguard_details": "Detailed interpretation of safeguard markers.",
    "decisions_pipeline": "Structured, stage-by-stage decisions captured during the run.",
    "decisions_timeline": "Chronological view of major run events and decisions.",
    "decisions_routing": "Ranked splice routing output, when routing was run.",
    "decisions_strategy_analysis": "LLM-written splice strategy analysis, when available.",
    "decisions_trace": "Raw agent trace from the splice strategy step, when available.",
    "outputs_collections": "Semantic collections of run outputs.",
    "outputs_artifacts": "File-level artifact inventory for the run directory.",
    "data_views": "Compact debugging views of the raw payloads.",
}

_ALLOWED_CONFIDENCE = {"high", "medium", "low", "unknown"}


def build_report_explainer_context(
    *,
    run_dir: str,
    payload: Dict[str, Any],
    structured_goal: Dict[str, Any],
    final_payload: Dict[str, Any],
    report_payload: Dict[str, Any],
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    request_text = _request_text_from_payload(payload, run_dir)
    qa_history = payload.get("qa_history") if isinstance(payload.get("qa_history"), list) else []
    branch_outcomes = report_payload.get("branch_outcomes") if isinstance(report_payload.get("branch_outcomes"), dict) else {}
    report_structured = report_payload.get("structured_goal") if isinstance(report_payload.get("structured_goal"), dict) else {}
    strategy_routing = report_payload.get("strategy_routing") if isinstance(report_payload.get("strategy_routing"), dict) else {}
    splicing = report_payload.get("splicing") if isinstance(report_payload.get("splicing"), dict) else {}
    splicing_eval = report_payload.get("splicing_eval") if isinstance(report_payload.get("splicing_eval"), dict) else {}

    return {
        "context_kind": "report_explanation_context",
        "schema_version": 1,
        "audience": "novice",
        "run_metadata": {
            "run_name": run_path.name,
            "run_dir": run_dir,
            "gene_symbol": report_structured.get("gene_symbol") or structured_goal.get("gene_symbol"),
            "gene_id": report_structured.get("gene_id") or structured_goal.get("gene_id"),
            "intent": report_structured.get("intent") or structured_goal.get("intent"),
            "desired_effect": report_structured.get("desired_effect") or structured_goal.get("desired_effect"),
            "species": report_structured.get("species") or structured_goal.get("species"),
            "target_context": report_structured.get("target_context") or structured_goal.get("target_context"),
            "user_request": request_text,
        },
        "field_semantics": {
            "branch_statuses": {
                "outputs_present": "The branch produced persisted outputs that are available in the run directory.",
                "no_candidates_after_constraints": "The branch ran, but the requested constraints removed the remaining search space.",
                "no_candidates_found": "The branch ran and did not produce qualifying candidates.",
                "skipped": "The branch was intentionally not run for this request.",
                "unknown": "The report payload did not preserve a definitive branch outcome.",
            },
            "run_splice_eval": "If false, no splice evaluation was performed and the report must not describe splice scores as executed results.",
            "run_splice_router": "If false, no strategy routing was performed and the report must not describe ranked routes as executed results.",
            "threshold_used": "Routing threshold applied to decide which ASO candidates count for a route goal.",
            "count": "Number of candidate ASOs retained for a route goal or output summary after filtering/thresholding.",
            "artifact_presence": "A verified existing file is evidence of delivery; a requested path alone is not.",
            "chemistry_source": "goal_configured and program_configured identify internal configuration, which can include defaults. Neither means user_requested. Attribute a particular architecture or length to the user only when the user_request or a recorded user answer explicitly states that value; otherwise describe it as the selected configuration. A generic request for standard chemistry does not specify a numerical architecture or length.",
            "isoform_uniqueness_score": ISOFORM_SCORE_DEFINITION,
            "filter_stage_counts": "initial_filtering counts the first composition screen; constraint_filtering counts only the later screen on survivors after scoring. A later removed_count=0 does not mean the initial screen removed zero, that all original windows passed, or that composition never limited the search. Identify the stage for every count and inspect both stage summaries before describing total attrition. Unavailable stage counts remain unknown.",
            "constraint_provenance": "Executed configuration and platform defaults are not necessarily user instructions. Attribute each length, count, GC, homopolymer, chemistry or threshold value to the user only when it appears in user_request or recorded user answers. A sufficient candidate count is not a maximum or hard cap. Default composition filters must be described as defaults even when they remove candidates.",
            "exon_nomenclature": "A verified_alternate_mapping relates the requested exon name to a different transcript ordinal at the same attested biological feature. Explain both names, reference and coordinates; do not present a verified alias as an unexplained wrong-exon discrepancy. Unresolved mappings remain unresolved.",
            "binding_region_candidate_sets": "These sets answer where each ASO physically binds, using each requested region's own containment/overlap rule. A splice-response route is the junction whose effect was scored, and can contain ASOs binding elsewhere. Do not use response-route labels as binding-site membership. Guide users to the separate requested binding-region sets when those are present.",
            "repeat_risk": "A zero repeat_row_count means no hits under that particular repeat annotation. It says nothing about genomic off-target counts, mismatched hits, or specificity. Never summarize repeat/off-target hits as none from this field. Read exact and mismatch counts from the actual scored candidate tables, preserve their units, and leave unavailable measurements unknown.",
            "recorded_screening_summary": "Authoritative summaries computed from the saved scored candidate tables, with separate exact off-target, mismatched off-target and alternate-repeat counts plus missing measurements. Use these measurements when explaining screening. Never infer absence of all hits from repeat_risk or claim the table-wide counts describe only final shortlisted rows.",
            "coordinate_conventions": "BED rows are zero-based half-open: length=end-start; native cDNA tile rows explicitly marked 1based_inclusive have length=end-start+1. Do not read a BED start as a 1-based base position.",
            "evidence_refs": "Short references to context keys or artifact paths that support an explanation.",
        },
        "tooltip_targets": [
            {"key": key, "description": description}
            for key, description in REPORT_TOOLTIP_TARGETS.items()
        ],
        "clarifications": [
            {
                "question": _stringify(item.get("question")),
                "answer": _stringify(item.get("answer")),
            }
            for item in qa_history[:8]
            if isinstance(item, dict) and (_stringify(item.get("question")) or _stringify(item.get("answer")))
        ],
        "transcript_context": _build_transcript_context(structured_goal, final_payload),
        "report_manifest": report_payload,
        "routing_summary": {
            "planner_plan": strategy_routing.get("planner_plan") if isinstance(strategy_routing.get("planner_plan"), dict) else {},
            "job_count": strategy_routing.get("job_count"),
            "goal_count": strategy_routing.get("goal_count"),
            "nonzero_goal_count": strategy_routing.get("nonzero_goal_count"),
            "zero_hit_goal_count": strategy_routing.get("zero_hit_goal_count"),
            "top_goals": [
                {
                    "rank": goal.get("rank"),
                    "goal_label": goal.get("goal_label"),
                    "rationale": goal.get("rationale"),
                    "count": goal.get("count"),
                    "threshold_used": goal.get("threshold_used"),
                    "threshold_pass_count": goal.get("threshold_pass_count"),
                    "status": goal.get("status"),
                    "warning": goal.get("warning"),
                    "fallback_selection_used": goal.get("fallback_selection_used"),
                }
                for goal in (strategy_routing.get("goals") or [])[:8]
                if isinstance(goal, dict)
            ],
            "final_route_order_source": (
                "strategy_routing.goals"
                if isinstance(strategy_routing.get("goals"), list) and strategy_routing.get("goals")
                else ""
            ),
        },
        "splicing_summary": {
            "strategy_model": splicing.get("model"),
            "strategy_analysis": _truncate_text(splicing.get("analysis"), 1800),
            "strategy_analysis_role": (
                "preliminary_planning_context_not_final_route_order"
                if isinstance(strategy_routing.get("goals"), list) and strategy_routing.get("goals")
                else "route_planning_context"
            ),
            "context_summary": splicing.get("context_summary") if isinstance(splicing.get("context_summary"), dict) else {},
            "eval_summary": {
                "path": splicing_eval.get("path"),
                "site_count": len(splicing_eval.get("sites") or []) if isinstance(splicing_eval.get("sites"), list) else splicing_eval.get("site_count"),
                "evaluated": splicing_eval.get("evaluated"),
                "nonzero_goal_count": splicing_eval.get("nonzero_goal_count"),
                "zero_hit_goal_count": splicing_eval.get("zero_hit_goal_count"),
            },
        },
        "branch_outcomes": branch_outcomes,
        "output_inventory": _build_output_inventory(report_payload, final_payload, structured_goal),
        "important_artifacts": _collect_important_artifacts(run_path),
    }


def parse_report_explainer_text(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def normalize_report_explainer(explainer: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(explainer, dict):
        return {}

    normalized: Dict[str, Any] = {
        "schema_version": 1,
        "audience": "novice",
        "summary": _normalize_summary(explainer.get("summary"), {}),
        "start_here": _normalize_string_list(explainer.get("start_here"), [], limit=6),
        "walkthrough": _normalize_step_list(explainer.get("walkthrough"), [], limit=12),
        "key_decisions": _normalize_decision_list(explainer.get("key_decisions"), [], limit=10),
        "branch_explanations": _normalize_branch_list(
            explainer.get("branch_explanations"),
            context,
            [],
        ),
        "output_guides": _normalize_output_guides(
            explainer.get("output_guides"),
            context,
            [],
        ),
        "caveats": _normalize_string_list(explainer.get("caveats"), [], limit=8),
        "glossary": _normalize_glossary(explainer.get("glossary"), [], limit=12),
        "tooltips": _normalize_tooltips(explainer.get("tooltips"), {}),
    }
    if not report_explainer_is_valid(normalized):
        return {}
    normalized["workflow_graph"] = _build_workflow_graph(normalized, context)
    return normalized if _workflow_graph_is_valid(normalized.get("workflow_graph")) else {}


def report_explainer_is_valid(explainer: Any) -> bool:
    if not isinstance(explainer, dict):
        return False
    summary = explainer.get("summary") if isinstance(explainer.get("summary"), dict) else {}
    summary_present = any(
        _stringify(summary.get(key))
        for key in ("headline", "plain_language_goal", "what_happened", "result")
    )
    walkthrough = explainer.get("walkthrough") if isinstance(explainer.get("walkthrough"), list) else []
    key_decisions = explainer.get("key_decisions") if isinstance(explainer.get("key_decisions"), list) else []
    branch_explanations = (
        explainer.get("branch_explanations")
        if isinstance(explainer.get("branch_explanations"), list)
        else []
    )
    return bool(summary_present and (walkthrough or key_decisions or branch_explanations))


def _workflow_graph_is_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    lanes = value.get("lanes") if isinstance(value.get("lanes"), list) else []
    nodes = value.get("nodes") if isinstance(value.get("nodes"), list) else []
    edges = value.get("edges") if isinstance(value.get("edges"), list) else []
    if not lanes or not nodes:
        return False
    node_ids = {str(node.get("id")) for node in nodes if isinstance(node, dict) and _stringify(node.get("id"))}
    if not node_ids:
        return False
    for edge in edges:
        if not isinstance(edge, dict):
            return False
        if _stringify(edge.get("source")) not in node_ids or _stringify(edge.get("target")) not in node_ids:
            return False
    return True


def _build_workflow_graph(explainer: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    report_manifest = context.get("report_manifest") if isinstance(context.get("report_manifest"), dict) else {}
    summary = explainer.get("summary") if isinstance(explainer.get("summary"), dict) else {}
    walkthrough = explainer.get("walkthrough") if isinstance(explainer.get("walkthrough"), list) else []
    decisions = explainer.get("key_decisions") if isinstance(explainer.get("key_decisions"), list) else []
    branch_explanations = (
        explainer.get("branch_explanations")
        if isinstance(explainer.get("branch_explanations"), list)
        else []
    )
    output_guides = explainer.get("output_guides") if isinstance(explainer.get("output_guides"), list) else []
    transcript_context = (
        context.get("transcript_context")
        if isinstance(context.get("transcript_context"), dict)
        else {}
    )
    clarifications = context.get("clarifications") if isinstance(context.get("clarifications"), list) else []
    routing_summary = (
        context.get("routing_summary")
        if isinstance(context.get("routing_summary"), dict)
        else {}
    )
    output_inventory = (
        context.get("output_inventory")
        if isinstance(context.get("output_inventory"), list)
        else []
    )
    branch_outcomes = (
        context.get("branch_outcomes")
        if isinstance(context.get("branch_outcomes"), dict)
        else {}
    )
    run_metadata = (
        context.get("run_metadata")
        if isinstance(context.get("run_metadata"), dict)
        else {}
    )

    steric_jobs_manifest = report_manifest.get("steric_jobs") if isinstance(report_manifest.get("steric_jobs"), list) else []
    steric_lane_jobs = _partition_steric_jobs_for_workflow(steric_jobs_manifest)
    active_branches = _expand_workflow_branches_for_graph(
        _active_workflow_branches(branch_explanations, branch_outcomes),
        steric_lane_jobs,
    )
    lanes = [{"id": "shared", "label": "Shared planning"}]
    lanes.extend(
        {"id": branch["key"], "label": branch["label"]}
        for branch in active_branches
    )

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    seen_ids: set[str] = set()

    def add_node(
        *,
        node_id: str,
        title: str,
        lane: str,
        column: int,
        tone: str,
        subtitle: str = "",
        description: str = "",
        why: str = "",
        evidence_refs: Optional[Sequence[str]] = None,
    ) -> str:
        if not node_id or node_id in seen_ids:
            return node_id
        node = {
            "id": node_id,
            "title": _stringify(title) or _humanize_key(node_id),
            "lane": lane or "shared",
            "lane_label": next((entry["label"] for entry in lanes if entry["id"] == lane), _humanize_key(lane)),
            "column": column,
            "tone": tone,
            "subtitle": _stringify(subtitle),
            "description": _stringify(description),
            "why": _stringify(why),
            "evidence_refs": [_stringify(ref) for ref in evidence_refs or [] if _stringify(ref)][:6],
        }
        nodes.append(node)
        seen_ids.add(node_id)
        return node_id

    def add_edge(source: Optional[str], target: Optional[str]) -> None:
        if not source or not target or source == target:
            return
        edge = {"source": source, "target": target}
        if edge not in edges:
            edges.append(edge)

    def output_guide_by_key(key: str) -> Dict[str, Any]:
        for guide in output_guides:
            if not isinstance(guide, dict):
                continue
            if _stringify(guide.get("key")) == key:
                return guide
        return {}

    def inventory_by_key(key: str) -> Dict[str, Any]:
        for item in output_inventory:
            if not isinstance(item, dict):
                continue
            if _stringify(item.get("key")) == key:
                return item
        return {}

    def walkthrough_match(*patterns: str, lane: str = "") -> Dict[str, Any]:
        wanted_lane = _stringify(lane).lower()
        lowered = [pattern.lower() for pattern in patterns if pattern]
        for item in walkthrough:
            if not isinstance(item, dict):
                continue
            item_lane = _stringify(item.get("lane")).lower() or "shared"
            if wanted_lane and item_lane != wanted_lane:
                continue
            haystack = " ".join(
                [
                    _stringify(item.get("title")),
                    _stringify(item.get("summary")),
                    _stringify(item.get("why_it_matters")),
                ]
            ).lower()
            if any(pattern in haystack for pattern in lowered):
                return item
        return {}

    def same_step(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if not left or not right:
            return False
        left_title = _stringify(left.get("title")).strip().lower()
        right_title = _stringify(right.get("title")).strip().lower()
        left_summary = _stringify(left.get("summary")).strip().lower()
        right_summary = _stringify(right.get("summary")).strip().lower()
        return bool(left_title and left_title == right_title and left_summary and left_summary == right_summary)

    def decision_match(*patterns: str) -> Dict[str, Any]:
        lowered = [pattern.lower() for pattern in patterns if pattern]
        for item in decisions:
            if not isinstance(item, dict):
                continue
            haystack = " ".join(
                [
                    _stringify(item.get("title")),
                    _stringify(item.get("decision")),
                    _stringify(item.get("why")),
                ]
            ).lower()
            if any(pattern in haystack for pattern in lowered):
                return item
        return {}

    chemistry = report_manifest.get("chemistry") if isinstance(report_manifest.get("chemistry"), dict) else {}
    chemistry_step = walkthrough_match("chemistry", lane="shared")
    chemistry_decision = decision_match("chemistry")

    def chemistry_recommendation(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        nested = payload.get("recommendation")
        if isinstance(nested, dict):
            return nested
        compact = {
            key: payload.get(key)
            for key in ("CHEMISTRY", "chemistry", "platform", "length", "rationale")
            if payload.get(key) not in (None, "", [], {})
        }
        return compact if compact else {}

    root_chemistry_recommendation = chemistry_recommendation(chemistry)

    def chemistry_branch_modality(branch_key: str) -> str:
        if branch_key in {"steric", "steric_splice", "steric_binding"}:
            return "steric"
        return branch_key

    def chemistry_matches_branch(branch_key: str, recommendation: Dict[str, Any]) -> bool:
        if not recommendation:
            return False
        modality_key = chemistry_branch_modality(branch_key)
        platform = _stringify(recommendation.get("platform")).strip().lower()
        chemistry_name = _stringify(
            recommendation.get("CHEMISTRY") or recommendation.get("chemistry")
        ).strip().lower()
        from aso_chemistry import chemistry_components, chemistry_label
        canonical = chemistry_label(chemistry_name)
        resolved_platform = chemistry_components(canonical)["platform"] if canonical else ""
        if modality_key == "steric":
            return resolved_platform == "steric_blocker" or "steric" in platform
        if modality_key == "gapmer":
            return resolved_platform == "gapmer" or "gapmer" in platform
        return False

    def chemistry_payload_for_branch(branch_key: str) -> Dict[str, Any]:
        by_modality = chemistry.get("by_modality") if isinstance(chemistry.get("by_modality"), dict) else {}
        modality_key = chemistry_branch_modality(branch_key)
        branch_payload = by_modality.get(modality_key)
        if isinstance(branch_payload, dict) and chemistry_recommendation(branch_payload):
            return branch_payload
        if chemistry_matches_branch(branch_key, root_chemistry_recommendation):
            return chemistry
        if len(active_branches) == 1 and root_chemistry_recommendation:
            only_branch = _stringify(active_branches[0].get("key"))
            if chemistry_branch_modality(only_branch) == modality_key:
                return chemistry
        return {}

    def add_branch_chemistry_node(
        branch_key: str,
        source_id: Optional[str],
        column: int,
        tone: str,
    ) -> tuple[Optional[str], int]:
        branch_payload = chemistry_payload_for_branch(branch_key)
        branch_recommendation = chemistry_recommendation(branch_payload)
        branch_label = _preferred_workflow_branch_label(branch_key, "")
        if branch_key == "steric_binding":
            branch_why = "Chemistry choice determines how the steric binding/blocking lane should be interpreted and executed."
        elif branch_key in {"steric", "steric_splice"}:
            branch_why = "Chemistry choice determines how the steric splice-modulation branch should be interpreted and executed."
        else:
            branch_why = "Chemistry choice determines which knockdown-oriented gapmer profile and oligo length the branch should use."
        evidence_refs = (
            chemistry_decision.get("evidence_refs")
            if isinstance(chemistry_decision.get("evidence_refs"), list)
            else ["report_manifest.chemistry"]
        )
        if branch_recommendation:
            chem_name = (
                _stringify(branch_recommendation.get("platform"))
                or _stringify(branch_recommendation.get("CHEMISTRY"))
                or _stringify(branch_recommendation.get("chemistry"))
            )
            chem_length = _stringify(branch_recommendation.get("length"))
            description = (
                _stringify(branch_recommendation.get("rationale"))
                or _stringify(chemistry_step.get("summary"))
                or _stringify(chemistry_decision.get("decision"))
            )
            if not description:
                if chem_name and chem_length:
                    description = f"The {branch_label.lower()} branch recommended {chem_name} at {chem_length} nt."
                elif chem_name:
                    description = f"The {branch_label.lower()} branch recommended {chem_name}."
                else:
                    description = f"The run produced {branch_label.lower()} chemistry guidance."
            chemistry_id = add_node(
                node_id=f"{branch_key}_chemistry",
                title=f"{branch_label} chemistry recommendation",
                lane=branch_key,
                column=column,
                tone=tone,
                subtitle=chem_name or "chemistry guidance",
                description=description,
                why=_stringify(chemistry_decision.get("why")) or _stringify(chemistry_step.get("why_it_matters")) or branch_why,
                evidence_refs=evidence_refs,
            )
            add_edge(source_id, chemistry_id)
            return chemistry_id, column + 1

        if root_chemistry_recommendation and len(active_branches) > 1:
            chemistry_id = add_node(
                node_id=f"{branch_key}_chemistry",
                title=f"{branch_label} chemistry context",
                lane=branch_key,
                column=column,
                tone=tone,
                subtitle="no dedicated branch-specific guidance",
                description=(
                    f"The run stored chemistry guidance only once at the request level, so it did not preserve a dedicated "
                    f"{branch_label.lower()} chemistry recommendation for this branch."
                ),
                why="This makes the branch structure explicit without implying that one chemistry recommendation applied equally to every request type.",
                evidence_refs=["report_manifest.chemistry"],
            )
            add_edge(source_id, chemistry_id)
            return chemistry_id, column + 1

        return source_id, column

    def steric_lane_primary_job(jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return next((job for job in jobs if isinstance(job, dict)), {})

    def steric_lane_subtitle(jobs: Sequence[Dict[str, Any]], fallback: str) -> str:
        primary_job = steric_lane_primary_job(jobs)
        if len(jobs) == 1 and primary_job:
            return _stringify(primary_job.get("label")) or fallback
        if len(jobs) > 1:
            return f"{len(jobs)} job(s)"
        return fallback

    def steric_binding_description(jobs: Sequence[Dict[str, Any]], step: Dict[str, Any]) -> str:
        if _stringify(step.get("summary")):
            return _stringify(step.get("summary"))
        primary_job = steric_lane_primary_job(jobs)
        primary_results = primary_job.get("results") if isinstance(primary_job.get("results"), dict) else {}
        row_count = primary_results.get("json_path_row_count")
        target_region = _stringify(primary_job.get("target_region"))
        if len(jobs) == 1:
            label = _stringify(primary_job.get("label")) or "Steric binding/blocking job"
            if row_count is not None:
                return f"{label} produced {row_count} tiled candidates" + (f" in {target_region}." if target_region else ".")
            return f"{label} produced non-splice steric candidate outputs."
        regions = sorted({
            _stringify(job.get("target_region"))
            for job in jobs
            if _stringify(job.get("target_region"))
        })
        region_text = f" across {', '.join(regions[:3])}" if regions else ""
        return f"{len(jobs)} non-splice steric job(s) produced binding/blocking candidate outputs{region_text}."

    def steric_binding_uniqueness_description(jobs: Sequence[Dict[str, Any]], guide: Dict[str, Any]) -> str:
        if _stringify(guide.get("what_it_is")):
            return _stringify(guide.get("what_it_is"))
        ranked_jobs = 0
        for job in jobs:
            results = job.get("results") if isinstance(job.get("results"), dict) else {}
            if _stringify(results.get("uniqueness_csv")):
                ranked_jobs += 1
        if ranked_jobs == 1:
            return "The non-splice steric candidates were ranked into a uniqueness-oriented output list."
        if ranked_jobs > 1:
            return f"Uniqueness-ranked candidate lists were produced for {ranked_jobs} non-splice steric jobs."
        return "The non-splice steric lane produced candidate outputs for review."

    shared_column = 0
    current_shared_id: Optional[str] = None

    goal_step = walkthrough_match("goal intake", "what the run was asked", "interpreted the request")
    goal_description = (
        _stringify(goal_step.get("summary"))
        or _stringify(summary.get("plain_language_goal"))
        or _stringify(run_metadata.get("user_request"))
        or _stringify(summary.get("headline"))
    )
    goal_why = _stringify(goal_step.get("why_it_matters")) or _stringify(summary.get("result"))
    current_shared_id = add_node(
        node_id="goal_intake",
        title=_stringify(goal_step.get("title")) or "Goal intake",
        lane="shared",
        column=shared_column,
        tone="shared",
        subtitle=_stringify(run_metadata.get("intent")),
        description=goal_description,
        why=goal_why,
        evidence_refs=(goal_step.get("evidence_refs") if isinstance(goal_step.get("evidence_refs"), list) else ["run_metadata.user_request"]),
    )
    shared_column += 1

    transcript_items = transcript_context.get("items") if isinstance(transcript_context.get("items"), list) else []
    clarification_step = walkthrough_match("clarification", lane="shared")
    transcript_step = walkthrough_match("transcript resolution", "transcript", "isoform", lane="shared")
    clarification_covers_transcripts = bool(
        clarifications
        and transcript_items
        and (
            same_step(clarification_step, transcript_step)
            or "transcript" in _stringify(clarification_step.get("title")).lower()
            or "isoform" in _stringify(clarification_step.get("title")).lower()
        )
    )

    if clarifications:
        clar_id = add_node(
            node_id="clarifications",
            title=(
                _stringify(clarification_step.get("title"))
                if clarification_step and not clarification_covers_transcripts
                else "Clarifications captured"
            ),
            lane="shared",
            column=shared_column,
            tone="shared",
            subtitle=f"{len(clarifications)} answer(s)",
            description=_stringify(clarification_step.get("summary")) or f"The run recorded {len(clarifications)} clarification answer(s) before downstream design work.",
            why=_stringify(clarification_step.get("why_it_matters")) or "Clarifications constrain transcript choices, branch selection, and route interpretation.",
            evidence_refs=(clarification_step.get("evidence_refs") if isinstance(clarification_step.get("evidence_refs"), list) else ["clarifications"]),
        )
        add_edge(current_shared_id, clar_id)
        current_shared_id = clar_id
        shared_column += 1

    if transcript_items:
        if same_step(transcript_step, goal_step) or same_step(transcript_step, clarification_step):
            transcript_step = {}
        transcript_description = _stringify(transcript_step.get("summary")) or " | ".join(
            _stringify(item.get("summary"))
            for item in transcript_items[:3]
            if isinstance(item, dict) and _stringify(item.get("summary"))
        )
        if clarification_covers_transcripts:
            current_shared_id = add_node(
                node_id="transcript_resolution",
                title="Transcript and isoform resolution",
                lane="shared",
                column=shared_column,
                tone="shared",
                subtitle=_stringify(transcript_context.get("policy")) or "reference selection",
                description=transcript_description,
                why=_stringify(transcript_step.get("why_it_matters")) or "This fixes what each isoform label means for routing, targeting, and output interpretation.",
                evidence_refs=(transcript_step.get("evidence_refs") if isinstance(transcript_step.get("evidence_refs"), list) else ["transcript_context.items"]),
            )
            add_edge("clarifications", current_shared_id)
            shared_column += 1
        else:
            transcript_id = add_node(
                node_id="transcript_resolution",
                title=_stringify(transcript_step.get("title")) or "Transcript and isoform resolution",
                lane="shared",
                column=shared_column,
                tone="shared",
                subtitle=_stringify(transcript_context.get("policy")) or "reference selection",
                description=transcript_description,
                why=_stringify(transcript_step.get("why_it_matters")) or "This fixes what each isoform label means for routing, targeting, and output interpretation.",
                evidence_refs=(transcript_step.get("evidence_refs") if isinstance(transcript_step.get("evidence_refs"), list) else ["transcript_context.items"]),
            )
            add_edge(current_shared_id, transcript_id)
            current_shared_id = transcript_id
            shared_column += 1

    split_source_id = current_shared_id
    branch_selection_step = walkthrough_match("design program", "branch", "workflow split", lane="shared")
    if len(active_branches) > 1:
        branch_titles = ", ".join(branch["label"] for branch in active_branches)
        split_id = add_node(
            node_id="branch_selection",
            title=_stringify(branch_selection_step.get("title")) or "Design branches selected",
            lane="shared",
            column=shared_column,
            tone="branch",
            subtitle=f"{len(active_branches)} branches",
            description=_stringify(branch_selection_step.get("summary")) or f"The run split into {branch_titles} as separate workstreams.",
            why=_stringify(branch_selection_step.get("why_it_matters")) or "Different design programs produce different decision paths and output types.",
            evidence_refs=(branch_selection_step.get("evidence_refs") if isinstance(branch_selection_step.get("evidence_refs"), list) else ["branch_outcomes"]),
        )
        add_edge(current_shared_id, split_id)
        split_source_id = split_id
        shared_column += 1

    branch_start_column = shared_column
    lane_last_nodes: Dict[str, str] = {}
    max_lane_column = branch_start_column

    steric_splice_branch = next((branch for branch in active_branches if branch["key"] == "steric_splice"), None)
    if steric_splice_branch:
        steric_column = branch_start_column
        splice_jobs = steric_lane_jobs.get("steric_splice") or []
        steric_job = steric_lane_primary_job(splice_jobs)
        tiling = report_manifest.get("tiling") if isinstance(report_manifest.get("tiling"), dict) else {}
        splicing_eval = report_manifest.get("splicing_eval") if isinstance(report_manifest.get("splicing_eval"), dict) else {}

        steric_source = split_source_id
        site_plan_step = walkthrough_match("site plan", "junction", "windows were explored", "splice-site plan", lane="steric")
        if site_plan_step or steric_job or report_manifest.get("splicing"):
            site_plan_id = add_node(
                node_id="steric_splice_site_plan",
                title=_stringify(site_plan_step.get("title")) or "Steric splice-site plan",
                lane="steric_splice",
                column=steric_column,
                tone="steric",
                subtitle=steric_lane_subtitle(splice_jobs, "steric planning"),
                description=_stringify(site_plan_step.get("summary")) or _stringify(steric_job.get("objective")) or "The run selected splice-site windows for steric exploration.",
                why=_stringify(site_plan_step.get("why_it_matters")) or "This defines which splice levers or windows the steric splice-modulation jobs will actually explore.",
                evidence_refs=(site_plan_step.get("evidence_refs") if isinstance(site_plan_step.get("evidence_refs"), list) else ["report_manifest.steric_jobs"]),
            )
            add_edge(steric_source, site_plan_id)
            steric_source = site_plan_id
            lane_last_nodes["steric_splice"] = site_plan_id
            steric_column += 1

        previous_steric_source = steric_source
        steric_source, steric_column = add_branch_chemistry_node("steric_splice", steric_source, steric_column, "steric")
        if steric_source and steric_source != previous_steric_source:
            lane_last_nodes["steric_splice"] = steric_source

        tiling_step = walkthrough_match("steric tiling", "candidate aso tiles", lane="steric")
        if tiling or steric_job:
            passed = ((tiling.get("initial_filtering") or {}).get("passed") or {}).get("count")
            total = (tiling.get("initial_filtering") or {}).get("total_rows")
            tiling_summary = _stringify(tiling_step.get("summary"))
            if not tiling_summary:
                if passed is not None and total is not None:
                    tiling_summary = f"Steric tiling kept {passed} of {total} candidates after initial filtering."
                else:
                    tiling_summary = "Steric tiling generated and filtered candidate ASO tiles."
            tiling_id = add_node(
                node_id="steric_splice_tiling",
                title=_stringify(tiling_step.get("title")) or "Steric tiling",
                lane="steric_splice",
                column=steric_column,
                tone="steric",
                subtitle="candidate generation",
                description=tiling_summary,
                why=_stringify(tiling_step.get("why_it_matters")) or "This stage defines the steric splice-modulation candidate pool before scoring and route prioritization.",
                evidence_refs=(tiling_step.get("evidence_refs") if isinstance(tiling_step.get("evidence_refs"), list) else ["report_manifest.tiling"]),
            )
            add_edge(steric_source, tiling_id)
            steric_source = tiling_id
            lane_last_nodes["steric_splice"] = tiling_id
            steric_column += 1

        splice_eval_step = walkthrough_match("splice evaluation", lane="steric")
        if splicing_eval:
            site_count = len(splicing_eval.get("sites") or []) if isinstance(splicing_eval.get("sites"), list) else splicing_eval.get("site_count")
            eval_id = add_node(
                node_id="steric_splice_evaluation",
                title=_stringify(splice_eval_step.get("title")) or "Splice evaluation",
                lane="steric_splice",
                column=steric_column,
                tone="steric",
                subtitle=f"{site_count or 0} site(s)",
                description=_stringify(splice_eval_step.get("summary")) or _stringify(output_guide_by_key("splicing_eval").get("what_it_is")) or "The steric candidates were scored for predicted splice effects.",
                why=_stringify(splice_eval_step.get("why_it_matters")) or _stringify(output_guide_by_key("splicing_eval").get("why_it_matters")) or "This connects steric splice-modulation candidates to predicted splice-effect evidence.",
                evidence_refs=(splice_eval_step.get("evidence_refs") if isinstance(splice_eval_step.get("evidence_refs"), list) else ["report_manifest.splicing_eval"]),
            )
            add_edge(steric_source, eval_id)
            steric_source = eval_id
            lane_last_nodes["steric_splice"] = eval_id
            steric_column += 1

        routing_step = walkthrough_match("strategy routing", "route goal", lane="steric")
        if routing_summary.get("goal_count") or report_manifest.get("strategy_routing"):
            top_goal = None
            for goal in routing_summary.get("top_goals") or []:
                if isinstance(goal, dict):
                    top_goal = goal
                    break
            routing_id = add_node(
                node_id="steric_splice_strategy_routing",
                title=_stringify(routing_step.get("title")) or "Strategy routing",
                lane="steric_splice",
                column=steric_column,
                tone="steric",
                subtitle=f"{routing_summary.get('goal_count') or 0} route goals",
                description=_stringify(routing_step.get("summary")) or _stringify(output_guide_by_key("strategy_routing").get("what_it_is")) or (
                    f"Strategy routing ranked {routing_summary.get('goal_count') or 0} splice-site goals." + (
                        f" Top route: {_stringify((top_goal or {}).get('goal_label'))}." if top_goal else ""
                    )
                ),
                why=_stringify(routing_step.get("why_it_matters")) or _stringify(output_guide_by_key("strategy_routing").get("why_it_matters")) or "Routing turns scored steric splice-modulation candidates into prioritized splice-route outputs.",
                evidence_refs=(routing_step.get("evidence_refs") if isinstance(routing_step.get("evidence_refs"), list) else ["report_manifest.strategy_routing"]),
            )
            add_edge(steric_source, routing_id)
            steric_source = routing_id
            lane_last_nodes["steric_splice"] = routing_id
            steric_column += 1

        if "steric_splice" not in lane_last_nodes:
            generic_id = add_node(
                node_id="steric_splice_branch",
                title=steric_splice_branch["label"],
                lane="steric_splice",
                column=steric_column,
                tone="steric",
                subtitle=_stringify(steric_splice_branch.get("status")),
                description=_stringify(steric_splice_branch.get("summary")) or "Steric splice-modulation work was part of this run.",
                why=_stringify(steric_splice_branch.get("why")),
                evidence_refs=steric_splice_branch.get("evidence_refs") or ["branch_outcomes.steric"],
            )
            add_edge(steric_source, generic_id)
            lane_last_nodes["steric_splice"] = generic_id
            steric_column += 1

        max_lane_column = max(max_lane_column, steric_column)

    steric_binding_branch = next((branch for branch in active_branches if branch["key"] == "steric_binding"), None)
    if steric_binding_branch:
        binding_column = branch_start_column
        binding_jobs = steric_lane_jobs.get("steric_binding") or []
        binding_job = steric_lane_primary_job(binding_jobs)
        binding_results = binding_job.get("results") if isinstance(binding_job.get("results"), dict) else {}

        binding_source = split_source_id
        previous_binding_source = binding_source
        binding_source, binding_column = add_branch_chemistry_node("steric_binding", binding_source, binding_column, "steric")
        if binding_source and binding_source != previous_binding_source:
            lane_last_nodes["steric_binding"] = binding_source

        binding_step = walkthrough_match("translation start", "translation-blocking", "block translation", "binding-only", lane="steric")
        if binding_job or binding_jobs:
            binding_id = add_node(
                node_id="steric_binding_tiling",
                title=_stringify(binding_step.get("title")) or "Steric binding/blocking candidates",
                lane="steric_binding",
                column=binding_column,
                tone="steric",
                subtitle=steric_lane_subtitle(binding_jobs, "binding/blocking lane"),
                description=steric_binding_description(binding_jobs, binding_step),
                why=_stringify(binding_step.get("why_it_matters")) or "These non-splice steric jobs are interpreted as binding/blocking candidates rather than splice-effect candidates.",
                evidence_refs=(binding_step.get("evidence_refs") if isinstance(binding_step.get("evidence_refs"), list) else ["report_manifest.steric_jobs"]),
            )
            add_edge(binding_source, binding_id)
            binding_source = binding_id
            lane_last_nodes["steric_binding"] = binding_id
            binding_column += 1

        uniqueness_guide = output_guide_by_key("steric_jobs")
        if _stringify(binding_results.get("uniqueness_csv")) or any(
            _stringify((job.get("results") or {}).get("uniqueness_csv"))
            for job in binding_jobs
            if isinstance(job, dict)
        ):
            uniqueness_id = add_node(
                node_id="steric_binding_uniqueness",
                title="Steric binding/blocking ranking",
                lane="steric_binding",
                column=binding_column,
                tone="steric",
                subtitle="ranked output",
                description=steric_binding_uniqueness_description(binding_jobs, uniqueness_guide),
                why=_stringify(uniqueness_guide.get("why_it_matters")) or "This is the prioritized list a user would inspect first for non-splice steric jobs.",
                evidence_refs=["report_manifest.steric_jobs"],
            )
            add_edge(binding_source, uniqueness_id)
            binding_source = uniqueness_id
            lane_last_nodes["steric_binding"] = uniqueness_id
            binding_column += 1

        if "steric_binding" not in lane_last_nodes:
            generic_binding_id = add_node(
                node_id="steric_binding_branch",
                title=steric_binding_branch["label"],
                lane="steric_binding",
                column=binding_column,
                tone="steric",
                subtitle=_stringify(steric_binding_branch.get("status")),
                description=_stringify(steric_binding_branch.get("summary")) or "Non-splice steric binding/blocking work was part of this run.",
                why=_stringify(steric_binding_branch.get("why")),
                evidence_refs=steric_binding_branch.get("evidence_refs") or ["report_manifest.steric_jobs"],
            )
            add_edge(binding_source, generic_binding_id)
            lane_last_nodes["steric_binding"] = generic_binding_id
            binding_column += 1

        max_lane_column = max(max_lane_column, binding_column)

    gapmer_branch = next((branch for branch in active_branches if branch["key"] == "gapmer"), None)
    if gapmer_branch:
        gapmer_column = branch_start_column
        gapmer_source = split_source_id
        gapmer_tiling = report_manifest.get("gapmer_tiling") if isinstance(report_manifest.get("gapmer_tiling"), dict) else {}
        gapmer_jobs = report_manifest.get("gapmer_jobs") if isinstance(report_manifest.get("gapmer_jobs"), list) else []
        gapmer_job = gapmer_jobs[0] if gapmer_jobs and isinstance(gapmer_jobs[0], dict) else {}
        gapmer_results = gapmer_job.get("results") if isinstance(gapmer_job.get("results"), dict) else {}

        previous_gapmer_source = gapmer_source
        gapmer_source, gapmer_column = add_branch_chemistry_node("gapmer", gapmer_source, gapmer_column, "gapmer")
        if gapmer_source and gapmer_source != previous_gapmer_source:
            lane_last_nodes["gapmer"] = gapmer_source

        gapmer_step = walkthrough_match("gapmer tiling", "knockdown candidate", lane="gapmer")
        if gapmer_tiling or gapmer_job:
            gapmer_count = gapmer_results.get("json_path_row_count") or gapmer_tiling.get("row_count")
            gapmer_target = _stringify(gapmer_results.get("transcript_id")) or _stringify(gapmer_tiling.get("transcript_id")) or _stringify(gapmer_tiling.get("primary_isoform"))
            gapmer_description = _stringify(gapmer_step.get("summary"))
            if not gapmer_description:
                if gapmer_count and gapmer_target:
                    gapmer_description = f"Gapmer tiling produced {gapmer_count} candidates for {gapmer_target}."
                else:
                    gapmer_description = "Gapmer tiling produced a knockdown-oriented candidate set."
            gapmer_tiling_id = add_node(
                node_id="gapmer_tiling",
                title="Gapmer tiling",
                lane="gapmer",
                column=gapmer_column,
                tone="gapmer",
                subtitle="knockdown candidates",
                description=gapmer_description,
                why=_stringify(gapmer_step.get("why_it_matters")) or "This stage creates the knockdown-oriented ASO candidate set for the gapmer branch.",
                evidence_refs=(gapmer_step.get("evidence_refs") if isinstance(gapmer_step.get("evidence_refs"), list) else ["report_manifest.gapmer_tiling"]),
            )
            add_edge(gapmer_source, gapmer_tiling_id)
            gapmer_source = gapmer_tiling_id
            lane_last_nodes["gapmer"] = gapmer_tiling_id
            gapmer_column += 1

        gapmer_guide = output_guide_by_key("gapmer_uniqueness_ranked")
        if gapmer_guide or gapmer_tiling.get("uniqueness_csv") or gapmer_results.get("uniqueness_csv"):
            uniqueness_id = add_node(
                node_id="gapmer_uniqueness",
                title=_stringify(gapmer_guide.get("title")) or "Gapmer uniqueness ranking",
                lane="gapmer",
                column=gapmer_column,
                tone="gapmer",
                subtitle="ranked output",
                description=_stringify(gapmer_guide.get("what_it_is")) or "The gapmer candidates were ranked into a uniqueness-oriented output list.",
                why=_stringify(gapmer_guide.get("why_it_matters")) or "This is the prioritized list that a user would inspect first on the gapmer side.",
                evidence_refs=(gapmer_guide.get("evidence_refs") if isinstance(gapmer_guide.get("evidence_refs"), list) else ["report_manifest.gapmer_tiling"]),
            )
            add_edge(gapmer_source, uniqueness_id)
            gapmer_source = uniqueness_id
            lane_last_nodes["gapmer"] = uniqueness_id
            gapmer_column += 1

        if "gapmer" not in lane_last_nodes:
            generic_gapmer_id = add_node(
                node_id="gapmer_branch",
                title=gapmer_branch["label"],
                lane="gapmer",
                column=gapmer_column,
                tone="gapmer",
                subtitle=_stringify(gapmer_branch.get("status")),
                description=_stringify(gapmer_branch.get("summary")) or "Gapmer branch work was part of this run.",
                why=_stringify(gapmer_branch.get("why")),
                evidence_refs=gapmer_branch.get("evidence_refs") or ["branch_outcomes.gapmer"],
            )
            add_edge(gapmer_source, generic_gapmer_id)
            lane_last_nodes["gapmer"] = generic_gapmer_id
            gapmer_column += 1

        max_lane_column = max(max_lane_column, gapmer_column)

    for branch in active_branches:
        if branch["key"] in {"steric", "gapmer"} or branch["key"] in lane_last_nodes:
            continue
        generic_id = add_node(
            node_id=f"{branch['key']}_branch",
            title=branch["label"],
            lane=branch["key"],
            column=branch_start_column,
            tone="branch",
            subtitle=_stringify(branch.get("status")),
            description=_stringify(branch.get("summary")) or f"{branch['label']} was part of the run.",
            why=_stringify(branch.get("why")),
            evidence_refs=branch.get("evidence_refs") or [f"branch_outcomes.{branch['key']}"],
        )
        add_edge(split_source_id, generic_id)
        lane_last_nodes[branch["key"]] = generic_id
        max_lane_column = max(max_lane_column, branch_start_column + 1)

    last_shared_id = current_shared_id
    if lane_last_nodes:
        merge_id = None
        if len(lane_last_nodes) > 1:
            merge_id = add_node(
                node_id="merged_branch_outputs",
                title="Merged branch outputs",
                lane="shared",
                column=max_lane_column,
                tone="branch",
                subtitle=f"{len(lane_last_nodes)} incoming paths",
                description="Separate branch results are brought back together here before final reporting.",
                why="This marks the point where branch-specific work becomes one combined report surface.",
                evidence_refs=["branch_outcomes"],
            )
            for node_id in lane_last_nodes.values():
                add_edge(node_id, merge_id)
            last_shared_id = merge_id
            shared_column = max(shared_column, max_lane_column + 1)
        else:
            last_shared_id = next(iter(lane_last_nodes.values()))
            shared_column = max(shared_column, max_lane_column)

    output_review_step = walkthrough_match("persisted outputs", "review", "inspect", lane="shared")
    if output_review_step:
        review_id = add_node(
            node_id="output_review",
            title=_stringify(output_review_step.get("title")) or "Output review",
            lane="shared",
            column=shared_column,
            tone="shared",
            subtitle="artifacts ready",
            description=_stringify(output_review_step.get("summary")),
            why=_stringify(output_review_step.get("why_it_matters")) or "This is the point where the run turns into inspectable artifacts for a human reader.",
            evidence_refs=(output_review_step.get("evidence_refs") if isinstance(output_review_step.get("evidence_refs"), list) else ["important_artifacts"]),
        )
        add_edge(last_shared_id, review_id)
        last_shared_id = review_id
        shared_column += 1

    report_id = add_node(
        node_id="reports_generated",
        title="Reports generated",
        lane="shared",
        column=max(shared_column, max_lane_column if lane_last_nodes else shared_column),
        tone="report",
        subtitle=f"{len(context.get('important_artifacts') or [])} tracked artifacts",
        description=_stringify(summary.get("result")) or "The run produced linked report artifacts and branch outputs.",
        why="This is the final point where a user can inspect the run outputs as one coherent package.",
        evidence_refs=["important_artifacts"],
    )
    add_edge(last_shared_id, report_id)

    return {
        "lanes": lanes,
        "nodes": nodes,
        "edges": edges,
    }


def _active_workflow_branches(
    branch_explanations: Sequence[Dict[str, Any]],
    branch_outcomes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    branch_map: Dict[str, Dict[str, Any]] = {}
    for item in branch_explanations:
        if not isinstance(item, dict):
            continue
        key = _slugify(item.get("branch"))
        if not key:
            continue
        status = _stringify(item.get("status")).lower() or _stringify((branch_outcomes.get(key) or {}).get("status")).lower()
        if status in {"skipped", "not_run"}:
            continue
        branch_map[key] = {
            "key": key,
            "label": _preferred_workflow_branch_label(key, item.get("title")),
            "status": status or "unknown",
            "summary": _stringify(item.get("summary")) or _stringify((branch_outcomes.get(key) or {}).get("reason")),
            "why": _stringify(item.get("why")),
            "evidence_refs": item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else [f"branch_outcomes.{key}"],
        }
    for key, value in branch_outcomes.items():
        if key in branch_map or not isinstance(value, dict):
            continue
        status = _stringify(value.get("status")).lower()
        if status in {"skipped", "not_run"}:
            continue
        branch_map[key] = {
            "key": key,
            "label": _preferred_workflow_branch_label(key, ""),
            "status": status or "unknown",
            "summary": _stringify(value.get("reason")),
            "why": _branch_status_explanation(status),
            "evidence_refs": [f"branch_outcomes.{key}"],
        }
    order = {"steric_splice": 0, "steric": 0, "steric_binding": 1, "gapmer": 2}
    return sorted(branch_map.values(), key=lambda item: (order.get(item["key"], 99), item["label"]))


def _preferred_workflow_branch_label(branch_key: str, raw_title: Any) -> str:
    cleaned = _stringify(raw_title).replace("-", " ")
    cleaned = re.sub(r"\bbranch\b", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        return cleaned
    labels = {
        "steric": "Steric",
        "steric_splice": "Steric splice modulation",
        "steric_binding": "Steric binding / blocking",
        "gapmer": "Gapmer knockdown",
    }
    return labels.get(branch_key, _humanize_key(branch_key))


def _workflow_steric_lane_key(job: Any) -> str:
    if not isinstance(job, dict):
        return ""
    if bool(job.get("run_splice_eval")) or bool(job.get("run_splice_router")):
        return "steric_splice"
    intent = _stringify(job.get("intent")).lower()
    if intent == "splice_modulation":
        return "steric_splice"
    return "steric_binding"


def _partition_steric_jobs_for_workflow(jobs: Any) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"steric_splice": [], "steric_binding": []}
    for job in jobs if isinstance(jobs, list) else []:
        if not isinstance(job, dict):
            continue
        lane_key = _workflow_steric_lane_key(job)
        if lane_key in grouped:
            grouped[lane_key].append(job)
    return grouped


def _expand_workflow_branches_for_graph(
    active_branches: Sequence[Dict[str, Any]],
    steric_lane_jobs: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for branch in active_branches:
        if not isinstance(branch, dict):
            continue
        branch_key = _stringify(branch.get("key"))
        if branch_key != "steric":
            expanded.append(branch)
            continue

        splice_jobs = steric_lane_jobs.get("steric_splice") or []
        binding_jobs = steric_lane_jobs.get("steric_binding") or []
        if splice_jobs:
            expanded.append({**branch, "key": "steric_splice", "label": _preferred_workflow_branch_label("steric_splice", "")})
        if binding_jobs:
            expanded.append({**branch, "key": "steric_binding", "label": _preferred_workflow_branch_label("steric_binding", "")})
        if not splice_jobs and not binding_jobs:
            expanded.append({**branch, "key": "steric_splice", "label": _preferred_workflow_branch_label("steric_splice", "")})

    order = {"steric_splice": 0, "steric_binding": 1, "gapmer": 2, "steric": 0}
    return sorted(expanded, key=lambda item: (order.get(_stringify(item.get("key")), 99), _stringify(item.get("label"))))


def build_recorded_report_explainer(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build navigation from recorded results; the canonical report explains science.

    A second generated narrative can contradict the final tables or an export
    created after the first report stage. Keep the rich workflow/file navigation,
    but derive its factual claims from the same recorded state as the report.
    """
    meta = context.get("run_metadata") or {}
    outcomes = context.get("branch_outcomes") or {}
    manifest = context.get("report_manifest") or {}
    branches = []
    decisions = []
    for name, record in outcomes.items():
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "unknown")
        reason = str(record.get("reason") or "")
        branches.append({"branch": name, "title": _humanize_key(name), "status": status,
                         "summary": reason or f"Recorded branch status: {status}.",
                         "why": _branch_status_explanation(status),
                         "evidence_refs": [f"branch_outcomes.{name}"]})
        decisions.append({"title": _humanize_key(name), "decision": f"Recorded status: {status}.",
                          "why": reason or _branch_status_explanation(status), "confidence": "unknown",
                          "evidence_refs": [f"branch_outcomes.{name}"]})
    inventory = [row for row in context.get("output_inventory") or [] if isinstance(row, dict)]
    guides = [{"key": item.get("key"), "title": item.get("title"),
               "what_it_is": item.get("summary"), "how_to_read": item.get("how_to_read"),
               "why_it_matters": item.get("why_it_matters"), "evidence_refs": item.get("evidence_refs", [])}
              for item in inventory]
    artifacts = [item for item in context.get("important_artifacts") or [] if isinstance(item, dict)]
    statuses = [f"{_humanize_key(name)}: {item.get('status') or 'unknown'}"
                for name, item in outcomes.items() if isinstance(item, dict)]
    walkthrough = [{"title": "Read the design report",
                    "summary": "The report contains the recorded candidate sets, reference identities, filtering scopes and scientific interpretation.",
                    "why_it_matters": "Interpret each result with its own reference, metric and uncertainty.",
                    "lane": "shared", "evidence_refs": ["report_manifest"]}]
    if artifacts:
        walkthrough.append({"title": "Inspect the saved artifacts",
                            "summary": "The output inventory links the files present at report generation.",
                            "why_it_matters": "The complete tables and sequence exports retain the underlying evidence.",
                            "lane": "shared", "evidence_refs": ["important_artifacts"]})
    raw = {"schema_version": 1, "audience": "novice",
           "summary": {"headline": f"{meta.get('gene_symbol') or 'ASO'} design report",
                       "plain_language_goal": str(meta.get("user_request") or ""),
                       "what_happened": "The report and linked artifacts record the executed design work.",
                       "result": "; ".join(statuses) or "Consult the report for the recorded outcome.",
                       "bullets": []},
           "start_here": ["Read the report, then inspect the linked candidate tables and reference evidence.",
                          "Binding-region membership and predicted splice-response routes describe different properties.",
                          "Use each screening metric's stated reference universe and mismatch policy."],
           "walkthrough": walkthrough, "key_decisions": decisions, "branch_explanations": branches,
           "output_guides": guides,
           "caveats": [str(x) for x in manifest.get("pipeline_alerts") or [] if isinstance(x, str)] +
                      ["Computational design and screening do not establish cellular activity, efficacy or safety.",
                       "An unavailable measurement is unknown; a recorded zero applies only to its stated screen."],
           "glossary": [], "tooltips": _fallback_tooltips(context)}
    result = normalize_report_explainer(raw, context)
    if result:
        result["source"] = "recorded_output_navigation"
    return result


def build_fallback_report_explainer(context: Dict[str, Any]) -> Dict[str, Any]:
    meta = context.get("run_metadata") if isinstance(context.get("run_metadata"), dict) else {}
    branch_outcomes = context.get("branch_outcomes") if isinstance(context.get("branch_outcomes"), dict) else {}
    output_inventory = context.get("output_inventory") if isinstance(context.get("output_inventory"), list) else []
    clarifications = context.get("clarifications") if isinstance(context.get("clarifications"), list) else []
    transcript_context = context.get("transcript_context") if isinstance(context.get("transcript_context"), dict) else {}
    routing_summary = context.get("routing_summary") if isinstance(context.get("routing_summary"), dict) else {}
    report_manifest = context.get("report_manifest") if isinstance(context.get("report_manifest"), dict) else {}
    tiling = report_manifest.get("tiling") if isinstance(report_manifest.get("tiling"), dict) else {}
    splicing_eval = report_manifest.get("splicing_eval") if isinstance(report_manifest.get("splicing_eval"), dict) else {}
    gapmer_tiling = report_manifest.get("gapmer_tiling") if isinstance(report_manifest.get("gapmer_tiling"), dict) else {}
    steric_jobs = report_manifest.get("steric_jobs") if isinstance(report_manifest.get("steric_jobs"), list) else []
    gapmer_jobs = report_manifest.get("gapmer_jobs") if isinstance(report_manifest.get("gapmer_jobs"), list) else []
    chemistry = report_manifest.get("chemistry") if isinstance(report_manifest.get("chemistry"), dict) else {}
    recommendation = chemistry.get("recommendation") if isinstance(chemistry.get("recommendation"), dict) else {}

    branch_labels = [
        f"{_humanize_key(branch)}: {str((entry or {}).get('status') or 'unknown')}"
        for branch, entry in sorted(branch_outcomes.items())
        if isinstance(entry, dict)
    ]
    available_outputs = [item.get("title") for item in output_inventory if isinstance(item, dict) and item.get("title")]
    transcript_lines = transcript_context.get("items") if isinstance(transcript_context.get("items"), list) else []
    top_route = None
    for goal in routing_summary.get("top_goals") or []:
        if isinstance(goal, dict):
            top_route = goal
            break

    summary_bullets = []
    if meta.get("intent"):
        summary_bullets.append(f"Intent: {meta['intent']}.")
    if meta.get("target_context"):
        summary_bullets.append(f"Context: {meta['target_context']}.")
    if branch_labels:
        summary_bullets.append("Branch outcomes: " + "; ".join(branch_labels[:4]) + ".")
    if top_route and top_route.get("goal_label"):
        summary_bullets.append(
            f"Top routed goal: {top_route.get('goal_label')} ({top_route.get('count') or 0} candidates)."
        )
    if available_outputs:
        summary_bullets.append("Major outputs: " + ", ".join(available_outputs[:5]) + ".")

    walkthrough: List[Dict[str, Any]] = []

    _append_walkthrough_step(
        walkthrough,
        title="Interpreted the request",
        summary=_fallback_goal_sentence(meta),
        why="This determines which design branches and outputs make sense for the run.",
        evidence_refs=[
            "run_metadata.user_request",
            "report_manifest.structured_goal.intent",
            "report_manifest.structured_goal.desired_effect",
        ],
        lane="shared",
    )

    if clarifications:
        _append_walkthrough_step(
            walkthrough,
            title="Captured clarification answers",
            summary=f"The run recorded {len(clarifications)} clarification answer(s) before committing to downstream design choices.",
            why="Clarifications often explain why one transcript, isoform label, or branch was chosen over another.",
            evidence_refs=["clarifications"],
            lane="shared",
        )

    if transcript_lines:
        transcript_summary = " | ".join(
            _stringify(item.get("summary"))
            for item in transcript_lines[:3]
            if isinstance(item, dict) and _stringify(item.get("summary"))
        )
        _append_walkthrough_step(
            walkthrough,
            title="Resolved transcript and isoform context",
            summary=transcript_summary or "Transcript or isoform context was selected for downstream targeting.",
            why="Transcript choice determines which splice sites, exons, or UTR regions the rest of the run refers to.",
            evidence_refs=["transcript_context.items", "report_manifest.structured_goal.transcript_policy"],
            lane="shared",
        )

    if recommendation:
        chemistry_name = (
            _stringify(recommendation.get("platform"))
            or _stringify(recommendation.get("CHEMISTRY"))
            or _stringify(recommendation.get("chemistry"))
        )
        length = _stringify(recommendation.get("length"))
        chemistry_summary = f"The run recommended {chemistry_name}." if chemistry_name else "The run produced chemistry guidance."
        if chemistry_name and length:
            chemistry_summary = f"The run recommended {chemistry_name} at {length} nt for the relevant design workstream."
        _append_walkthrough_step(
            walkthrough,
            title="Selected ASO chemistry guidance",
            summary=chemistry_summary,
            why="Chemistry choice determines whether the ASOs are intended to block splicing sterically or to drive transcript knockdown.",
            evidence_refs=["report_manifest.chemistry"],
            lane="shared",
        )

    steric_job = steric_jobs[0] if steric_jobs and isinstance(steric_jobs[0], dict) else {}
    steric_results = steric_job.get("results") if isinstance(steric_job.get("results"), dict) else {}
    initial_filtering = tiling.get("initial_filtering") if isinstance(tiling.get("initial_filtering"), dict) else {}
    if tiling or steric_job or isinstance(branch_outcomes.get("steric"), dict):
        steric_total = _as_int(initial_filtering.get("total_rows")) or _as_int(steric_results.get("json_path_row_count"))
        steric_passed = _as_int(((initial_filtering.get("passed") or {}).get("count"))) or _as_int(steric_results.get("filtered_tiles_path_row_count"))
        region_count = _as_int(steric_job.get("region_count"))
        steric_parts: List[str] = []
        if region_count is not None:
            steric_parts.append(f"planned {region_count} region(s)")
        if steric_total is not None and steric_passed is not None:
            steric_parts.append(f"kept {steric_passed} of {steric_total} tiled candidates after initial filtering")
        elif steric_total is not None:
            steric_parts.append(f"generated {steric_total} tiled candidates")
        steric_summary = "Steric tiling ran"
        if steric_parts:
            steric_summary += " and " + "; ".join(steric_parts) + "."
        else:
            steric_summary += " and produced persisted candidate outputs."
        _append_walkthrough_step(
            walkthrough,
            title="Ran steric tiling across splice-relevant windows",
            summary=steric_summary,
            why="This defines the steric candidate space before splice scoring and route ranking.",
            evidence_refs=[
                "report_manifest.tiling",
                "report_manifest.steric_jobs",
            ],
            lane="steric",
        )

    if splicing_eval:
        site_count = len(splicing_eval.get("sites") or []) if isinstance(splicing_eval.get("sites"), list) else _as_int(splicing_eval.get("site_count"))
        evaluated = _as_int(splicing_eval.get("evaluated")) or _as_int(splicing_eval.get("actual_evaluated"))
        repeat_risk = splicing_eval.get("repeat_risk") if isinstance(splicing_eval.get("repeat_risk"), dict) else {}
        splice_summary = "Splice evaluation ran for the steric workstream."
        if evaluated is not None and site_count is not None:
            splice_summary = f"Splice evaluation scored {evaluated} steric candidates across {site_count} splice site(s)."
        elif site_count is not None:
            splice_summary = f"Splice evaluation ran across {site_count} splice site(s)."
        if repeat_risk:
            repeat_rows = _as_int(repeat_risk.get("repeat_row_count"))
            if repeat_rows:
                splice_summary += (
                    f" Repeat-aware off-target scoring annotated {repeat_rows} candidate(s) with alternate exact hits; "
                    "same-gene or same-transcript repeats are flagged for functional splice risk rather than automatically discarded."
                )
        _append_walkthrough_step(
            walkthrough,
            title="Scored steric candidates with splice evaluation",
            summary=splice_summary,
            why=(
                "This is the stage that estimates predicted splice effects before route ranking. "
                "A Times_mapped value above 1 inside the same gene or transcript is not automatically disqualifying; "
                "the key question is whether alternate repeat hits create unwanted splice effects."
            ),
            evidence_refs=["report_manifest.splicing_eval", "report_manifest.tiling.scored_tiles_path"],
            lane="steric",
        )

    if routing_summary.get("goal_count") or report_manifest.get("strategy_routing"):
        route_goal_count = _as_int(routing_summary.get("goal_count"))
        top_goal_label = _stringify((top_route or {}).get("goal_label"))
        route_summary = "Strategy routing ranked splice-site goals for the steric candidates."
        if route_goal_count is not None and top_goal_label:
            route_summary = f"Strategy routing ranked {route_goal_count} splice-site goal(s); the top route was {top_goal_label}."
        elif route_goal_count is not None:
            route_summary = f"Strategy routing ranked {route_goal_count} splice-site goal(s)."
        _append_walkthrough_step(
            walkthrough,
            title="Ranked splice route goals",
            summary=route_summary,
            why="Routing turns scored steric candidates into prioritized splice levers and per-goal candidate lists.",
            evidence_refs=["routing_summary", "report_manifest.strategy_routing"],
            lane="steric",
        )

    gapmer_job = gapmer_jobs[0] if gapmer_jobs and isinstance(gapmer_jobs[0], dict) else {}
    gapmer_results = gapmer_job.get("results") if isinstance(gapmer_job.get("results"), dict) else {}
    if gapmer_tiling or gapmer_job or isinstance(branch_outcomes.get("gapmer"), dict):
        gapmer_count = _as_int(gapmer_results.get("json_path_row_count")) or _as_int(gapmer_tiling.get("row_count"))
        gapmer_transcript = (
            _stringify(gapmer_results.get("transcript_id"))
            or _stringify(gapmer_tiling.get("transcript_id"))
            or _stringify(gapmer_tiling.get("primary_isoform"))
        )
        gapmer_summary = "Gapmer tiling produced a knockdown-oriented candidate set."
        if gapmer_count is not None and gapmer_transcript:
            gapmer_summary = f"Gapmer tiling produced {gapmer_count} knockdown candidates for {gapmer_transcript}."
        elif gapmer_count is not None:
            gapmer_summary = f"Gapmer tiling produced {gapmer_count} knockdown candidates."
        _append_walkthrough_step(
            walkthrough,
            title="Generated a gapmer knockdown candidate set",
            summary=gapmer_summary,
            why="This branch provides the knockdown-oriented modality separately from steric splice modulation.",
            evidence_refs=["report_manifest.gapmer_tiling", "report_manifest.gapmer_jobs"],
            lane="gapmer",
        )

    if available_outputs:
        _append_walkthrough_step(
            walkthrough,
            title="Persisted outputs for review",
            summary="The run saved interpretable outputs including " + ", ".join(available_outputs[:5]) + ".",
            why="These are the main artifacts a new user should inspect to understand what the pipeline produced.",
            evidence_refs=["output_inventory", "important_artifacts"],
            lane="shared",
        )

    key_decisions: List[Dict[str, Any]] = []
    if transcript_lines:
        key_decisions.append(
            {
                "title": "Transcript selection",
                "decision": " | ".join(
                    _stringify(item.get("summary"))
                    for item in transcript_lines[:3]
                    if isinstance(item, dict) and _stringify(item.get("summary"))
                ) or "Transcript context was selected for downstream work.",
                "why": "This anchors every downstream coordinate, splice-site label, and isoform-specific statement in the report.",
                "confidence": "medium",
                "evidence_refs": ["transcript_context.items", "report_manifest.structured_goal.transcript_policy"],
            }
        )
    steric_job_flags = steric_job if steric_job else {}
    gapmer_job_flags = gapmer_job if gapmer_job else {}
    if steric_job_flags or gapmer_job_flags:
        key_decisions.append(
            {
                "title": "Branch execution policy",
                "decision": (
                    f"Steric branch: splice_eval={bool(steric_job_flags.get('run_splice_eval'))}, "
                    f"splice_router={bool(steric_job_flags.get('run_splice_router'))}; "
                    f"Gapmer branch: splice_eval={bool(gapmer_job_flags.get('run_splice_eval'))}, "
                    f"splice_router={bool(gapmer_job_flags.get('run_splice_router'))}."
                ),
                "why": "These flags determine why steric outputs can include splice scoring and route ranking while gapmer outputs may stop at tiling and uniqueness ranking.",
                "confidence": "high",
                "evidence_refs": ["report_manifest.steric_jobs", "report_manifest.gapmer_jobs"],
            }
        )
    if top_route and top_route.get("goal_label"):
        key_decisions.append(
            {
                "title": "Route prioritization",
                "decision": f"The top ranked route was {top_route.get('goal_label')}.",
                "why": _stringify(top_route.get("rationale")) or "The route ranking metadata indicates it was prioritized over alternatives.",
                "confidence": "medium",
                "evidence_refs": ["routing_summary.top_goals[0]", "report_manifest.strategy_routing"],
            }
        )
    chemistry = report_manifest.get("chemistry") if isinstance(report_manifest.get("chemistry"), dict) else {}
    if recommendation:
        chem_text = recommendation.get("CHEMISTRY") or recommendation.get("chemistry") or recommendation.get("platform")
        if chem_text:
            key_decisions.append(
                {
                    "title": "Chemistry recommendation",
                    "decision": f"The report recommends {chem_text}.",
                    "why": "Chemistry is chosen to match the requested mechanism, such as steric blocking versus RNase H-mediated knockdown.",
                    "confidence": "medium",
                    "evidence_refs": ["report_manifest.chemistry"],
                }
            )

    branch_explanations: List[Dict[str, Any]] = []
    for branch, entry in sorted(branch_outcomes.items()):
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "unknown").strip() or "unknown"
        reason = _stringify(entry.get("reason")) or "No detailed reason was preserved in the report payload."
        branch_explanations.append(
            {
                "branch": branch,
                "title": f"{_humanize_key(branch)} branch",
                "status": status,
                "summary": reason,
                "why": _branch_status_explanation(status),
                "evidence_refs": [f"branch_outcomes.{branch}"],
            }
        )

    output_guides: List[Dict[str, Any]] = []
    for item in output_inventory[:8]:
        if not isinstance(item, dict):
            continue
        output_guides.append(
            {
                "key": _stringify(item.get("key")) or _slugify(item.get("title") or "output"),
                "title": _stringify(item.get("title")) or "Output",
                "what_it_is": _stringify(item.get("summary")) or "This is a persisted output from the run.",
                "how_to_read": _stringify(item.get("how_to_read")) or "Read the title, counts, and linked file paths first, then inspect the structured output or table if you need detail.",
                "why_it_matters": _stringify(item.get("why_it_matters")) or "It captures one important part of what the run produced.",
                "evidence_refs": item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else [f"output_inventory[{len(output_guides)}]"],
            }
        )

    caveats = []
    for warning in report_manifest.get("pipeline_alerts") or []:
        if isinstance(warning, str) and warning.strip():
            caveats.append(warning.strip())
    if not caveats:
        caveats.append("No explicit pipeline alerts were preserved in the compact report payload.")

    glossary = _default_glossary(report_manifest)
    tooltips = _fallback_tooltips(context)

    return {
        "schema_version": 1,
        "audience": "novice",
        "summary": {
            "headline": _fallback_goal_sentence(meta),
            "plain_language_goal": _fallback_goal_sentence(meta),
            "what_happened": _fallback_what_happened(
                transcript_lines=transcript_lines,
                clarifications=clarifications,
                recommendation=recommendation,
                tiling=tiling,
                splicing_eval=splicing_eval,
                routing_summary=routing_summary,
                gapmer_tiling=gapmer_tiling,
            ),
            "result": _fallback_result_sentence(branch_labels, available_outputs),
            "bullets": summary_bullets[:5],
        },
        "start_here": [
            "Read the TL;DR first to understand the goal, what ran, and what was produced.",
            "Use the Decisions / Workflow tab to follow the run step by step before drilling into maps or raw artifacts.",
            "Open the output guides before reading structured outputs or file inventories.",
        ],
        "walkthrough": walkthrough[:10],
        "key_decisions": key_decisions[:6],
        "branch_explanations": branch_explanations[:6],
        "output_guides": output_guides[:8],
        "caveats": caveats[:6],
        "glossary": glossary[:10],
        "tooltips": tooltips,
    }


def _append_walkthrough_step(
    steps: List[Dict[str, Any]],
    *,
    title: str,
    summary: str,
    why: str,
    evidence_refs: Sequence[str],
    lane: str = "shared",
) -> None:
    if not (_stringify(title) or _stringify(summary) or _stringify(why)):
        return
    steps.append(
        {
            "title": _stringify(title) or "Step",
            "lane": _stringify(lane) or "shared",
            "summary": _stringify(summary),
            "why_it_matters": _stringify(why),
            "evidence_refs": [_stringify(ref) for ref in evidence_refs if _stringify(ref)][:5],
        }
    )


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fallback_what_happened(
    *,
    transcript_lines: Sequence[Dict[str, Any]],
    clarifications: Sequence[Dict[str, Any]],
    recommendation: Dict[str, Any],
    tiling: Dict[str, Any],
    splicing_eval: Dict[str, Any],
    routing_summary: Dict[str, Any],
    gapmer_tiling: Dict[str, Any],
) -> str:
    parts: List[str] = []
    if clarifications:
        parts.append(f"captured {len(clarifications)} clarification answer(s)")
    if transcript_lines:
        parts.append("resolved transcript and isoform context")
    if recommendation:
        chemistry_name = (
            _stringify(recommendation.get("platform"))
            or _stringify(recommendation.get("CHEMISTRY"))
            or _stringify(recommendation.get("chemistry"))
        )
        if chemistry_name:
            parts.append(f"selected chemistry guidance ({chemistry_name})")
    if tiling:
        parts.append("ran steric tiling")
    if splicing_eval:
        site_count = len(splicing_eval.get("sites") or []) if isinstance(splicing_eval.get("sites"), list) else _as_int(splicing_eval.get("site_count"))
        if site_count is not None:
            parts.append(f"scored steric candidates across {site_count} splice site(s)")
        else:
            parts.append("ran splice evaluation")
    if routing_summary.get("goal_count"):
        parts.append(f"ranked {routing_summary.get('goal_count')} splice route goal(s)")
    if gapmer_tiling:
        parts.append("generated a gapmer knockdown candidate set")
    if not parts:
        return "The pipeline interpreted the request, selected transcript context where needed, executed the relevant design branches, and wrote the resulting artifacts."
    if len(parts) == 1:
        return "The pipeline " + parts[0] + "."
    return "The pipeline " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def _source_search_markdown(report_payload: Dict[str, Any]) -> str:
    """Expose recorded dataset coverage, including completed searches with zero hits."""
    records = []
    for branch in (report_payload.get("branch_outcomes") or {}).values():
        records.extend(branch.get("source_searches") or [])
        if isinstance(branch.get("source_search"), dict):
            records.append(branch["source_search"])
    lines = []
    seen = set()
    for record in records:
        dataset = record.get("selected_dataset") or {}
        detail = {key: value for key, value in {
            "source": record.get("source_type"), "target": record.get("rbp"),
            "assay": record.get("assay"), "search_status": record.get("search_status"),
            "result": record.get("status"), "error": record.get("error"),
            "experiment": dataset.get("experiment_accession"), "file": dataset.get("file_accession"),
            "biosample": dataset.get("biosample_term_name"), "assembly": dataset.get("assembly"),
            "overlapping_peaks": record.get("peak_count"), "searched_gene_span_bed0": record.get("gene_span"),
            "feature_scope": record.get("feature_scope"),
        }.items() if value is not None and value != ""}
        serialized = json.dumps(detail, ensure_ascii=False, sort_keys=True)
        if not detail or serialized in seen:
            continue
        seen.add(serialized)
        lines.extend(["```json", json.dumps(detail, ensure_ascii=False, indent=2), "```"])
        url = dataset.get("download_url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            lines.append(f"[Recorded source dataset]({url})")
    if not lines:
        return ""
    return "\n\n".join(["## Source search evidence", *lines,
        "Coverage is limited to the recorded datasets and searched intervals. Zero overlapping peaks do not establish absence in unsearched datasets or a biological effect."])


def _selected_off_target_budget_markdown(generated_set: Dict[str, Any]) -> str:
    """State the actual selected-panel mismatch budgets without treating zero as missing."""
    rows = generated_set.get("rows") if isinstance(generated_set, dict) else None
    if not isinstance(rows, list) or not rows:
        return ""
    budgets = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("Off_target_mismatch_budget", row.get("off_target_mismatch_budget"))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            budgets.append(value)
        else:
            budgets.append(None)
    if not budgets or all(value is None for value in budgets):
        return ""
    known = sorted({value for value in budgets if value is not None})
    if len(known) == 1 and len(budgets) == len(rows) and all(value is not None for value in budgets):
        budget = known[0]
        scope = (
            "exact matches (0 mismatches) only; one- and higher-mismatch bins were not searched and are unknown"
            if budget == 0 else
            f"ungapped matches through {budget} mismatch(es); higher-mismatch bins were not searched and are unknown"
        )
        return f"Selected panel: every retained candidate has a recorded off-target search budget of {budget} mismatch(es), covering {scope}."
    labels = [
        f"rank {row.get('selection_rank', index)}: {value if value is not None else 'unavailable'}"
        for index, (row, value) in enumerate(zip(rows, budgets), 1)
        if isinstance(row, dict)
    ]
    return ("Selected-panel off-target mismatch budgets: " + "; ".join(labels)
            + ". Counts beyond each recorded budget are unsearched and unknown, not zero.")


def render_report_markdown(explainer: Dict[str, Any], report_payload: Dict[str, Any]) -> str:
    deterministic_summary = _build_design_output_summary_markdown(report_payload)
    if deterministic_summary:
        from .reference_provenance import reference_markdown
        from .binding_region_reports import binding_region_markdown
        return "\n\n".join(part for part in [deterministic_summary, binding_region_markdown(report_payload.get("binding_region_candidate_sets")), reference_markdown(report_payload.get("reference_provenance")), _source_search_markdown(report_payload)] if part)

    summary = explainer.get("summary") if isinstance(explainer.get("summary"), dict) else {}
    lines: List[str] = []

    lines.append("# TL;DR")
    headline = _stringify(summary.get("headline")) or "Run summary unavailable."
    lines.append(headline)
    lines.append("")
    for key in ("plain_language_goal", "what_happened", "result"):
        value = _stringify(summary.get(key))
        if value and value != headline:
            lines.append(value)
            lines.append("")
    bullets = _normalize_string_list(summary.get("bullets"), [], limit=6)
    if bullets:
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")

    start_here = _normalize_string_list(explainer.get("start_here"), [], limit=6)
    if start_here:
        lines.append("## How To Read This Run")
        for item in start_here:
            lines.append(f"- {item}")
        lines.append("")

    walkthrough = explainer.get("walkthrough") if isinstance(explainer.get("walkthrough"), list) else []
    if walkthrough:
        lines.append("## What Happened")
        for index, item in enumerate(walkthrough, start=1):
            if not isinstance(item, dict):
                continue
            title = _stringify(item.get("title")) or f"Step {index}"
            lines.append(f"{index}. **{title}**")
            summary_text = _stringify(item.get("summary"))
            if summary_text:
                lines.append(f"   {summary_text}")
            why = _stringify(item.get("why_it_matters"))
            if why:
                lines.append(f"   Why it matters: {why}")
            evidence = _normalize_string_list(item.get("evidence_refs"), [], limit=4)
            if evidence:
                lines.append(f"   Evidence: {', '.join(evidence)}")
        lines.append("")

    decisions = explainer.get("key_decisions") if isinstance(explainer.get("key_decisions"), list) else []
    if decisions:
        lines.append("## Key Decisions")
        for item in decisions:
            if not isinstance(item, dict):
                continue
            title = _stringify(item.get("title")) or "Decision"
            lines.append(f"### {title}")
            decision = _stringify(item.get("decision"))
            why = _stringify(item.get("why"))
            confidence = _stringify(item.get("confidence")) or "unknown"
            if decision:
                lines.append(decision)
                lines.append("")
            if why:
                lines.append(f"- Why: {why}")
            if confidence:
                lines.append(f"- Confidence: {confidence}")
            evidence = _normalize_string_list(item.get("evidence_refs"), [], limit=5)
            if evidence:
                lines.append(f"- Evidence: {', '.join(evidence)}")
            lines.append("")

    branches = explainer.get("branch_explanations") if isinstance(explainer.get("branch_explanations"), list) else []
    if branches:
        lines.append("## Branch Outcomes")
        for item in branches:
            if not isinstance(item, dict):
                continue
            title = _stringify(item.get("title")) or _humanize_key(item.get("branch"))
            status = _stringify(item.get("status")) or "unknown"
            lines.append(f"### {title}")
            lines.append(f"- Status: {status}")
            summary_text = _stringify(item.get("summary"))
            if summary_text:
                lines.append(f"- Summary: {summary_text}")
            why = _stringify(item.get("why"))
            if why:
                lines.append(f"- Why it matters: {why}")
            evidence = _normalize_string_list(item.get("evidence_refs"), [], limit=4)
            if evidence:
                lines.append(f"- Evidence: {', '.join(evidence)}")
            lines.append("")

    skipped_jobs = _collect_skipped_jobs(report_payload) if isinstance(report_payload, dict) else []
    if skipped_jobs:
        lines.append("## Skipped Jobs")
        for item in skipped_jobs:
            lines.append(f"- {item['modality']} job '{item['label']}': {item['reason']}")
        lines.append("")

    output_guides = explainer.get("output_guides") if isinstance(explainer.get("output_guides"), list) else []
    if output_guides:
        lines.append("## How To Read The Outputs")
        for item in output_guides:
            if not isinstance(item, dict):
                continue
            title = _stringify(item.get("title")) or "Output"
            lines.append(f"### {title}")
            for field, label in (
                ("what_it_is", "What it is"),
                ("how_to_read", "How to read it"),
                ("why_it_matters", "Why it matters"),
            ):
                value = _stringify(item.get(field))
                if value:
                    lines.append(f"- {label}: {value}")
            evidence = _normalize_string_list(item.get("evidence_refs"), [], limit=4)
            if evidence:
                lines.append(f"- Evidence: {', '.join(evidence)}")
            lines.append("")

    caveats = _normalize_string_list(explainer.get("caveats"), [], limit=10)
    if caveats:
        lines.append("## Caveats")
        for item in caveats:
            lines.append(f"- {item}")
        lines.append("")

    glossary = explainer.get("glossary") if isinstance(explainer.get("glossary"), list) else []
    if glossary:
        lines.append("## Glossary")
        for item in glossary:
            if not isinstance(item, dict):
                continue
            term = _stringify(item.get("term"))
            definition = _stringify(item.get("definition"))
            if term and definition:
                lines.append(f"- **{term}**: {definition}")
        lines.append("")

    if isinstance(report_payload, dict):
        meta = report_payload.get("structured_goal") if isinstance(report_payload.get("structured_goal"), dict) else {}
        lines.append("## Run Metadata")
        for label, value in (
            ("Gene", meta.get("gene_symbol")),
            ("Gene ID", meta.get("gene_id")),
            ("Intent", meta.get("intent")),
            ("Species", meta.get("species")),
            ("Target context", meta.get("target_context")),
        ):
            text = _stringify(value)
            if text:
                lines.append(f"- {label}: {text}")
        lines.append("")

    source_search = _source_search_markdown(report_payload)
    if source_search:
        lines.extend([source_search, ""])
    run_root = _infer_report_run_root(report_payload) if isinstance(report_payload, dict) else None
    return _sanitize_report_paths("\n".join(line.rstrip() for line in lines).strip(), run_root) + "\n"


def _build_transcript_context(structured_goal: Dict[str, Any], final_payload: Dict[str, Any]) -> Dict[str, Any]:
    source = final_payload if isinstance(final_payload.get("isoform_transcripts"), dict) else structured_goal
    isoforms = source.get("isoform_transcripts") if isinstance(source.get("isoform_transcripts"), dict) else {}
    items: List[Dict[str, Any]] = []
    seen_transcript_ids: set[str] = set()
    gapmer_primary = _stringify(final_payload.get("gapmer_primary_transcript_id"))
    if gapmer_primary:
        seen_transcript_ids.add(gapmer_primary)
        items.append(
            {
                "label": "gapmer_primary",
                "transcript_id": gapmer_primary,
                "summary": f"Gapmer tiling used transcript {gapmer_primary} as the primary template.",
            }
        )
    for label, entry in sorted(isoforms.items()):
        if not isinstance(entry, dict):
            continue
        transcript_id = _stringify(entry.get("transcript_id") or entry.get("id"))
        if not transcript_id:
            continue
        if transcript_id in seen_transcript_ids:
            continue
        seen_transcript_ids.add(transcript_id)
        match_info = entry.get("match_info") if isinstance(entry.get("match_info"), dict) else {}
        parts = [f"{label}: {transcript_id}"]
        if match_info.get("method"):
            parts.append(f"method={match_info.get('method')}")
        if match_info.get("confidence"):
            parts.append(f"confidence={match_info.get('confidence')}")
        items.append(
            {
                "label": label,
                "transcript_id": transcript_id,
                "match_method": _stringify(match_info.get("method")),
                "confidence": _stringify(match_info.get("confidence")),
                "summary": "; ".join(parts),
                "exon_nomenclature_resolutions": entry.get("exon_nomenclature_resolutions") or (entry.get("metadata") or {}).get("exon_nomenclature_resolutions") or {},
            }
        )
    return {
        "policy": _stringify(structured_goal.get("transcript_policy") or final_payload.get("transcript_policy")),
        "isoform_resolution_locked": bool(
            final_payload.get("isoform_resolution_locked") or structured_goal.get("isoform_resolution_locked")
        ),
        "items": items[:12],
    }


def _collect_skipped_jobs(report_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    skipped: List[Dict[str, str]] = []
    for modality, source_key in (("Steric", "steric_jobs"), ("Gapmer", "gapmer_jobs")):
        jobs = report_payload.get(source_key) if isinstance(report_payload, dict) else []
        for job in jobs if isinstance(jobs, list) else []:
            if not isinstance(job, dict):
                continue
            status = (_stringify(job.get("status")) or "").lower()
            reason = _stringify(job.get("reason"))
            if status != "skipped" and not reason:
                continue
            label = _stringify(job.get("label")) or _stringify(job.get("job_id")) or f"{modality} job"
            skipped.append(
                {
                    "modality": modality,
                    "label": label,
                    "reason": reason or "No rationale was recorded for this skipped job.",
                }
            )
    return skipped


def _job_inventory_summary(jobs: Any) -> str:
    if not isinstance(jobs, list) or not jobs:
        return "0 job(s) with per-job outputs and metadata."
    normalized_jobs = [job for job in jobs if isinstance(job, dict)]
    skipped = [job for job in normalized_jobs if (_stringify(job.get("status")) or "").lower() == "skipped" or _stringify(job.get("reason"))]
    total = len(normalized_jobs)
    skipped_count = len(skipped)
    if skipped_count <= 0:
        return f"{total} job(s) with per-job outputs and metadata."
    active_count = max(0, total - skipped_count)
    if active_count > 0:
        summary = f"{total} job(s): {active_count} remained active and {skipped_count} were skipped."
    else:
        summary = f"All {total} job(s) were skipped."
    first_reason = _stringify(skipped[0].get("reason")) if skipped else ""
    if first_reason:
        summary += f" Skip rationale: {first_reason}"
    return summary


def _build_output_inventory(
    report_payload: Dict[str, Any],
    final_payload: Dict[str, Any],
    structured_goal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    strategy = report_payload.get("strategy_routing") if isinstance(report_payload.get("strategy_routing"), dict) else {}
    if strategy:
        planner = strategy.get("planner_plan") if isinstance(strategy.get("planner_plan"), dict) else {}
        ranking = planner.get("ranking") if isinstance(planner.get("ranking"), dict) else {}
        outputs.append(
            {
                "key": "strategy_routing",
                "title": "Strategy routing",
                "summary": f"Ranked splice route goals for the run ({strategy.get('goal_count') or 0} goals across {strategy.get('job_count') or 0} jobs).",
                "how_to_read": "Read the rank, goal label, rationale, candidate count, and threshold together. A route with count 0 did not retain usable candidates.",
                "why_it_matters": "This shows which splice levers the pipeline prioritized and why.",
                "evidence_refs": ["report_manifest.strategy_routing"],
                "model": ranking.get("model"),
            }
        )
    splicing_eval = report_payload.get("splicing_eval") if isinstance(report_payload.get("splicing_eval"), dict) else {}
    if splicing_eval:
        outputs.append(
            {
                "key": "splicing_eval",
                "title": "Splice evaluation",
                "summary": f"Splice evaluation on {len(splicing_eval.get('sites') or []) if isinstance(splicing_eval.get('sites'), list) else splicing_eval.get('site_count') or 0} site(s).",
                "how_to_read": "Use the scored path and site count to see what was evaluated. Nonzero-vs-zero counts tell you whether route goals showed measurable predicted effects.",
                "why_it_matters": "This connects candidate ASOs to predicted splice effects when splice evaluation was run.",
                "evidence_refs": ["report_manifest.splicing_eval"],
            }
        )
    for key, title in (
        ("tiling", "Steric tiling"),
        ("gapmer_tiling", "Gapmer tiling"),
    ):
        payload = report_payload.get(key)
        if not isinstance(payload, dict):
            continue
        passed = ((payload.get("initial_filtering") or {}).get("passed") or {}).get("count")
        outputs.append(
            {
                "key": key,
                "title": title,
                "summary": f"{title} outputs with filter summaries and persisted tile files." + (f" Passed count: {passed}." if passed is not None else ""),
                "how_to_read": "Start with the kept or passed counts, then inspect scored or filtered file paths if you need the candidate-level details.",
                "why_it_matters": "These outputs show the concrete candidate space that survived tiling and filtering.",
                "evidence_refs": [f"report_manifest.{key}"],
            }
        )
    for key, title, source_key in (
        ("steric_jobs", "Steric jobs", "steric_jobs"),
        ("gapmer_jobs", "Gapmer jobs", "gapmer_jobs"),
    ):
        jobs = report_payload.get(source_key)
        if isinstance(jobs, list) and jobs:
            outputs.append(
                {
                    "key": key,
                    "title": title,
                    "summary": _job_inventory_summary(jobs),
                    "how_to_read": "Each job corresponds to one design objective or region set. Check the job status and rationale first, then compare labels, region counts, and output directories.",
                    "why_it_matters": "Jobs are the bridge between the user's intent and the actual output files that were produced or intentionally skipped.",
                    "evidence_refs": [f"report_manifest.{source_key}"],
                }
            )
    for key, title in (
        ("rbp_binding_result", "RBP dataset selection"),
        ("mirna_binding_result", "miRNA target-site selection"),
        ("motif_tiling_result", "Motif targeting"),
        ("patient_reference", "Patient reference"),
        ("snapgene_annotation", "SnapGene annotation"),
    ):
        payload = final_payload.get(key) or structured_goal.get(key)
        if isinstance(payload, dict):
            outputs.append(
                {
                    "key": key,
                    "title": title,
                    "summary": f"{title} context preserved in the run payload.",
                    "how_to_read": "Treat this as context that shaped targeting rather than as a ranked candidate list.",
                    "why_it_matters": "It explains why the pipeline focused on certain regions or isoforms.",
                    "evidence_refs": [key],
                }
            )
    return outputs


def _collect_important_artifacts(run_path: Path) -> List[Dict[str, Any]]:
    candidates = [
        "report.md",
        "report.html",
        "artifacts/intake/report_explainer.json",
        "report_explainer.json",
        "goal_intake_report.html",
        "artifacts/design/strategy_router_summary.json",
        "strategy_router_summary.json",
        "artifacts/design/tiling/tiles.json",
        "tiling/tiles.json",
        "artifacts/candidates/scored_tiles.json",
        "tiling/scored_tiles.json",
        "artifacts/candidates/gapmer_scored_uniqueness.json",
        "gapmer_tiling/gapmer_scored_uniqueness.json",
        "artifacts/candidates/generated_aso_set.json",
        "generated_aso_set.json",
        "artifacts/design/steric_regions.bed",
        "steric_regions.bed",
        "artifacts/design/snapgene/aso_export.gb",
        "snapgene/aso_export.gb",
    ]
    source_manifest = run_path / "artifacts/design/region_sources/manifest.json"
    if source_manifest.is_file():
        candidates.append(source_manifest.relative_to(run_path).as_posix())
        source_exports = json.loads(source_manifest.read_text())
        for record in source_exports.get("records") or []:
            source_path = Path(str(record.get("path") or ""))
            if source_path.is_file() and source_path.is_relative_to(source_manifest.parent):
                candidates.append(source_path.relative_to(run_path).as_posix())
    artifacts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for relative in candidates:
        path = resolve_run_path(run_path, *Path(relative).parts)
        if not path.exists():
            continue
        try:
            shown = path.relative_to(run_path).as_posix()
        except Exception:
            shown = relative
        if shown in seen:
            continue
        seen.add(shown)
        artifacts.append(
            {
                "path": shown,
                "kind": path.suffix.lstrip(".") or "file",
            }
        )
    return artifacts


_RUN_ARTIFACT_DIR_NAMES = {
    "artifacts",
    "candidates",
    "design",
    "intake",
    "controller",
    "child_runs",
    "gapmer_tiling",
    "junction_gapmer_tiling",
    "tiling",
    "gapmer_jobs",
    "steric_jobs",
    "strategy_routes",
    "strategy_router_debug",
    "snapgene",
    "v2_tasks",
    "v3_tasks",
}


def _known_artifact_run_root(path: Path) -> Optional[Path]:
    parts = path.parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        if idx > 0:
            try:
                return Path(*parts[:idx])
            except Exception:
                return None
    for idx, part in enumerate(parts):
        if part in _RUN_ARTIFACT_DIR_NAMES and idx > 0:
            try:
                return Path(*parts[:idx])
            except Exception:
                return None
    return None


def _collect_report_path_values(value: Any) -> List[str]:
    paths: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key or "").lower()
            if isinstance(item, str):
                text = item.strip()
                if text and (
                    key_text.endswith("path")
                    or key_text.endswith("dir")
                    or key_text in {"path", "output_dir", "json_path", "csv_path"}
                ):
                    paths.append(text)
            elif isinstance(item, (dict, list)):
                paths.extend(_collect_report_path_values(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_collect_report_path_values(item))
    return paths


def _infer_report_run_root(report_payload: Dict[str, Any]) -> Optional[Path]:
    path_values = [
        Path(path).expanduser()
        for path in _collect_report_path_values(report_payload)
        if isinstance(path, str) and path.strip() and Path(path).expanduser().is_absolute()
    ]
    if not path_values:
        return None
    artifact_roots = [
        root
        for root in (_known_artifact_run_root(path) for path in path_values)
        if root is not None
    ]
    if artifact_roots:
        ranked = sorted(
            artifact_roots,
            key=lambda root: (
                -sum(1 for candidate in artifact_roots if candidate == root),
                -len(root.parts),
                str(root),
            ),
        )
        return ranked[0]
    try:
        existing = [path for path in path_values if path.exists()]
        candidates = existing or path_values
        common = Path(os.path.commonpath([str(path if path.is_dir() else path.parent) for path in candidates]))
    except Exception:
        common = path_values[0].parent
    root = common
    while root.name in _RUN_ARTIFACT_DIR_NAMES or root.parent.name in _RUN_ARTIFACT_DIR_NAMES:
        if root.parent == root:
            break
        root = root.parent
    return root


def _relative_report_path(path_value: Any, run_root: Optional[Path]) -> str:
    text = _stringify(path_value)
    if not text:
        return ""
    path = Path(text).expanduser()
    if run_root:
        try:
            return path.relative_to(run_root).as_posix()
        except Exception:
            pass
    if run_root and path.is_absolute():
        try:
            return path.resolve().relative_to(run_root.resolve()).as_posix()
        except Exception:
            pass
    if run_root and run_root.name:
        parts = path.parts
        if run_root.name in parts:
            idx = parts.index(run_root.name)
            if idx + 1 < len(parts):
                return Path(*parts[idx + 1:]).as_posix()
    if "/runs/" in text:
        suffix = text.split("/runs/", 1)[1]
        parts = suffix.split("/", 1)
        if len(parts) == 2:
            return parts[1]
    return text


def _sanitize_report_paths(text: str, run_root: Optional[Path]) -> str:
    if not text or not run_root:
        return text
    roots = [str(run_root)]
    try:
        resolved = str(run_root.resolve())
        if resolved not in roots:
            roots.append(resolved)
    except Exception:
        pass
    sanitized = text
    for root_text in sorted(roots, key=len, reverse=True):
        sanitized = sanitized.replace(root_text + "/", "").replace(root_text, ".")
    if run_root.name:
        sanitized = sanitized.replace(f"runs/{run_root.name}/", "")
        sanitized = sanitized.replace(f"runs/{run_root.name}", ".")
    return sanitized


def _load_report_rows(path_value: Any) -> List[Dict[str, Any]]:
    path_text = _stringify(path_value)
    if not path_text:
        return []
    path = Path(path_text).expanduser()
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _load_report_json(path_value: Any) -> Dict[str, Any]:
    path_text = _stringify(path_value)
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_artifact_json(run_root: Optional[Path], relative_path: str) -> Dict[str, Any]:
    if not run_root:
        return {}
    return _load_report_json(resolve_run_path(run_root, *Path(relative_path).parts))


def _run_artifact_path(run_root: Optional[Path], relative_path: str) -> str:
    if not run_root:
        return ""
    path = resolve_run_path(run_root, *Path(relative_path).parts)
    if not path.exists():
        return ""
    try:
        return path.relative_to(run_root).as_posix()
    except Exception:
        return Path(relative_path).as_posix()


def _run_request_text(run_root: Optional[Path]) -> str:
    if not run_root:
        return ""
    request_path = run_root / "request.txt"
    if not request_path.exists():
        return ""
    try:
        from .report_presentation import display_request
        envelope = _run_artifact_json(run_root, 'request_envelope.json')
        return display_request(run_root, envelope, request_path.read_text(encoding="utf-8", errors="replace").strip())
    except Exception:
        return ""


def _append_report_file_line(lines: List[str], path_text: str, description: str) -> bool:
    if not path_text:
        return False
    lines.append(f"- `{path_text}`: {description}.")
    return True


def _candidate_name(row: Dict[str, Any]) -> str:
    return _stringify(row.get("Name") or row.get("name") or row.get("id") or row.get("aso_id") or row.get("site_label"))


def _candidate_sequence(row: Dict[str, Any]) -> str:
    return _stringify(row.get("Sequence") or row.get("sequence") or row.get("seq"))


def _candidate_rank(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("selection_rank"))
    except Exception:
        return 10**9


_OFF_TARGET_GENE_COUNT_KEYS = (
    "off_target_gene_count",
    "Times_mapped_off_target_genes",
    "times_mapped_off_target_genes",
    "worst_species_off_target_gene_count",
)
_TARGET_ISOFORM_HIT_KEYS = (
    "isoform_uniqueness_count",
    "isoform_hit_count",
)


def _row_int_from_keys(row: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        value = _as_int(row.get(key))
        if value is not None:
            return value
    return None


def _candidate_offtarget_count(row: Dict[str, Any]) -> Optional[int]:
    """Return other-gene exact-match count. Never treat isoform/Times_mapped hits as off-targets."""
    return exact_gene_count(row)


def _candidate_isoform_hit_count(row: Dict[str, Any]) -> Optional[int]:
    """Return target-gene isoform hit count for pan-isoform coverage."""
    return _row_int_from_keys(row, _TARGET_ISOFORM_HIT_KEYS)


def _candidate_isoform_score(row: Dict[str, Any]) -> Optional[float]:
    # Normalized candidates retain additional measured fields under their
    # source annotations. An explicit current unknown still takes precedence.
    if "isoform_uniqueness_score" in row:
        value = row["isoform_uniqueness_score"]
    else:
        annotations = row.get("source_annotations")
        value = annotations.get("isoform_uniqueness_score") if isinstance(annotations, dict) else None
    if value in (None, "", [], {}):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_report_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _fmt_report_int(value: Any) -> str:
    try:
        return str(int(value))
    except Exception:
        return "unknown"


def _clean_report_sentence(value: Any) -> str:
    text = _stringify(value).strip()
    return text[:-1] if text.endswith(".") else text


def _top_ranked_candidates(rows: Sequence[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    ranked = [row for row in rows if _candidate_sequence(row)]
    return sorted(
        ranked,
        key=lambda row: (
            _candidate_rank(row),
            _candidate_offtarget_count(row) if _candidate_offtarget_count(row) is not None else 10**9,
            _candidate_name(row),
        ),
    )[:limit]


def _render_candidate_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    lines = [
        "| rank | candidate | source sequence / role (5′–3′) | isoform score class | isoform score | matched / catalog isoforms | coverage fraction | off-target genes | conservation (phastCons / phyloP) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rank = "" if _candidate_rank(row) == 10**9 else str(_candidate_rank(row))
        score = _fmt_report_float(_candidate_isoform_score(row))
        isoform_hits = _candidate_isoform_hit_count(row)
        catalog_size = row.get('isoform_catalog_size')
        isoform_class = str(row.get('isoform_uniqueness_class') or 'unknown')
        coverage = _fmt_report_float(row.get('isoform_coverage_fraction'))
        off_count = _candidate_offtarget_count(row)
        lines.append(
            f"| {rank} | {_candidate_name(row) or 'candidate'} | {_candidate_sequence(row)} ({generated_sequence_fields(row)['sequence_role']}); synthesized ASO: {generated_sequence_fields(row).get('synthesized_antisense_sequence', 'unverified')} | "
            f"{isoform_class} | {score or 'NA'} | "
            f"{isoform_hits if isoform_hits is not None else 'NA'} / {catalog_size if catalog_size is not None else 'NA'} | "
            f"{coverage or 'NA'} | {off_count if off_count is not None else 'NA'} | "
            f"{_fmt_report_float(row.get('Phastcon') if row.get('Phastcon') is not None else row.get('phastcons')) or 'NA'} / "
            f"{_fmt_report_float(row.get('PhyloP') if row.get('PhyloP') is not None else row.get('phylop')) or 'NA'} |"
        )
    return lines


def _report_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _render_allele_specific_candidate_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    lines = [
        "| rank | candidate | mutant-matched ASO (5′→3′) | variant position in ASO | in DNA gap | WT mismatches | off-target genes |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        rank = "" if _candidate_rank(row) == 10**9 else str(_candidate_rank(row))
        aso_sequence = _stringify(row.get("mutant_aso_sequence") or row.get("Synthesis"))
        variant_position = _fmt_report_int(row.get("variant_position_in_aso"))
        wildtype_mismatches = _fmt_report_int(row.get("wildtype_mismatch_count"))
        off_count = _candidate_offtarget_count(row)
        lines.append(
            f"| {rank} | {_candidate_name(row) or 'candidate'} | {aso_sequence or 'NA'} | "
            f"{variant_position} | {'yes' if _report_bool(row.get('variant_in_gap')) else 'no'} | "
            f"{wildtype_mismatches} | {off_count if off_count is not None else 'NA'} |"
        )
    return lines


def _row_splice_effect(row: Dict[str, Any]) -> Optional[float]:
    values = []
    for key in ("shared_context_abs_delta", "delta_inclusion_deletion", "delta_inclusion_replacement"):
        value = row.get(key)
        try:
            values.append(abs(float(value)))
        except Exception:
            continue
    return max(values) if values else None


def _top_splice_effect_candidates(rows: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if isinstance(row, dict) and _candidate_sequence(row) and _row_splice_effect(row) is not None]
    return sorted(candidates, key=lambda row: (_row_splice_effect(row) or 0.0), reverse=True)[:limit]


def _render_splice_effect_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    lines = [
        "| candidate | source sequence / role (5′–3′) | route/site | abs splice effect | off-target genes |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        route = _stringify(row.get("shared_context_label") or row.get("route_label") or row.get("selected_site_kind"))
        off_count = _candidate_offtarget_count(row)
        lines.append(
            f"| {_candidate_name(row) or 'candidate'} | {_candidate_sequence(row)} ({generated_sequence_fields(row)['sequence_role']}); synthesized ASO: {generated_sequence_fields(row).get('synthesized_antisense_sequence', 'unverified')} | {route or 'NA'} | "
            f"{_fmt_report_float(_row_splice_effect(row), digits=4) or 'NA'} | {off_count if off_count is not None else 'NA'} |"
        )
    return lines


def _route_goal_rows(strategy_router: Dict[str, Any], run_root: Optional[Path], limit: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    goals = strategy_router.get("goals") if isinstance(strategy_router.get("goals"), list) else []
    for goal in sorted([goal for goal in goals if isinstance(goal, dict)], key=lambda item: int(item.get("rank") or 10**9))[:limit]:
        rows.append(
            {
                "rank": goal.get("rank"),
                "goal_label": goal.get("goal_label"),
                "count": goal.get("count"),
                "directional_delta": goal.get("observed_directional_max_delta"),
                "opposite_delta": goal.get("observed_opposite_max_delta"),
                "confidence": goal.get("ranking_confidence"),
                "csv_path": _relative_report_path(goal.get("csv_path") or goal.get("output_path"), run_root),
                "rationale": goal.get("rationale"),
            }
        )
    return rows


def _render_route_goal_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    lines = [
        "| rank | route | candidates | directional delta | opposite delta | confidence | file |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {_fmt_report_int(row.get('rank'))} | {_stringify(row.get('goal_label')) or 'route'} | "
            f"{_fmt_report_int(row.get('count'))} | {_fmt_report_float(row.get('directional_delta')) or 'NA'} | "
            f"{_fmt_report_float(row.get('opposite_delta')) or 'NA'} | {_stringify(row.get('confidence')) or 'NA'} | "
            f"`{_stringify(row.get('csv_path')) or 'see artifacts/design/strategy_routes/'}` |"
        )
    return lines


def _count_gene_wide_rows(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if _stringify(row.get("isoform_uniqueness_class")).lower() == "gene_wide")


def _count_zero_offtarget_rows(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if _candidate_offtarget_count(row) == 0)


def _cross_species_report_evidence(
    filters: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Keep conservation observations separate from hard-filter eligibility.

    A non-required filter passes every row regardless of whether its target is
    conserved. Only the row's tri-state measurement can establish conservation.
    """
    species = sorted({str(item) for entry in filters
                      if isinstance(entry.get("requested_species"), list)
                      for item in entry["requested_species"]})
    exact = sum(row.get("cross_species_pass") is True for row in rows)
    nonmatching = sum(row.get("cross_species_pass") is False for row in rows)
    observed_species = {str(key) for row in rows
                        if isinstance(row.get("cross_species_matches"), dict)
                        for key in row["cross_species_matches"]}
    return {
        "active": len(species) > 1 or len(observed_species) > 1 or bool(exact + nonmatching),
        "required": any(entry.get("required") is True for entry in filters),
        "requested_species": species,
        "total": len(rows),
        "exact": exact,
        "nonmatching": nonmatching,
        "unknown": len(rows) - exact - nonmatching,
    }


def _artifact_if_exists(path_value: Any, run_root: Optional[Path]) -> str:
    path_text = _stringify(path_value)
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if path.exists():
        return _relative_report_path(str(path), run_root)
    return ""


def _sibling_artifact(path_value: Any, suffix: str, run_root: Optional[Path]) -> str:
    path_text = _stringify(path_value)
    if not path_text:
        return ""
    path = Path(path_text).expanduser().with_suffix(suffix)
    return _relative_report_path(str(path), run_root)


def _candidate_shortfall_lines(
    goal: Dict[str, Any],
    generated_set: Dict[str, Any],
    trace: Dict[str, Any],
    route_rows: List[Dict[str, Any]],
) -> List[str]:
    """State an observed panel deficit and its bounded candidate-stage evidence."""
    minimum = goal.get("candidate_minimum")
    if type(minimum) is not int or minimum <= 0 or not generated_set:
        return []
    delivered = generated_set.get("distinct_molecule_count")
    if type(delivered) is not int:
        delivered = generated_set.get("unique_sequence_count")
    if type(delivered) is not int or delivered >= minimum:
        return []
    lines = [
        f"- **Requested panel shortfall:** {delivered} distinct delivered molecule(s) versus "
        f"a minimum of {minimum}; deficit {minimum - delivered}. The requested candidate "
        "count was not fulfilled."
    ]
    trace_rows = trace.get("rows") if isinstance(trace, dict) else None
    if isinstance(trace_rows, list):
        initial_removed = sum(
            row.get("first_drop_stage") == "initial_filter"
            for row in trace_rows if isinstance(row, dict))
        final_unselected = sum(
            row.get("first_drop_stage") == "final_output"
            for row in trace_rows if isinstance(row, dict))
        route_ineligible = 0
        for row in trace_rows:
            if not isinstance(row, dict):
                continue
            stages = row.get("stage_results")
            route = stages.get("routing_eligibility") if isinstance(stages, dict) else None
            route_ineligible += isinstance(route, dict) and route.get("status") == "fail"
        lines.append(
            f"- Bounded search trace: {len(trace_rows)} candidate sequence row(s) "
            f"entered the recorded funnel; {initial_removed} failed initial filters, "
            f"{final_unselected} passed to scoring but were not selected for final output"
            + (f" ({route_ineligible} lacked recorded route eligibility)." if route_ineligible
               else ".")
        )
    if route_rows:
        counts = [str(row.get("count")) for row in route_rows
                  if isinstance(row, dict) and row.get("count") is not None]
        if counts:
            lines.append(
                "- Ranked route candidate counts: " + ", ".join(counts)
                + ". Route pools can contain the same molecules and must not be summed "
                "to infer distinct available ASOs."
            )
    lines.append(
        "- This is a shortfall within the recorded computational search and filters, "
        "not evidence that no other viable ASOs exist outside that search. Unselected "
        "tiles are not validated replacement recommendations."
    )
    return lines


def _build_design_output_summary_markdown(report_payload: Dict[str, Any]) -> str:
    if not isinstance(report_payload, dict):
        return ""
    run_root = _infer_report_run_root(report_payload)
    meta = report_payload.get("structured_goal") if isinstance(report_payload.get("structured_goal"), dict) else {}
    gapmer = report_payload.get("gapmer_tiling") if isinstance(report_payload.get("gapmer_tiling"), dict) else {}
    tiling = report_payload.get("tiling") if isinstance(report_payload.get("tiling"), dict) else {}
    scored_path = _stringify(gapmer.get("scored_tiles_path") or gapmer.get("json_path") or tiling.get("scored_tiles_path"))
    rows = _load_report_rows(scored_path)
    if not rows and not gapmer and not tiling:
        return ""
    allele_specific_rows = [row for row in rows if _report_bool(row.get("allele_specific"))]
    allele_specific_design = bool(allele_specific_rows)
    allele_gap_count = sum(
        1 for row in allele_specific_rows if _report_bool(row.get("variant_in_gap"))
    )

    generated_set = _run_artifact_json(run_root, "generated_aso_set.json")
    generated_filter_trace = _run_artifact_json(run_root, "generated_filter_trace.json")
    strategy_router = _run_artifact_json(run_root, "strategy_router_summary.json")
    route_rows = _route_goal_rows(strategy_router, run_root)
    binding_sets = report_payload.get("binding_region_candidate_sets") or {}
    has_requested_bindings = bool(binding_sets.get("sets"))
    splice_design = bool(route_rows or meta.get("run_splice_eval") or str(meta.get("intent")) == "splice_modulation")
    modality = "gapmer" if gapmer else "steric splice-switching" if splice_design else "steric binding-site blocking"
    lines: List[str] = ["# Run Output Summary", ""]
    gene = _stringify(meta.get("gene_symbol") or meta.get("gene_id") or "target")
    intent = _stringify(meta.get("intent") or "ASO design")
    species = _stringify(meta.get("species"))
    context = _stringify(meta.get("target_context"))
    request_text = _run_request_text(run_root)
    if run_root and run_root.name:
        lines.append(f"Output folder: `{run_root.name}`.")
    lines.append(
        f"This run designed {modality} ASO candidates for {gene}"
        f"{f' ({species})' if species else ''}, with intent `{intent}`."
    )
    if context:
        lines.append(f"Target context: {_clean_report_sentence(context)}.")
    if request_text:
        lines.extend(["", "Original request:", "", _sanitize_report_paths(request_text, run_root)])
    lines.append("")

    lines.append("## How To Read The Rankings")
    if gapmer:
        if allele_specific_design:
            lines.extend(
                [
                    "- `selection_rank`: lower is better. Required mutant-versus-wild-type evidence is ranked before generic sequence proxies.",
                    "- `mutant_aso_sequence` is the candidate ASO in 5′→3′ orientation; `mutant_target_sequence` and `wildtype_target_sequence` are the aligned target strands.",
                    "- `wildtype_mismatch_count` records sequence-level discrimination from the wild-type target; `variant_position_in_aso` and `variant_in_gap` show where that mismatch falls in the gapmer.",
                    "- Central DNA-gap placement is prioritized over terminal or wing placement when the other evidence is comparable.",
                    "- `off_target_gene_count` counts distinct off-target genes. `Off_target_exact_count` is the workflow exact-hit count (inspect its count unit and source loci/gene lists); `repeat_hit_count_exact` counts alternate exact repeat loci. Do not equate loci, transcripts and genes.",
                ]
            )
        else:
            lines.extend(
                [
                    "- `selection_rank`: lower is better. Start here for a general shortlist.",
                    "- `isoform_uniqueness_class`: `gene_wide` means standard knockdown candidates were scored for pan-isoform coverage.",
                    "- " + ISOFORM_SCORE_DEFINITION,
                    "- `isoform_uniqueness_count` / `isoform_hit_count`: number of annotated **target-gene isoforms** containing the site. Higher is better for pan-isoform knockdown. This is not an off-target count.",
                    "- `Times_mapped`: legacy alias for that same target-isoform hit count on gapmer cDNA tiles. Do not read it as other-gene off-targets.",
                    "- `off_target_gene_count` counts distinct off-target genes. `Off_target_exact_count` is the workflow exact-hit count (inspect its count unit and source loci/gene lists); `repeat_hit_count_exact` counts alternate exact repeat loci. Do not equate loci, transcripts and genes.",
                ]
            )
    elif splice_design:
        lines.extend(
            [
                ("- Start with the requested physical binding-region sets. Route ranks describe predicted response at a scored splice site, which can differ from the ASO binding location."
                 if has_requested_bindings else "- Route `rank`: lower is better within the recorded response objective; verify the physical binding location before choosing individual ASOs."),
                "- `directional delta`: model-predicted splice effect in the requested direction. Higher is stronger.",
                "- `opposite delta`: strongest predicted effect in the wrong direction. Lower is better.",
                "- Route candidate count: non-zero routes produced a route-specific shortlist; zero-count routes are not chooser-ready.",
                "- `Off_target_exact_count` counts exact off-target hits in the source search; inspect the loci/gene lists and count unit. `repeat_hit_count_exact` counts alternate exact repeat loci. Distinct genes require `off_target_gene_count` or a deduplicated gene list. Do not equate these counts or treat `Times_mapped` as off-target genes.",
            ]
        )
    else:
        lines.append("- Candidates are ranked binding-site blockers. Inspect the requested site overlaps, sequence constraints, chemistry and measured screening annotations. Splice-route ranks and splice deltas are not evidence for this mechanism.")
    lines.extend(["", "## Main Files To Open"])
    emitted_path = False
    if gapmer:
        emitted_path |= _append_report_file_line(
            lines,
            _sibling_artifact(gapmer.get("scored_tiles_path"), ".xlsx", run_root),
            "best spreadsheet for browsing, filtering, and selecting candidate gapmers",
        )
        emitted_path |= _append_report_file_line(
            lines,
            _relative_report_path(gapmer.get("uniqueness_csv"), run_root),
            "ranked candidate table in CSV form",
        )
        emitted_path |= _append_report_file_line(
            lines,
            _relative_report_path(gapmer.get("scored_tiles_path"), run_root),
            "complete scored/ranked table in JSON form",
        )
    elif tiling:
        emitted_path |= _append_report_file_line(
            lines,
            _sibling_artifact(tiling.get("scored_tiles_path") or tiling.get("json_path"), ".xlsx", run_root),
            "spreadsheet for browsing scored steric/splice candidates",
        )
        emitted_path |= _append_report_file_line(
            lines,
            _relative_report_path(tiling.get("scored_tiles_path") or tiling.get("json_path"), run_root),
            "scored tile table with splice and off-target annotations",
        )
        emitted_path |= _append_report_file_line(
            lines,
            _run_artifact_path(run_root, "strategy_router_summary.json"),
            "route-level ranking summary; start here for splice-switching route choice",
        )
        if route_rows and not has_requested_bindings:
            emitted_path |= _append_report_file_line(
                lines,
                _stringify(route_rows[0].get("csv_path")),
                "top predicted-response candidate set; verify physical binding locations",
            )
        emitted_path |= _append_report_file_line(
            lines,
            _relative_report_path(tiling.get("json_path"), run_root),
            "initial tile table",
        )
    emitted_path |= _append_report_file_line(lines, _run_artifact_path(run_root, "generated_aso_set.json"), "final generated ASO set used by downstream reports")
    emitted_path |= _append_report_file_line(lines, _run_artifact_path(run_root, "generated_filter_trace.json"), "candidate-stage trace showing where rows entered, passed, or were filtered")
    if run_root:
        for filename, description in (
            ("design_panel.md", "selected panel with physical antisense sequences and target-accessibility evidence"),
            ("design_panel.csv", "selected panel with chemistry, transcript positions and species-specific screening scope"),
        ):
            panel_path = run_root / "artifacts" / "candidates" / filename
            if panel_path.is_file():
                emitted_path |= _append_report_file_line(lines, _relative_report_path(str(panel_path), run_root), description)
    if gapmer and run_root:
        filtered_paths = [
            *sorted(run_root.glob("gapmer_jobs/*/filtered_*.json")),
            *sorted(run_root.glob("artifacts/design/gapmer_jobs/*/filtered_*.json")),
        ]
        for filtered in filtered_paths[:4]:
            emitted_path |= _append_report_file_line(
                lines,
                _relative_report_path(str(filtered), run_root),
                "filter bucket for candidates removed before final selection",
            )
    if not emitted_path:
        lines.append("- `report.md`: run-level summary.")
    lines.append("")

    row_count = len(rows) or int(gapmer.get("row_count") or tiling.get("row_count") or 0)
    lines.append("## General Result")
    if generated_filter_trace.get("row_count") not in (None, "", [], {}):
        lines.append(f"- {generated_filter_trace.get('row_count')} candidate row(s) appeared in the stage trace before final output filtering.")
    if generated_set:
        lines.append(
            f"- Final generated ASO set: {generated_set.get('row_count', 'unknown')} retained row(s), "
            f"{generated_set.get('unique_sequence_count', 'unknown')} unique sequence(s), "
            f"{generated_set.get('excluded_row_count', 0)} excluded row(s)."
        )
        lines.extend(_candidate_shortfall_lines(
            meta, generated_set, generated_filter_trace, route_rows))
    if row_count:
        lines.append(f"- {row_count} scored candidate row(s) were available for review in the main scored table.")
    if rows:
        if allele_specific_design:
            variant_overlap_count = sum(
                1 for row in allele_specific_rows if _fmt_report_int(row.get("variant_position_in_aso")) != "unknown"
            )
            wt_mismatch_count = sum(
                1
                for row in allele_specific_rows
                if _fmt_report_int(row.get("wildtype_mismatch_count")) not in {"unknown", "0"}
            )
            lines.append(
                f"- Allele evidence: {variant_overlap_count}/{len(allele_specific_rows)} candidates overlap the variant, "
                f"{wt_mismatch_count}/{len(allele_specific_rows)} differ from the wild-type target, and "
                f"{allele_gap_count}/{len(allele_specific_rows)} place the variant in the DNA gap."
            )
            if allele_gap_count == 0:
                lines.append(
                    "- No retained candidate places the discriminating base in the DNA gap; the retained set is a lower-confidence allele-selective shortlist."
                )
        else:
            gene_wide_count = _count_gene_wide_rows(rows)
            if gene_wide_count:
                lines.append(f"- {gene_wide_count}/{len(rows)} scored rows use `gene_wide` isoform conservation.")
        zero_off = _count_zero_offtarget_rows(rows)
        measured_off = sum(_candidate_offtarget_count(row) is not None for row in rows)
        lines.append(f"- {zero_off}/{measured_off} rows with reported exact-match gene counts have zero annotated other-gene hits in the stated search; {len(rows)-measured_off} rows have unknown gene counts. This is not a safety assessment.")
    cross_species_filters = (
        [
            item for item in generated_set.get("eligibility_filters", [])
            if isinstance(item, dict) and _stringify(item.get("name")) == "cross_species_exact_match"
        ]
        if isinstance(generated_set.get("eligibility_filters"), list)
        else []
    )
    conservation_rows = generated_set.get("rows")
    if not isinstance(conservation_rows, list):
        conservation_rows = rows
    conservation = _cross_species_report_evidence(cross_species_filters, conservation_rows)
    for filter_row in cross_species_filters[:1]:
        species_text = ", ".join(filter_row.get("requested_species") or [])
        if filter_row.get("required"):
            lines.append(
                f"- Cross-species exact matching was required: {filter_row.get('passing_count', 'unknown')}/"
                f"{filter_row.get('total_count', 'unknown')} candidate(s) passed{f' for {species_text}' if species_text else ''}."
            )
        else:
            lines.append(f"- Cross-species exact matching was not required{f' for this {species_text} run' if species_text else ''}.")
    if conservation["active"]:
        lines.append(
            f"- Conservation evidence for the {conservation['total']} displayed candidate row(s): "
            f"{conservation['exact']} have recorded exact matches across the requested species, "
            f"{conservation['nonmatching']} were checked without a full cross-species match, and "
            f"{conservation['unknown']} have unavailable or incomplete conservation evidence. "
            "These are sequence observations, separate from hard-filter eligibility and biological activity."
        )
    filtering = gapmer.get("initial_filtering") if isinstance(gapmer.get("initial_filtering"), dict) else {}
    if filtering:
        kept = ((filtering.get("passed") or {}) if isinstance(filtering.get("passed"), dict) else {}).get("count")
        total = filtering.get("total_rows")
        if kept not in (None, "", [], {}) or total not in (None, "", [], {}):
            lines.append(f"- Initial sequence filters kept {kept if kept is not None else 'unknown'} of {total if total is not None else 'unknown'} candidates.")
    constraint_filtering = gapmer.get("constraint_filtering") if isinstance(gapmer.get("constraint_filtering"), dict) else (
        tiling.get("constraint_filtering") if isinstance(tiling.get("constraint_filtering"), dict) else {}
    )
    if constraint_filtering:
        removed = constraint_filtering.get("removed_count")
        kept = constraint_filtering.get("kept_count")
        if removed not in (None, "", [], {}) or kept not in (None, "", [], {}):
            lines.append(f"- Constraint filtering kept {kept if kept is not None else 'unknown'} and removed {removed if removed is not None else 'unknown'} candidate(s).")
    if route_rows:
        nonempty_routes = sum(1 for row in route_rows if str(row.get("count") or "0") not in {"0", "unknown"})
        lines.append(f"- Strategy routing produced {len(route_rows)} ranked route(s); {nonempty_routes} route(s) have non-empty candidate sets.")
    lines.append("")

    access_rows = [row for row in rows if isinstance(row.get("target_accessibility"), dict)]
    if access_rows:
        lines.extend(["## Target-RNA Accessibility and Screening Scope", "",
                      "RNAplfold predicts the probability that the full target binding window is unpaired. Higher values indicate greater predicted accessibility within the recorded transcript scope. These predictions do not establish cellular accessibility or RNase H activity.", "",
                      "| Species | Candidates with accessibility measurements | Candidates with complete off-target screening |",
                      "|---|---:|---:|"])
        species_keys = sorted({key for row in access_rows for key in (row["target_accessibility"].get("measurements") or {})})
        for key in species_keys:
            measured = sum(bool((row["target_accessibility"].get("measurements") or {}).get(key)) for row in access_rows)
            screened = sum((row.get("off_target_screening_status_by_species") or {}).get(key) == "complete" for row in access_rows)
            lines.append(f"| {key} | {measured}/{len(access_rows)} | {screened}/{len(access_rows)} |")
        lines.extend(["", "Exact conservation and off-target screening are separate checks. Missing screening coverage is unavailable, not zero off-targets. The panel exports preserve the per-species measurements, transcript identities, raw RNAplfold matrices and actual screening status.", ""])

    selected_screening_scope = _selected_off_target_budget_markdown(generated_set)
    if selected_screening_scope:
        lines.extend(["## Selected-Panel Off-Target Search Budget", "", selected_screening_scope, ""])

    lines.append("## Where Users Should Look To Choose ASOs")
    if gapmer:
        gapmer_xlsx = _run_artifact_path(run_root, "gapmer_tiling/gapmer_scored_uniqueness.xlsx") or "artifacts/candidates/gapmer_scored_uniqueness.xlsx"
        gapmer_csv = _run_artifact_path(run_root, "gapmer_tiling/gapmer_uniqueness_ranked.csv") or "artifacts/candidates/gapmer_uniqueness_ranked.csv"
        lines.append(f"- Start with `{gapmer_xlsx}` when manually browsing candidates.")
        if allele_specific_design:
            if allele_gap_count:
                lines.append("- Start with low `selection_rank`, confirm the recorded `wildtype_mismatch_count` and `variant_in_gap` evidence, and inspect off-target counts within their stated search scope. Apply the recorded eligibility filters.")
            else:
                lines.append("- No retained candidate has `variant_in_gap=true`. Review the filter trace and filtered candidates, and consider a different oligo architecture, length, or chemistry before promoting the wing-mismatch candidates.")
        else:
            lines.append("- Use the recorded `selection_rank` and eligibility filters. Inspect off-target counts within their stated search scope and interpret `isoform_uniqueness_score` using its recorded definition. A screening count does not introduce an additional cutoff. Do not filter `Times_mapped` or `isoform_uniqueness_count` as if they were off-targets.")
        lines.append(f"- Use `{gapmer_csv}` for spreadsheet workflows and the scored JSON for scripted review.")
        if conservation["required"]:
            lines.append("- Keep the cross-species fields visible; final choices should preserve the required species evidence.")
        elif conservation["active"]:
            lines.append("- Review the recorded cross-species matches alongside selection ranks. Conservation is optional: candidates without a supported cross-species match remain eligible when they satisfy the other requested constraints. Missing species evidence remains unknown.")
        else:
            lines.append("- No cross-species conservation evidence is recorded for these candidates.")
    elif route_rows:
        router_rel = _run_artifact_path(run_root, "strategy_router_summary.json") or "artifacts/design/strategy_router_summary.json"
        routes_rel = _run_artifact_path(run_root, "strategy_routes") or "artifacts/design/strategy_routes"
        if has_requested_bindings:
            lines.append(f"- Start with `{_relative_report_path(binding_sets.get('artifact_path'), run_root)}` and the candidate panel. Apply the stated containment/overlap rule for each requested physical region.")
        lines.append(f"- Review predicted splice responses in `{router_rel}` and the rank-ordered response CSV/XLSX files under `{routes_rel}/`.")
        lines.append("- A predicted response-site coordinate describes the measured model output; it does not relocate the ASO binding interval or establish experimental activity.")
        lines.append("- Favor routes with strong directional delta, low opposite-direction signal, non-empty candidate counts, and confidence/rationale consistent with the biology.")
    else:
        lines.append("- Use the delivered candidate CSV/JSON/XLSX tables and target-site annotations. No ranked splice-route output was produced.")
    lines.append("")

    if route_rows:
        lines.append("## Predicted Splice-Response Candidate Sets")
        lines.extend(_render_route_goal_table(route_rows))
        lines.append("")

    if gapmer:
        from .candidate_presentation import distinct_ranked_rows
        top_rows = _top_ranked_candidates(distinct_ranked_rows(generated_set.get("rows") or rows), limit=10)
        if top_rows:
            lines.append(
                "## Top Allele-Specific Candidates"
                if allele_specific_design
                else "## Top Normal-Ranking Candidates"
            )
            lines.extend(
                _render_allele_specific_candidate_table(top_rows)
                if allele_specific_design
                else _render_candidate_table(top_rows)
            )
            lines.append("")
    else:
        top_splice_rows = _top_splice_effect_candidates(rows, limit=8)
        if top_splice_rows:
            lines.append("## Strongest Raw Splice-Effect Rows")
            lines.append("Use these as a signal check only; route CSV/XLSX files are the better final chooser views.")
            lines.append("")
            lines.extend(_render_splice_effect_table(top_splice_rows))
            lines.append("")

    lines.append("## Practical Interpretation")
    if gapmer:
        if row_count:
            if allele_specific_design:
                if allele_gap_count:
                    lines.append(
                        f"For {gene}, use the allele-specific fields as the primary decision surface. The best initial shortlist is the low-`selection_rank` set "
                        "with the recorded mutant-versus-wild-type mismatch and central DNA-gap placement evidence, subject to the requested eligibility filters. Inspect the actual off-target screening coverage and counts."
                    )
                else:
                    lines.append(
                        f"For {gene}, the retained rows have sequence-level mutant-versus-wild-type discrimination but no DNA-gap mismatch placement. "
                        "Treat them as lower-confidence fallback candidates, not as a strong allele-selective shortlist."
                    )
                lines.append(
                    "A single sequence mismatch does not establish zero wild-type knockdown. Validate mutant and wild-type dose-response curves experimentally before treating a candidate as allele-selective."
                )
            else:
                lines.append(
                    f"For {gene}, use the recorded candidate ranks and requested eligibility filters to choose a shortlist. "
                    "Interpret each isoform score using its recorded selectivity or catalog-coverage definition, and retain the stated uncertainty of off-target screening."
                )
        if conservation["required"]:
            lines.append("Because this run required cross-species identity, do not promote candidates that lack the required species evidence even if their sequence metrics look good.")
        elif conservation["active"]:
            lines.append("Use the recorded conservation evidence for the requested prioritization while retaining otherwise eligible candidates without a cross-species match. An unavailable species reference does not establish absence of conservation. Exact matches do not establish cross-species activity.")
        else:
            lines.append("Cross-species conservation is unestablished in these outputs; it should not be inferred from single-species sequence scores.")
    elif route_rows:
        lines.append(
            f"For {gene}, make route choice first. The route table tells you which splice intervention produced usable candidate sets; "
            "then inspect the top route-specific CSV/XLSX files for the final ASO shortlist."
        )
        if route_rows:
            rationale = _stringify(route_rows[0].get("rationale"))
            if rationale:
                lines.append(f"Top-route rationale: {rationale}")
    else:
        lines.append(f"For {gene}, these are computational candidates for the requested steric-blocking sites. Sequence overlap and annotations support a binding hypothesis; cellular occupancy and the intended regulatory effect remain unmeasured.")
    lines.append("")

    caveats = []
    branch_outcomes = report_payload.get("branch_outcomes") if isinstance(report_payload.get("branch_outcomes"), dict) else {}
    for branch, outcome in branch_outcomes.items():
        if not isinstance(outcome, dict):
            continue
        status = _stringify(outcome.get("status")).lower()
        reason = _stringify(outcome.get("reason"))
        if status in {"no_candidates_after_constraints", "no_candidates_found", "skipped", "failed"} and reason:
            caveats.append(f"{_humanize_key(branch)}: {reason}")
    if caveats:
        lines.append("## Caveats")
        for item in caveats[:8]:
            lines.append(f"- {item}")
        lines.append("")

    markdown = "\n".join(lines).strip() + "\n"
    return _sanitize_report_paths(markdown, run_root)


def _normalize_summary(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "headline": _stringify(source.get("headline")) or _stringify(fallback.get("headline")),
        "plain_language_goal": _stringify(source.get("plain_language_goal")) or _stringify(fallback.get("plain_language_goal")),
        "what_happened": _stringify(source.get("what_happened")) or _stringify(fallback.get("what_happened")),
        "result": _stringify(source.get("result")) or _stringify(fallback.get("result")),
        "bullets": _normalize_string_list(source.get("bullets"), fallback.get("bullets") or [], limit=6),
    }


def _normalize_step_list(value: Any, fallback: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = _stringify(item.get("title"))
        lane = _stringify(item.get("lane")).lower() or "shared"
        summary = _stringify(item.get("summary"))
        why = _stringify(item.get("why_it_matters"))
        if not (title or summary or why):
            continue
        steps.append(
            {
                "title": title or "Step",
                "lane": lane,
                "summary": summary,
                "why_it_matters": why,
                "evidence_refs": _normalize_string_list(item.get("evidence_refs"), [], limit=5),
            }
        )
    return steps[:limit] or list(fallback)[:limit]


def _normalize_decision_list(value: Any, fallback: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        title = _stringify(item.get("title"))
        decision = _stringify(item.get("decision"))
        why = _stringify(item.get("why"))
        if not (title or decision or why):
            continue
        confidence = _stringify(item.get("confidence")).lower()
        if confidence not in _ALLOWED_CONFIDENCE:
            confidence = "unknown"
        decisions.append(
            {
                "title": title or "Decision",
                "decision": decision,
                "why": why,
                "confidence": confidence,
                "evidence_refs": _normalize_string_list(item.get("evidence_refs"), [], limit=5),
            }
        )
    return decisions[:limit] or list(fallback)[:limit]


def _normalize_branch_list(value: Any, context: Dict[str, Any], fallback: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    known = context.get("branch_outcomes") if isinstance(context.get("branch_outcomes"), dict) else {}
    items: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        branch = _stringify(item.get("branch"))
        if not branch or branch not in known:
            continue
        known_entry = known.get(branch) if isinstance(known.get(branch), dict) else {}
        status = _stringify(item.get("status")) or _stringify(known_entry.get("status")) or "unknown"
        items.append(
            {
                "branch": branch,
                "title": _stringify(item.get("title")) or f"{_humanize_key(branch)} branch",
                "status": status,
                "summary": _stringify(item.get("summary")) or _stringify(known_entry.get("reason")),
                "why": _stringify(item.get("why")) or _branch_status_explanation(status),
                "evidence_refs": _normalize_string_list(item.get("evidence_refs"), [f"branch_outcomes.{branch}"], limit=4),
            }
        )
    return items or list(fallback)


def _normalize_output_guides(value: Any, context: Dict[str, Any], fallback: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    known_outputs = {}
    for item in context.get("output_inventory") if isinstance(context.get("output_inventory"), list) else []:
        if isinstance(item, dict) and item.get("key"):
            known_outputs[str(item["key"])] = item
    guides: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        key = _stringify(item.get("key"))
        if not key:
            continue
        known = known_outputs.get(key, {})
        title = _stringify(item.get("title")) or _stringify(known.get("title")) or _humanize_key(key)
        what = _stringify(item.get("what_it_is")) or _stringify(known.get("summary"))
        how = _stringify(item.get("how_to_read")) or _stringify(known.get("how_to_read"))
        why = _stringify(item.get("why_it_matters")) or _stringify(known.get("why_it_matters"))
        if not (title or what or how or why):
            continue
        guides.append(
            {
                "key": key,
                "title": title or _humanize_key(key),
                "what_it_is": what,
                "how_to_read": how,
                "why_it_matters": why,
                "evidence_refs": _normalize_string_list(item.get("evidence_refs"), known.get("evidence_refs") or [f"output_inventory.{key}"], limit=4),
            }
        )
    return guides or list(fallback)


def _normalize_glossary(value: Any, fallback: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        term = _stringify(item.get("term"))
        definition = _stringify(item.get("definition"))
        if not term or not definition:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({"term": term, "definition": definition})
    if items:
        return items[:limit]
    return list(fallback)[:limit]


def _normalize_tooltips(value: Any, fallback: Dict[str, str]) -> Dict[str, str]:
    tooltips: Dict[str, str] = {}
    source = value if isinstance(value, dict) else {}
    for key in REPORT_TOOLTIP_TARGETS:
        text = _stringify(source.get(key))
        if text:
            tooltips[key] = text
    if tooltips:
        return tooltips
    return {key: text for key, text in fallback.items() if key in REPORT_TOOLTIP_TARGETS and _stringify(text)}


def _normalize_string_list(value: Any, fallback: Sequence[str], limit: int) -> List[str]:
    items: List[str] = []
    for item in value if isinstance(value, list) else []:
        text = _stringify(item)
        if text:
            items.append(text)
    if items:
        return items[:limit]
    fallback_items = [_stringify(item) for item in fallback]
    return [item for item in fallback_items if item][:limit]


def _request_text_from_payload(payload: Dict[str, Any], run_dir: str) -> str:
    from .report_presentation import display_request
    envelope = payload.get('request_envelope') or (payload.get('final_payload') or {}).get('request_envelope') or {}
    if isinstance(payload, dict):
        text = _stringify(payload.get("request"))
        if text:
            return display_request(run_dir, envelope, text)
    request_path = Path(run_dir) / "request.txt"
    if not request_path.exists():
        return ""
    try:
        return display_request(run_dir, envelope, request_path.read_text(encoding="utf-8").strip())
    except Exception:
        return ""


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _stringify(value)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _fallback_goal_sentence(meta: Dict[str, Any]) -> str:
    gene = _stringify(meta.get("gene_symbol")) or "the requested gene"
    intent = _stringify(meta.get("intent")) or "the requested objective"
    context = _stringify(meta.get("target_context"))
    if context:
        return f"This run focused on {gene} with intent '{intent}' in the context '{context}'."
    return f"This run focused on {gene} with intent '{intent}'."


def _fallback_result_sentence(branch_labels: Sequence[str], outputs: Sequence[str]) -> str:
    parts = []
    if branch_labels:
        parts.append("Branch outcomes were: " + "; ".join(branch_labels[:4]) + ".")
    if outputs:
        parts.append("The main persisted outputs were " + ", ".join(outputs[:5]) + ".")
    if not parts:
        parts.append("The run payload did not preserve a rich output summary.")
    return " ".join(parts)


def _branch_status_explanation(status: str) -> str:
    status = (status or "").strip().lower()
    if status == "outputs_present":
        return "This branch produced outputs that the user can inspect directly."
    if status == "no_candidates_after_constraints":
        return "The branch ran, but the requested filters or constraints eliminated the remaining candidate space."
    if status == "no_candidates_found":
        return "The branch ran, but it did not return any qualifying candidates."
    if status == "skipped":
        return "This branch was not executed because the request or earlier decisions did not require it."
    return "The compact report payload did not preserve a more specific interpretation for this branch."


def _default_glossary(report_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    terms = [
        {"term": "steric ASO", "definition": "An antisense oligo intended to block binding or splicing without triggering RNase H cleavage."},
        {"term": "gapmer", "definition": "An antisense oligo design that typically aims to recruit RNase H and reduce transcript abundance."},
    ]
    if report_payload.get("strategy_routing"):
        terms.extend(
            [
                {"term": "route goal", "definition": "A splice-targeting objective such as increasing or decreasing a specific donor or acceptor site."},
                {"term": "safeguard", "definition": "A nearby or competing splice site that the run tries to avoid perturbing while targeting the chosen site."},
                {"term": "threshold", "definition": "The cutoff used to decide whether a candidate counts as supporting a route goal."},
            ]
        )
    splicing_eval = report_payload.get("splicing_eval")
    if isinstance(splicing_eval, dict) and splicing_eval:
        terms.extend(
            [
                {"term": "donor site", "definition": "The splice site at the start of an intron where splicing can be increased or decreased."},
                {"term": "acceptor site", "definition": "The splice site at the end of an intron where splicing can be increased or decreased."},
            ]
        )
    if report_payload.get("gapmer_tiling"):
        terms.extend(
            [
                {
                    "term": "target isoforms",
                    "definition": "How many annotated isoforms of the intended gene contain this oligo site. Higher is better for pan-isoform knockdown.",
                },
                {
                    "term": "Times_mapped",
                    "definition": "On gapmer cDNA tiles, a legacy alias for that target-isoform hit count. It is not a count of off-target genes.",
                },
                {
                    "term": "off-target genes",
                    "definition": "Other genes with an exact match to the oligo. Prefer 0. Do not confuse this with target-isoform hit counts.",
                },
            ]
        )
    return terms


def _fallback_tooltips(context: Dict[str, Any]) -> Dict[str, str]:
    meta = context.get("run_metadata") if isinstance(context.get("run_metadata"), dict) else {}
    gene = _stringify(meta.get("gene_symbol")) or "this gene"
    routing_summary = context.get("routing_summary") if isinstance(context.get("routing_summary"), dict) else {}
    output_inventory = context.get("output_inventory") if isinstance(context.get("output_inventory"), list) else []
    branch_outcomes = context.get("branch_outcomes") if isinstance(context.get("branch_outcomes"), dict) else {}
    return {
        "guide_summary": f"This section explains the {gene} run in plain language before you look at raw pipeline details.",
        "guide_walkthrough": "These steps describe what the run actually did, in order, using only the evidence preserved in the run context.",
        "guide_decisions": "Each decision card explains one important choice the pipeline made and why that choice mattered for the final outputs.",
        "guide_outputs": f"This run preserved {len(output_inventory)} major output category or categories. Use these guides before opening raw files.",
        "guide_caveats": "Caveats collect warnings, uncertainty, timeouts, and places where the run should be interpreted carefully.",
        "guide_glossary": "The glossary only defines terms that are likely to be confusing in this specific run.",
        "overview_snapshot": f"This snapshot summarizes the overall {gene} run before you drill into maps, decisions, or file inventories.",
        "overview_branches": f"These cards show branch statuses for {len(branch_outcomes)} recorded branch or branches.",
        "overview_alerts": "Alerts are preserved warnings or blockers carried through the report payload.",
        "overview_markdown": "This markdown file is rendered from the structured explainer so the narrative stays grounded in the same evidence as the HTML guide.",
        "targets_summary": "This summary rolls up the targeted sites and recommended ASO outputs produced for the run.",
        "maps_targets": f"This map shows where the selected targets land across the transcript models for {gene}.",
        "maps_target_details": "This panel explains the currently selected target in a more readable way than the structured output.",
        "maps_target_records": "These records keep the full per-target detail when the visual map alone is not enough.",
        "maps_safeguards": "Safeguards matter when the run used splice routing and needed to preserve other nearby splice behaviors.",
        "maps_safeguard_details": "This panel explains what each safeguard marker represents for the selected target.",
        "decisions_pipeline": "These are the structured stage-level notes preserved by the pipeline itself.",
        "decisions_timeline": "This timeline turns the run into a readable sequence of events and decisions.",
        "decisions_routing": f"This run preserved {routing_summary.get('goal_count') or 0} routed goal or goals after routing.",
        "decisions_strategy_analysis": "This section shows the preserved splice strategy reasoning when a splice strategy analysis was generated.",
        "decisions_trace": "The trace exposes raw intermediate agent outputs that sit behind the condensed explanations.",
        "outputs_collections": "These cards group semantically related outputs so users can understand what exists before opening raw files.",
        "outputs_artifacts": "This is the file-level audit trail for the run directory.",
        "data_views": "These compact payload views are mainly for debugging or for users who need the raw structure behind the explanations.",
    }


def _humanize_key(value: Any) -> str:
    text = _stringify(value)
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text.replace("_", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else ""


def _slugify(value: Any) -> str:
    text = _stringify(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()
