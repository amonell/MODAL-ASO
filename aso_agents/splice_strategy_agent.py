from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Set

from agents import Agent, Runner, ModelSettings


def _call_agent_json(
    agent: Agent,
    payload: Dict[str, Any],
    *,
    max_steps: int = 6,
    traces: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run an agent and parse JSON output with helpful error messaging."""
    prompt = json.dumps(payload, ensure_ascii=False)
    result = Runner._run_sync(agent, input=prompt, max_steps=max_steps)
    output = result.final_output
    if not isinstance(output, str):
        raise RuntimeError(f"{agent.name} did not return a string output.")
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{agent.name} returned non-JSON output: {output}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{agent.name} must return a JSON object.")
    if traces is not None:
        traces.append(
            {
                "agent": agent.name,
                "kind": "json_agent",
                "prompt": payload,
                "final_output": data,
                "responses": result.responses,
            }
        )
    return data


HIGH_REASONING = ModelSettings(reasoning={"effort": "high"})
HIGH_REASONING_JSON = ModelSettings(
    reasoning={"effort": "high"},
    extra_body={
        "text": {"format": {"type": "json_object"}},
    },
)


_mixed_junction_decider_agent = Agent(
    name="Splice Junction Direction Decider",
    instructions=(
        "Reference-scope contract: reference_scope records physical sharing only within its compared transcript catalog; outside-catalog specificity is unknown. Objective metadata, including unique_to_pathogenic and covers_all_isoforms, concerns selected_design_transcripts only and cannot establish gene-wide uniqueness.\n"
        "You choose whether mixed-status splice junctions should be increased or decreased to satisfy the user's ASO goal.\n"
        "Input JSON: {\"user_request\": string, \"junctions\": [{\"label\": string, \"isoforms\": [{\"label\": string, \"role\": \"beneficial\"|\"pathogenic\"|\"neutral\", \"targetable\": bool}], \"all_isoforms_shared\": bool}]}\n"
        "Output JSON: {\"decisions\": [{\"label\": string, \"direction\": \"increase\"|\"decrease\"|\"skip\", \"rationale\": string}]}\n"
        "Return only a JSON object. Do not include markdown fences or prose.\n"
        "Guidelines:\n"
        "- Focus on the roles provided: increase beneficial isoforms when they are the priority, decrease pathogenic isoforms to suppress them.\n"
        "- When both beneficial and pathogenic isoforms share the junction, reason about the user_request to decide which outcome offers the bigger net benefit.\n"
        "- Neutral isoforms neither help nor hurt; prefer the outcome that advances the explicit goals even if neutrals move with the target.\n"
        "- If every isoform shares the junction and moving it would not create a useful differential (e.g., increasing all isoforms equally), return direction=\"skip\".\n"
        "- Keep rationales concise (<=120 characters) and mention key trade-offs.\n"
    ),
    tools=[],
    model="gpt-5.2-2025-12-11",
    model_settings=HIGH_REASONING_JSON,
)


def resolve_mixed_junction_directions(
    user_request: str,
    junctions: List[Dict[str, Any]],
    traces: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not junctions:
        return {}

    payload = {
        "user_request": user_request,
        "junctions": junctions,
    }

    result = _call_agent_json(
        _mixed_junction_decider_agent,
        payload,
        max_steps=4,
        traces=traces,
    )

    decisions: Dict[str, Dict[str, Any]] = {}
    for item in result.get("decisions", []):
        label = str(item.get("label") or "").strip()
        direction = str(item.get("direction") or "").strip().lower()
        if not label or direction not in {"increase", "decrease", "skip"}:
            continue
        decisions[label] = {
            "direction": direction,
            "rationale": item.get("rationale"),
        }
    return decisions


_ambiguous_safeguard_decider_agent = Agent(
    name="Safeguard Relevance Decider",
    instructions=(
        "You decide whether ambiguous splice-site safeguards should be kept or dropped for a specific primary target.\n"
        "Input JSON: {\"user_request\": string, \"isoform_objectives\": [{\"isoform_label\": string, \"desired_abundance\": \"increase\"|\"decrease\", \"role\": string?}], "
        "\"cases\": [{\"primary_label\": string, \"candidate_label\": string, \"classification_reason\": string, "
        "\"primary\": {\"direction\": string|null, \"present_in\": [string], \"absent_in\": [string], \"beneficial_isoforms\": [string], \"pathogenic_isoforms\": [string]}, "
        "\"candidate\": {\"direction\": string|null, \"present_in\": [string], \"absent_in\": [string], \"beneficial_isoforms\": [string], \"pathogenic_isoforms\": [string], "
        "\"conflict_between_objectives\": bool, \"shared_with_beneficial\": bool, \"unique_to_pathogenic\": bool, \"unique_to_beneficial\": bool, "
        "\"same_bundle_as_primary\": bool, \"same_direction_as_primary\": bool}}]}\n"
        "Output JSON: {\"decisions\": [{\"primary_label\": string, \"candidate_label\": string, \"action\": \"keep\"|\"drop\", \"rationale\": string}]}\n"
        "Return only a JSON object. Do not include markdown fences or prose.\n"
        "Guidelines:\n"
        "- Assess whether the candidate guard tracks an unwanted effect relative to the user's objectives.\n"
        "- Distinct physical junctions can have different measured effects even when their transcript membership or desired directions match. Those fields alone do not establish redundant measurements or coupled movement.\n"
        "- candidate.guard_kind describes the monitored direction (neutral, no_increase, or no_decrease); candidate.guard_policy=advisory means breaches remain visible without removing candidates. Such a guard can document collateral risk at a shared site; it does not assert that the primary route spares every isoform.\n"
        "- Drop a guard for demonstrated irrelevance, a duplicate physical measurement, or a conflict with the requested monitoring policy. Explain that evidence; do not infer redundancy from transcript membership alone.\n"
        "- Prefer keeping unique beneficial-only or pathogenic-only junctions when they oppose unwanted collateral movement.\n"
        "- Treat classification_reason as a hint about why the case was escalated; use the isoform bundles to make the final call.\n"
        "- Do not assume that upstream or downstream junctions move with the primary target without evidence of that relationship.\n"
        "- Rationales must be concise (<=120 characters) and mention the key reason.\n"
    ),
    tools=[],
    model="gpt-5.2-2025-12-11",
    model_settings=HIGH_REASONING_JSON,
)


def resolve_ambiguous_safeguards(
    user_request: str,
    isoform_objectives: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    traces: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not cases:
        return {}

    payload = {
        "user_request": user_request,
        "isoform_objectives": isoform_objectives,
        "cases": cases,
    }

    result = _call_agent_json(
        _ambiguous_safeguard_decider_agent,
        payload,
        max_steps=4,
        traces=traces,
    )

    decisions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in result.get("decisions", []):
        primary_label = str(item.get("primary_label") or "").strip()
        candidate_label = str(item.get("candidate_label") or "").strip()
        action = str(item.get("action") or "").strip().lower()
        if not primary_label or not candidate_label or action not in {"keep", "drop"}:
            continue
        decisions[(primary_label, candidate_label)] = {
            "action": action,
            "rationale": item.get("rationale"),
        }
    return decisions


_target_identifier_agent = Agent(
    name="Splice Target Identifier",
    instructions=(
        "You list every steric hindrance splice junction opportunity that can advance the isoform objectives.\n"
        "Input JSON: {\"context\": string}. The context includes the original user_request, isoform objectives, steric_site_candidates (with label_hint, canonical_label, and recommended_directions), splice_site_plan entries, and notes.\n"
        "Output JSON: {\"targets\": [{\"label\": string, \"site_kind\": \"acceptor\"|\"donor\", \"chrom\": string, \"position\": int|null, \"isoforms_present\": [string], \"isoforms_absent\": [string], \"desired_direction\": \"increase\"|\"decrease\", \"coupled_with\": [string], \"rationale\": string}], \"notes\": string?}.\n"
        "Return only a JSON object. Do not include markdown fences or prose.\n"
        "Rules:\n"
        "- Emit at least one target entry for every steric_site_candidate. When a junction applies to multiple isoforms, create separate entries per isoform/direction pair (use recommended_directions when provided).\n"
        "- Set the label field to canonical_label (e.g., 'donor@156138607') exactly as provided. If canonical_label is null, fall back to label_hint without modification.\n"
        "- Copy site_kind, chrom, position, isoforms_present, and isoforms_absent from the candidate. Maintain coupled_with ONLY to reference the paired junction for the same isoform/direction.\n"
        "- desired_direction is an optimization hypothesis, not a predicted biological outcome. Prefer recommended_directions; if none are supplied, align with the explicit isoform or exon objectives. Context/fallback sites may be explored, but their rationale must state when the desired isoform consequence is unestablished.\n"
        "- Shared boundaries and delta_per_isoform encode anatomy and requested outcomes. They do not establish selective conversion between isoforms or reduction of an isoform that also contains the boundary. Require separate directional evidence before asserting that effect; otherwise describe an exploratory hypothesis with unresolved direction. A site elsewhere in the gene is not the requested alternative exon merely because its annotation ordinal matches a clinical exon label.\n"
        "- Use coupled_with ONLY to reference the partner junction (same isoform & direction) when every involved candidate is unique_to_pathogenic and conflict_between_objectives is false; otherwise keep it empty and emit single-junction fallbacks instead.\n"
        "- Treat an intended increase at an increased isoform's own distinguishing junction as a supported primary route, "
        "not a fallback merely because beneficial_isoforms are present. Mark conflicting or shared routes as fallbacks "
        "when their proposed direction opposes another requested or preserved outcome, and explain that conflict.\n"
        "- Never add targets that list the isoform in terminal_donor_isoforms unless another isoform remains in present_in; in that case note the terminal isoform in the rationale but direct modulation only toward the non-terminal isoforms.\n"
        "- Skip non-junction elements; focus strictly on donor/acceptor windows even when they behave as shared or secondary options.\n"
        "- Keep rationale terse (<140 characters) indicating primary leverage vs shared fallback and the expected isoform shift. Reference label_hint or isoform names in the rationale instead of modifying the label, and mention net_effect_hint when risk exists.\n"
        "- Do not invent new isoforms, genomic positions, or directions beyond what the context supports."
    ),
    tools=[],
    model="gpt-5.2-2025-12-11",
    model_settings=HIGH_REASONING_JSON,
)


_target_ranker_agent = Agent(
    name="Splice Target Ranker",
    instructions=(
        "You transform splice targets into an ordered list of single-junction routing goals.\n"
        "Input JSON: {\"context\": string, \"targets\": [...]}. Targets come from the identifier agent and include net_leverage, pathogenic_gain, beneficial_penalty, and terminal_donor_isoforms metadata.\n"
        "Output JSON: {\"ranked_targets\": [{\"rank\": int, \"bundle\": [{\"label\": string, \"desired_direction\": string}], \"primary_effect\": string, \"rationale\": string, \"confidence\": \"high\"|\"medium\"|\"low\"}], \"notes\": string?}.\n"
        "Return only a JSON object. Do not include markdown fences or prose.\n"
        "Guidelines:\n"
        "- Emit exactly one target per canonical label; never combine multiple junctions in the same bundle.\n"
        "- Rank strictly by descending net_leverage (higher pathogenic_gain minus beneficial_penalty first). Use beneficial_penalty as the primary tie-breaker (fewer collateral isoforms first); if still tied, prefer targets affecting fewer terminal_donor_isoforms, then fall back to original target order.\n"
        "- Demote (never drop) any target whose metadata shows conflict_between_objectives or shared_with_beneficial; keep them as late-ranked fallbacks with rationales that acknowledge the trade-off, still respecting the net_leverage ordering.\n"
        "- Copy labels verbatim, reuse the desired_direction from the targets, keep rationale <=160 characters, and avoid new terminology."
        " Desired directions and leverage derived from objectives are not effect measurements. Preserve unresolved mechanism qualifiers from the input; never turn a shared-site exploratory hypothesis into an established isoform shift."
    ),
    tools=[],
    model="gpt-5.2-2025-12-11",
    model_settings=HIGH_REASONING_JSON,
)


_strategy_report_agent = Agent(
    name="Splice Strategy Author",
    instructions=(
        "Compose the final splice strategy report in concise Markdown.\n"
        "Input JSON: {\"context\": string, \"ranked_targets\": [...], \"global_notes\": string?}.\n"
        "Output MUST include the headings 'Strategy Overview' and 'Ranked Steric Hindrance Targets'. Include 'Routing Notes' only when you have actionable guidance.\n"
        "Formatting requirements:\n"
        "- Under Strategy Overview, list one bullet per isoform objective: '<isoform>: <increase|decrease> - <<=140 char rationale>'.\n"
        "- Under Ranked Steric Hindrance Targets, enumerate every bundle in order. Each entry must follow exactly this template:\n"
        "  <rank>) <direction> <isoform> <acceptor|donor> usage (include 'acceptor and donor' when both appear)\n"
        "  - Primary targets: <comma-separated labels with direction in parentheses>\n"
        "  - Safeguards: <comma-separated guard summaries>\n"
        "  - Notes: <<=120 char note clarifying whether this is the main bundle or a fallback>\n"
        "- Routing Notes (when present) should summarize how combined bundles and singles should be treated; omit the section if there is nothing new to add.\n"
        "- Do not invent extra ranks or auxiliary commentary. Keep the wording tight and actionable."
    ),
    tools=[],
    model="gpt-5.2-2025-12-11",
    model_settings=HIGH_REASONING,
)


def generate_splice_strategy_report(
    context_prompt: str,
    context_data: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the splice strategy agent pipeline and return the Markdown report plus execution trace."""
    diagnostic_trace: List[Dict[str, Any]] = []

    targets = _call_agent_json(
        _target_identifier_agent,
        {"context": context_prompt},
        max_steps=6,
        traces=diagnostic_trace,
    )

    target_direction_map: Dict[str, Set[str]] = {}
    for target_entry in targets.get("targets", []) or []:
        if not isinstance(target_entry, dict):
            continue
        label = str(target_entry.get("label") or "").strip()
        direction = str(target_entry.get("desired_direction") or "").strip().lower()
        if not label or direction not in {"increase", "decrease", "neutral"}:
            continue
        target_direction_map.setdefault(label, set()).add(direction)

    ranked = _call_agent_json(
        _target_ranker_agent,
        {"context": context_prompt, "targets": targets.get("targets", [])},
        max_steps=6,
        traces=diagnostic_trace,
    )

    candidate_map: Dict[str, Dict[str, Any]] = {}
    for entry in context_data.get("steric_site_candidates", []) or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("canonical_label") or entry.get("label_hint")
        if not key:
            continue
        candidate_map[str(key)] = entry

    ranked_targets = ranked.get("ranked_targets", []) or []

    global_notes: Optional[str] = None

    bond_map: Dict[str, Dict[str, Any]] = {}
    for site in context_data.get("splice_site_plan") or []:
        if not isinstance(site, dict):
            continue
        kind = (site.get("site") or "").strip().lower()
        position = site.get("position")
        if kind not in {"acceptor", "donor"} or position is None:
            continue
        canonical = f"{kind}@{position}"
        bond_map[canonical] = site

    ranked_direction_map: Dict[str, str] = {}
    for entry in ranked_targets:
        bundle = entry.get("bundle") or []
        if not bundle:
            continue
        first = bundle[0]
        label = str(first.get("label") or "").strip()
        direction = str(first.get("desired_direction") or "").strip().lower()
        if label and direction in {"increase", "decrease", "neutral"}:
            ranked_direction_map[label] = direction

    for entry in ranked_targets:
        bundle = entry.get("bundle") or []
        if not bundle:
            continue
        first = bundle[0]
        label = str(first.get("label") or "")
        if not label:
            continue
        metadata = candidate_map.get(label)
        if not isinstance(metadata, dict):
            metadata = candidate_map.get(str(first.get("label")))
        if not isinstance(metadata, dict):
            continue
        isoforms_present = set(metadata.get("present_in") or [])
        safeguards: List[Dict[str, Any]] = []
        for canonical, site in bond_map.items():
            if canonical == label:
                continue
            guard_value: Optional[str] = ranked_direction_map.get(canonical)
            if guard_value not in {"increase", "decrease", "neutral"}:
                direction_set = target_direction_map.get(canonical)
                if direction_set:
                    viable_directions = {item for item in direction_set if item in {"increase", "decrease", "neutral"}}
                    if len(viable_directions) == 1:
                        guard_value = viable_directions.pop()
            if guard_value not in {"increase", "decrease", "neutral"}:
                site_meta = candidate_map.get(canonical)
                if isinstance(site_meta, dict):
                    decision_direction = str(site_meta.get("decision_direction") or "").strip().lower()
                    if decision_direction in {"increase", "decrease", "neutral"}:
                        guard_value = decision_direction
                    else:
                        for rec in site_meta.get("recommended_directions", []) or []:
                            direction_hint = str(rec.get("direction") or "").strip().lower()
                            if direction_hint in {"increase", "decrease", "neutral"}:
                                guard_value = direction_hint
                                break
            if guard_value not in {"increase", "decrease", "neutral"}:
                guard_value = "neutral"
            guard_entry = {
                "label": canonical,
                "guard": guard_value,
            }
            safeguards.append(guard_entry)
        entry["safeguards"] = safeguards

    final_result = Runner._run_sync(
        _strategy_report_agent,
        input=json.dumps(
            {
                "context": context_prompt,
                "ranked_targets": ranked.get("ranked_targets", []),
                "global_notes": global_notes,
            },
            ensure_ascii=False,
        ),
        max_steps=6,

    )

    output = final_result.final_output
    if not isinstance(output, str):
        raise RuntimeError("Splice Strategy Author returned non-string output.")
    diagnostic_trace.append(
        {
            "agent": _strategy_report_agent.name,
            "kind": "report_agent",
            "prompt": {
                "context": context_prompt,
                "ranked_targets": ranked.get("ranked_targets", []),
                "global_notes": global_notes,
            },
            "final_output": output,
            "responses": final_result.responses,
        }
    )
    return output.strip(), diagnostic_trace
