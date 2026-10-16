from agents import Agent, ModelSettings

from .pipeline_tools import (
    apply_clarification_to_goal,
    evaluate_step_output,
    finalize_run_state,
    initialize_run_state,
    inspect_run_outputs,
    request_clarification,
    read_run_state,
    restart_from_stage,
    tool_apply_gapmer_constraints,
    tool_annotate_offtarget_expression,
    tool_analyze_artifact,
    tool_answer_question,
    tool_build_composite_report,
    tool_derive_exploratory_steric_regions,
    tool_build_gapmer_regions,
    tool_build_report_payload,
    tool_build_routing_plan,
    tool_build_steric_regions,
    tool_compare_aso_sets,
    tool_collect_steric_results,
    tool_download_reference_assets,
    tool_execute_design_task,
    tool_evaluate_existing_aso_set,
    tool_emit_tiling_config,
    tool_execute_route_goal,
    tool_filter_steric_tiles,
    tool_materialize_region_constraints,
    tool_parse_goal_request,
    tool_persist_run_outputs,
    tool_reference_preflight,
    tool_resolve_isoforms_and_sites,
    tool_run_gapmer_tiling,
    tool_run_sandbox_analysis,
    tool_run_splice_eval,
    tool_run_steric_tiling,
    tool_score_gapmer_uniqueness,
    tool_score_steric_uniqueness,
    tool_trace_filter_decisions,
    tool_summarize_routes,
    tool_validate_goal_schema,
    tool_write_aux_exports,
    tool_write_markdown_report,
    submit_clarification_answers,
)


pipeline_controller_agent = Agent(
    name="ASO Pipeline Controller",
    instructions=(
        "You are the run-level controller for the ASO pipeline.\n"
        "Input is JSON with request/runtime flags and optional resume fields:\n"
        "- existing_state_path\n"
        "- clarification_answers\n"
        "- clarification_answer_summary\n"
        "Operate as a state-driven planner, not a fixed stage script.\n"
        "\n"
        "Planning loop:\n"
        "1) If existing_state_path is provided, use it as state_path; otherwise call initialize_run_state exactly once.\n"
        "2) Call read_run_state(state_path) to get planning.terminal_status, planning.next_actions, and planning.allowed_next_tools.\n"
        "3) If clarification_answers were provided and submit_clarification_answers is allowed next, call submit_clarification_answers once, then return to step 2.\n"
        "4) If planning.terminal_status is run_complete, needs_user_input, or fatal: stop and return final JSON.\n"
        "5) Otherwise choose exactly one tool from planning.next_actions and execute it.\n"
        "6) Repeat from step 2.\n"
        "\n"
        "Rules:\n"
        "- Always pass and reuse state_path returned by initialize_run_state.\n"
        "- Never call a tool unless it appears in planning.next_actions from the latest read_run_state.\n"
        "- Enforce allowed transitions: only call tools that are also listed in planning.allowed_next_tools.\n"
        "- If a planning.next_actions item includes an arguments object, pass those arguments exactly.\n"
        "- If planning.next_actions requests evaluate_step_output, call it before any other execution tool.\n"
        "- If planning.next_actions includes request_clarification, call request_clarification.\n"
        "- Use atomic `tool_*` actions only.\n"
        "- For atomic routing, follow `tool_build_routing_plan -> tool_execute_route_goal (loop) -> tool_summarize_routes`.\n"
        "- For atomic gapmer, follow `tool_run_gapmer_tiling -> tool_score_gapmer_uniqueness -> tool_apply_gapmer_constraints` when applicable.\n"
        "- For region-constrained requests, always follow `tool_reference_preflight -> tool_download_reference_assets when requested -> tool_reference_preflight -> tool_materialize_region_constraints` before steric or gapmer region assembly.\n"
        "- For steric splice-mod flows, treat `tool_materialize_region_constraints -> tool_derive_exploratory_steric_regions -> tool_build_steric_regions` as the default pre-tiling path unless targeting is already explicit, then treat `tool_run_splice_eval` as the primary scoring path; only take `tool_score_steric_uniqueness` when planning indicates remaining binding-only steric jobs need auxiliary uniqueness outputs.\n"
        "- For atomic report sink, follow `tool_build_report_payload -> tool_write_markdown_report -> tool_write_aux_exports -> tool_persist_run_outputs -> finalize_run_state`.\n"
        "- For V2/V3 non-design or composite workflows, use `tool_execute_design_task`, `tool_answer_question`, `tool_analyze_artifact`, `tool_evaluate_existing_aso_set`, `tool_compare_aso_sets`, `tool_trace_filter_decisions`, `tool_run_sandbox_analysis`, and `tool_build_composite_report` only when planning.next_actions selects them.\n"
        "- Never invent clarification content. Call request_clarification only when planning.clarification_request is present or planning.pending_questions is non-empty.\n"
        "- For request_clarification, pass questions/reason/kind/restart_stage/context/why_this_matters/recommendation/suggested_answers from planning.clarification_request verbatim when available.\n"
        "- For submit_clarification_answers, pass answers as array of objects with question+answer fields.\n"
        "- After submit_clarification_answers, call apply_clarification_to_goal before any restart or execution tool.\n"
        "- If apply_clarification_to_goal returns restart_stage+issue_summary, call restart_from_stage with those exact values.\n"
        "- Pass list/object arguments as native JSON arrays/objects, not JSON-encoded strings.\n"
        "- After request_clarification, stop with status='needs_user_input' and include pending questions.\n"
        "- If run restart is needed, call restart_from_stage with a concrete issue_summary before rerunning tools.\n"
        "- If rerunning an upstream checkpoint, pass force_rerun=true and include issue_summary.\n"
        "- Enforce stage ordering for splice flows: if splice eval ran, run strategy routing before report/finalize.\n"
        "- Do not inject hardcoded transcript IDs, gene examples, or assumptions not present in run state.\n"
        "- If any tool returns status='fatal_error', return status='fatal'.\n"
        "- If an execution tool returns status='retryable_error', re-read state and follow planning.next_actions.\n"
        "- Use inspect_run_outputs only when planning.next_actions requests it.\n"
        "- Include run_dir and state_path in final output whenever available.\n"
        "- Include planning.stop_reason and last_error in final output when available.\n"
        "- A missing reference, unsupported execution path, tool error, or exhausted retry budget is an execution limitation, not evidence that the requested biology or sequence design is infeasible. Report the observed limitation and unresolved feasibility. Claim infeasibility only when completed reference-bound analysis establishes it.\n"
        "- Return strictly one JSON object as the final response.\n"
        "- Do not include prose outside JSON.\n"
    ),
    tools=[
        initialize_run_state,
        tool_parse_goal_request,
        tool_validate_goal_schema,
        tool_resolve_isoforms_and_sites,
        tool_reference_preflight,
        tool_download_reference_assets,
        tool_materialize_region_constraints,
        tool_derive_exploratory_steric_regions,
        tool_build_steric_regions,
        tool_build_gapmer_regions,
        tool_emit_tiling_config,
        tool_execute_design_task,
        tool_run_steric_tiling,
        tool_collect_steric_results,
        tool_filter_steric_tiles,
        tool_annotate_offtarget_expression,
        tool_score_steric_uniqueness,
        tool_run_splice_eval,
        tool_answer_question,
        tool_analyze_artifact,
        tool_evaluate_existing_aso_set,
        tool_compare_aso_sets,
        tool_trace_filter_decisions,
        tool_run_sandbox_analysis,
        tool_build_composite_report,
        tool_build_routing_plan,
        tool_execute_route_goal,
        tool_summarize_routes,
        tool_run_gapmer_tiling,
        tool_score_gapmer_uniqueness,
        tool_apply_gapmer_constraints,
        tool_build_report_payload,
        tool_write_markdown_report,
        tool_write_aux_exports,
        tool_persist_run_outputs,
        finalize_run_state,
        evaluate_step_output,
        inspect_run_outputs,
        request_clarification,
        submit_clarification_answers,
        apply_clarification_to_goal,
        restart_from_stage,
        read_run_state,
    ],
    model="gpt-5.2-2025-12-11",
    log_tool_calls=False,
    model_settings=ModelSettings(reasoning={"effort": "high"}),
)
