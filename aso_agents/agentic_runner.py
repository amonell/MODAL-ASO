import json
import os
from pathlib import Path
import signal
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, Optional

from agents import Runner

from .pipeline_controller_agent import pipeline_controller_agent
from .pipeline_tools import (
    _derive_next_actions,
    _event,
    _load_state,
    _state_path_to_obj,
    submit_clarification_answers,
)


_DETERMINISTIC_CONTROLLER_TOOL_NAMES = {
    "apply_clarification_to_goal",
    "evaluate_step_output",
    "finalize_run_state",
    "inspect_run_outputs",
    "request_clarification",
    "restart_from_stage",
    "tool_analyze_artifact",
    "tool_annotate_offtarget_expression",
    "tool_answer_question",
    "tool_apply_gapmer_constraints",
    "tool_build_composite_report",
    "tool_build_gapmer_regions",
    "tool_build_report_payload",
    "tool_build_routing_plan",
    "tool_build_steric_regions",
    "tool_collect_steric_results",
    "tool_compare_aso_sets",
    "tool_derive_exploratory_steric_regions",
    "tool_download_reference_assets",
    "tool_emit_tiling_config",
    "tool_evaluate_existing_aso_set",
    "tool_execute_design_task",
    "tool_execute_route_goal",
    "tool_filter_steric_tiles",
    "tool_materialize_region_constraints",
    "tool_parse_goal_request",
    "tool_persist_run_outputs",
    "tool_reference_preflight",
    "tool_resolve_isoforms_and_sites",
    "tool_run_gapmer_tiling",
    "tool_run_sandbox_analysis",
    "tool_run_splice_eval",
    "tool_run_steric_tiling",
    "tool_score_gapmer_uniqueness",
    "tool_score_steric_uniqueness",
    "tool_summarize_routes",
    "tool_trace_filter_decisions",
    "tool_validate_goal_schema",
    "tool_write_aux_exports",
    "tool_write_markdown_report",
}


class _RunnerTerminationSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(f"received signal {self.signum}")


def _runner_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_runner_history(state: Dict[str, Any], entry: Dict[str, Any]) -> None:
    history = state.get("runner_history")
    if not isinstance(history, list):
        history = []
        state["runner_history"] = history
    history.append(entry)
    if len(history) > 200:
        del history[:-200]


def _live_state_path(state_path: Path | str) -> Path:
    """Return the current state file, following goal_*_GOAL renames when needed."""
    return _state_path_to_obj(str(state_path))


def _mutate_runner_state(
    state_path: Path,
    snapshot_reason: str,
    mutator: Callable[[Dict[str, Any]], None],
) -> None:
    live_path = _live_state_path(state_path)
    try:
        state = _load_state(str(live_path))
    except Exception:
        return
    if not isinstance(state, dict):
        return
    try:
        mutator(state)
        state["updated_at"] = _runner_timestamp()
        tmp_path = live_path.with_name(f".{live_path.name}.{os.getpid()}.{snapshot_reason}.tmp")
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, live_path)
    except Exception:
        return


def _mark_deterministic_tool_started(state_path: Path, tool_name: str) -> None:
    now = _runner_timestamp()

    def _mutator(state: Dict[str, Any]) -> None:
        lease = {
            "status": "running",
            "tool": tool_name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "host": socket.gethostname(),
            "started_at": now,
            "heartbeat_at": now,
        }
        state["active_runner"] = lease
        _append_runner_history(state, {"event": "started", **lease})
        _event(state, "runner_active", f"Started deterministic tool {tool_name} with pid {os.getpid()}.")

    _mutate_runner_state(state_path, f"{tool_name}_started", _mutator)


def _mark_deterministic_tool_finished(state_path: Path, tool_name: str, result_status: str) -> None:
    now = _runner_timestamp()

    def _mutator(state: Dict[str, Any]) -> None:
        lease = state.get("active_runner") if isinstance(state.get("active_runner"), dict) else {}
        finished = {
            "event": "finished",
            "status": str(result_status or "unknown"),
            "tool": tool_name,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": lease.get("started_at") if isinstance(lease, dict) else None,
            "ended_at": now,
        }
        state["last_runner"] = finished
        state["active_runner"] = None
        _append_runner_history(state, finished)
        _event(state, "runner_idle", f"Finished deterministic tool {tool_name} with status {finished['status']}.")

    _mutate_runner_state(state_path, f"{tool_name}_finished", _mutator)


def _mark_deterministic_tool_failed(
    state_path: Path,
    tool_name: str,
    *,
    kind: str,
    message: str,
    status: str = "failed",
) -> None:
    now = _runner_timestamp()

    def _mutator(state: Dict[str, Any]) -> None:
        state["status"] = status
        state["phase"] = "controller_tool_failed"
        state["last_error"] = {
            "kind": kind,
            "tool": tool_name,
            "message": message,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "ts": now,
        }
        failed = {
            "event": "failed",
            "status": kind,
            "tool": tool_name,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "ended_at": now,
        }
        state["last_runner"] = failed
        state["active_runner"] = None
        _append_runner_history(state, failed)
        _event(state, "controller_tool_failed", f"Deterministic tool {tool_name} stopped: {message}", level="error")

    _mutate_runner_state(state_path, f"{tool_name}_{kind}", _mutator)


@contextmanager
def _termination_signal_guard() -> Iterator[None]:
    previous_handlers: Dict[int, Any] = {}

    def _handler(signum: int, _frame: Any) -> None:
        raise _RunnerTerminationSignal(signum)

    for signum in (getattr(signal, "SIGHUP", None), getattr(signal, "SIGTERM", None)):
        if not isinstance(signum, int):
            continue
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handler)
        except Exception:
            previous_handlers.pop(signum, None)
    try:
        yield
    finally:
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except Exception:
                continue


def _parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            decoder = json.JSONDecoder()
            idx = 0
            objects: list = []
            text = value.strip()
            while idx < len(text):
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                if idx >= len(text):
                    break
                try:
                    obj, end = decoder.raw_decode(text, idx)
                except Exception:
                    idx += 1
                    continue
                if isinstance(obj, dict):
                    objects.append(obj)
                idx = end
            if objects and isinstance(objects[-1], dict):
                recovered = dict(objects[-1])
                recovered.setdefault("raw_output", value)
                recovered.setdefault("json_recovered", True)
                return recovered
            return {"status": "fatal", "error": "controller_non_json_output", "raw_output": value}
        if isinstance(parsed, dict):
            return parsed
        return {"status": "fatal", "error": "controller_output_not_object", "raw_output": value}
    return {"status": "fatal", "error": f"controller_output_type_{type(value).__name__}"}


def _recover_state_path_from_history(
    responses: Any,
    *,
    fallback_state_path: Optional[str] = None,
) -> Optional[str]:
    if isinstance(fallback_state_path, str) and fallback_state_path.strip():
        return str(Path(fallback_state_path).expanduser())
    if not isinstance(responses, list):
        return None
    for item in reversed(responses):
        if not isinstance(item, dict):
            continue
        parsed = item.get("parsed")
        if not isinstance(parsed, dict):
            continue
        tool_calls = parsed.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in reversed(tool_calls):
            if not isinstance(call, dict):
                continue
            raw_args = call.get("arguments")
            args_obj: Dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args_obj = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {}
                if isinstance(parsed_args, dict):
                    args_obj = parsed_args
            candidate = args_obj.get("state_path")
            if isinstance(candidate, str) and candidate.strip():
                return str(Path(candidate).expanduser())
    return None


def _canonical_state_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(_state_path_to_obj(value.strip()))


def _authoritative_state_path(
    *,
    existing_state_path: Optional[str],
    controller_state_path: Any,
    responses: Any,
) -> Optional[str]:
    """Resolve state identity without trusting free-form controller output.

    A resumed run is bound to the state supplied by its caller. For a newly
    initialized run, prefer paths used by actual tool calls and only fall back
    to the controller's final response when no tool history is available.
    """
    existing = _canonical_state_path(existing_state_path)
    if existing and Path(existing).is_file():
        return existing

    recovered = _recover_state_path_from_history(responses)
    recovered = _canonical_state_path(recovered)
    reported = _canonical_state_path(controller_state_path)
    for candidate in (recovered, reported):
        if candidate and Path(candidate).is_file():
            return candidate
    return recovered or reported


def _infer_status_from_state(state_path: Path) -> Dict[str, Any]:
    live_path = _live_state_path(state_path)
    try:
        payload = json.loads(live_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "fatal", "error": f"state_read_failed: {exc}"}
    if not isinstance(payload, dict):
        return {"status": "fatal", "error": "state_not_object"}

    state_status = str(payload.get("status") or "").strip().lower()
    run_dir = payload.get("run_dir")
    response: Dict[str, Any] = {
        "state_path": str(live_path),
        "run_dir": run_dir,
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        paths = artifacts.get("paths")
        if isinstance(paths, dict):
            exists = artifacts.get("exists") if isinstance(artifacts.get("exists"), dict) else {}
            response["artifacts"] = {
                key: value
                for key, value in paths.items()
                if bool(exists.get(key))
            }
    planning = payload.get("planning")
    derived_planning: Dict[str, Any] = {}
    try:
        derived_planning = _derive_next_actions(payload)
    except Exception:
        derived_planning = {}
    if isinstance(derived_planning, dict) and derived_planning:
        planning = derived_planning
    if isinstance(planning, dict):
        response["planning"] = planning
    planning_terminal = (
        str(planning.get("terminal_status") or "").strip().lower()
        if isinstance(planning, dict)
        else ""
    )
    pending_questions = payload.get("pending_questions")
    if isinstance(pending_questions, list) and pending_questions:
        response["pending_questions"] = pending_questions
    state_phase = str(payload.get("phase") or "").strip().lower()
    response["state_status"] = state_status
    if state_phase:
        response["state_phase"] = state_phase
    replay = payload.get("replay")
    if isinstance(replay, dict):
        commands = replay.get("commands")
        if isinstance(commands, list):
            latest_logs: Dict[str, str] = {}
            for entry in commands:
                if not isinstance(entry, dict):
                    continue
                stage = entry.get("stage")
                log_path = entry.get("log_path")
                if isinstance(stage, str) and isinstance(log_path, str) and stage.strip() and log_path.strip():
                    latest_logs[stage] = log_path
            if latest_logs:
                response["logs"] = latest_logs
    if planning_terminal == "needs_user_input":
        response["status"] = "needs_user_input"
        return response
    if planning_terminal == "out_of_scope" or state_status == "out_of_scope":
        response["status"] = "out_of_scope"
        clarification = payload.get("clarification") if isinstance(payload.get("clarification"), dict) else {}
        reason = str(clarification.get("reason") or "").strip()
        if reason:
            response["reason"] = reason
        return response
    if planning_terminal == "fatal":
        response["status"] = "fatal"
        response["error"] = payload.get("last_error") or "controller_failed"
        if isinstance(payload.get("integrity"), dict):
            response["integrity"] = payload.get("integrity")
        return response
    if planning_terminal == "run_complete":
        response["status"] = "run_complete"
        if isinstance(payload.get("integrity"), dict):
            response["integrity"] = payload.get("integrity")
        return response
    if planning_terminal == "in_progress":
        response["status"] = "in_progress"
        return response
    if state_status == "completed":
        integrity = payload.get("integrity") if isinstance(payload.get("integrity"), dict) else None
        if integrity is not None:
            response["integrity"] = integrity
            if integrity.get("ok") is False:
                response["status"] = "fatal"
                response["error"] = {"kind": "integrity_check_failed", "blocking_errors": integrity.get("blocking_errors") or []}
                return response
        response["status"] = "run_complete"
        return response
    if state_status in {"intake_partial", "needs_user_input", "awaiting_user_input"}:
        response["status"] = "needs_user_input"
        return response
    if state_status in {"failed", "resolution_failed", "pipeline_failed", "intake_failed"}:
        response["status"] = "fatal"
        response["error"] = payload.get("last_error") or "controller_failed"
        if isinstance(payload.get("integrity"), dict):
            response["integrity"] = payload.get("integrity")
        return response
    if state_status in {"initialized", "intake_ready", "resolution_ready"}:
        response["status"] = "in_progress"
        return response
    response["status"] = "in_progress"
    return response


def _first_deterministic_action(planning: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    actions = planning.get("next_actions")
    if not isinstance(actions, list):
        return None
    allowed_raw = planning.get("allowed_next_tools")
    allowed = {
        str(item or "").strip()
        for item in allowed_raw
        if str(item or "").strip()
    } if isinstance(allowed_raw, list) else set()
    for item in actions:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "").strip()
        if not tool_name or tool_name not in _DETERMINISTIC_CONTROLLER_TOOL_NAMES:
            continue
        if allowed and tool_name not in allowed:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if tool_name == "restart_from_stage" and not arguments:
            continue
        return item
    return None


def _run_planned_next_action_once(state_path: Path) -> Optional[Dict[str, Any]]:
    state_path = _live_state_path(state_path)
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    planning = _derive_next_actions(payload)
    if not isinstance(planning, dict):
        return None
    terminal = str(planning.get("terminal_status") or "").strip().lower()
    if terminal and terminal != "in_progress":
        inferred = _infer_status_from_state(state_path)
        inferred["deterministic_controller"] = True
        return inferred
    action = _first_deterministic_action(planning)
    if not isinstance(action, dict):
        return None
    tool_name = str(action.get("tool") or "").strip()
    arguments = dict(action.get("arguments") if isinstance(action.get("arguments"), dict) else {})

    from . import pipeline_tools as tool_module

    tool = getattr(tool_module, tool_name, None)
    if not callable(tool):
        return None
    arguments.setdefault("state_path", str(state_path))
    _mark_deterministic_tool_started(state_path, tool_name)
    try:
        with _termination_signal_guard():
            tool_result = tool(**arguments)
    except KeyboardInterrupt:
        live_path = _live_state_path(state_path)
        _mark_deterministic_tool_failed(
            live_path,
            tool_name,
            kind="deterministic_tool_interrupted",
            message="controller interrupted by KeyboardInterrupt",
        )
        raise
    except _RunnerTerminationSignal as exc:
        live_path = _live_state_path(state_path)
        _mark_deterministic_tool_failed(
            live_path,
            tool_name,
            kind="deterministic_tool_interrupted",
            message=f"controller received signal {exc.signum}",
        )
        raise KeyboardInterrupt(f"controller received signal {exc.signum}") from exc
    except Exception as exc:
        live_path = _live_state_path(state_path)
        _mark_deterministic_tool_failed(
            live_path,
            tool_name,
            kind="deterministic_tool_exception",
            message=str(exc),
        )
        return {
            "status": "fatal",
            "state_path": str(live_path),
            "error": {
                "kind": "deterministic_tool_exception",
                "tool": tool_name,
                "message": str(exc),
            },
            "deterministic_controller": True,
            "deterministic_tool": tool_name,
        }
    if not isinstance(tool_result, dict):
        tool_result = {"status": "unknown", "raw_result": tool_result}
    tool_status = str(tool_result.get("status") or "").strip().lower()
    live_path = _live_state_path(str(tool_result.get("state_path") or state_path))
    _mark_deterministic_tool_finished(live_path, tool_name, tool_status or "unknown")
    result = _infer_status_from_state(live_path)
    if tool_status == "fatal_error":
        result["status"] = "fatal"
        result["error"] = tool_result
    result["deterministic_controller"] = True
    result["deterministic_tool"] = tool_name
    result["tool_result_status"] = tool_status or None
    result["state_path"] = str(live_path)
    if tool_result.get("run_dir"):
        result["run_dir"] = tool_result.get("run_dir")
    return result


def _write_decision_log(
    state_path: Path,
    controller_input: Dict[str, Any],
    responses: Any,
    final_output: Any,
) -> str:
    decision_log_path = state_path.parent / "decision_log.md"
    lines = [
        "# Agentic Decision Log",
        "",
        f"- controller_agent: {pipeline_controller_agent.name}",
        f"- state_path: {state_path}",
        "",
        "## Input",
        "```json",
        json.dumps(controller_input, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Tool Calls",
    ]
    response_list = responses if isinstance(responses, list) else []
    if not response_list:
        lines.append("- none")
    else:
        for idx, item in enumerate(response_list, start=1):
            parsed = item.get("parsed") if isinstance(item, dict) else None
            tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            lines.append(f"### Step {idx}")
            for call in tool_calls:
                name = call.get("name")
                arguments = call.get("arguments")
                if isinstance(arguments, str):
                    arg_text = arguments
                else:
                    arg_text = json.dumps(arguments, ensure_ascii=False)
                lines.append(f"- `{name}` args: `{arg_text}`")
    lines.extend(
        [
            "",
            "## Final Output",
            "```json",
            json.dumps(final_output, ensure_ascii=False, indent=2)
            if not isinstance(final_output, str)
            else final_output,
            "```",
            "",
        ]
    )
    decision_log_path.write_text("\n".join(lines), encoding="utf-8")
    return str(decision_log_path)


def run_agentic_pipeline(
    *,
    user_request: str,
    skip_router: bool = False,
    skip_ensembl_check: bool = False,
    router_max_steps: int = 25,
    router_log_tool_calls: bool = False,
    router_debug_dir: Optional[str] = None,
    router_print_summary: bool = False,
    existing_state_path: Optional[str] = None,
    clarification_answers: Optional[list[Dict[str, Any]]] = None,
    clarification_answer_summary: str = "",
    max_steps: int = 120,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    existing_state_path_resolved = _canonical_state_path(existing_state_path)

    # Deterministically apply externally provided clarification answers before
    # controller execution so user input is not re-serialized by the LLM.
    if (
        existing_state_path_resolved
        and isinstance(clarification_answers, list)
        and clarification_answers
    ):
        submit_result = submit_clarification_answers(
            state_path=existing_state_path_resolved,
            answers=clarification_answers,
            answer_summary=(clarification_answer_summary or "").strip(),
        )
        submit_status = str(submit_result.get("status") or "").strip().lower()
        if submit_status != "ok":
            return {
                "status": "fatal",
                "error": {
                    "kind": "clarification_submit_failed",
                    "details": submit_result,
                },
                "state_path": existing_state_path_resolved,
            }
        # This path is entered only with answers supplied by the embedding host.
        # Controller tool calls use submit_clarification_answers directly and
        # must not acquire user-evidence provenance by answering themselves.
        def _mark_external_answers(state: Dict[str, Any]) -> None:
            clarification = state.get("clarification")
            if isinstance(clarification, dict):
                clarification["answer_source"] = "external_user"
            history = state.get("clarification_history")
            if isinstance(history, list) and history and isinstance(history[-1], dict):
                history[-1]["answer_source"] = "external_user"

        _mutate_runner_state(
            Path(existing_state_path_resolved),
            "external_clarification_provenance",
            _mark_external_answers,
        )
        try:
            marked = _load_state(str(_live_state_path(existing_state_path_resolved)))
        except Exception:
            marked = {}
        if (marked.get("clarification") or {}).get("answer_source") != "external_user":
            return {
                "status": "fatal",
                "error": {"kind": "external_clarification_provenance_not_recorded"},
                "state_path": existing_state_path_resolved,
            }
        # Prevent the controller from re-submitting the same answers.
        clarification_answers = None
        if not clarification_answer_summary:
            clarification_answer_summary = "Clarification answers were pre-applied by runner."

    controller_input = {
        "request": user_request,
        "flags": {
            "skip_router": bool(skip_router),
            "skip_ensembl_check": bool(skip_ensembl_check),
            "router_max_steps": int(max(1, router_max_steps)),
            "router_log_tool_calls": bool(router_log_tool_calls),
            "router_debug_dir": (router_debug_dir or "").strip(),
            "router_print_summary": bool(router_print_summary),
        },
    }
    if existing_state_path_resolved:
        controller_input["existing_state_path"] = existing_state_path_resolved
    if isinstance(clarification_answers, list) and clarification_answers:
        controller_input["clarification_answers"] = clarification_answers
    if clarification_answer_summary:
        controller_input["clarification_answer_summary"] = clarification_answer_summary

    if existing_state_path_resolved:
        deterministic = _run_planned_next_action_once(Path(existing_state_path_resolved))
        if isinstance(deterministic, dict):
            deterministic.setdefault("controller_agent", "deterministic_planned_action")
            deterministic.setdefault("controller_steps", 0)
            return deterministic

    run_result = Runner._run_sync(
        pipeline_controller_agent,
        input=json.dumps(controller_input, ensure_ascii=False),
        max_steps=max_steps,
        event_callback=event_callback,
    )
    parsed = _parse_json_object(run_result.final_output)
    parsed.setdefault("controller_agent", pipeline_controller_agent.name)
    parsed.setdefault("controller_steps", len(run_result.responses or []))

    controller_state_path = parsed.get("state_path")
    state_path = _authoritative_state_path(
        existing_state_path=existing_state_path_resolved,
        controller_state_path=controller_state_path,
        responses=run_result.responses or [],
    )
    if state_path:
        reported_canonical = _canonical_state_path(controller_state_path)
        if reported_canonical and reported_canonical != state_path:
            parsed["controller_reported_state_path"] = reported_canonical
        parsed["state_path"] = state_path
    if isinstance(state_path, str) and state_path.strip():
        try:
            resolved_state = _live_state_path(state_path)
            trace_path = resolved_state.parent / "agent_trace.json"
            trace_payload = {
                "trace_format": "compact_v1",
                "controller_input": controller_input,
                "controller_agent": pipeline_controller_agent.name,
                "controller_steps": len(run_result.responses or []),
                "responses": run_result.responses or [],
                "final_output": run_result.final_output,
            }
            trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            parsed["agent_trace_path"] = str(trace_path)
            decision_log_path = _write_decision_log(
                resolved_state,
                controller_input,
                run_result.responses or [],
                run_result.final_output,
            )
            parsed["decision_log_path"] = decision_log_path
            parsed["controller_status_raw"] = parsed.get("status")
            parsed.update(_infer_status_from_state(resolved_state))
        except Exception as exc:
            parsed["agent_trace_error"] = str(exc)
    else:
        status_text = str(parsed.get("status") or "").strip().lower()
        if status_text not in {"run_complete", "needs_user_input", "fatal"}:
            parsed["status"] = "fatal"
            parsed.setdefault("error", "controller_missing_state_path")
    return parsed
