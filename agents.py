from __future__ import annotations

import copy
import json
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, get_origin, get_args

if TYPE_CHECKING:
    from aso_agents.llm_provider import ProviderConfig


class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Callable]] = None,
        model: str = "gpt-5-mini-2025-08-07",
        log_tool_calls: bool = False,
        model_settings: Optional["ModelSettings"] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.log_tool_calls = log_tool_calls
        self.model_settings = model_settings


def _pytype_to_jsonschema(py_t) -> Dict[str, Any]:
    origin = get_origin(py_t)
    args = get_args(py_t)
    if origin is list or origin is List:
        item_type = args[0] if args else str
        return {"type": "array", "items": _pytype_to_jsonschema(item_type)}
    if py_t in (str, Optional[str]):
        return {"type": "string"}
    if py_t in (int, Optional[int]):
        return {"type": "integer"}
    if py_t in (float, Optional[float]):
        return {"type": "number"}
    if py_t in (bool, Optional[bool]):
        return {"type": "boolean"}
    # Fallback
    return {"type": "string"}


def _build_function_schema(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        ann = param.annotation if param.annotation is not inspect._empty else str
        properties[name] = _pytype_to_jsonschema(ann)
        if param.default is inspect._empty:
            required.append(name)
    doc = (func.__doc__ or "").strip()
    name = getattr(func, "_tool_name", func.__name__)
    schema: Dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }
    if doc:
        schema["description"] = doc[:800]
    return schema


def function_tool(func: Callable) -> Callable:
    func._is_tool = True  # type: ignore[attr-defined]
    func._tool_name = func.__name__  # type: ignore[attr-defined]
    return func


class _RunResult:
    def __init__(self, final_output: Any, responses: Optional[List[Dict[str, Any]]] = None):
        self.final_output = final_output
        self.responses = responses or []


def _serialize_model_setting(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_model_setting(item) for item in value]
    for attr in ("model_dump", "to_dict", "dict"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:
                continue
    if hasattr(value, "__dict__"):
        try:
            return {
                key: _serialize_model_setting(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass
    return value


def _response_output_items(data: Dict[str, Any]) -> List[Any]:
    outputs = data.get("output")
    if outputs is None:
        nested_response = data.get("response")
        if isinstance(nested_response, dict):
            outputs = nested_response.get("output")
    return outputs if isinstance(outputs, list) else []


def _response_output_types(data: Dict[str, Any]) -> List[str]:
    outputs = _response_output_items(data)
    types: List[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "unknown").strip() or "unknown"
        if item_type == "message":
            content = item.get("content") or []
            segment_types: List[str] = []
            for segment in content:
                if not isinstance(segment, dict):
                    continue
                seg_type = str(segment.get("type") or "").strip()
                if seg_type:
                    segment_types.append(seg_type)
            if segment_types:
                types.append(f"message[{','.join(segment_types[:4])}]")
            else:
                types.append("message")
        else:
            types.append(item_type)
    return types[:24]


def _compact_history_entry(
    data: Dict[str, Any],
    parsed: Dict[str, Any],
    attribution: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    parsed_out = dict(parsed) if isinstance(parsed, dict) else {}
    content = str(parsed_out.get("content") or "").strip()
    if content and len(content) > 2000:
        parsed_out["content"] = content[:2000] + "... <truncated>"
    response_meta: Dict[str, Any] = {
        "id": data.get("id"),
        "created_at": data.get("created_at"),
        "status": data.get("status"),
        "model": data.get("model"),
        "parallel_tool_calls": data.get("parallel_tool_calls"),
        "tool_choice": data.get("tool_choice"),
        "output_types": _response_output_types(data),
    }
    if attribution:
        response_meta.update(
            {
                "provider": attribution.get("provider"),
                "requested_model": attribution.get("requested_model"),
                "resolved_model": attribution.get("resolved_model"),
            }
        )
    tools = data.get("tools")
    if isinstance(tools, list):
        response_meta["tool_count"] = len(tools)
    incomplete_details = data.get("incomplete_details")
    if incomplete_details:
        response_meta["incomplete_details"] = incomplete_details
    return {
        "response": response_meta,
        "parsed": parsed_out,
    }


@dataclass
class ModelSettings:
    reasoning: Optional[Any] = None
    verbosity: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = None

    def apply(self, payload: Dict[str, Any]) -> None:
        if self.reasoning is not None:
            payload["reasoning"] = _serialize_model_setting(self.reasoning)
        if self.verbosity is not None:
            payload["verbosity"] = self.verbosity
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.extra_body:
            for key, value in self.extra_body.items():
                payload[key] = _serialize_model_setting(value)


class Runner:
    @staticmethod
    def _emit_event(event_callback: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
        if not callable(event_callback):
            return
        try:
            event_callback(payload)
        except Exception:
            pass

    @staticmethod
    def _responses(
        payload: Dict[str, Any],
        config: Optional[ProviderConfig] = None,
    ) -> Dict[str, Any]:
        from aso_agents.llm_provider import create_response

        return create_response(payload, config)

    @staticmethod
    def _function_call_output_items(
        tool_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        input_items: List[Dict[str, Any]] = []
        for idx, output in enumerate(tool_outputs, start=1):
            call_id = output.get("call_id") or output.get("tool_call_id") or output.get("id")
            if not call_id:
                raise RuntimeError(f"Missing tool call id for tool output #{idx}")
            tool_output = output.get("output", "")
            if not isinstance(tool_output, str):
                try:
                    tool_output = json.dumps(tool_output, ensure_ascii=False)
                except Exception:
                    tool_output = str(tool_output)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(call_id),
                    "output": tool_output,
                }
            )
        return input_items

    @staticmethod
    def _submit_tool_outputs(
        response_id: str,
        tool_outputs: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        model_settings: Optional["ModelSettings"] = None,
        config: Optional[ProviderConfig] = None,
    ) -> Dict[str, Any]:
        from aso_agents.llm_provider import get_provider_config

        resolved_config = get_provider_config() if config is None else config
        if resolved_config.continuation_mode != "previous_response_id":
            raise RuntimeError(
                "Stateless continuation requires the complete accumulated response input."
            )
        if not isinstance(response_id, str) or not response_id.strip():
            raise RuntimeError("Missing response id for stateful tool continuation")
        payload: Dict[str, Any] = {
            "previous_response_id": response_id,
            "input": Runner._function_call_output_items(tool_outputs),
        }
        if model:
            payload["model"] = model
        if tools:
            payload["tools"] = tools
        if model_settings:
            model_settings.apply(payload)
        return Runner._responses(payload, config=resolved_config)

    @staticmethod
    def _message_to_response_format(message: Dict[str, Any]) -> Dict[str, Any]:
        role = message.get("role", "user")
        content_segments: List[Dict[str, Any]] = []

        def _append_text(text_value: Any, *, role: str) -> None:
            if text_value is None:
                return
            if isinstance(text_value, str):
                trimmed = text_value.strip()
                if trimmed:
                    segment_type = "input_text" if role in {"system", "user"} else "output_text"
                    content_segments.append({"type": segment_type, "text": text_value})
            elif isinstance(text_value, Sequence):
                for element in text_value:
                    _append_text(element, role=role)

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_output = message.get("content")
            if tool_output is None:
                tool_output = ""
            if not isinstance(tool_output, str):
                try:
                    tool_output = json.dumps(tool_output, ensure_ascii=False)
                except Exception:
                    tool_output = str(tool_output)
            segment: Dict[str, Any] = {
                "type": "tool_result",
                "tool_call_id": tool_call_id,
                "output": tool_output,
            }
            if message.get("is_error"):
                segment["is_error"] = True
            return {"role": "tool", "tool_call_id": tool_call_id, "content": [segment]}

        _append_text(message.get("content"), role=role)

        if not content_segments:
            default_type = "input_text" if role in {"system", "user"} else "output_text"
            content_segments.append({"type": default_type, "text": ""})

        return {"role": role, "content": content_segments}

    @staticmethod
    def _messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [Runner._message_to_response_format(msg) for msg in messages]

    @staticmethod
    def _parse_response_output(data: Dict[str, Any]) -> Dict[str, Any]:
        outputs = _response_output_items(data)
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                contents = item.get("content") or []
                for segment in contents:
                    if not isinstance(segment, dict):
                        continue
                    seg_type = segment.get("type")
                    if seg_type in {"output_text", "input_text", "text"}:
                        text_parts.append(segment.get("text", ""))
            elif item_type in {"function_call", "tool_call"}:
                call_id = item.get("call_id")
                item_id = item.get("id")
                name = item.get("name")
                arguments = item.get("arguments")
                tool_calls.append({
                    "id": item_id or call_id or f"tool-call-{abs(hash(str(arguments))) % 10_000_000}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                })
            elif item_type in {"output_text", "text"}:
                text_parts.append(item.get("text", ""))
        content = "".join(text_parts).strip() if text_parts else ""
        return {"content": content, "tool_calls": tool_calls}

    @staticmethod
    def _run_sync(
        agent: Agent,
        input: str,
        temperature: float = 0.2,
        max_steps: int = 5,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> _RunResult:
        from aso_agents.llm_provider import get_provider_config

        tool_funcs: Dict[str, Callable] = {}
        tools_spec: List[Dict[str, Any]] = []
        for t in agent.tools:
            name = getattr(t, "_tool_name", None)
            if not callable(t) or not getattr(t, "_is_tool", False) or not isinstance(name, str) or not name.strip():
                raise ValueError(f"unregistered_declared_tool:{getattr(t, '__name__', type(t).__name__)}")
            if name in tool_funcs:
                raise ValueError(f"duplicate_declared_tool:{name}")
            tool_funcs[name] = t
            tools_spec.append(_build_function_schema(t))

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": agent.instructions},
            {"role": "user", "content": input},
        ]
        config = get_provider_config()
        requested_model = agent.model.strip()
        attribution = {
            "provider": config.name,
            "requested_model": requested_model,
            "resolved_model": config.resolve_model(requested_model),
        }
        initial_input = Runner._messages_to_responses_input(messages)
        conversation_input = copy.deepcopy(initial_input)
        payload: Dict[str, Any] = {
            "model": agent.model,
            "input": initial_input,
        }
        if tools_spec:
            payload["tools"] = tools_spec
        if getattr(agent, "model_settings", None):
            agent.model_settings.apply(payload)

        response_history: List[Dict[str, Any]] = []
        data = Runner._responses(payload, config=config)

        # Each permitted continuation is already paid for. Inspect its response
        # even at the limit, without executing another tool batch or nudge.
        for step in range(max_steps + 1):
            if config.continuation_mode == "stateless":
                conversation_input.extend(copy.deepcopy(_response_output_items(data)))
            parsed = Runner._parse_response_output(data)
            msg_content = parsed.get("content", "")
            tool_calls = parsed.get("tool_calls") or []
            response_history.append(_compact_history_entry(data, parsed, attribution))

            if tool_calls:
                if step == max_steps:
                    return _RunResult(
                        final_output={"error": "Max tool-calling steps exceeded"},
                        responses=response_history[:],
                    )
                tool_outputs: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    call_id = tc.get("call_id")
                    if not call_id:
                        raise RuntimeError("Missing tool call id in model response")
                    name = tc.get("name")
                    arg_str = tc.get("arguments", "{}")
                    try:
                        args = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
                    except Exception:
                        args = {}
                    Runner._emit_event(
                        event_callback,
                        {
                            "type": "tool_call",
                            "agent": agent.name,
                            "tool": name,
                            "arguments": args,
                        },
                    )
                    if getattr(agent, "log_tool_calls", True):
                        try:
                            print(f"[TOOL CALL] agent={agent.name} tool={name} args={json.dumps(args, ensure_ascii=False)}")
                        except Exception:
                            print(f"[TOOL CALL] agent={agent.name} tool={name} args=<unserializable>")
                    func = tool_funcs.get(name)
                    if not func:
                        tool_result = {"error": f"Unknown tool: {name}"}
                    else:
                        try:
                            tool_result = func(**args)
                        except Exception as e:
                            tool_result = {"error": f"Tool '{name}' failed: {e}"}
                    Runner._emit_event(
                        event_callback,
                        {
                            "type": "tool_result",
                            "agent": agent.name,
                            "tool": name,
                            "arguments": args,
                            "result": tool_result,
                        },
                    )
                    # Evaluation telemetry is deliberately best-effort and
                    # inactive during ordinary product runs.
                    try:
                        from aso_agents.telemetry import record_tool_execution

                        record_tool_execution(
                            agent=agent.name,
                            tool=name,
                            call_id=call_id,
                            arguments=args,
                            result=tool_result,
                        )
                    except Exception:
                        pass
                    if getattr(agent, "log_tool_calls", True):
                        try:
                            result_str = json.dumps(tool_result, ensure_ascii=False)
                            if len(result_str) > 2000:
                                result_str = result_str[:2000] + "... <truncated>"
                            print(f"[TOOL RESULT] agent={agent.name} tool={name} result={result_str}")
                        except Exception:
                            print(f"[TOOL RESULT] agent={agent.name} tool={name} result=<unserializable>")
                    output_payload: Any
                    if isinstance(tool_result, str):
                        output_payload = tool_result
                    else:
                        try:
                            output_payload = json.dumps(tool_result, ensure_ascii=False)
                        except Exception:
                            output_payload = str(tool_result)
                    tool_outputs.append({
                        "call_id": call_id,
                        "output": output_payload,
                    })
                function_output_items = Runner._function_call_output_items(tool_outputs)
                followup_payload: Dict[str, Any] = {"model": agent.model}
                if config.continuation_mode == "previous_response_id":
                    response_id = data.get("id")
                    if not isinstance(response_id, str) or not response_id.strip():
                        return _RunResult(final_output={
                            "error": "missing_response_id",
                            "details": data,
                        }, responses=response_history[:])
                    followup_payload["previous_response_id"] = response_id
                    followup_payload["input"] = function_output_items
                else:
                    conversation_input.extend(copy.deepcopy(function_output_items))
                    followup_payload["input"] = copy.deepcopy(conversation_input)
                if tools_spec:
                    followup_payload["tools"] = tools_spec
                if getattr(agent, "model_settings", None):
                    agent.model_settings.apply(followup_payload)
                data = Runner._responses(followup_payload, config=config)
                continue

            status = data.get("status")
            if status == "incomplete":
                rejection = data.get("structured_output_rejection")
                if (
                    (data.get("incomplete_details") or {}).get("reason") == "invalid_structured_output"
                    and isinstance(rejection, dict)
                ):
                    return _RunResult(final_output={
                        "error": "invalid_structured_output",
                        "reason": (
                            f"Array {rejection.get('property')!r} violated "
                            f"{rejection.get('keyword')} at item {rejection.get('item_index')}. "
                            "Generation stopped; emit a complete JSON object with only "
                            "distinct necessary entries within its schema limits."
                        ),
                        "structured_output_rejection": rejection,
                    }, responses=response_history[:])
                return _RunResult(final_output={
                    "error": "incomplete_response",
                    "status": status,
                    "reason": (data.get("incomplete_details") or {}).get("reason"),
                }, responses=response_history[:])

            if msg_content:
                Runner._emit_event(
                    event_callback,
                    {
                        "type": "final_output",
                        "agent": agent.name,
                        "content": msg_content,
                    },
                )
                return _RunResult(final_output=msg_content, responses=response_history[:])

            # Some model turns can be empty (no text, no tool calls). Nudge once and continue.
            if step == max_steps:
                break
            response_id = data.get("id")
            nudge_items = Runner._messages_to_responses_input(
                [
                    {
                        "role": "user",
                        "content": "Continue. Either call the next tool or return your final JSON answer.",
                    }
                ]
            )
            followup_payload = {"model": agent.model}
            if config.continuation_mode == "stateless":
                conversation_input.extend(copy.deepcopy(nudge_items))
                followup_payload["input"] = copy.deepcopy(conversation_input)
            elif isinstance(response_id, str) and response_id.strip():
                followup_payload["previous_response_id"] = response_id
                followup_payload["input"] = nudge_items
            else:
                return _RunResult(
                    final_output={
                        "error": "empty_response_no_tool_calls",
                        "status": status,
                    },
                    responses=response_history[:],
                )
            if tools_spec:
                followup_payload["tools"] = tools_spec
            if getattr(agent, "model_settings", None):
                agent.model_settings.apply(followup_payload)
            data = Runner._responses(followup_payload, config=config)
            continue

        return _RunResult(final_output={"error": "Max tool-calling steps exceeded"}, responses=response_history[:])

    @staticmethod
    async def run(
        agent: Agent,
        input: str,
        temperature: float = 0.2,
        max_steps: int = 5,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> _RunResult:
        # Simple async wrapper around the synchronous implementation (temperature ignored for API compatibility)
        return Runner._run_sync(
            agent,
            input,
            temperature=1.0,
            max_steps=max_steps,
            event_callback=event_callback,
        )
