#!/usr/bin/env python3
"""Regenerate explainer, markdown, and HTML report artifacts for an existing run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aso_agents.llm_provider import (
    provider_credential_help,
    provider_credentials_available,
    selected_provider_name,
)


_NO_LLM_CREDENTIAL = "REPORT_REGEN_REUSE_ONLY"
_PROVIDER_CREDENTIAL_NAMES = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _prepare_provider_credentials(
    *,
    no_llm: bool,
    env: MutableMapping[str, str] | None = None,
) -> None:
    target_env = os.environ if env is None else env
    if no_llm:
        credential_name = _PROVIDER_CREDENTIAL_NAMES[selected_provider_name(target_env)]
        if not str(target_env.get(credential_name) or "").strip():
            target_env[credential_name] = _NO_LLM_CREDENTIAL
        return
    if not provider_credentials_available(target_env):
        raise SystemExit(
            f"{provider_credential_help(target_env)} A selected-provider credential is required "
            "to regenerate the explainer. Use --no-llm only if a valid "
            "report_explainer.json already exists."
        )


def _load_state_run_dir(state_path: Path) -> str:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"State file is not a JSON object: {state_path}")
    run_dir = payload.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise SystemExit(f"State file is missing run_dir: {state_path}")
    return run_dir.strip()


def _resolve_run_dir(run_arg: str | None, state_arg: str | None) -> Path:
    if run_arg and state_arg:
        raise SystemExit("Provide either a run directory or --state-path, not both.")
    if state_arg:
        state_path = Path(state_arg).expanduser().resolve()
        if not state_path.exists():
            raise SystemExit(f"State path not found: {state_path}")
        return Path(_load_state_run_dir(state_path)).expanduser().resolve()
    if not run_arg:
        raise SystemExit("Provide a run directory or --state-path.")
    run_dir = Path(run_arg).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate report_explainer.json, report.md, and report.html for an existing run."
    )
    parser.add_argument("run_dir", nargs="?", help="Existing run directory under runs/")
    parser.add_argument(
        "--state-path",
        dest="state_path",
        default=None,
        help="Existing agentic state path (run_state.json). The run_dir will be derived from it.",
    )
    parser.add_argument(
        "--no-llm",
        dest="no_llm",
        action="store_true",
        help="Skip the LLM call and reuse the existing report_explainer.json for markdown/HTML regeneration.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.state_path)
    _prepare_provider_credentials(no_llm=args.no_llm)

    from aso_agents.run_layout import resolve_run_path
    from aso_agents.pipeline_tools import (
        _build_report_payload_manifest,
        _generate_report_explainer,
        _load_goal_context_from_run_dir,
        _save_goal_payload,
        _write_json,
    )
    from aso_agents.report_explainer import (
        build_report_explainer_context,
        normalize_report_explainer,
        report_explainer_is_valid,
        render_report_markdown,
    )
    from run_goal_intake import _generate_goal_intake_html_report

    context = _load_goal_context_from_run_dir(str(run_dir))
    payload: Dict[str, Any] = context["payload"]
    final_payload: Dict[str, Any] = context["final_payload"]
    structured_goal: Dict[str, Any] = context["structured_goal"]

    report_payload = _build_report_payload_manifest(
        structured_goal=structured_goal,
        final_payload=final_payload,
    )
    final_payload["report_payload"] = report_payload
    explanation_context = build_report_explainer_context(
        run_dir=str(run_dir),
        payload=payload,
        structured_goal=structured_goal,
        final_payload=final_payload,
        report_payload=report_payload,
    )
    report_explainer_path = resolve_run_path(run_dir, "report_explainer.json")

    if args.no_llm:
        if not report_explainer_path.exists():
            raise SystemExit(
                "No existing report_explainer.json found. Remove --no-llm to regenerate it. "
                f"{provider_credential_help()}"
            )
        existing_raw = json.loads(report_explainer_path.read_text(encoding="utf-8"))
        existing_explainer = normalize_report_explainer(existing_raw, explanation_context)
        if not report_explainer_is_valid(existing_explainer):
            raise SystemExit(
                "Existing report_explainer.json is missing or invalid. Regenerate with the LLM instead of using --no-llm."
            )
        explainer_result = {
            "explainer": existing_explainer,
            "generation_error": "",
            "source": "existing_report_explainer",
        }
    else:
        explainer_result = _generate_report_explainer(
            run_dir=str(run_dir),
            payload=payload,
            structured_goal=structured_goal,
            final_payload=final_payload,
            report_payload=report_payload,
        )
    explainer = explainer_result["explainer"]
    if not report_explainer_is_valid(explainer):
        raise SystemExit(
            f"Explainer regeneration failed: {explainer_result.get('generation_error') or 'report_explainer_invalid_or_incomplete'}"
        )
    report_md_path = run_dir / "report.md"

    _write_json(report_explainer_path, explainer)
    report_md_path.write_text(render_report_markdown(explainer, report_payload), encoding="utf-8")

    final_payload["report_explainer_path"] = str(report_explainer_path)
    final_payload["report_md_path"] = str(report_md_path)
    structured_goal["report_explainer_path"] = str(report_explainer_path)
    structured_goal["report_md_path"] = str(report_md_path)
    _save_goal_payload(str(run_dir), payload, structured_goal, final_payload)

    html_path = _generate_goal_intake_html_report(str(run_dir), structured_goal)
    if not html_path:
        raise SystemExit("HTML regeneration failed.")

    print(f"Run directory: {run_dir}")
    print(f"Explainer: {report_explainer_path}")
    print(f"Markdown: {report_md_path}")
    print(f"HTML: {html_path}")
    if args.no_llm:
        print("Explainer mode: reused existing report_explainer.json (--no-llm).")


if __name__ == "__main__":
    main()
