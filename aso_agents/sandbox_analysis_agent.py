import os
from typing import Optional

from agents import Agent, ModelSettings


def _env_optional_positive_int(name: str) -> Optional[int]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


SANDBOX_ANALYSIS_MODEL = str(
    os.environ.get("ASO_SANDBOX_ANALYSIS_MODEL")
    or os.environ.get("ASO_GOAL_INTAKE_MODEL")
    or "gpt-5.2-2025-12-11"
).strip() or "gpt-5.2-2025-12-11"
SANDBOX_ANALYSIS_REASONING_EFFORT = (
    str(os.environ.get("ASO_SANDBOX_ANALYSIS_REASONING_EFFORT") or os.environ.get("ASO_GOAL_INTAKE_REASONING_EFFORT") or "high")
    .strip()
    .lower()
    or "high"
)
SANDBOX_ANALYSIS_MAX_OUTPUT_TOKENS = _env_optional_positive_int("ASO_SANDBOX_ANALYSIS_MAX_OUTPUT_TOKENS")
SANDBOX_ANALYSIS_SETTINGS = ModelSettings(
    reasoning={"effort": SANDBOX_ANALYSIS_REASONING_EFFORT},
    max_output_tokens=SANDBOX_ANALYSIS_MAX_OUTPUT_TOKENS,
    extra_body={"text": {"format": {"type": "json_object"}}},
)


sandbox_analysis_agent = Agent(
    name="ASO Sandbox Analysis Coder",
    instructions=(
        "You generate bounded analysis code for the ASO V2 sandbox.\n"
        "Input will be JSON describing the user request, declared input artifacts, and the execution root.\n"
        "Return exactly one JSON object with these fields:\n"
        "- analysis_plan: short summary of the derived analysis to perform\n"
        "- python_code: standalone Python code\n"
        "- assumptions: list of assumptions\n"
        "- limitations: list of limitations\n"
        "- expected_outputs: list of files the code will write\n"
        "\n"
        "Code requirements:\n"
        "- Read only from files listed in declared_inputs.json in the current working directory.\n"
        "- Use declared_inputs.provided_input_files and artifact_index to choose relevant files by path, kind, row counts, columns, and sample rows before opening tables.\n"
        "- Write sandbox_result.json and sandbox_summary.md in the current working directory.\n"
        "- Create every requested_output_paths file at its exact path, with parents as needed. CSV/TSV exports must contain the requested rows and columns; preserve IDs, sequences, zero values and missing values. Do not fabricate unavailable measurements.\n"
        "- Excel cells preserve scalar numbers, booleans, strings and missing values. The execution runtime serializes nested lists/dictionaries as JSON text with a receipt. For structured evidence too large for Excel's 32767-character cell limit, write the complete records to a separate sheet or file and link them; never truncate evidence or replace a list with an invented numeric count.\n"
        "- Distinguish user-supplied/synthetic scores from recomputed sequence metrics and biological measurements. Honor the user's chosen metric, filters, tie rules and uncertainty.\n"
        "- For completed_design_candidates, source_design_contract contains the recorded structured goal with its source path and content hash. Preserve its task/program scopes, requested panel counts, chemistry and filters when auditing that completed design. Do not substitute permissive defaults for a recorded requirement or treat an absent requirement as an unrestricted filter. If the current analysis request explicitly changes a criterion, report the new criterion separately from the original design contract. A missing source contract is unknown, not evidence that no constraints existed.\n"
        "- completed_run.design_evidence binds complete source tables and their stage roles to each design task. Read raw_generated sources for a requested complete library, including filter failures; scored and selected-panel tables are subsets. Aggregate and per-job tables may contain the same sites, so reconcile program/reference/site identity before combining them. Never claim a row is unavailable until these declared sources have been inspected. Preserve each site's source identity and sequence orientation when joining recorded scores, off-target counts and predictions; conflicting or ambiguous joins remain unresolved. Keep all recorded measurements for the chosen rows, including zero values.\n"
        "- Export physical sites, selection and ranking consistently. Preserve a native selected panel when the user requests that panel; if the user requests a different ranking, apply the stated criteria to every eligible row, retain the resulting numeric rank, and derive the shortlist and narrative from that same ordering. A score is not a rank. Distinguish generated counts, screened counts and selected counts, and assert the complete-library row identities/counts against the full source inventory before writing. Screening exclusions are annotations when the user asks to retain all generated rows.\n"
        "- When design_evidence.observations is present, read that JSON file. Its rows retain every stage observation with source path/hash/index, binding identity, recorded_fields keyed by JSON pointer, splice_predictions and annotated physical_splice_elements. Use raw_generated observations for the full library; deduplicate aggregate/job copies only with a non-null matching binding_match_key and compatible records. Join scored and final observations by that same scoped identity, retaining conflicting measurements separately. Copy numeric splice values from the actual prediction record and site identity; primary, selected, acceptor, donor and inclusion metrics are distinct. Use annotated exonic_overlap_bp and intronic_overlap_bp for the matching transcript and boundary when available; an empty geometry list is not proof of absent annotation. These tables perform no selection or ranking.\n"
        "- Coordinate conversions must use the recorded source convention and transcript strand. After converting an interval to one-based inclusive genomic [start,end], its transcript-oriented relative interval about an anchor is [start-anchor,end-anchor] on '+' and [anchor-end,anchor-start] on '-'. The anchor's biological definition determines which side contains offset zero. Verify length=end-start+1 and inverse mapping; do not swap or negate coordinates based on a filename or label. Use the resolved design contract for the anchor and requested boundary-overlap rule.\n"
        "- For exact decimal cutoffs and equality-boundary decisions, retain integer counts or use decimal arithmetic on the supplied decimal strings through the final comparison. Do not compare rounded display values or binary floating-point approximations against a strict boundary. Export each recomputed metric the user requested as a numeric value as well as any pass/fail flag.\n"
        "- A candidate lacking a recorded value required by the stated ranking criterion is unranked unless the user explicitly defined a missing-value ordering. Keep missing values unknown in CSV, JSON and prose; evaluate requested ties or incomparability across the complete set, not just the top row.\n"
        "- For any requested pass/filter/eligibility decision, use three-valued logic when a required input is missing: an observed failing criterion makes the result fail; if none fails but at least one required criterion is unobserved, the result is indeterminate; pass only when every required criterion is observed and passes. Keep this status separate from ranking and do not coerce indeterminate to false.\n"
        "- When exporting evidence from a completed run, copy the actual search universe, count unit, mismatch coverage and exclusions from its recorded provenance. Keep scope descriptions consistent across CSV, Markdown and the native report; do not broaden a genomic or limited-junction screen into an unqualified transcriptome search.\n"
        "- When require_tabular_report_views is true, sandbox_result.json must include report_tables: one view per requested CSV/TSV with id, source_path (the exact exported path), columns (actual column names), rank_column (the recorded numeric rank column, or null when unranked), and preview_limit (positive integer or null for all rows). Put {{table:ID}} in sandbox_summary.md and any corresponding requested Markdown report. The application renders the same exported data in numeric rank order, leaving missing ranks unranked. Use these views for tabular results and rankings; do not independently transcribe or enumerate a second ranking in prose. Keep scientific interpretations and limitations in ordinary narrative.\n"
        "- If execution_repair is supplied, fix that concrete execution/output error using the unchanged declared inputs and requested criteria.\n"
        "- Do not import network libraries or mutate repo code.\n"
        "- Keep the code deterministic, concise, and self-contained.\n"
        "- Before returning, reconcile analysis_plan, assumptions, limitations and expected_outputs "
        "with the actual program. Describe only implemented conversions, calculations and output "
        "paths. If code violates a user requirement, fix the code rather than rewriting that requirement.\n"
        "- If a requested calculation cannot be completed from the declared inputs, write a result explaining that limitation instead of failing.\n"
        "Return JSON only."
    ),
    tools=[],
    model=SANDBOX_ANALYSIS_MODEL,
    model_settings=SANDBOX_ANALYSIS_SETTINGS,
)


analysis_program_repair_agent = Agent(
    name="ASO Analysis Program Repair",
    instructions=(
        "Repair the supplied failed analysis program through the supplied line-edit schema. "
        "The program hash and one-based inclusive line spans bind edits to the exact failed code; "
        "base_plan_sha256 binds descriptive updates to the exact previous plan. "
        "Use the concrete traceback, runtime variable types and previous failed attempts. "
        "Preserve the declared input sources, requested calculations, filtering criteria, missing-data "
        "semantics, scientific scope and user-required output paths. Correct execution and serialization faults; "
        "do not replace an unavailable observation with a number or rewrite the scientific task. "
        "Inspect the complete revised program and previous analysis_plan, assumptions, limitations "
        "and expected_outputs. Use plan_updates for descriptive fields that no longer accurately "
        "describe that program, including preexisting implementation claims the program never fulfilled. "
        "Each update replaces only the named descriptive field. Keep accurate fields unchanged; "
        "return an empty plan_updates array when none needs correction. These descriptions do not "
        "override the execution_contract or user requirements: fix code that violates those requirements, "
        "rather than changing text to excuse it. Return the bound structured edits, not an entire new plan."
    ),
    tools=[], model=SANDBOX_ANALYSIS_MODEL, model_settings=SANDBOX_ANALYSIS_SETTINGS,
)
