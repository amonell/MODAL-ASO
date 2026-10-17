"""Versioned contracts shared by the live evaluation runner and graders."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .clarification_provenance import (
    INDEPENDENT_BIOLOGICAL_SOURCE,
    VALID_CLARIFICATION_SOURCE_CLASSES,
    clarification_rule_admission,
    classify_clarification_evidence_refs,
)

EVAL_CASE_SCHEMA_VERSION = 1
EVAL_RUN_SCHEMA_VERSION = 1
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
STAGE_ORDER = (
    "request_graph",
    "structured_goal",
    "resolution",
    "targeting",
    "routing",
    "candidates",
    "e2e",
)
VALID_TIERS = frozenset({"pr", "nightly", "weekly", "silver"})
VALID_DEPENDENCY_MODES = frozenset({"frozen", "live"})
VALID_GOLD_STATUSES = frozenset({"draft", "approved"})
VALID_ROUTING_DIRECTIONS = frozenset({"increase", "decrease", "neutral", "skip"})
CLARIFICATION_QUESTION_MATCH_KEYS = frozenset({"contains_all", "contains_any"})
CLARIFICATION_STATE_MATCH_KEYS = frozenset(
    {
        "clarification_kinds_any",
        "missing_fields_all",
        "missing_fields_any",
        "question",
    }
)
CLARIFICATION_TEMPLATE_FIELDS = frozenset({"reference_data_root"})
REQUIRED_TRANSCRIPT_ID_GROUP_KEYS = frozenset({"any_of", "evidence_refs"})


class EvalContractError(ValueError):
    """Raised when a versioned evaluation contract is malformed."""


class EvalVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    UNGRADED = "ungraded"


@dataclass(frozen=True)
class EvalCase:
    """One validated, immutable live evaluation case."""

    path: Path
    raw: Mapping[str, Any]
    case_id: str
    description: str
    tags: tuple[str, ...]
    tiers: tuple[str, ...]
    request: str
    stop_after: str
    dependency_mode: str
    clarifications: tuple[Mapping[str, Any], ...]
    required_input_artifacts: tuple[Mapping[str, Any], ...]
    gold: Mapping[str, Any]
    qualitative_rubric: Mapping[str, Any]
    gold_status: str
    reviewer: str
    evidence_refs: tuple[str, ...]
    digest: str

    @property
    def approved(self) -> bool:
        return self.gold_status == "approved"

    @property
    def silver(self) -> bool:
        return self.belongs_to("silver")

    def belongs_to(self, tier: str) -> bool:
        return tier in self.tiers


@dataclass(frozen=True)
class EvalProfile:
    """Runtime policy for one automation tier."""

    name: str
    model: str
    dependency_mode_override: str | None = None
    advisory: bool = True
    fresh_samples: int = 1
    run_judge: bool = True
    provider_sort: str | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    request_timeout_seconds: float | None = None
    max_retries: int | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_nonempty_string(
    payload: Mapping[str, Any], key: str, *, path: Path
) -> str:
    value = payload.get(key)
    text = str(value or "").strip()
    if not text:
        raise EvalContractError(f"{path}: {key} must be a non-empty string")
    return text


def _normalize_string_list(value: Any, *, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvalContractError(f"{path}: {field} must be a list")
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            raise EvalContractError(f"{path}: {field} cannot contain empty values")
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _validate_question_matcher(value: Any, *, field: str, path: Path) -> None:
    if not isinstance(value, Mapping):
        raise EvalContractError(f"{path}: {field} must be an object")
    unknown = set(value) - CLARIFICATION_QUESTION_MATCH_KEYS
    if unknown:
        raise EvalContractError(
            f"{path}: {field} has unsupported keys: {sorted(unknown)}"
        )
    selectors = 0
    for key in ("contains_all", "contains_any"):
        if key not in value:
            continue
        _normalize_string_list(value[key], field=f"{field}.{key}", path=path)
        selectors += 1
    if not selectors:
        raise EvalContractError(f"{path}: {field} must contain at least one selector")


def _validate_clarification_rules(
    value: Any,
    *,
    path: Path,
    case_evidence_refs: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise EvalContractError(f"{path}: clarifications must be a list of objects")
    normalized: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_rule in enumerate(value):
        rule = dict(raw_rule)
        field = f"clarifications[{index}]"
        rule_id = str(rule.get("id") or f"rule_{index + 1}").strip()
        if not CASE_ID_PATTERN.fullmatch(rule_id.lower()):
            raise EvalContractError(
                f"{path}: {field}.id is not identifier-safe: {rule_id!r}"
            )
        if rule_id in seen_ids:
            raise EvalContractError(
                f"{path}: duplicate clarification rule id {rule_id!r}"
            )
        seen_ids.add(rule_id)

        answer = str(rule.get("answer") or "").strip()
        answer_template = str(rule.get("answer_template") or "").strip()
        if bool(answer) == bool(answer_template):
            raise EvalContractError(
                f"{path}: {field} must define exactly one of answer or answer_template"
            )
        if answer_template:
            template_fields = {
                match.group(1)
                for match in re.finditer(
                    r"\{([A-Za-z_][A-Za-z0-9_]*)\}", answer_template
                )
            }
            unknown = template_fields - CLARIFICATION_TEMPLATE_FIELDS
            if unknown:
                raise EvalContractError(
                    f"{path}: {field}.answer_template has unsupported fields: {sorted(unknown)}"
                )
            if not template_fields:
                raise EvalContractError(
                    f"{path}: {field}.answer_template must contain a runtime field"
                )

        match = rule.get("match")
        if match is None:
            legacy_match = {
                key: rule[key]
                for key in CLARIFICATION_QUESTION_MATCH_KEYS
                if key in rule
            }
            _validate_question_matcher(legacy_match, field=field, path=path)
        else:
            if not isinstance(match, Mapping):
                raise EvalContractError(f"{path}: {field}.match must be an object")
            unknown = set(match) - CLARIFICATION_STATE_MATCH_KEYS
            if unknown:
                raise EvalContractError(
                    f"{path}: {field}.match has unsupported keys: {sorted(unknown)}"
                )
            selectors = 0
            if "question" in match:
                _validate_question_matcher(
                    match["question"], field=f"{field}.match.question", path=path
                )
                selectors += 1
            for key in (
                "clarification_kinds_any",
                "missing_fields_all",
                "missing_fields_any",
            ):
                if key not in match:
                    continue
                _normalize_string_list(
                    match[key], field=f"{field}.match.{key}", path=path
                )
                selectors += 1
            if not selectors:
                raise EvalContractError(
                    f"{path}: {field}.match must contain at least one selector"
                )

        rule_evidence = rule.get("evidence_refs")
        if rule_evidence is None:
            if not case_evidence_refs:
                raise EvalContractError(
                    f"{path}: {field} requires evidence_refs on the rule or case"
                )
        else:
            normalized_rule_evidence = _normalize_string_list(
                rule_evidence, field=f"{field}.evidence_refs", path=path
            )
            if not normalized_rule_evidence:
                raise EvalContractError(
                    f"{path}: {field}.evidence_refs must contain at least one source"
                )
        declared_source_class = str(rule.get("source_class") or "").strip()
        if declared_source_class:
            if declared_source_class not in VALID_CLARIFICATION_SOURCE_CLASSES:
                raise EvalContractError(
                    f"{path}: {field}.source_class is unsupported: "
                    f"{declared_source_class!r}"
                )
            if rule_evidence is None:
                raise EvalContractError(
                    f"{path}: {field}.source_class requires rule-scoped evidence_refs"
                )
            admission = clarification_rule_admission(rule)
            if not admission["admissible"]:
                raise EvalContractError(
                    f"{path}: {field}.source_class/evidence_refs are not admissible: "
                    f"{admission['reason']}"
                )
        normalized.append(rule)
    return tuple(normalized)


def _validate_required_transcript_id_groups(
    gold: Mapping[str, Any], *, path: Path
) -> None:
    """Validate source-backed alternatives for required transcript identities.

    Flat ``transcript_ids`` remain exact all-of requirements.  A required group
    represents one additional biological identity for which any independently
    sourced identifier is acceptable.  Keeping the evidence on the group makes
    corpus alternatives auditable without exposing them to the system under test.
    """

    targeting = gold.get("targeting")
    if not isinstance(targeting, Mapping):
        return
    raw_groups = targeting.get("required_transcript_id_groups")
    if raw_groups is None:
        return
    field = "gold.targeting.required_transcript_id_groups"
    if not isinstance(raw_groups, list) or not raw_groups:
        raise EvalContractError(f"{path}: {field} must be a non-empty list")

    identifiers_seen: set[str] = set()
    for index, raw_group in enumerate(raw_groups):
        group_field = f"{field}[{index}]"
        if not isinstance(raw_group, Mapping):
            raise EvalContractError(f"{path}: {group_field} must be an object")
        unknown = set(raw_group) - REQUIRED_TRANSCRIPT_ID_GROUP_KEYS
        if unknown:
            raise EvalContractError(
                f"{path}: {group_field} has unsupported keys: {sorted(unknown)}"
            )
        alternatives = _normalize_string_list(
            raw_group.get("any_of"), field=f"{group_field}.any_of", path=path
        )
        if len(alternatives) < 2:
            raise EvalContractError(
                f"{path}: {group_field}.any_of requires at least two alternatives"
            )
        normalized_ids = {item.casefold() for item in alternatives}
        overlap = identifiers_seen.intersection(normalized_ids)
        if overlap:
            raise EvalContractError(
                f"{path}: {field} alternatives must be disjoint; repeated values: "
                f"{sorted(overlap)}"
            )
        identifiers_seen.update(normalized_ids)

        evidence_refs = _normalize_string_list(
            raw_group.get("evidence_refs"),
            field=f"{group_field}.evidence_refs",
            path=path,
        )
        if not evidence_refs:
            raise EvalContractError(
                f"{path}: {group_field}.evidence_refs must contain at least one source"
            )
        admission = classify_clarification_evidence_refs(evidence_refs)
        if (
            not admission["admissible"]
            or admission["source_class"] != INDEPENDENT_BIOLOGICAL_SOURCE
        ):
            raise EvalContractError(
                f"{path}: {group_field}.evidence_refs must be independent "
                f"biological sources: {admission['reason']}"
            )


def validate_case(payload: Mapping[str, Any], *, path: Path) -> EvalCase:
    if payload.get("schema_version") != EVAL_CASE_SCHEMA_VERSION:
        raise EvalContractError(
            f"{path}: unsupported schema_version={payload.get('schema_version')!r}; "
            f"expected {EVAL_CASE_SCHEMA_VERSION}"
        )
    case_id = _require_nonempty_string(payload, "id", path=path).lower()
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise EvalContractError(f"{path}: id is not filesystem-safe: {case_id!r}")
    description = _require_nonempty_string(payload, "description", path=path)
    request = _require_nonempty_string(payload, "request", path=path)
    tags = _normalize_string_list(payload.get("tags"), field="tags", path=path)
    tiers = _normalize_string_list(payload.get("tiers"), field="tiers", path=path)
    invalid_tiers = sorted(set(tiers) - VALID_TIERS)
    if invalid_tiers:
        raise EvalContractError(f"{path}: unsupported tiers: {invalid_tiers}")
    stop_after = _require_nonempty_string(payload, "stop_after", path=path).lower()
    if stop_after not in STAGE_ORDER:
        raise EvalContractError(f"{path}: unsupported stop_after={stop_after!r}")
    dependency_mode = _require_nonempty_string(
        payload, "dependency_mode", path=path
    ).lower()
    if dependency_mode not in VALID_DEPENDENCY_MODES:
        raise EvalContractError(
            f"{path}: unsupported dependency_mode={dependency_mode!r}"
        )
    gold = payload.get("gold")
    if not isinstance(gold, Mapping):
        raise EvalContractError(f"{path}: gold must be an object")
    _validate_required_transcript_id_groups(gold, path=path)
    rubric = payload.get("qualitative_rubric")
    if not isinstance(rubric, Mapping):
        raise EvalContractError(f"{path}: qualitative_rubric must be an object")
    gold_status = _require_nonempty_string(payload, "gold_status", path=path).lower()
    if gold_status not in VALID_GOLD_STATUSES:
        raise EvalContractError(f"{path}: unsupported gold_status={gold_status!r}")
    reviewer = str(payload.get("reviewer") or "").strip()
    if gold_status == "approved" and not reviewer:
        raise EvalContractError(f"{path}: approved gold requires reviewer")
    evidence_refs = _normalize_string_list(
        payload.get("evidence_refs", []), field="evidence_refs", path=path
    )
    if gold_status == "approved" and not evidence_refs:
        raise EvalContractError(
            f"{path}: approved gold requires at least one evidence_ref"
        )
    cutoff_index = STAGE_ORDER.index(stop_after)
    executable_sections = [
        section
        for section, pipeline_stage in {
            "task_graph": "request_graph",
            "structured_goal": "structured_goal",
            "targeting": "targeting",
            "routing": "routing",
            "candidates": "candidates",
            "e2e": "e2e",
        }.items()
        if isinstance(gold.get(section), Mapping)
        and bool(gold.get(section))
        and STAGE_ORDER.index(pipeline_stage) <= cutoff_index
    ]
    if gold_status == "approved" and not executable_sections:
        raise EvalContractError(
            f"{path}: approved gold requires non-empty checks at or before stop_after"
        )
    if "silver" in tiers:
        if "silver" not in tags:
            raise EvalContractError(f"{path}: silver-tier cases require the silver tag")
        if gold_status != "draft":
            raise EvalContractError(f"{path}: silver-tier cases must remain draft gold")
        if not evidence_refs:
            raise EvalContractError(
                f"{path}: silver-tier cases require source evidence_refs"
            )
        if not executable_sections:
            raise EvalContractError(
                f"{path}: silver-tier cases require non-empty checks at or before stop_after"
            )
        routing_gold = gold.get("routing")
        if isinstance(routing_gold, Mapping):
            raw_directions = routing_gold.get(
                "directions", routing_gold.get("expected_directions")
            )
            if raw_directions is not None:
                directions = _normalize_string_list(
                    raw_directions,
                    field="gold.routing.directions",
                    path=path,
                )
                invalid_directions = sorted(set(directions) - VALID_ROUTING_DIRECTIONS)
                if invalid_directions:
                    raise EvalContractError(
                        f"{path}: silver gold routing directions must use the "
                        f"normalized route vocabulary; unsupported values: "
                        f"{invalid_directions}"
                    )
    clarifications = _validate_clarification_rules(
        payload.get("clarifications", []),
        path=path,
        case_evidence_refs=evidence_refs,
    )
    raw_required_inputs = payload.get("required_input_artifacts", [])
    if not isinstance(raw_required_inputs, list) or not all(
        isinstance(item, Mapping) for item in raw_required_inputs
    ):
        raise EvalContractError(
            f"{path}: required_input_artifacts must be a list of objects"
        )
    required_input_artifacts: list[Mapping[str, Any]] = []
    seen_artifact_ids: set[str] = set()
    for index, raw_artifact in enumerate(raw_required_inputs):
        artifact = dict(raw_artifact)
        field = f"required_input_artifacts[{index}]"
        artifact_id = _require_nonempty_string(
            artifact,
            "artifact_id",
            path=path,
        ).lower()
        if not CASE_ID_PATTERN.fullmatch(artifact_id):
            raise EvalContractError(
                f"{path}: {field}.artifact_id is not identifier-safe: {artifact_id!r}"
            )
        if artifact_id in seen_artifact_ids:
            raise EvalContractError(
                f"{path}: duplicate required input artifact ID {artifact_id!r}"
            )
        seen_artifact_ids.add(artifact_id)
        relative_path_text = _require_nonempty_string(
            artifact,
            "relative_path",
            path=path,
        )
        relative_path = Path(relative_path_text)
        if (
            relative_path.is_absolute()
            or relative_path == Path(".")
            or ".." in relative_path.parts
        ):
            raise EvalContractError(
                f"{path}: {field}.relative_path must stay below the benchmark "
                "input root"
            )
        description_text = _require_nonempty_string(
            artifact,
            "description",
            path=path,
        )
        expected_sha256 = str(artifact.get("sha256") or "").strip().lower()
        if expected_sha256:
            try:
                int(expected_sha256, 16)
            except ValueError as exc:
                raise EvalContractError(
                    f"{path}: {field}.sha256 must be a hexadecimal SHA-256 digest"
                ) from exc
            if len(expected_sha256) != 64:
                raise EvalContractError(
                    f"{path}: {field}.sha256 must contain 64 hexadecimal characters"
                )
        required_input_artifacts.append(
            {
                "artifact_id": artifact_id,
                "relative_path": relative_path.as_posix(),
                "description": description_text,
                **({"sha256": expected_sha256} if expected_sha256 else {}),
            }
        )
    raw_copy = json.loads(json.dumps(payload, ensure_ascii=False))
    return EvalCase(
        path=path,
        raw=raw_copy,
        case_id=case_id,
        description=description,
        tags=tags,
        tiers=tiers,
        request=request,
        stop_after=stop_after,
        dependency_mode=dependency_mode,
        clarifications=clarifications,
        required_input_artifacts=tuple(required_input_artifacts),
        gold=dict(gold),
        qualitative_rubric=dict(rubric),
        gold_status=gold_status,
        reviewer=reviewer,
        evidence_refs=evidence_refs,
        digest=stable_digest(raw_copy),
    )


def load_case(path: Path) -> EvalCase:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvalContractError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvalContractError(f"{path}: case file must contain an object")
    return validate_case(payload, path=path)


def load_cases(case_root: Path, *, tier: str | None = None) -> list[EvalCase]:
    if tier is not None and tier not in VALID_TIERS:
        raise EvalContractError(f"unsupported tier={tier!r}")
    if not case_root.exists():
        raise EvalContractError(f"case root does not exist: {case_root}")
    cases: list[EvalCase] = []
    seen: dict[str, Path] = {}
    for path in sorted(case_root.glob("*.json")):
        if path.name == "index.json":
            continue
        case = load_case(path)
        if case.case_id in seen:
            raise EvalContractError(
                f"duplicate case id {case.case_id!r}: {seen[case.case_id]} and {path}"
            )
        if path.stem != case.case_id:
            raise EvalContractError(
                f"{path}: filename must match case id ({case.case_id}.json)"
            )
        seen[case.case_id] = path
        if tier is None or case.belongs_to(tier):
            cases.append(case)
    if not cases:
        suffix = f" for tier {tier!r}" if tier else ""
        raise EvalContractError(
            f"no evaluation cases found beneath {case_root}{suffix}"
        )
    return cases


def select_cases(
    cases: Sequence[EvalCase],
    *,
    case_ids: Iterable[str] = (),
) -> list[EvalCase]:
    wanted = {
        str(case_id).strip().lower() for case_id in case_ids if str(case_id).strip()
    }
    if not wanted:
        return list(cases)
    selected = [case for case in cases if case.case_id in wanted]
    missing = sorted(wanted - {case.case_id for case in selected})
    if missing:
        raise EvalContractError(f"unknown case id(s): {', '.join(missing)}")
    return selected


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
