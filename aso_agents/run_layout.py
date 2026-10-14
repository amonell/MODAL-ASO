"""Canonical run-folder layout for MODAL-ASO outputs.

Root of every run is only:
  request.txt, report.md, report.html, artifacts/

Everything else lives under artifacts/{candidates,design,intake,controller}.
Reads accept the current layout first, then the pre-migration root layout so
existing run folders remain inspectable. New writes use the current layout only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

RunDir = Union[str, Path]

LAYOUT_VERSION = 1
ARTIFACTS_DIRNAME = "artifacts"
CANDIDATES_DIRNAME = "candidates"
DESIGN_DIRNAME = "design"
INTAKE_DIRNAME = "intake"
CONTROLLER_DIRNAME = "controller"
LAYOUT_MANIFEST_NAME = "layout.json"

ROOT_HUMAN_FILES = ("request.txt", "report.md", "report.html")
ROOT_ALLOWED_ENTRIES = frozenset((*ROOT_HUMAN_FILES, ARTIFACTS_DIRNAME, "child_runs"))

INTAKE_BASENAMES = frozenset(
    {
        "goal_intake.json",
        "structured_goal.json",
        "request_envelope.json",
        "task_index.json",
        "report_explainer.json",
        "composite_report.md",
        "composite_report.json",
        "answer_question.md",
        "answer_question.json",
        "artifact_analysis.md",
        "artifact_analysis.json",
        "existing_aso_evaluation.md",
        "existing_aso_evaluation.json",
        "existing_aso_evaluation.xlsx",
        "artifact_index.json",
    }
)

CANDIDATE_EXACT_NAMES = frozenset(
    {
        "generated_aso_set.json",
        "generated_filter_trace.json",
        "gapmer_uniqueness_ranked.csv",
        "gapmer_uniqueness_ranked.xlsx",
        "gapmer_scored_uniqueness.json",
        "gapmer_scored_uniqueness.filtered.json",
        "gapmer_scored_uniqueness.xlsx",
        "steric_uniqueness_ranked.csv",
        "steric_uniqueness_ranked.xlsx",
        "steric_scored_uniqueness.json",
        "steric_scored_uniqueness.filtered.json",
        "steric_scored_uniqueness.xlsx",
        "scored_tiles.json",
        "scored_tiles.filtered.json",
        "scored_tiles.xlsx",
        "scored_tiles.filtered.xlsx",
    }
)

DESIGN_DIR_NAMES = frozenset(
    {
        "tiling",
        "gapmer_tiling",
        "junction_gapmer_tiling",
        "gapmer_jobs",
        "steric_jobs",
        "strategy_routes",
        "strategy_router_debug",
        "snapgene",
        "v2_tasks",
        "v3_tasks",
        "patient_reference",
    }
)

DESIGN_FILE_NAMES = frozenset(
    {
        "strategy_router_summary.json",
        "steric_regions.bed",
        "tiling_config.json",
        "splice_site_comparisons.csv",
        "splice_strategy_analysis.md",
        "splice_strategy_input.json",
        "splice_strategy_prompt.txt",
        "splice_strategy_agent_trace.json",
    }
)

TILING_ROOT_DIR_NAMES = frozenset({"tiling", "gapmer_tiling", "junction_gapmer_tiling"})

_HTML_ALIASES = ("report.html", "goal_intake_report.html")

LOGICAL_PATHS: Dict[str, Tuple[str, ...]] = {
    "request_txt": ("request.txt",),
    "report_md": ("report.md",),
    "report_html": ("report.html",),
    "layout_json": (ARTIFACTS_DIRNAME, LAYOUT_MANIFEST_NAME),
    "goal_intake_json": (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, "goal_intake.json"),
    "structured_goal_json": (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, "structured_goal.json"),
    "request_envelope_json": (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, "request_envelope.json"),
    "task_index_json": (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, "task_index.json"),
    "report_explainer_json": (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, "report_explainer.json"),
    "generated_aso_set_json": (ARTIFACTS_DIRNAME, CANDIDATES_DIRNAME, "generated_aso_set.json"),
    "generated_filter_trace_json": (ARTIFACTS_DIRNAME, CANDIDATES_DIRNAME, "generated_filter_trace.json"),
    "candidates_dir": (ARTIFACTS_DIRNAME, CANDIDATES_DIRNAME),
    "design_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME),
    "intake_dir": (ARTIFACTS_DIRNAME, INTAKE_DIRNAME),
    "controller_dir": (ARTIFACTS_DIRNAME, CONTROLLER_DIRNAME),
    "tiling_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "tiling"),
    "gapmer_tiling_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "gapmer_tiling"),
    "steric_jobs_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "steric_jobs"),
    "gapmer_jobs_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "gapmer_jobs"),
    "strategy_routes_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "strategy_routes"),
    "strategy_router_debug_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "strategy_router_debug"),
    "snapgene_dir": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "snapgene"),
    "strategy_router_summary_json": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "strategy_router_summary.json"),
    "steric_regions_bed": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "steric_regions.bed"),
    "tiling_config_json": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "tiling_config.json"),
    "splicing_eval_csv": (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, "splice_site_comparisons.csv"),
}


def run_root(run_dir: RunDir) -> Path:
    return Path(run_dir).expanduser()


def _stem_basename(name: str) -> str:
    text = str(name or "")
    if text.endswith(".tmp"):
        return text[: -len(".tmp")]
    return text


def _is_candidate_table_name(name: str) -> bool:
    lower = str(name or "").lower()
    if not lower or "unfiltered" in lower:
        return False
    if lower in CANDIDATE_EXACT_NAMES:
        return True
    if "uniqueness_ranked" in lower:
        return True
    if "scored_uniqueness" in lower:
        return True
    return False


def _normalize_parts(parts: Sequence[str]) -> Tuple[str, ...]:
    normalized: List[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text or text in {".", "/"}:
            continue
        for piece in Path(text).parts:
            if piece in {".", "/"}:
                continue
            normalized.append(piece)
    return tuple(normalized)


def canonical_relparts(*parts: str) -> Tuple[str, ...]:
    """Map a legacy or current run-relative path to the current layout."""
    rel = _normalize_parts(parts)
    if not rel:
        return rel
    if rel[0] == ARTIFACTS_DIRNAME:
        if len(rel) >= 2 and rel[1] == "goal_intake_report.html":
            return ("report.html",)
        return rel
    first = rel[0]
    rest = rel[1:]
    if first in {"goal_intake_report.html", "report.html"} and not rest:
        return ("report.html",)
    if first in ROOT_HUMAN_FILES and not rest:
        return rel
    if first == "child_runs":
        return rel
    if first.startswith("agentic_"):
        return (ARTIFACTS_DIRNAME, CONTROLLER_DIRNAME, *rel)
    if first == "agentic_state":
        return (ARTIFACTS_DIRNAME, CONTROLLER_DIRNAME, *rel)
    intake_name = _stem_basename(first)
    if intake_name in INTAKE_BASENAMES and not rest:
        return (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, first)
    if first in CANDIDATE_EXACT_NAMES and not rest:
        return (ARTIFACTS_DIRNAME, CANDIDATES_DIRNAME, first)
    if _is_candidate_table_name(first) and not rest:
        return (ARTIFACTS_DIRNAME, CANDIDATES_DIRNAME, first)
    if first in TILING_ROOT_DIR_NAMES:
        if len(rel) == 2 and _is_candidate_table_name(rel[1]):
            return (ARTIFACTS_DIRNAME, CANDIDATES_DIRNAME, rel[1])
        return (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, *rel)
    if first in DESIGN_DIR_NAMES:
        return (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, *rel)
    if first in DESIGN_FILE_NAMES and not rest:
        return (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, first)
    if first.startswith("splice_strategy_") and not rest:
        return (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, first)
    if first.startswith("reference_download_") and not rest:
        return (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, first)
    if not rest and Path(first).suffix.lower() in {".bed", ".gb", ".gbk"}:
        return (ARTIFACTS_DIRNAME, DESIGN_DIRNAME, first)
    if not rest and first not in ROOT_HUMAN_FILES:
        suffix = Path(first).suffix.lower()
        if suffix in {".json", ".md", ".xlsx", ".csv", ".tsv", ".txt", ".html"}:
            return (ARTIFACTS_DIRNAME, INTAKE_DIRNAME, first)
    return rel


def canonical_run_path(run_dir: RunDir, *parts: str) -> Path:
    root = run_root(run_dir)
    rel = canonical_relparts(*parts)
    return root.joinpath(*rel) if rel else root


def ensure_run_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def legacy_relparts(*parts: str) -> Tuple[str, ...]:
    rel = _normalize_parts(parts)
    if not rel:
        return rel
    if rel[0] == ARTIFACTS_DIRNAME:
        remainder = rel[1:]
        if not remainder:
            return rel
        if remainder[0] in {CANDIDATES_DIRNAME, DESIGN_DIRNAME, INTAKE_DIRNAME, CONTROLLER_DIRNAME}:
            remainder = remainder[1:]
        if remainder:
            return remainder
        return rel
    if rel == ("report.html",):
        return ("goal_intake_report.html",)
    return rel


def _layout_manifest_path(run_dir: RunDir) -> Path:
    return run_root(run_dir) / ARTIFACTS_DIRNAME / LAYOUT_MANIFEST_NAME


def load_layout_manifest(run_dir: RunDir) -> Dict[str, Any]:
    path = _layout_manifest_path(run_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_relative(run_dir: RunDir, *parts: str) -> Optional[str]:
    payload = load_layout_manifest(run_dir)
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    if not isinstance(paths, dict):
        return None
    joined = "/".join(_normalize_parts(parts))
    logical_key = joined.replace("/", "_").replace(".", "_")
    for key in (joined, logical_key, parts[-1] if parts else ""):
        value = paths.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def iter_run_path_candidates(run_dir: RunDir, *parts: str) -> Iterator[Path]:
    root = run_root(run_dir)
    seen: set[str] = set()

    def _emit(path: Path) -> Iterator[Path]:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        yield path

    manifest_rel = _manifest_relative(run_dir, *parts)
    if manifest_rel:
        yield from _emit(root.joinpath(*Path(manifest_rel).parts))
    canonical = canonical_run_path(run_dir, *parts)
    yield from _emit(canonical)
    rel = _normalize_parts(parts)
    if rel:
        yield from _emit(root.joinpath(*rel))
    legacy = legacy_relparts(*parts)
    if legacy:
        yield from _emit(root.joinpath(*legacy))
    if rel == ("report.html",) or (rel and rel[-1] in _HTML_ALIASES):
        for name in _HTML_ALIASES:
            yield from _emit(root / name)
    if rel and rel[-1] in CANDIDATE_EXACT_NAMES:
        name = rel[-1]
        yield from _emit(root / name)
        for folder in ("tiling", "gapmer_tiling", "junction_gapmer_tiling"):
            yield from _emit(root / folder / name)
            yield from _emit(root / ARTIFACTS_DIRNAME / DESIGN_DIRNAME / folder / name)
        yield from _emit(root / ARTIFACTS_DIRNAME / CANDIDATES_DIRNAME / name)


def resolve_run_path(run_dir: RunDir, *parts: str) -> Path:
    """Return the first existing mapped path, else the canonical write path."""
    existing: Optional[Path] = None
    canonical: Optional[Path] = None
    for index, candidate in enumerate(iter_run_path_candidates(run_dir, *parts)):
        if index == 0:
            canonical = candidate
        if candidate.exists():
            existing = candidate
            break
    if existing is not None:
        return existing
    return canonical if canonical is not None else canonical_run_path(run_dir, *parts)


def resolve_run_file(run_dir: RunDir, *parts: str) -> Optional[Path]:
    path = resolve_run_path(run_dir, *parts)
    return path if path.exists() else None


def relative_run_path(run_dir: RunDir, path: Union[str, Path]) -> str:
    root = run_root(run_dir)
    candidate = Path(path)
    try:
        return candidate.relative_to(root).as_posix()
    except Exception:
        pass
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def write_layout_manifest(run_dir: RunDir) -> Optional[Path]:
    root = run_root(run_dir)
    paths: Dict[str, str] = {}
    for logical, rel in LOGICAL_PATHS.items():
        path = root.joinpath(*rel)
        if path.exists():
            paths[logical] = Path(*rel).as_posix()
    extra_names = (
        "gapmer_uniqueness_ranked.csv",
        "gapmer_uniqueness_ranked.xlsx",
        "gapmer_scored_uniqueness.json",
        "gapmer_scored_uniqueness.xlsx",
        "steric_uniqueness_ranked.csv",
        "steric_uniqueness_ranked.xlsx",
        "steric_scored_uniqueness.json",
        "scored_tiles.json",
    )
    for name in extra_names:
        path = canonical_run_path(root, name)
        if path.exists():
            paths[name] = relative_run_path(root, path)
    if not paths:
        return None
    manifest_path = _layout_manifest_path(root)
    ensure_run_parent(manifest_path)
    payload = {
        "version": LAYOUT_VERSION,
        "paths": paths,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def is_controller_rel(rel_parts: Sequence[str]) -> bool:
    parts = [str(part) for part in rel_parts]
    if not parts:
        return False
    if parts[0].startswith("agentic_"):
        return True
    if "controller" in parts:
        return True
    if "state_snapshots" in parts:
        return True
    return False


def infer_run_dir(path: Union[str, Path]) -> Optional[Path]:
    """Best-effort run root from a file or directory inside a run."""
    current = Path(path).expanduser()
    try:
        resolved = current.resolve()
    except Exception:
        resolved = current
    parts = resolved.parts
    if ARTIFACTS_DIRNAME in parts:
        idx = parts.index(ARTIFACTS_DIRNAME)
        if idx > 0:
            return Path(*parts[:idx])
    search_from = resolved if resolved.is_dir() else resolved.parent
    for parent in [search_from, *search_from.parents]:
        if (parent / "request.txt").exists() or (parent / "report.md").exists():
            return parent
        if parent.name.startswith("goal_"):
            return parent
    return None


def scored_table_output_path(source_json_path: Union[str, Path], output_name: str) -> Path:
    """Write run-level ranked/scored tables under artifacts/candidates.

    Job copies stay next to the source JSON in the job directory.
    """
    source = Path(source_json_path)
    name = str(output_name or "").strip() or source.name
    parent_name = source.parent.name
    run_dir = infer_run_dir(source)
    if run_dir is not None and _is_candidate_table_name(name) and parent_name in TILING_ROOT_DIR_NAMES:
        return canonical_run_path(run_dir, name)
    return source.parent / name
