from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

_BASES = ("A", "C", "G", "T")
DEFAULT_HOMOPOLYMER_LIMIT = 4

_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_BINDING_MORE_PATTERN = re.compile(
    r"(?:more than|greater than|over)\s+(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:total\s+)?binding sites?",
    re.IGNORECASE,
)
_BINDING_MAX_PATTERN = re.compile(
    r"(?:at most|no more than|not more than|<=|<=|maximum of|max(?:imum)?\s+of)\s+(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:total\s+)?binding sites?",
    re.IGNORECASE,
)
_BINDING_EXACT_PATTERN = re.compile(
    r"(?:limit(?:ed)? to|exactly|only)\s+(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:total\s+)?binding sites?",
    re.IGNORECASE,
)
_OFFTARGET_MAX_PATTERN = re.compile(
    r"(?:at most|no more than|not more than|<=|maximum of|max(?:imum)?\s+(?:of\s+)?)"
    r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:off[-\s]?target\s+)?(?:mappings?|hits?|binding sites?)",
    re.IGNORECASE,
)
_OFFTARGET_LESS_THAN_PATTERN = re.compile(
    r"(?:fewer than|less than|under)\s+"
    r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:off[-\s]?target\s+)?(?:mappings?|hits?|binding sites?)",
    re.IGNORECASE,
)
_GC_GT_LT_PATTERN = re.compile(
    r"\bgc(?:\s+content)?\s*(?:is|:)?\s*"
    r"(?:greater\s+than|more\s+than|>)\s*(?P<min>\d+(?:\.\d+)?\s*%?)\s*"
    r"(?:,?\s*(?:and|to|through)?\s*)"
    r"(?:less\s+than|under|<)\s*(?P<max>\d+(?:\.\d+)?\s*%?)",
    re.IGNORECASE,
)
_GC_BETWEEN_PATTERN = re.compile(
    r"\bgc(?:\s+content)?\s*(?:is|:)?\s*"
    r"(?:between|from)\s*(?P<min>\d+(?:\.\d+)?\s*%?)\s*(?:and|to|-)\s*(?P<max>\d+(?:\.\d+)?\s*%?)",
    re.IGNORECASE,
)
_GC_EXACT_PATTERN = re.compile(
    r"\bgc(?:\s+content)?\s*(?:is|:)?\s*"
    r"(?:exactly|equal(?:s)?|=)\s*(?P<value>\d+(?:\.\d+)?\s*%?)",
    re.IGNORECASE,
)
_HOMOPOLYMER_EXCLUDE_PATTERN = re.compile(
    r"(?:exclude|avoid|disallow|filter\s+out|remove)\s+"
    r"(?:sequences?\s+with\s+)?"
    r"(?:(?P<base>[ACGT])\s*)?"
    r"(?:homo(?:polymer)?\s*)?"
    r"(?:runs?|stretches?)\s+"
    r"(?:of\s+)?(?:length\s+)?"
    r"(?P<op>greater\s+than\s+or\s+equal\s+to|at\s+least|>=|≥|greater\s+than|more\s+than|over|>)?\s*"
    r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)",
    re.IGNORECASE,
)


def _word_to_int(token: Any) -> Optional[int]:
    if token is None:
        return None
    if isinstance(token, int):
        return token
    if isinstance(token, float) and token.is_integer():
        return int(token)
    text_raw = str(token).strip()
    if not text_raw:
        return None
    text = text_raw.lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    cleaned = text.replace(",", "").rstrip("+").strip()
    prefix = ""
    while cleaned and cleaned[0] in {">", "<", "=", "~", ">=", "<="}:
        prefix += cleaned[0]
        cleaned = cleaned[1:]
    cleaned = cleaned.strip()
    match = re.search(r"\d+", cleaned)
    if match:
        value = int(match.group())
        if prefix.startswith(">"):
            value += 1
        return value
    return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            num = int(round(float(value)))
        except Exception:
            return None
        return num if num > 0 else None
    text = str(value).strip()
    if not text:
        return None
    digits: List[str] = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    candidate = int("".join(digits))
    return candidate if candidate > 0 else None


def _coerce_nonnegative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            num = int(round(float(value)))
        except Exception:
            return None
        return num if num >= 0 else None
    text = str(value).strip()
    if not text:
        return None
    text_lower = text.lower()
    if text_lower in _NUMBER_WORDS:
        return _NUMBER_WORDS[text_lower]
    digits: List[str] = []
    for ch in text.replace(",", ""):
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    candidate = int("".join(digits))
    return candidate if candidate >= 0 else None


def _normalize_per_base_limits(per_base_limits: Any, *, fallback: Optional[int] = None) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    if isinstance(per_base_limits, dict):
        for base, value in per_base_limits.items():
            base_norm = str(base).strip().upper()
            if base_norm not in _BASES:
                continue
            value_int = _coerce_nonnegative_int(value)
            if value_int is None:
                continue
            existing = normalized.get(base_norm)
            if existing is None or value_int < existing:
                normalized[base_norm] = value_int
    if fallback is not None:
        for base in _BASES:
            if base not in normalized:
                normalized[base] = fallback
    return normalized


def _infer_per_base_limits_from_sequences(sequences: Optional[List[Any]]) -> Dict[str, int]:
    inferred: Dict[str, int] = {}
    if not isinstance(sequences, list):
        return inferred
    for seq in sequences:
        seq_norm = str(seq).strip().upper()
        if not seq_norm:
            continue
        unique = set(seq_norm)
        if len(unique) != 1:
            continue
        base = seq_norm[0]
        if base not in _BASES:
            continue
        limit = len(seq_norm) - 1
        if limit <= 0:
            continue
        current = inferred.get(base)
        if current is None or limit < current:
            inferred[base] = limit
    return inferred


def _coerce_fraction(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        percent = False
        if text.endswith("%"):
            percent = True
            text = text[:-1]
        try:
            val = float(text)
        except ValueError:
            return None
        if percent:
            val /= 100.0
    if val > 1.0 and val <= 100.0:
        val /= 100.0
    if val < 0:
        return None
    if val > 1.0:
        val = 1.0
    return round(val, 6)


def _coerce_constraint_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "include", "included", "inclusive"}:
        return True
    if text in {"false", "0", "no", "n", "exclude", "excluded", "exclusive"}:
        return False
    return None


def _merge_gc_constraint(
    target: Dict[str, Any],
    min_val: Optional[float],
    max_val: Optional[float],
    source: str,
    *,
    min_inclusive: Optional[bool] = None,
    max_inclusive: Optional[bool] = None,
) -> None:
    if min_val is None and max_val is None:
        return
    entry = target.setdefault("gc_content", {})
    entry.setdefault("sources", [])
    if source not in entry["sources"]:
        entry["sources"].append(source)
    if min_val is not None:
        entry["min"] = min_val
        target["gc_min"] = min_val
        entry["min_percent"] = round(min_val * 100.0, 2)
        if min_inclusive is not None:
            entry["min_inclusive"] = bool(min_inclusive)
    if max_val is not None:
        entry["max"] = max_val
        target["gc_max"] = max_val
        entry["max_percent"] = round(max_val * 100.0, 2)
        if max_inclusive is not None:
            entry["max_inclusive"] = bool(max_inclusive)
    if "bounds" in entry:
        entry.pop("bounds")


def _merge_binding_constraint(target: Dict[str, Any], max_sites: Optional[int], source: str) -> None:
    if max_sites is None:
        return
    existing = target.get("max_binding_sites")
    if existing is None or max_sites < existing:
        target["max_binding_sites"] = max_sites
    meta = target.setdefault("binding_sites", {})
    meta.setdefault("sources", [])
    if source not in meta["sources"]:
        meta["sources"].append(source)
    meta["max"] = target["max_binding_sites"]


def _build_homopolymer_sequences(max_run: int) -> List[str]:
    min_disallowed = max(1, max_run + 1)
    bases = ["A", "C", "G", "T"]
    return [base * min_disallowed for base in bases]


def _max_homopolymer_runs(sequence: str) -> Dict[str, int]:
    runs: Dict[str, int] = {base: 0 for base in _BASES}
    if not sequence:
        return runs
    seq = str(sequence).upper()
    prev = ""
    current = 0
    for ch in seq:
        if ch == prev:
            current += 1
        else:
            if prev in runs:
                runs[prev] = max(runs[prev], current)
            prev = ch
            current = 1
    if prev in runs:
        runs[prev] = max(runs[prev], current)
    return runs


def get_homopolymer_per_base_limits(
    constraints: Optional[Dict[str, Any]],
    *,
    fallback_default: Optional[int] = None,
) -> Dict[str, int]:
    """Resolve per-base homopolymer limits from a constraints object.

    - Honors explicit per-base values from constraints.homopolymers.per_base_max.
    - If only a global max_run/homopolymer_max_run is present, it applies to all bases.
    - When per-base limits are present, they are honored directly.
    - Optionally fills missing bases with fallback_default.
    """
    if not isinstance(constraints, dict):
        return {}

    homo_entry = constraints.get("homopolymers")
    per_base_limits = {}
    if isinstance(homo_entry, dict):
        per_base_limits = _normalize_per_base_limits(homo_entry.get("per_base_max"))

    global_limit = _coerce_positive_int(constraints.get("homopolymer_max_run"))
    if global_limit is None and isinstance(homo_entry, dict):
        global_limit = _coerce_positive_int(homo_entry.get("max_run"))

    if global_limit is not None and not per_base_limits:
        for base in _BASES:
            per_base_limits[base] = min(per_base_limits.get(base, global_limit), global_limit)

    if fallback_default is not None:
        for base in _BASES:
            if base not in per_base_limits:
                per_base_limits[base] = fallback_default

    return per_base_limits


def candidate_constraint_sequence(row: Dict[str, Any]) -> Any:
    """Use the synthesized molecule for oligo sequence constraints.

    Legacy untyped sequence-only rows retain their existing interpretation.
    An explicitly labeled target string is complemented only when no physical
    synthesis field is available; the source row is never modified.
    """
    annotations = row.get("source_annotations")
    if not isinstance(annotations, dict):
        annotations = {}
    for container in (row, annotations):
        for key in ("synthesized_antisense_sequence", "Synthesis", "chemistry_synthesis", "antisense_5to3"):
            if container.get(key):
                return container[key]
    sequence = next((row[key] for key in ("Sequence", "sequence", "Seq", "seq", "ASO", "Oligo")
                     if row.get(key)), None)
    if sequence and row.get("sequence_role") in {"transcript_oriented_target_binding_site", "target_site_sequence"}:
        return str(sequence).upper().replace("U", "T").translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return sequence


def sequence_passes_homopolymer_limits(sequence: Any, per_base_limits: Dict[str, Any]) -> bool:
    """Return True when sequence run lengths satisfy per-base max limits."""
    if not per_base_limits:
        return True
    normalized_limits = _normalize_per_base_limits(per_base_limits)
    if not normalized_limits:
        return True
    seq = str(sequence or "").strip().upper()
    if not seq:
        return False
    observed = _max_homopolymer_runs(seq)
    for base, limit in normalized_limits.items():
        if observed.get(base, 0) > limit:
            return False
    return True


def _merge_homopolymer_constraint(
    target: Dict[str, Any],
    limit: Optional[int],
    source: str,
    extra_sequences: Optional[List[Any]] = None,
    per_base_limits: Optional[Dict[str, Any]] = None,
) -> None:
    if limit is None and not extra_sequences and not per_base_limits:
        return
    entry = target.setdefault("homopolymers", {})
    entry.setdefault("sources", [])
    existing_per_base = _normalize_per_base_limits(entry.get("per_base_max"))
    existing_sources = {
        str(item)
        for item in entry.get("sources", [])
        if str(item).strip()
    }
    default_like_existing = (
        existing_per_base == {base: DEFAULT_HOMOPOLYMER_LIMIT for base in _BASES}
        and bool(existing_sources)
        and existing_sources.issubset(
            {
                "structured_goal.parameter_constraints",
                "structured_goal.top_level",
                "inferred_default",
            }
        )
        and entry.get("max_run") in (None, "")
    )
    if source not in entry["sources"]:
        entry["sources"].append(source)

    def _merge_per_base(target_map: Dict[str, int], incoming: Dict[str, int]) -> None:
        for base, value in incoming.items():
            if value is None:
                continue
            current = target_map.get(base)
            if (
                current is not None
                and value > current
                and default_like_existing
                and source not in {"structured_goal.parameter_constraints", "structured_goal.top_level"}
            ):
                target_map[base] = value
            elif current is None or value < current:
                target_map[base] = value

    merged_per_base: Dict[str, int] = dict(existing_per_base)
    inferred_from_sequences = _infer_per_base_limits_from_sequences(extra_sequences if isinstance(extra_sequences, list) else [])
    _merge_per_base(merged_per_base, inferred_from_sequences)
    _merge_per_base(merged_per_base, _normalize_per_base_limits(per_base_limits))
    if limit is not None and not merged_per_base:
        limit = max(0, int(limit))
        for base in _BASES:
            merged_per_base[base] = min(merged_per_base.get(base, limit), limit)
    if merged_per_base:
        entry["per_base_max"] = merged_per_base
    # Keep global max_run only when it was explicitly provided.
    # Do not derive a global cap from per-base minima.
    if limit is not None and not merged_per_base:
        entry["max_run"] = limit
        target["homopolymer_max_run"] = limit
    else:
        entry.pop("max_run", None)
        if "homopolymer_max_run" in target:
            target.pop("homopolymer_max_run", None)
    avoid_sequences = entry.get("avoid_sequences") or []
    derived_sequences: List[str] = []
    if merged_per_base:
        for base, max_run in merged_per_base.items():
            if max_run is None:
                continue
            derived_sequences.append(str(base) * (max_run + 1))
    elif entry.get("max_run") is not None:
        derived_sequences.extend(_build_homopolymer_sequences(entry["max_run"]))
    if extra_sequences:
        derived_sequences.extend(extra_sequences)
    combined = []
    for seq in avoid_sequences + derived_sequences:
        seq_norm = str(seq).upper()
        if seq_norm and seq_norm not in combined:
            combined.append(seq_norm)
    entry["avoid_sequences"] = combined


def _extract_homopolymer_from_constraints_obj(obj: Any) -> Optional[int]:
    if not isinstance(obj, dict):
        return None
    for key in (
        "homopolymer_max_run",
        "homopolymer_limit",
        "max_homopolymer_run",
        "max_homopolymer",
        "max_run",
    ):
        if key in obj:
            return _coerce_positive_int(obj.get(key))
    nested = obj.get("homopolymers") or obj.get("homopolymer")
    if isinstance(nested, dict):
        for key in ("max_run", "limit", "max"):
            if key in nested:
                return _coerce_positive_int(nested.get(key))
    return None


def _extract_allow_homopolymer_runs_from_constraints_obj(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for key in (
        "allow_homopolymer_runs",
        "allow_homopolymers",
        "allow_homopolymer",
        "allow_all_homopolymers",
    ):
        if _coerce_constraint_bool(obj.get(key)) is True:
            return True
    nested = obj.get("homopolymers") or obj.get("homopolymer")
    if isinstance(nested, dict):
        return _extract_allow_homopolymer_runs_from_constraints_obj(nested)
    return False


def _extract_per_base_homopolymer_from_constraints_obj(obj: Any) -> Dict[str, int]:
    if not isinstance(obj, dict):
        return {}
    limits: Dict[str, int] = {}

    def _merge_limit(base_value: Any, limit_value: Any) -> None:
        base = str(base_value or "").strip().upper()
        if base not in _BASES:
            return
        limit = _coerce_nonnegative_int(limit_value)
        if limit is None:
            return
        current = limits.get(base)
        if current is None or limit < current:
            limits[base] = limit

    def _ingest_runs(runs: Any) -> None:
        if isinstance(runs, dict) and any(str(key).strip().upper() in _BASES for key in runs):
            for base, value in runs.items():
                _merge_limit(base, value)
            return
        entries = [runs] if isinstance(runs, dict) else runs
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            base = (
                entry.get("base")
                or entry.get("nucleotide")
                or entry.get("nt")
                or entry.get("letter")
            )
            max_allowed = (
                entry.get("max_allowed_run_length")
                or entry.get("max_run_length")
                or entry.get("max_run")
                or entry.get("max")
                or entry.get("limit")
            )
            if max_allowed is None:
                min_disallowed = (
                    entry.get("min_disallowed_run_length")
                    or entry.get("min_run_length")
                    or entry.get("min_excluded_run_length")
                    or entry.get("excluded_run_length")
                    or entry.get("run_length")
                )
                min_disallowed_int = _coerce_positive_int(min_disallowed)
                if min_disallowed_int is not None:
                    max_allowed = max(0, min_disallowed_int - 1)
            _merge_limit(base, max_allowed)

    _ingest_runs(obj.get("disallowed_homopolymer_runs"))
    _ingest_runs(obj.get("homopolymer_runs"))
    _ingest_runs(obj.get("homopolymer_exclusions"))
    _ingest_runs(obj.get("exclude_homopolymers"))
    _ingest_runs(obj.get("excluded_homopolymers"))
    _ingest_runs(obj.get("max_homopolymer_runs"))
    _ingest_runs(obj.get("max_homopolymer_run_lengths"))
    _ingest_runs(obj.get("homopolymer_run_limits"))

    sequence_constraints = obj.get("sequence_constraints")
    if isinstance(sequence_constraints, dict):
        _ingest_runs(sequence_constraints.get("disallowed_homopolymer_runs"))
        _ingest_runs(sequence_constraints.get("homopolymer_runs"))
        _ingest_runs(sequence_constraints.get("homopolymer_exclusions"))
        _ingest_runs(sequence_constraints.get("exclude_homopolymers"))
        _ingest_runs(sequence_constraints.get("excluded_homopolymers"))
        _ingest_runs(sequence_constraints.get("max_homopolymer_runs"))
        _ingest_runs(sequence_constraints.get("max_homopolymer_run_lengths"))
        _ingest_runs(sequence_constraints.get("homopolymer_run_limits"))
    elif isinstance(sequence_constraints, list):
        for item in sequence_constraints:
            if isinstance(item, dict):
                _ingest_runs(item.get("disallowed_homopolymer_runs"))
                _ingest_runs(item.get("homopolymer_runs"))
                _ingest_runs(item.get("homopolymer_exclusions"))
                _ingest_runs(item.get("exclude_homopolymers"))
                _ingest_runs(item.get("excluded_homopolymers"))
                _ingest_runs(item.get("max_homopolymer_runs"))
                _ingest_runs(item.get("max_homopolymer_run_lengths"))
                _ingest_runs(item.get("homopolymer_run_limits"))

    return limits


def get_homopolymer_sequences(constraints: Optional[Dict[str, Any]]) -> List[str]:
    """Return a deduplicated list of homopolymer sequences to avoid based on constraints."""
    sequences: List[str] = []
    if not isinstance(constraints, dict):
        return sequences
    entry = constraints.get("homopolymers")
    if isinstance(entry, dict):
        per_base_limits = _normalize_per_base_limits(entry.get("per_base_max"))
        for base, max_run in per_base_limits.items():
            sequences.append(str(base) * (max_run + 1))
        seqs = entry.get("avoid_sequences")
        if isinstance(seqs, list):
            for seq in seqs:
                seq_norm = str(seq).upper()
                if seq_norm:
                    sequences.append(seq_norm)
    extra = constraints.get("avoid_sequences")
    if isinstance(extra, list):
        for seq in extra:
            seq_norm = str(seq).upper()
            if seq_norm:
                sequences.append(seq_norm)
    limit = _coerce_positive_int(constraints.get("homopolymer_max_run"))
    if limit is None and isinstance(entry, dict):
        limit = _coerce_positive_int(entry.get("max_run"))
    if limit is not None:
        sequences.extend(_build_homopolymer_sequences(limit))
    # Preserve order but remove duplicates
    deduped: List[str] = []
    for seq in sequences:
        if seq not in deduped:
            deduped.append(seq)
    return deduped


def _extract_gc_from_constraints_obj(obj: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(obj, dict):
        return None, None

    def _coerce_range_pair(value: Any) -> Tuple[Optional[float], Optional[float]]:
        if isinstance(value, dict):
            return _extract_gc_from_constraints_obj(value)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return _coerce_fraction(value[0]), _coerce_fraction(value[1])
        if isinstance(value, str):
            match = re.search(
                r"(?P<min>\d+(?:\.\d+)?\s*%?)\s*(?:-|to|through|and)\s*(?P<max>\d+(?:\.\d+)?\s*%?)",
                value,
                flags=re.IGNORECASE,
            )
            if match:
                return _coerce_fraction(match.group("min")), _coerce_fraction(match.group("max"))
        return None, None

    def _try_keys(container: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
        for key in keys:
            if key in container:
                val = _coerce_fraction(container.get(key))
                if val is not None:
                    return val
        return None

    range_min, range_max = (None, None)
    for key in (
        "gc_percent_range",
        "gc_range_percent",
        "gc_content_percent_range",
        "gc_content_range_percent",
        "gc_content_range",
        "gc_range",
        "gc_bounds",
        "gc_content_bounds",
        "gc",
    ):
        if key not in obj:
            continue
        range_min, range_max = _coerce_range_pair(obj.get(key))
        if range_min is not None or range_max is not None:
            break

    min_val = _try_keys(
        obj,
        (
            "gc_min",
            "min",
            "lower",
            "gc_min_percent",
            "min_percent",
        ),
    )
    if min_val is None:
        min_val = _try_keys(
            obj,
            (
                "gc_content_min",
                "gc_content_min_fraction",
                "gc_min_fraction",
                "gc_minimum",
                "min_exclusive",
                "min_fraction_exclusive",
            ),
        )
    if min_val is None:
        min_val = _try_keys(obj, ("gc_lower",))
    if min_val is None:
        min_val = range_min

    max_val = _try_keys(
        obj,
        (
            "gc_max",
            "max",
            "upper",
            "gc_max_percent",
            "max_percent",
        ),
    )
    if max_val is None:
        max_val = _try_keys(
            obj,
            (
                "gc_content_max",
                "gc_content_max_fraction",
                "gc_max_fraction",
                "gc_maximum",
                "max_exclusive",
                "max_fraction_exclusive",
            ),
        )
    if max_val is None:
        max_val = _try_keys(obj, ("gc_upper",))
    if max_val is None:
        max_val = range_max

    if min_val is None or max_val is None:
        nested = (
            obj.get("gc_content")
            or obj.get("gc_content_constraint")
            or obj.get("gc_content_fraction")
            or obj.get("gc_fraction")
            or obj.get("gc_content_range")
            or obj.get("gc_range")
            or obj.get("gc")
        )
        if isinstance(nested, dict):
            nested_min, nested_max = _extract_gc_from_constraints_obj(nested)
            if min_val is None:
                min_val = nested_min
            if max_val is None:
                max_val = nested_max

    return min_val, max_val


def _extract_gc_inclusive_flags_from_constraints_obj(obj: Any) -> Tuple[Optional[bool], Optional[bool]]:
    if not isinstance(obj, dict):
        return None, None

    min_inclusive = _coerce_constraint_bool(obj.get("min_inclusive"))
    if min_inclusive is None:
        min_inclusive = _coerce_constraint_bool(obj.get("gc_min_inclusive"))
    if min_inclusive is None and any(
        key in obj and _coerce_fraction(obj.get(key)) is not None
        for key in (
            "min_exclusive",
            "min_fraction_exclusive",
            "gc_min_exclusive",
            "gc_fraction_min_exclusive",
            "gc_min_fraction_exclusive",
        )
    ):
        min_inclusive = False

    max_inclusive = _coerce_constraint_bool(obj.get("max_inclusive"))
    if max_inclusive is None:
        max_inclusive = _coerce_constraint_bool(obj.get("gc_max_inclusive"))
    if max_inclusive is None and any(
        key in obj and _coerce_fraction(obj.get(key)) is not None
        for key in (
            "max_exclusive",
            "max_fraction_exclusive",
            "gc_max_exclusive",
            "gc_fraction_max_exclusive",
            "gc_max_fraction_exclusive",
        )
    ):
        max_inclusive = False

    nested = (
        obj.get("gc_content")
        or obj.get("gc_content_constraint")
        or obj.get("gc_content_fraction")
        or obj.get("gc_fraction")
        or obj.get("gc_content_range")
        or obj.get("gc_range")
        or obj.get("gc")
    )
    if isinstance(nested, dict):
        nested_min_inclusive, nested_max_inclusive = _extract_gc_inclusive_flags_from_constraints_obj(nested)
        if min_inclusive is None:
            min_inclusive = nested_min_inclusive
        if max_inclusive is None:
            max_inclusive = nested_max_inclusive
    return min_inclusive, max_inclusive


def _extract_binding_from_constraints_obj(obj: Any) -> Optional[int]:
    if isinstance(obj, dict):
        for key in (
            "no_exact_off_target_gene_hits",
            "no_exact_offtarget_gene_hits",
            "no_exact_off_target_hits",
            "no_exact_offtarget_hits",
            "no_off_target_gene_hits",
            "no_offtarget_gene_hits",
            "no_off_target_hits",
            "no_offtarget_hits",
            "zero_exact_off_target_gene_hits",
            "zero_exact_offtarget_gene_hits",
            "zero_off_target_hits",
            "zero_offtarget_hits",
        ):
            if _coerce_constraint_bool(obj.get(key)) is True:
                return 0
        for key in (
            "max_binding_sites",
            "max_sites",
            "times_mapped_max",
            "max_times_mapped",
            "max_offtarget_mappings",
            "max_off_target_mappings",
            "max_offtarget_mapping",
            "max_off_target_mapping",
            "max_offtarget_sites",
            "max_off_target_sites",
            "max_offtargets",
            "max_off_targets",
            "max_total_mappings",
            "max_total_mapping",
            "max_mappings",
            "max_mapping",
            "total_mappings_max",
            "max",
        ):
            if key in obj:
                return _word_to_int(obj.get(key))
        nested = (
            obj.get("binding_sites")
            or obj.get("binding")
            or obj.get("offtarget_mappings")
            or obj.get("off_target_mappings")
            or obj.get("offtarget_constraints")
            or obj.get("off_target_constraints")
            or obj.get("offtarget_filtering")
            or obj.get("off_target_filtering")
            or obj.get("offtargets")
            or obj.get("off_targets")
            or obj.get("off_target_policy")
            or obj.get("offtarget_policy")
            or obj.get("mapping_requirements")
            or obj.get("mapping_policy")
        )
        if isinstance(nested, dict):
            return _extract_binding_from_constraints_obj(nested)
    return None


_OFFTARGET_NONE_PATTERN = re.compile(
    r"\b(?:no|zero|without)\s+"
    r"(?:exact\s+)?(?:off[-\s]?target\s+)?(?:gene\s+)?"
    r"(?:hits?|mappings?|binding\s+sites?)\b",
    re.IGNORECASE,
)


def _extract_binding_from_texts(texts: List[str]) -> Optional[int]:
    max_sites: Optional[int] = None
    for text in texts:
        if not text:
            continue
        if _OFFTARGET_NONE_PATTERN.search(text):
            max_sites = 0
            break
        match = _BINDING_MAX_PATTERN.search(text)
        if match:
            candidate = _word_to_int(match.group("value"))
            if candidate is not None:
                max_sites = candidate
                break
        match = _OFFTARGET_MAX_PATTERN.search(text)
        if match:
            candidate = _word_to_int(match.group("value"))
            if candidate is not None:
                max_sites = candidate
                break
        match = _OFFTARGET_LESS_THAN_PATTERN.search(text)
        if match:
            candidate = _word_to_int(match.group("value"))
            if candidate is not None:
                max_sites = max(0, candidate - 1)
                break
        match = _BINDING_EXACT_PATTERN.search(text)
        if match:
            candidate = _word_to_int(match.group("value"))
            if candidate is not None:
                max_sites = candidate
                break
    if max_sites is None:
        for text in texts:
            if not text:
                continue
            match = _BINDING_MORE_PATTERN.search(text)
            if match:
                candidate = _word_to_int(match.group("value"))
                if candidate is not None:
                    max_sites = candidate
                    break
    return max_sites


def _extract_gc_from_texts(texts: List[str]) -> Tuple[Optional[float], Optional[float], bool, bool]:
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        match = _GC_GT_LT_PATTERN.search(text)
        if match:
            min_val = _coerce_fraction(match.group("min"))
            max_val = _coerce_fraction(match.group("max"))
            if min_val is not None or max_val is not None:
                return min_val, max_val, True, True
        match = _GC_BETWEEN_PATTERN.search(text)
        if match:
            min_val = _coerce_fraction(match.group("min"))
            max_val = _coerce_fraction(match.group("max"))
            if min_val is not None or max_val is not None:
                return min_val, max_val, False, False
        match = _GC_EXACT_PATTERN.search(text)
        if match:
            val = _coerce_fraction(match.group("value"))
            if val is not None:
                return val, val, False, False
    return None, None, False, False


def _extract_homopolymer_from_texts(texts: List[str]) -> Dict[str, int]:
    limits: Dict[str, int] = {}
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        for match in _HOMOPOLYMER_EXCLUDE_PATTERN.finditer(text):
            base = str(match.group("base") or "").strip().upper()
            if base not in _BASES:
                continue
            value = _word_to_int(match.group("value"))
            if value is None:
                continue
            op = str(match.group("op") or "").lower()
            if op in {"greater than", "more than", "over", ">"}:
                max_allowed = value
            else:
                max_allowed = max(0, value - 1)
            current = limits.get(base)
            if current is None or max_allowed < current:
                limits[base] = max_allowed
    return limits


def parse_parameter_constraints(user_request: str, structured_goal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract user-driven parameter constraints (e.g., GC range, binding site limits)."""
    constraints: Dict[str, Any] = {}

    def _merge_constraint_object(obj: Any, source: str) -> None:
        if not isinstance(obj, dict):
            return
        gc_from_obj = _extract_gc_from_constraints_obj(obj)
        min_inclusive, max_inclusive = _extract_gc_inclusive_flags_from_constraints_obj(obj)
        _merge_gc_constraint(
            constraints,
            gc_from_obj[0],
            gc_from_obj[1],
            source,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
        )
        _merge_binding_constraint(constraints, _extract_binding_from_constraints_obj(obj), source)
        homo_obj = obj.get("homopolymers") or obj.get("homopolymer")
        extra_sequences = None
        per_base_limits = None
        if isinstance(homo_obj, dict):
            extra_sequences = homo_obj.get("avoid_sequences")
            per_base_limits = _normalize_per_base_limits(homo_obj.get("per_base_max"))
        if extra_sequences is None and isinstance(obj.get("avoid_sequences"), list):
            extra_sequences = obj.get("avoid_sequences")
        if per_base_limits is None:
            per_base_limits = _normalize_per_base_limits(obj.get("homopolymer_limits"))
        explicit_per_base = _extract_per_base_homopolymer_from_constraints_obj(obj)
        if explicit_per_base:
            combined_per_base = dict(per_base_limits or {})
            for base, limit in explicit_per_base.items():
                current = combined_per_base.get(base)
                if current is None or limit < current:
                    combined_per_base[base] = limit
            per_base_limits = combined_per_base
        _merge_homopolymer_constraint(
            constraints,
            _extract_homopolymer_from_constraints_obj(obj),
            source,
            extra_sequences if isinstance(extra_sequences, list) else None,
            per_base_limits,
        )
        if _extract_allow_homopolymer_runs_from_constraints_obj(obj):
            constraints["allow_homopolymer_runs"] = True

    # 1) Existing normalized constraints on the goal
    if structured_goal:
        existing = structured_goal.get("parameter_constraints")
        if isinstance(existing, dict):
            gc_entry = existing.get("gc_content") or {}
            min_val = _coerce_fraction(gc_entry.get("min"))
            max_val = _coerce_fraction(gc_entry.get("max"))
            if existing.get("gc_min") is not None:
                min_val = _coerce_fraction(existing.get("gc_min"))
            if existing.get("gc_max") is not None:
                max_val = _coerce_fraction(existing.get("gc_max"))
            min_inclusive, max_inclusive = _extract_gc_inclusive_flags_from_constraints_obj(existing)
            _merge_gc_constraint(
                constraints,
                min_val,
                max_val,
                "structured_goal.parameter_constraints",
                min_inclusive=min_inclusive,
                max_inclusive=max_inclusive,
            )

            binding_max = existing.get("max_binding_sites")
            if isinstance(binding_max, dict):
                binding_val = _word_to_int(binding_max.get("value") or binding_max.get("max"))
            else:
                binding_val = _word_to_int(binding_max)
            _merge_binding_constraint(constraints, binding_val, "structured_goal.parameter_constraints")
            homo_entry = existing.get("homopolymers")
            extra_sequences = None
            limit_from_entry: Optional[int] = None
            per_base_limits = None
            if isinstance(homo_entry, dict):
                extra_sequences = homo_entry.get("avoid_sequences")
                limit_from_entry = _coerce_positive_int(homo_entry.get("max_run"))
                per_base_limits = _normalize_per_base_limits(homo_entry.get("per_base_max"))
            if limit_from_entry is None:
                limit_from_entry = _extract_homopolymer_from_constraints_obj(existing)
            _merge_homopolymer_constraint(
                constraints,
                limit_from_entry,
                "structured_goal.parameter_constraints",
                extra_sequences if isinstance(extra_sequences, list) else None,
                per_base_limits,
            )
            if _extract_allow_homopolymer_runs_from_constraints_obj(existing):
                constraints["allow_homopolymer_runs"] = True

    # 2) Top-level fields on structured_goal
    if structured_goal:
        top_gc_min = _coerce_fraction(structured_goal.get("gc_min"))
        top_gc_max = _coerce_fraction(structured_goal.get("gc_max"))
        _merge_gc_constraint(constraints, top_gc_min, top_gc_max, "structured_goal.top_level")
        top_gc_from_obj = _extract_gc_from_constraints_obj(structured_goal)
        top_gc_min_inclusive, top_gc_max_inclusive = _extract_gc_inclusive_flags_from_constraints_obj(structured_goal)
        _merge_gc_constraint(
            constraints,
            top_gc_from_obj[0],
            top_gc_from_obj[1],
            "structured_goal.top_level",
            min_inclusive=top_gc_min_inclusive,
            max_inclusive=top_gc_max_inclusive,
        )

        binding_val = _word_to_int(
            structured_goal.get("max_binding_sites")
            or structured_goal.get("times_mapped_max")
            or structured_goal.get("max_offtarget_mappings")
            or structured_goal.get("max_off_target_mappings")
            or structured_goal.get("max_offtargets")
            or structured_goal.get("max_off_targets")
            or structured_goal.get("max_total_mappings")
            or structured_goal.get("max_mappings")
            or structured_goal.get("total_mappings_max")
        )
        _merge_binding_constraint(constraints, binding_val, "structured_goal.top_level")
        top_homo_limit = _coerce_positive_int(structured_goal.get("homopolymer_max_run"))
        homo_entry = structured_goal.get("homopolymers")
        homo_sequences = homo_entry.get("avoid_sequences") if isinstance(homo_entry, dict) else None
        per_base_limits = _normalize_per_base_limits(homo_entry.get("per_base_max")) if isinstance(homo_entry, dict) else None
        _merge_homopolymer_constraint(
            constraints,
            top_homo_limit,
            "structured_goal.top_level",
            homo_sequences if isinstance(homo_sequences, list) else None,
            per_base_limits,
        )
        if _extract_allow_homopolymer_runs_from_constraints_obj(structured_goal):
            constraints["allow_homopolymer_runs"] = True

        sg_constraints = structured_goal.get("constraints")
        if isinstance(sg_constraints, dict):
            _merge_constraint_object(sg_constraints, "structured_goal.constraints")
        elif isinstance(sg_constraints, list):
            for item in sg_constraints:
                if isinstance(item, dict):
                    _merge_constraint_object(item, "structured_goal.constraints_list")
        for block_name in ("filters", "sequence_filters", "sequence_constraints", "design_constraints", "filter_constraints", "candidate_filters"):
            block = structured_goal.get(block_name)
            if isinstance(block, dict):
                _merge_constraint_object(block, f"structured_goal.{block_name}")
            elif isinstance(block, list):
                for item in block:
                    _merge_constraint_object(item, f"structured_goal.{block_name}")
        for block_name in (
            "offtarget_constraints",
            "off_target_constraints",
            "offtarget_filtering",
            "off_target_filtering",
        ):
            block = structured_goal.get(block_name)
            if isinstance(block, dict):
                _merge_constraint_object(block, f"structured_goal.{block_name}")
            elif isinstance(block, list):
                for item in block:
                    _merge_constraint_object(item, f"structured_goal.{block_name}")
        programs = structured_goal.get("design_programs")
        if isinstance(programs, list):
            for program in programs:
                _merge_constraint_object(program, "structured_goal.design_programs")

    # 3) Natural language text (user request + optional constraint text)
    text_sources: List[str] = []
    if user_request:
        text_sources.append(user_request)
    if structured_goal:
        sg_constraints = structured_goal.get("constraints")
        if isinstance(sg_constraints, str):
            text_sources.append(sg_constraints)
        elif isinstance(sg_constraints, list):
            for item in sg_constraints:
                if isinstance(item, str):
                    text_sources.append(item)

    binding_from_text = _extract_binding_from_texts(text_sources)
    _merge_binding_constraint(constraints, binding_from_text, "user_text")
    gc_min_text, gc_max_text, gc_min_exclusive, gc_max_exclusive = _extract_gc_from_texts(text_sources)
    _merge_gc_constraint(
        constraints,
        gc_min_text,
        gc_max_text,
        "user_text",
        min_inclusive=False if gc_min_exclusive else None,
        max_inclusive=False if gc_max_exclusive else None,
    )
    homopolymer_from_text = _extract_homopolymer_from_texts(text_sources)
    _merge_homopolymer_constraint(
        constraints,
        None,
        "user_text",
        None,
        homopolymer_from_text,
    )

    _finalize_canonical_parameter_constraints(constraints)
    return constraints


def _dedupe_sources(values: List[Any]) -> List[str]:
    sources: List[str] = []
    for value in values:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [str(item) for item in value if str(item).strip()]
        else:
            continue
        for candidate in candidates:
            text = str(candidate).strip()
            if text and text not in sources:
                sources.append(text)
    return sources


def _finalize_canonical_parameter_constraints(constraints: Dict[str, Any]) -> None:
    """Collapse accepted aliases into the exact parameter constraint schema."""
    if not isinstance(constraints, dict):
        return

    min_val, max_val = _collect_gc_bounds_from_constraints(constraints)
    min_inclusive, max_inclusive = _extract_gc_inclusive_flags_from_constraints_obj(constraints)
    if min_val is not None and max_val is not None and min_val > max_val:
        min_val, max_val = max_val, min_val
        min_inclusive, max_inclusive = max_inclusive, min_inclusive

    gc_sources: List[Any] = []
    for key in ("gc_content", "gc_content_constraint", "gc_content_fraction", "gc_fraction", "gc_content_range", "gc_range", "gc"):
        entry = constraints.get(key)
        if isinstance(entry, dict):
            gc_sources.append(entry.get("sources"))
    if isinstance(constraints.get("gc_content"), dict):
        gc_sources.append(constraints["gc_content"].get("source"))
    sources = _dedupe_sources(gc_sources)

    for key in (
        "gc_content",
        "gc_content_constraint",
        "gc_content_fraction",
        "gc_content_constraint",
        "gc_fraction",
        "gc_content_range",
        "gc_content_constraint",
        "gc_range",
        "gc",
        "gc_bounds",
        "gcFraction",
        "gcFractionBounds",
        "gc_fraction_min_exclusive",
        "gc_fraction_max_exclusive",
        "gc_min_fraction_exclusive",
        "gc_max_fraction_exclusive",
        "min_fraction_exclusive",
        "max_fraction_exclusive",
        "gc_min_exclusive",
        "gc_max_exclusive",
        "gc_content_min",
        "gc_content_max",
        "gc_content_min_fraction",
        "gc_content_max_fraction",
        "gc_min_fraction",
        "gc_max_fraction",
        "gc_minimum",
        "gc_maximum",
        "min_exclusive",
        "max_exclusive",
        "min_inclusive",
        "max_inclusive",
    ):
        constraints.pop(key, None)

    for key in ("gc_min", "gc_max", "gc_min_percent", "gc_max_percent"):
        constraints.pop(key, None)

    if min_val is not None or max_val is not None:
        gc_entry: Dict[str, Any] = {}
        if sources:
            gc_entry["sources"] = sources
        if min_val is not None:
            gc_entry["min"] = float(min_val)
            gc_entry["min_percent"] = round(float(min_val) * 100.0, 2)
            gc_entry["min_inclusive"] = bool(True if min_inclusive is None else min_inclusive)
            constraints["gc_min"] = float(min_val)
            constraints["gc_min_percent"] = gc_entry["min_percent"]
        if max_val is not None:
            gc_entry["max"] = float(max_val)
            gc_entry["max_percent"] = round(float(max_val) * 100.0, 2)
            gc_entry["max_inclusive"] = bool(True if max_inclusive is None else max_inclusive)
            constraints["gc_max"] = float(max_val)
            constraints["gc_max_percent"] = gc_entry["max_percent"]
        constraints["gc_content"] = gc_entry

    binding_val = _extract_binding_from_constraints_obj(constraints)
    if binding_val is not None:
        _merge_binding_constraint(constraints, binding_val, "canonicalization")
    for key in (
        "times_mapped_max",
        "max_offtarget_mappings",
        "max_off_target_mappings",
        "offtarget_mappings",
        "off_target_mappings",
        "max_offtarget_mapping",
        "max_off_target_mapping",
        "max_offtarget_sites",
        "max_off_target_sites",
        "max_offtargets",
        "max_off_targets",
        "max_total_mappings",
        "max_total_mapping",
        "max_mappings",
        "max_mapping",
        "total_mappings_max",
    ):
        constraints.pop(key, None)
    if constraints.get("max_binding_sites") is not None:
        binding_limit = _word_to_int(constraints.get("max_binding_sites"))
        if binding_limit is not None:
            constraints["max_binding_sites"] = binding_limit
            binding_entry = constraints.setdefault("binding_sites", {})
            if isinstance(binding_entry, dict):
                binding_entry["max"] = binding_limit

    allow_value = (
        _coerce_constraint_bool(constraints.get("allow_homopolymer_runs"))
        or _coerce_constraint_bool(constraints.get("allow_homopolymers"))
        or _coerce_constraint_bool(constraints.get("allow_homopolymer"))
        or _coerce_constraint_bool(constraints.get("allow_all_homopolymers"))
    )
    for key in ("allow_homopolymers", "allow_homopolymer", "allow_all_homopolymers"):
        constraints.pop(key, None)
    if allow_value is True:
        constraints["allow_homopolymer_runs"] = True

    homo_entry = constraints.get("homopolymers")
    if isinstance(homo_entry, dict):
        per_base = _normalize_per_base_limits(homo_entry.get("per_base_max"))
        if per_base:
            homo_entry["per_base_max"] = per_base
        else:
            homo_entry.pop("per_base_max", None)
        avoid_sequences = []
        for seq in homo_entry.get("avoid_sequences") or []:
            seq_norm = str(seq).strip().upper()
            if seq_norm and seq_norm not in avoid_sequences:
                avoid_sequences.append(seq_norm)
        if avoid_sequences:
            homo_entry["avoid_sequences"] = avoid_sequences
        else:
            homo_entry.pop("avoid_sequences", None)
        if not homo_entry:
            constraints.pop("homopolymers", None)
    constraints.pop("homopolymer", None)


def apply_parameter_constraints_to_goal(structured_goal: Dict[str, Any], user_request: str) -> Dict[str, Any]:
    """Parse constraints and attach normalized values onto structured_goal."""
    constraints = parse_parameter_constraints(user_request, structured_goal)
    structured_goal["parameter_constraints"] = constraints
    gc_entry = constraints.get("gc_content")
    if isinstance(gc_entry, dict):
        if gc_entry.get("min") is not None:
            structured_goal["gc_min"] = gc_entry["min"]
        if gc_entry.get("max") is not None:
            structured_goal["gc_max"] = gc_entry["max"]
    max_binding = constraints.get("max_binding_sites")
    if isinstance(max_binding, dict):
        value = _word_to_int(max_binding.get("value") or max_binding.get("max"))
    else:
        value = _word_to_int(max_binding)
    if value is not None:
        structured_goal["max_binding_sites"] = value
    else:
        structured_goal.pop("max_binding_sites", None)
    homo_entry = constraints.get("homopolymers")
    homopolymer_limit = _coerce_positive_int(constraints.get("homopolymer_max_run"))
    if homopolymer_limit is None and isinstance(homo_entry, dict):
        homopolymer_limit = _coerce_positive_int(homo_entry.get("max_run"))
    per_base_limits = get_homopolymer_per_base_limits(
        constraints,
        fallback_default=DEFAULT_HOMOPOLYMER_LIMIT,
    )
    if homopolymer_limit is not None:
        structured_goal["homopolymer_max_run"] = homopolymer_limit
    else:
        structured_goal.pop("homopolymer_max_run", None)
    if per_base_limits:
        homo_block = structured_goal.setdefault("homopolymers", {})
        homo_block["per_base_max"] = per_base_limits
        if homopolymer_limit is not None:
            homo_block["max_run"] = homopolymer_limit
        else:
            homo_block.pop("max_run", None)
        structured_goal["homopolymer_limits"] = per_base_limits
    else:
        structured_goal.pop("homopolymer_limits", None)
    if constraints.get("allow_homopolymer_runs") is True:
        structured_goal["allow_homopolymer_runs"] = True
    _strip_parameter_constraint_aliases_from_goal(structured_goal)
    return structured_goal


def _strip_parameter_constraint_aliases_from_goal(structured_goal: Dict[str, Any]) -> None:
    """Remove non-canonical executable constraint keys after parameter normalization."""
    if not isinstance(structured_goal, dict):
        return

    top_level_aliases = (
        "times_mapped_max",
        "max_offtarget_mappings",
        "max_off_target_mappings",
        "max_offtarget_mapping",
        "max_off_target_mapping",
        "max_offtarget_sites",
        "max_off_target_sites",
        "max_offtargets",
        "max_off_targets",
        "max_total_mappings",
        "max_total_mapping",
        "max_mappings",
        "max_mapping",
        "total_mappings_max",
        "gc_content_fraction",
        "gc_content_constraint",
        "gc_fraction",
        "gc_content_range",
        "gc_range",
        "gc_bounds",
        "gcFraction",
        "gcFractionBounds",
        "gc_fraction_min_exclusive",
        "gc_fraction_max_exclusive",
        "gc_min_fraction_exclusive",
        "gc_max_fraction_exclusive",
        "min_fraction_exclusive",
        "max_fraction_exclusive",
        "gc_min_exclusive",
        "gc_max_exclusive",
        "gc_content_min",
        "gc_content_max",
        "gc_content_min_fraction",
        "gc_content_max_fraction",
        "gc_min_fraction",
        "gc_max_fraction",
        "gc_minimum",
        "gc_maximum",
        "min_exclusive",
        "max_exclusive",
        "allow_homopolymers",
        "allow_homopolymer",
        "allow_all_homopolymers",
    )
    for key in top_level_aliases:
        structured_goal.pop(key, None)

    executable_constraint_keys = set(top_level_aliases) | {
        "gc_content",
        "gc_min",
        "gc_max",
        "gc_min_percent",
        "gc_max_percent",
        "min",
        "max",
        "lower",
        "upper",
        "min_percent",
        "max_percent",
        "min_inclusive",
        "max_inclusive",
        "max_binding_sites",
        "binding_sites",
        "max_sites",
        "binding",
        "offtarget_mappings",
        "off_target_mappings",
        "offtarget_constraints",
        "off_target_constraints",
        "offtarget_filtering",
        "off_target_filtering",
        "offtargets",
        "off_targets",
        "homopolymer_max_run",
        "homopolymer_limit",
        "max_homopolymer_run",
        "max_homopolymer",
        "max_run",
        "homopolymers",
        "homopolymer",
        "homopolymer_limits",
        "avoid_sequences",
        "disallowed_homopolymer_runs",
        "homopolymer_runs",
        "homopolymer_exclusions",
        "exclude_homopolymers",
        "excluded_homopolymers",
        "allow_homopolymer_runs",
    }

    def _clean_block(block: Dict[str, Any]) -> bool:
        for key in list(block.keys()):
            if key in executable_constraint_keys:
                block.pop(key, None)
        return bool(block)

    for block_name in (
        "constraints",
        "filters",
        "sequence_filters",
        "sequence_constraints",
        "design_constraints",
        "filter_constraints",
        "candidate_filters",
        "offtarget_constraints",
        "off_target_constraints",
        "offtarget_filtering",
        "off_target_filtering",
    ):
        block = structured_goal.get(block_name)
        if isinstance(block, dict):
            if not _clean_block(block):
                structured_goal.pop(block_name, None)
        elif isinstance(block, list):
            cleaned = []
            for item in block:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    if _clean_block(item_copy):
                        cleaned.append(item_copy)
                elif item:
                    cleaned.append(item)
            if cleaned:
                structured_goal[block_name] = cleaned
            else:
                structured_goal.pop(block_name, None)

    programs = structured_goal.get("design_programs")
    if isinstance(programs, list):
        cleaned_programs = []
        for program in programs:
            if not isinstance(program, dict):
                continue
            program_copy = dict(program)
            _clean_block(program_copy)
            cleaned_programs.append(program_copy)
        structured_goal["design_programs"] = cleaned_programs


def _collect_gc_bounds_from_constraints(constraints: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return explicit GC min/max fractions when present, or (None, None)."""
    if not isinstance(constraints, dict):
        return None, None

    gc_entry = constraints.get("gc_content")
    if not isinstance(gc_entry, dict):
        alt_entry = constraints.get("gc_content_fraction")
        if isinstance(alt_entry, dict):
            gc_entry = alt_entry
    if not isinstance(gc_entry, dict):
        alt_entry = constraints.get("gc_fraction")
        if isinstance(alt_entry, dict):
            gc_entry = alt_entry

    min_val = _coerce_fraction(
        constraints.get("gc_min")
        if constraints.get("gc_min") is not None
        else constraints.get("gc_fraction_min_exclusive")
        if constraints.get("gc_fraction_min_exclusive") is not None
        else constraints.get("gc_min_fraction_exclusive")
        if constraints.get("gc_min_fraction_exclusive") is not None
        else constraints.get("min_fraction_exclusive")
    )
    max_val = _coerce_fraction(
        constraints.get("gc_max")
        if constraints.get("gc_max") is not None
        else constraints.get("gc_fraction_max_exclusive")
        if constraints.get("gc_fraction_max_exclusive") is not None
        else constraints.get("gc_max_fraction_exclusive")
        if constraints.get("gc_max_fraction_exclusive") is not None
        else constraints.get("max_fraction_exclusive")
    )

    if min_val is None and isinstance(gc_entry, dict):
        min_val = _coerce_fraction(gc_entry.get("min"))
        if min_val is None:
            min_val = _coerce_fraction(gc_entry.get("lower"))
        if min_val is None:
            min_val = _coerce_fraction(gc_entry.get("gc_min_percent"))
        if min_val is None:
            min_val = _coerce_fraction(gc_entry.get("min_percent"))
        if min_val is None:
            min_val = _coerce_fraction(gc_entry.get("min_exclusive"))
        if min_val is None:
            min_val = _coerce_fraction(gc_entry.get("min_fraction_exclusive"))
        if min_val is None:
            nested = gc_entry.get("bounds")
            if isinstance(nested, dict):
                min_val = _coerce_fraction(nested.get("min"))
                if min_val is None:
                    min_val = _coerce_fraction(nested.get("lower"))
    if max_val is None and isinstance(gc_entry, dict):
        max_val = _coerce_fraction(gc_entry.get("max"))
        if max_val is None:
            max_val = _coerce_fraction(gc_entry.get("upper"))
        if max_val is None:
            max_val = _coerce_fraction(gc_entry.get("gc_max_percent"))
        if max_val is None:
            max_val = _coerce_fraction(gc_entry.get("max_percent"))
        if max_val is None:
            max_val = _coerce_fraction(gc_entry.get("max_exclusive"))
        if max_val is None:
            max_val = _coerce_fraction(gc_entry.get("max_fraction_exclusive"))
        if max_val is None:
            nested = gc_entry.get("bounds")
            if isinstance(nested, dict):
                max_val = _coerce_fraction(nested.get("max"))
                if max_val is None:
                    max_val = _coerce_fraction(nested.get("upper"))

    if min_val is None:
        fraction_entry = constraints.get("gc_content_fraction")
        if not isinstance(fraction_entry, dict):
            fraction_entry = constraints.get("gc_fraction")
        if isinstance(fraction_entry, dict):
            for key in ("min", "lower", "min_percent", "min_exclusive", "min_fraction_exclusive"):
                min_val = _coerce_fraction(fraction_entry.get(key))
                if min_val is not None:
                    break
    if max_val is None:
        fraction_entry = constraints.get("gc_content_fraction")
        if not isinstance(fraction_entry, dict):
            fraction_entry = constraints.get("gc_fraction")
        if isinstance(fraction_entry, dict):
            for key in ("max", "upper", "max_percent", "max_exclusive", "max_fraction_exclusive"):
                max_val = _coerce_fraction(fraction_entry.get(key))
                if max_val is not None:
                    break

    return min_val, max_val


def get_gc_inclusive_flags(constraints: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return whether explicit GC min/max bounds are inclusive."""
    min_val, max_val = _collect_gc_bounds_from_constraints(constraints)
    min_inclusive, max_inclusive = _extract_gc_inclusive_flags_from_constraints_obj(constraints)
    if min_val is not None and min_inclusive is None:
        min_inclusive = True
    if max_val is not None and max_inclusive is None:
        max_inclusive = True
    return bool(True if min_inclusive is None else min_inclusive), bool(True if max_inclusive is None else max_inclusive)


def get_gc_bounds(constraints: Dict[str, Any], default_min: float = 0.3, default_max: float = 0.65) -> Tuple[float, float]:
    """Return GC min/max fractions derived from constraints (defaults to 0.3-0.65).

    Accepts values expressed either as fractions (0-1), percentages (0-100), numeric
    strings, or nested within gc_content / gc_bounds blocks.
    """

    min_val, max_val = _collect_gc_bounds_from_constraints(constraints)

    if min_val is None:
        min_val = _coerce_fraction(default_min)
    if max_val is None:
        max_val = _coerce_fraction(default_max)

    if min_val is None:
        min_val = 0.0
    if max_val is None:
        max_val = 1.0

    if min_val > max_val:
        min_val, max_val = max_val, min_val

    return float(min_val), float(max_val)


def get_max_binding_sites(constraints: Dict[str, Any]) -> Optional[int]:
    value = constraints.get("max_binding_sites")
    if isinstance(value, dict):
        return _word_to_int(value.get("value") or value.get("max"))
    return _word_to_int(value)


def _count_list_field(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, list):
        return len([v for v in value if str(v).strip()]) if value else 0
    if isinstance(value, (int, float)):
        try:
            return int(round(float(value)))
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return 0
    if text in {"[]", "{}"}:
        return 0
    text = text.strip("[]")
    if not text:
        return 0
    parts = [p.strip() for p in re.split(r"[;,]+", text) if p.strip()]
    return len(parts)


def get_binding_count_from_row(row: Dict[str, Any]) -> Optional[int]:
    """Return the binding count used for hard off-target constraints.

    Gapmer ``Times_mapped`` is a legacy target-isoform hit count, not an
    other-gene off-target count. Prefer explicit off-target annotations when
    they are present, including a valid zero, and retain ``Times_mapped`` as a
    fallback for steric rows whose mapper exposes only that field.
    """
    if row.get("off_target_count_status") in {"incomplete", "unavailable"}:
        return None
    preferred_keys = (
        "off_target_mapping_count_lower_bound",
        "Times_mapped_off_target_genes",
        "times_mapped_off_target_genes",
        "off_target_gene_count",
        "Off_target_gene_count",
        "worst_species_off_target_gene_count",
    )
    fallback_keys = (
        "Times_mapped",
        "times_mapped",
        "Binding_sites",
        "binding_sites",
    )
    for keys in (preferred_keys, fallback_keys):
        for key in keys:
            if key not in row or row.get(key) in (None, ""):
                continue
            binding_val = _word_to_int(row.get(key))
            if binding_val is not None:
                return int(binding_val)
        if keys is preferred_keys:
            source_text = " ".join(
                str(row.get(key) or "")
                for key in ("source", "source_modality", "Source", "modality")
            ).lower()
            if "gapmer" in source_text:
                # Gapmer Times_mapped records target-isoform coverage. The
                # off-target count is populated by the later genomic mapper.
                return None
    return None


def evaluate_row_against_constraints(row: Dict[str, Any], constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check whether a row satisfies parameter constraints, returning (keep, reasons)."""
    reasons: List[str] = []
    gc_min, gc_max = _collect_gc_bounds_from_constraints(constraints)
    gc_min_inclusive, gc_max_inclusive = get_gc_inclusive_flags(constraints)

    if gc_min is not None or gc_max is not None:
        gc_val = _coerce_fraction(row.get("GC") or row.get("gc") or row.get("GC_content"))
        if gc_val is None:
            reasons.append("gc_missing")
        else:
            if gc_min is not None and (
                gc_val < gc_min - 1e-6
                if gc_min_inclusive
                else gc_val <= gc_min + 1e-6
            ):
                reasons.append("gc_below_min")
            if gc_max is not None and (
                gc_val > gc_max + 1e-6
                if gc_max_inclusive
                else gc_val >= gc_max - 1e-6
            ):
                reasons.append("gc_above_max")

    max_binding = get_max_binding_sites(constraints)
    if max_binding is not None:
        binding_val = get_binding_count_from_row(row)
        if binding_val is None:
            # Missing binding count -> treat as failure to ensure constraint enforcement
            reasons.append("binding_sites_missing")
        elif binding_val > max_binding:
            reasons.append("binding_sites_exceed")

    per_base_limits = get_homopolymer_per_base_limits(
        constraints,
        fallback_default=DEFAULT_HOMOPOLYMER_LIMIT,
    )
    if per_base_limits:
        seq = candidate_constraint_sequence(row)
        if seq:
            homopolymer_ok = sequence_passes_homopolymer_limits(seq, per_base_limits)
        else:
            # Fallback when raw sequence is unavailable.
            no_homo_value = row.get("No_homopolymer") or row.get("no_homopolymer")
            if isinstance(no_homo_value, bool):
                homopolymer_ok = no_homo_value
            else:
                homopolymer_ok = str(no_homo_value).strip().lower() in {"true", "1", "yes"}
        if not homopolymer_ok:
            reasons.append("homopolymer_run_exceeds_limit")

    return (len(reasons) == 0), reasons


def filter_tiles_json(
    json_path: str,
    constraints: Dict[str, Any],
    *,
    output_suffix: str = "constraint_filtered",
) -> Dict[str, Any]:
    """Filter tiles JSON by constraints; returns summary with output_path and counts."""
    if not isinstance(constraints, dict) or not constraints:
        return {
            "applied": False,
            "changed": False,
            "output_path": json_path,
            "kept_count": None,
            "removed_count": 0,
            "reason_counts": {},
        }

    if not os.path.exists(json_path):
        return {
            "applied": False,
            "changed": False,
            "output_path": json_path,
            "error": f"json_path_not_found: {json_path}",
        }

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("rows") or []
    filtered_rows: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        keep, reasons = evaluate_row_against_constraints(row, constraints)
        if keep:
            filtered_rows.append(row)
        else:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    removed_count = len(rows) - len(filtered_rows)
    changed = removed_count > 0

    if changed:
        base, ext = os.path.splitext(json_path)
        output_path = f"{base}.{output_suffix}{ext or '.json'}"
        payload["rows"] = filtered_rows
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        output_path = json_path

    return {
        "applied": True,
        "changed": changed,
        "output_path": output_path,
        "kept_count": len(filtered_rows),
        "removed_count": removed_count,
        "total_count": len(rows),
        "reason_counts": reason_counts,
    }
