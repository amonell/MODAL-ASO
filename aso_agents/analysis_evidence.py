"""Check copied scalar facts against completed native evaluation observations.

This does not grade biological conclusions or require every table to contain
every source row. It checks only columns explicitly sharing recorded field
names, and only when the physical record and value are unambiguous.
"""
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _facts(row):
    values = {}

    def visit(obj, prefix=()):
        for key, value in obj.items():
            path = (*prefix, str(key))
            if isinstance(value, dict):
                visit(value, path)
            elif value is not None and isinstance(value, (str, int, float, bool)):
                for name in {str(key), "_".join(path)}:
                    values.setdefault(name, []).append(value)

    visit(row)
    return {key: group[0] for key, group in values.items()
            if all(type(value) is type(group[0]) and value == group[0] for value in group)}


def _identifiers(row):
    return {str(row[key]).strip() for key in ("aso_id", "name", "id")
            if row.get(key) is not None and str(row[key]).strip()}


def _sequence(row):
    for key in ("sequence", "sequence_5to3", "input_sequence_5to3"):
        if row.get(key):
            return "".join(str(row[key]).upper().split())
    return None


def _agrees(expected, observed):
    text = str(observed).strip()
    if not text:
        return False
    if isinstance(expected, bool):
        return text.lower() in ({"true", "yes", "1"} if expected else {"false", "no", "0"})
    if isinstance(expected, (int, float)):
        try:
            number = Decimal(text)
            actual = Decimal(str(expected))
            if not number.is_finite() or not actual.is_finite():
                return False
            if isinstance(expected, int):
                return number == actual
            # Preserve ordinary display rounding, not a fixed numerical epsilon.
            quantum = Decimal(1).scaleb(number.as_tuple().exponent)
            return abs(number - actual) <= abs(quantum) / 2
        except InvalidOperation:
            return False
    return text == str(expected).strip()


def validate_recorded_export_facts(payload: dict, paths: list[Path]) -> None:
    if payload.get("operation") != "render_requested_outputs":
        return
    declared = payload.get("declared_inputs") or {}
    completed = declared.get("completed_run") or {}
    records = [row for entry in completed.get("evaluation_evidence") or []
               if isinstance(entry, dict) for row in entry.get("rows") or []
               if isinstance(row, dict)]
    if not records:
        return
    indexed = [(row, _identifiers(row), _sequence(row), _facts(row)) for row in records]
    errors = []
    for path in paths:
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for line, row in enumerate(csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","), 2):
                identifiers, sequence = _identifiers(row), _sequence(row)
                if not identifiers and not sequence:
                    continue
                matches = [facts for _, names, bases, facts in indexed
                           if (not identifiers or bool(identifiers & names))
                           and (not sequence or sequence == bases)]
                if not matches:
                    continue
                # Duplicate names across tasks/references must not borrow a
                # conflicting sibling observation. Only shared facts qualify.
                for key, observed in row.items():
                    if not key or not all(key in facts for facts in matches):
                        continue
                    expected = matches[0][key]
                    if not all(type(facts[key]) is type(expected) and facts[key] == expected for facts in matches):
                        continue
                    if not _agrees(expected, observed):
                        errors.append(f"{path.name}:{line} {key}: exported {observed!r}, recorded {expected!r}")
                    if len(errors) >= 10:
                        break
                if len(errors) >= 10:
                    break
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError("Requested export contradicts recorded evaluation facts: " + "; ".join(errors)
                         + ". Read completed_run.evaluation_evidence or its source_json_path using the actual schema. "
                         "Do not substitute an empty collection, zero or missing binding for an absent JSON key. "
                         "Keep derived calculations separately labeled; preserve the native observations.")
