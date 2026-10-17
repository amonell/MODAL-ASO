"""Source-bound correction of numeric sequence-metric notes in reports."""
from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from fractions import Fraction
from pathlib import Path


def _source_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        fields = set(reader.fieldnames or [])
        if not {"id", "sequence", "reported_gc_fraction", "computed_gc_fraction"} <= fields:
            return None
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids) or any(not item for item in ids):
        return None
    return rows


def _decimal_matches_fraction(raw: str, exact: Fraction) -> bool:
    try:
        declared = Decimal(str(raw).strip())
        if not declared.is_finite():
            return False
        unit = Decimal(1).scaleb(declared.as_tuple().exponent)
        with localcontext() as context:
            context.prec = max(60, len(str(raw)) + 15)
            rounded = (Decimal(exact.numerator) / Decimal(exact.denominator)).quantize(
                unit, rounding=ROUND_HALF_UP
            )
        return declared == rounded
    except (InvalidOperation, ValueError, OverflowError):
        return False


def _exact_fraction_text(value: Fraction) -> str:
    divisor = value.denominator
    for prime in (2, 5):
        while divisor % prime == 0:
            divisor //= prime
    if divisor == 1:
        with localcontext() as context:
            context.prec = max(60, len(str(value.denominator)) + 15)
            decimal = Decimal(value.numerator) / Decimal(value.denominator)
        return f"{value.numerator}/{value.denominator} ({decimal})"
    return f"{value.numerator}/{value.denominator}"


def _metric_discrepancies(rows):
    differences = []
    for row in rows:
        sequence = str(row["sequence"] or "").strip().upper()
        if not sequence or set(sequence) - set("ACGTU"):
            raise ValueError(f"Sequence metric source has unsupported bases for row {row['id']}.")
        exact = Fraction(sum(base in "GC" for base in sequence), len(sequence))
        if not _decimal_matches_fraction(row["computed_gc_fraction"], exact):
            raise ValueError(
                f"Computed GC in the exported table disagrees with the sequence for row {row['id']}. "
                "Repair the source table before rendering its report."
            )
        computed_length = row.get("computed_length")
        if computed_length not in (None, ""):
            try:
                if int(computed_length) != len(sequence):
                    raise ValueError
            except ValueError:
                raise ValueError(
                    f"Computed length in the exported table disagrees with the sequence for row {row['id']}."
                ) from None
        reported_length = row.get("reported_length")
        if reported_length not in (None, ""):
            try:
                length_agrees = int(reported_length) == len(sequence)
            except ValueError:
                length_agrees = False
            if not length_agrees:
                differences.append((row["id"], "length", reported_length, str(len(sequence)), "different"))
        reported_gc = str(row["reported_gc_fraction"] or "").strip()
        if reported_gc:
            try:
                observed = Fraction(Decimal(reported_gc))
            except (InvalidOperation, ValueError):
                observed = None
            if observed != exact:
                relation = "rounded" if _decimal_matches_fraction(reported_gc, exact) else "different"
                differences.append((row["id"], "GC fraction", reported_gc,
                                    _exact_fraction_text(exact), relation))
    return differences


def _contains_identifier(text: str, identifier: str) -> bool:
    start = 0
    while (position := text.find(identifier, start)) >= 0:
        before = text[position - 1] if position else ""
        after_index = position + len(identifier)
        after = text[after_index] if after_index < len(text) else ""
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        start = position + 1
    return False


def canonicalize_metric_notes(texts: dict[str, str], export_tables: dict[str, Path]):
    """Replace a row-linked free-text GC comparison with exact sequence facts.

    Only a bullet that cites one of the table's row IDs and explicitly
    compares reported with computed GC is replaced. Other narrative and all
    raw reported values are preserved. An audit receipt keeps the old line.
    """
    sources = [(path, rows) for path in export_tables.values()
               if (rows := _source_rows(path)) is not None]
    if not sources:
        return texts, []
    sources.sort(key=lambda item: len(item[1]), reverse=True)
    source, rows = sources[0]
    if len(sources) > 1 and len(sources[1][1]) == len(rows):
        return texts, []
    by_id = {row["id"]: row for row in rows}
    for _, subset in sources[1:]:
        if any(row["id"] not in by_id or any(
            row[field] != by_id[row["id"]][field]
            for field in ("sequence", "reported_gc_fraction", "computed_gc_fraction")
        ) for row in subset):
            return texts, []
    identifiers = {row["id"] for row in rows}
    def is_source_linked_gc_comparison(line: str) -> bool:
        lower = line.lower()
        return (
            line.lstrip().startswith(("- ", "* "))
            and (_contains_identifier(lower, "gc") or "gc_fraction" in lower)
            and _contains_identifier(lower, "reported")
            and (_contains_identifier(lower, "computed")
                 or _contains_identifier(lower, "recalculated"))
            and any(_contains_identifier(line, identifier) for identifier in identifiers)
        )
    if not any(is_source_linked_gc_comparison(line)
               for text in texts.values() for line in text.splitlines()):
        return texts, []
    differences = _metric_discrepancies(rows)
    if not differences:
        return texts, []
    table = ["| Row | Metric | Reported | Computed from sequence | Relation |",
             "|---|---|---:|---:|---|"]
    table.extend("| " + " | ".join(str(cell) for cell in item) + " |" for item in differences)
    note = ["- Sequence-metric comparison from the exported rows (reported inputs are preserved):",
            "", *table, "",
            "A rounded reported fraction can differ from its exact sequence fraction without being an error.", ""]
    amended, receipts = {}, []
    for destination, text in texts.items():
        output = []
        replaced = []
        for line in text.splitlines():
            if is_source_linked_gc_comparison(line):
                output.extend(note)
                replaced.append(line)
            else:
                output.append(line)
        amended[destination] = "\n".join(output) + ("\n" if text.endswith("\n") else "")
        if replaced:
            receipts.append({"destination": destination, "source": str(source),
                             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "replaced_lines": replaced, "metric_rows": len(differences)})
    return amended, receipts
