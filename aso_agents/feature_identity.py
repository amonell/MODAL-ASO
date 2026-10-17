"""Keep a feature's identifier distinct from lengths and reference identifiers."""
from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any


_EXON_NUMBER = re.compile(
    r"\bexon(?:[-_]|\s*#?\s*)(\d+)(?![\w.])", re.IGNORECASE
)
_ORDINAL_EXON = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+exon\b", re.IGNORECASE)
_LENGTH_OR_RANGE = re.compile(
    r"\s*(?:[-–]?\s*(?:nt|bp|nucleotides?|bases?|base[- ]pairs?)\b"
    r"|(?:[-–/,]|and\b|or\b|to\b)\s*\d)", re.IGNORECASE
)


def exon_number(value: Any) -> int | None:
    """Return one explicit positive exon number, never a descriptive number.

    A named or ambiguous exon remains a label for reference-backed resolution.
    In particular, lengths, transcript versions and nonstandard labels must not
    become transcript ordinals merely because they contain digits.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Real):
        return int(value) if math.isfinite(value) and value > 0 and int(value) == value else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    bare = re.fullmatch(r"#?\s*(\d+)(?:\s*\([^)]*\))?", text)
    if bare:
        number = int(bare.group(1))
        return number if number > 0 else None
    numbers = []
    for pattern in (_EXON_NUMBER, _ORDINAL_EXON):
        for match in pattern.finditer(text):
            if _LENGTH_OR_RANGE.match(text[match.end():]):
                return None
            number = int(match.group(1))
            if number <= 0:
                return None
            numbers.append(number)
    return numbers[0] if numbers and len(set(numbers)) == 1 else None
