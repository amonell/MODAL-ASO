"""Lossless text handling for Excel's per-cell character limit."""

EXCEL_TEXT_UNITS = 32767
PART_TEXT_UNITS = 30000


def excel_text_units(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def excel_text_parts(text: str) -> list[str]:
    """Split at character boundaries, including supplementary Unicode."""
    parts = []
    start = 0
    units = 0
    for index, character in enumerate(text):
        width = 2 if ord(character) > 0xffff else 1
        if units + width > PART_TEXT_UNITS:
            parts.append(text[start:index])
            start = index
            units = 0
        units += width
    parts.append(text[start:])
    return parts
