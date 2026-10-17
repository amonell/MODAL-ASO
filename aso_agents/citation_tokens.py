"""Literal citation-placeholder grammar shared by delivery checks."""


def citation_placeholders(text):
    """Return complete ``{{token}}`` spans in source order, including malformed ones."""
    if not isinstance(text, str):
        return []
    tokens = []
    offset = 0
    while True:
        start = text.find('{{', offset)
        if start < 0:
            return tokens
        end = text.find('}}', start + 2)
        if end < 0:
            return tokens
        content = text[start + 2:end]
        tokens.append(content)
        offset = end + 2
