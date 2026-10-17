"""Keep intervention direction separate from asserted biological roles."""


def explicit_biological_role(value):
    """Preserve an explicit role; absence is contextual, never inferred by action."""
    role = str(value or "").strip().lower()
    return role if role in {"beneficial", "pathogenic", "context"} else "context"
