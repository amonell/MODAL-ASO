"""Physical delivery checks, independent of scientific result assessment."""
import json
import csv
import hashlib
import shutil
import re
from pathlib import Path


NATIVE_EXPORT_FORMATS = {
    "report_md": {".md", ".markdown"},
    "report_html": {".html"},
    "design_panel_csv": {".csv"},
    "comparison_table_csv": {".csv"},
    "generated_aso_set_json": {".json"},
}


def requested_artifact_sha256(path: Path) -> str:
    """Bind a completed delivery to bytes, independently of file timestamps."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def requested_file_destination(spec: dict, path: Path) -> Path:
    """Give a directory request a concrete, producer-bound filename.

    Only filesystem facts or a trailing separator establish a directory.
    Extensionless filenames remain filenames; labels never establish science.
    """
    raw = str(spec.get("path") or "")
    if not (path.is_dir() or raw.endswith("/")):
        return path
    fmt = str(spec.get("format") or "").strip().lower().lstrip(".")
    if not fmt:
        key = spec.get("source_artifact") or spec.get("kind")
        suffixes = NATIVE_EXPORT_FORMATS.get(key, set())
        if key in {"report", "design_report", "report_markdown"}:
            fmt = "md"
        elif suffixes:
            fmt = sorted(suffixes, key=lambda value: (len(value), value))[0][1:]
    if not re.fullmatch(r"[a-z0-9]{1,16}", fmt):
        raise ValueError(f"requested_output_directory_format_required:{path}")
    identity = {key: spec.get(key) for key in
                ("task_id", "kind", "label", "source_artifact", "source_scope")}
    label = spec.get("label") or spec.get("source_artifact") or spec.get("kind") or "result"
    stem = str(spec.get("task_id") or "result") + "_" + str(label)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:80] or "result"
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
    return path / f"{stem}_{token}.{fmt}"


def native_export_key(spec: dict, destination: Path) -> str | None:
    """Resolve explicit artifact roles, never infer a table from its filename."""
    key = spec.get("source_artifact")
    if "source_artifact" not in spec:
        kind = spec.get("kind")
        if kind in NATIVE_EXPORT_FORMATS:
            key = kind
        elif kind in {"report", "design_report", "report_markdown", "report_html"}:
            key = {".md": "report_md", ".markdown": "report_md", ".html": "report_html"}.get(destination.suffix.lower())
        elif kind in {"candidate_table", "design_panel"} and destination.suffix.lower() == ".csv":
            key = "design_panel_csv"
    if key and destination.suffix.lower() not in NATIVE_EXPORT_FORMATS.get(key, set()):
        raise ValueError(f"requested_output_native_format_mismatch:{key}:{destination}")
    return key


def copy_native_export(source: Path, destination: Path, *, scope: dict | None = None,
                       source_root: Path | None = None) -> dict:
    """Reuse the producer's exact observations and conclusions, with provenance."""
    errors = requested_artifact_errors([source])
    if errors:
        raise ValueError("requested_output_native_source_unavailable:" + ";".join(errors))
    if scope:
        if source.suffix.lower() != '.csv' or destination.suffix.lower() != '.csv':
            raise ValueError('requested_output_scope_unsupported:' + str(source))
        allowed = {'program_ids': 'program_id', 'modalities': 'source_modality'}
        if set(scope) - set(allowed):
            raise ValueError('requested_output_scope_unknown_field')
        filters = {}
        for key, values in scope.items():
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError('requested_output_scope_invalid:' + key)
            if values:
                filters[allowed[key]] = set(values)
        if filters:
            if source.resolve() == destination.resolve():
                raise ValueError('requested_output_scope_would_replace_source:' + str(source))
            with source.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                for key in filters:
                    if key not in fields:
                        raise ValueError('requested_output_scope_column_missing:' + key)
                rows = list(reader)
            for key in filters:
                if rows and not any(row.get(key) for row in rows):
                    raise ValueError('requested_output_scope_identity_unavailable:' + key)
            def scoped_value(row, key):
                value = row.get(key)
                # Native splice-route records describe their source stage;
                # their molecular design modality is still steric.
                return 'steric' if key == 'source_modality' and value == 'splice_mod' else value
            selected = [row for row in rows if all(scoped_value(row, key) in values for key, values in filters.items())]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(selected)
            return {'source':str(source), 'destination':str(destination),
                    'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'method':'native_artifact_row_selection','scope':scope,
                    'row_count':len(selected),'source_row_count':len(rows)}
    if (source.resolve() != destination.resolve()
            and source.suffix.lower() in {'.md', '.markdown', '.html', '.htm'}):
        from .report_exports import export_native_report
        return export_native_report(source, destination, source_root=source_root or source.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return {"source": str(source), "destination": str(destination),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "method": "native_artifact_copy"}


def requested_artifact_errors(paths: list[Path]) -> list[str]:
    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"requested_output_missing:{path}")
        elif path.stat().st_size == 0:
            errors.append(f"requested_output_empty:{path}")
        elif path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text())
            except (OSError, ValueError):
                errors.append(f"requested_output_invalid_json:{path}")
    return errors
