"""Bounded execution and artifact verification for generated data analyses."""
from __future__ import annotations

import ast
import csv
import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys
from typing import Callable


def _inexact_decimal_boundary_lines(code: str) -> list[int]:
    """Find strict numeric gates built from binary floats and decimal literals.

    This is a conservative syntax-level guard, independent of column names,
    threshold values, gene names and requested outcome. It covers both a
    fraction computed with ``/`` and a decimal parsed with ``float``. A
    generated program can instead compare ``Fraction`` or ``Decimal`` values
    formed from the source strings, then format them for display afterwards.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    def scope_nodes(root: ast.AST):
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                stack.append(child)

    scopes = [tree, *(node for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))]
    lines: set[int] = set()
    for scope in scopes:
        assignments: dict[str, list[ast.AST]] = {}
        nodes = list(scope_nodes(scope))
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)

        def contains(node: ast.AST, *, decimal_literal: bool, seen: frozenset[str] = frozenset()) -> bool:
            if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
                return any(contains(value, decimal_literal=decimal_literal, seen=seen | {node.id})
                           for value in assignments[node.id])
            if decimal_literal:
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    return not node.value.is_integer()
            else:
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    return True
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
                    return True
            return any(contains(child, decimal_literal=decimal_literal, seen=seen)
                       for child in ast.iter_child_nodes(node))

        for node in nodes:
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                continue
            operands = [node.left, *node.comparators]
            if any(contains(item, decimal_literal=True) for item in operands) and any(
                contains(item, decimal_literal=False) for item in operands
            ):
                lines.add(node.lineno)
    return sorted(lines)


def _file_signature(path: Path) -> tuple[int, str] | None:
    """Fingerprint content; cached network timestamps cannot prove a write."""
    if path.is_symlink() or not path.is_file():
        return None
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _archive_previous_outputs(paths: list[Path], local: Path, archived: list[dict]) -> None:
    """Preserve existing declared outputs before requiring new files to appear.

    Only explicit outputs are removed from their old locations. Conventional
    basename aliases may be inputs and are never moved by this helper.
    """
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        signature = _file_signature(path)
        if signature is None:
            raise ValueError(f'Declared output is not a regular file: {path}')
        backup = local / 'previous_outputs' / str(len(archived))
        backup.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as source, backup.open('xb') as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if _file_signature(backup) != signature or _file_signature(path) != signature:
            raise ValueError(f'Previous output changed while being archived: {path}')
        entry = {'path': str(path), 'archive': str(backup), 'sha256': signature[1]}
        # Keep recovery provenance even if a subsequent output cannot be moved
        # or the process exits before its final execution receipt is written.
        with backup.with_suffix('.json').open('x') as metadata:
            metadata.write(json.dumps(entry, indent=2))
            metadata.flush()
            os.fsync(metadata.fileno())
        archived.append(entry)
        path.unlink()


def _promote_same_named_exports(
    directory: Path, required_outputs: list[Path], before: dict[Path, tuple[int, str] | None],
) -> list[tuple[Path, Path]]:
    """Deliver only fresh, unambiguous files saved in conventional work locations.

    A generated program may use the declared export's basename relative to its
    working directory or its output/ subdirectory. The requested absolute path
    remains authoritative. No renamed, stale, multiply written, symlinked or
    already delivered file is substituted.
    """
    names = [path.name for path in required_outputs]
    promoted = []
    for output in required_outputs:
        if names.count(output.name) != 1 or output.exists():
            continue
        candidates = [directory / output.name, directory / "output" / output.name]
        fresh = [source for source in candidates
                 if source != output
                 and not source.parent.is_symlink()
                 and (signature := _file_signature(source)) is not None
                 and signature != before.get(source)]
        if len(fresh) != 1:
            continue
        source = fresh[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        promoted.append((source, output))
    return promoted


def _validate_completed_evaluation_export_rows(payload: dict, required_outputs: list[Path]) -> None:
    """Do not accept header-only exports of completed physical evaluations.

    The native evaluator supplies a structured molecule count to the final
    output writer. If that count is positive, an all-empty set of requested
    tables is a delivery failure, even when every file exists and a generated
    program exits successfully. The writer can repair its source traversal on
    the next attempt; no observations or rows are synthesized here.
    """
    if payload.get("operation") != "render_requested_outputs":
        return
    declared = payload.get("declared_inputs")
    completed = declared.get("completed_run") if isinstance(declared, dict) else None
    evidence = completed.get("evaluation_evidence") if isinstance(completed, dict) else None
    source_rows = sum(
        entry["total_rows"] for entry in evidence or []
        if isinstance(entry, dict) and type(entry.get("total_rows")) is int
        and entry["total_rows"] > 0
    )
    if not source_rows:
        return
    tables = [path for path in required_outputs if path.suffix.lower() in {".csv", ".tsv"}]
    if not tables:
        return
    exported_rows = []
    for path in tables:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            next(reader, None)
            exported_rows.append(sum(1 for row in reader if any(value.strip() for value in row)))
    if not any(exported_rows):
        raise ValueError(
            "All requested CSV/TSV exports contain zero data rows, although "
            f"completed native evaluation evidence contains {source_rows} physical molecule rows. "
            "Read the declared completed_run.evaluation_evidence rows or its source_json_path, "
            "then write a source-grounded table; do not treat a header-only file as complete."
        )


def execute_analysis_program(
    *, directory: Path, payload: dict, generate: Callable[[dict], dict],
    required_outputs: list[Path], timeout_seconds: int | None = None, max_attempts: int = 3,
    readonly_outputs: list[Path] | None = None,
    source_tables: list[Path] | None = None,
    validate_result: Callable[[dict], None] | None = None,
    review_program: Callable[[dict], dict] | None = None,
) -> dict:
    """Repair execution/serialization faults without substituting scientific data.

    Every generated program and error stays on disk. Success requires a valid
    result, an actual summary and all explicitly requested files. An exit-zero
    program that only promises files is incomplete.
    """
    directory = directory.resolve()
    required_outputs = [path.resolve() for path in required_outputs]
    readonly_outputs = [path.resolve() for path in (readonly_outputs or [])]
    source_tables = [path.resolve() for path in (source_tables or [])]
    if not set(source_tables).issubset(readonly_outputs):
        raise ValueError('Report source tables must be protected read-only inputs')
    directory.mkdir(parents=True, exist_ok=True)
    result_path, summary_path = directory / 'sandbox_result.json', directory / 'sandbox_summary.md'
    expected_outputs = [result_path, summary_path, *required_outputs]
    if set(expected_outputs) & set(readonly_outputs):
        raise ValueError('Generated output paths must not replace read-only input files')
    errors, attempts = [], []
    last_program = None
    request = dict(payload)
    request['requested_output_paths'] = [str(p) for p in required_outputs]
    request['readonly_output_paths'] = [str(p) for p in readonly_outputs]
    from .analysis_helpers import shared_analysis_helpers, validate_shared_helper_bindings
    execution_contract = {'working_directory': str(directory),
        'result_json': str(result_path), 'summary_markdown': str(summary_path),
        'required_output_files': [str(p) for p in [result_path, summary_path, *required_outputs]],
        'relative_path_base': str(directory), 'readonly_output_files': [str(p) for p in readonly_outputs],
        'report_source_tables': [str(p) for p in source_tables],
        'report_table_views': {
            'declaration': 'sandbox_result.json.report_tables',
            'allowed_source_paths': [str(p) for p in required_outputs
                                     if p.suffix.lower() in {'.csv', '.tsv', '.xlsx'}]
                                    + [str(p) for p in source_tables],
            'workbook_identity': 'For XLSX, source_sheet is the exact worksheet name; '
                                 'header_row is a positive one-based row, default 1. '
                                 'Use unique nonempty headers and computed values, not formulas.',
            'rendering': 'Write {{table:ID}} markers in Markdown and declare id/source_path, '
                         'optional columns, rank_column and preview_limit. The host renders '
                         'after execution; do not import or implement a Markdown binder.',
        },
        'shared_analysis_helpers': shared_analysis_helpers()}
    request['execution_contract'] = execution_contract
    previous = [int(p.name) for p in (directory / "execution_attempts").glob("*") if p.is_dir() and p.name.isdigit()]
    offset = max(previous, default=0)
    for attempt in range(offset + 1, offset + max_attempts + 1):
        local = directory / 'execution_attempts' / f'{attempt:02d}'
        local.mkdir(parents=True, exist_ok=False)
        protected = {}
        for index, path in enumerate(readonly_outputs):
            backup = local / 'preserved_outputs' / str(index)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            protected[path] = backup
        plan = generate(request)
        code = str(plan.get('python_code') or '').strip()
        (local / 'plan.json').write_text(json.dumps(plan, indent=2))
        code_path = directory / 'generated_code.py'
        code_path.write_text(code)
        (local / 'generated_code.py').write_text(code)
        same_named_before = {
            source: _file_signature(source)
            for path in required_outputs
            for source in (directory / path.name, directory / "output" / path.name)
        }
        stdout = stderr = ''
        exit_code = 1
        promoted: list[tuple[Path, Path]] = []
        archived_outputs: list[dict] = []
        contract_review = None
        try:
            if not code:
                detail = plan.get('error')
                raise ValueError('No executable analysis program was returned'
                                 + (': ' + json.dumps(detail, ensure_ascii=False) if detail else ''))
            from .analysis_dependencies import validate_program_dependencies
            from .analysis_control_flow import validate_program_control_flow
            preflight_errors = []
            for validator in (validate_program_dependencies, validate_program_control_flow,
                              validate_shared_helper_bindings):
                try:
                    validator(code)
                except ValueError as exc:
                    preflight_errors.append(str(exc))
            inexact_lines = _inexact_decimal_boundary_lines(code)
            if inexact_lines:
                preflight_errors.append(
                    'Inexact decimal-boundary comparison at generated code lines '
                    + ', '.join(map(str, inexact_lines))
                    + '. Retain source decimal strings and integer counts as '
                    'Decimal or Fraction through every threshold comparison; '
                    'convert only for final display or serialization.'
                )
            from .analysis_input_paths import declared_input_path_errors
            manifest_path = directory / 'declared_inputs.json'
            if manifest_path.is_file():
                preflight_errors.extend(declared_input_path_errors(
                    code, directory, json.loads(manifest_path.read_text())))
            if preflight_errors:
                raise ValueError('Analysis preflight: ' + '; '.join(preflight_errors))
            diagnostic_path = local / "runtime_error.json"
            # A file's presence after a clean start proves creation without
            # relying on delayed/coarse/unstable filesystem timestamps. Keep
            # the previous bytes as evidence, including on unsuccessful retries.
            _archive_previous_outputs(expected_outputs, local, archived_outputs)
            process = subprocess.run([sys.executable, str(Path(__file__).with_name("analysis_runtime.py")),
                str(code_path.resolve()), str(diagnostic_path.resolve())], cwd=directory,
                capture_output=True, text=True, timeout=timeout_seconds)
            stdout, stderr, exit_code = process.stdout, process.stderr, process.returncode
            if exit_code:
                raise ValueError(f'Analysis program exited {exit_code}: {stderr[-6000:]}')
            # Require a real result and summary before moving same-named
            # working-directory exports to their declared delivery paths.
            if result_path.is_file():
                preview = json.loads(result_path.read_text())
                if isinstance(preview, dict):
                    from .design_workbooks import render_design_workbook_plans
                    render_design_workbook_plans(payload, preview, required_outputs)
                    result_path.write_text(json.dumps(preview, indent=2))
                if isinstance(preview, dict) and summary_path.is_file() and summary_path.read_text().strip():
                    promoted = _promote_same_named_exports(directory, required_outputs, same_named_before)
            missing = [str(p) for p in [result_path, summary_path, *required_outputs] if not p.is_file()]
            if missing:
                if any(item['path'] in missing for item in archived_outputs):
                    raise ValueError('The current attempt did not produce its output files: ' + ', '.join(missing))
                raise ValueError('Requested output files were not created: ' + ', '.join(missing))
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict):
                raise ValueError('sandbox_result.json must contain an object')
            if validate_result is not None:
                validate_result(result)
            from .tabular_reports import render_report_tables
            render_report_tables(result, required_outputs, summary_path,
                                 required=bool(payload.get('require_tabular_report_views')),
                                 source_tables=source_tables)
            _validate_completed_evaluation_export_rows(payload, required_outputs)
            from .analysis_evidence import validate_recorded_export_facts
            validate_recorded_export_facts(payload, required_outputs)
            from .design_workbooks import validate_design_workbooks
            validate_design_workbooks(payload, result, required_outputs)
            if result.get('report_table_rendering') or result.get('design_fact_validation'):
                result_path.write_text(json.dumps(result, indent=2))
            summary = summary_path.read_text().strip()
            if not summary:
                raise ValueError('Analysis summary is empty')
            if review_program is not None:
                from .analysis_review import assess_review, program_digest
                review_input = {
                    "request_text": str(payload.get("request_text") or ""),
                    "task_type": payload.get("task_type"),
                    "request_envelope": payload.get("request_envelope"),
                    "declared_inputs": payload.get("declared_inputs"),
                    "execution_contract": execution_contract,
                    "program": code, "program_sha256": program_digest(code),
                    "result": result, "summary": summary,
                    "output_paths": [str(path) for path in required_outputs],
                }
                try:
                    raw_review = review_program(review_input)
                except Exception as exc:
                    # A review transport failure is not a scientific finding or
                    # a reason to rewrite a successfully executed program.
                    raw_review = {"error": f"{type(exc).__name__}: {exc}"}
                contract_review = assess_review(raw_review, code, review_input["request_text"])
                review_path = local / "contract_review.json"
                review_path.write_text(json.dumps({
                    "program_sha256": program_digest(code),
                    "raw_review": raw_review, **contract_review,
                }, indent=2))
                if contract_review["findings"]:
                    raise ValueError("Analysis request requirements not implemented: "
                                     + json.dumps(contract_review["findings"], ensure_ascii=False))
                result["analysis_contract_review"] = {
                    "status": contract_review["status"], "receipt_path": str(review_path)}
                result_path.write_text(json.dumps(result, indent=2))
            error = None
        except (ValueError, SyntaxError, OSError, subprocess.TimeoutExpired) as exc:
            error = f'{type(exc).__name__}: {exc}'
            for _, output in promoted:
                output.unlink(missing_ok=True)
            if isinstance(exc, subprocess.TimeoutExpired):
                exit_code = 124
                stdout = exc.stdout or ''
                stderr = exc.stderr or ''
                stdout = stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
            result, summary = {}, ''
        changed_readonly = []
        def file_digest(path):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()
        for index, (path, backup) in enumerate(protected.items()):
            try:
                changed = path.is_symlink() or not path.is_file() or file_digest(path) != file_digest(backup)
            except OSError:
                changed = True
            if changed:
                changed_readonly.append(str(path))
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_dir() and not path.is_symlink():
                    shutil.move(str(path), str(local / f'rejected_output_directory_{index}'))
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    restored = Path(handle.name)
                shutil.copy2(backup, restored)
                restored.replace(path)
        if changed_readonly:
            error = (error + '; ' if error else '') + 'Modified previously delivered files; original contents restored: ' + ', '.join(changed_readonly)
            for _, output in promoted:
                output.unlink(missing_ok=True)
            result, summary = {}, ''
        for name, text in [('stdout.log', stdout), ('stderr.log', stderr)]:
            (local / name).write_text(text)
            (directory / name).write_text(text)
        diagnostic_path = local / "runtime_error.json"
        diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else {}
        receipt = {'attempt':attempt, 'exit_code':exit_code, 'error':error,
                   'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                   'program_path': str(local / 'generated_code.py'),
                   'plan_path': str(local / 'plan.json'),
                   'plan_file_sha256': hashlib.sha256((local / 'plan.json').read_bytes()).hexdigest(),
                   'runtime_diagnostics': diagnostics,
                   'required_outputs':[str(p) for p in required_outputs]}
        serialization_path = local / 'serialization.json'
        if serialization_path.is_file():
            receipt['serialization'] = json.loads(serialization_path.read_text())
        if archived_outputs:
            receipt['previous_output_archives'] = archived_outputs
        if contract_review is not None:
            receipt['contract_review'] = {
                'status': contract_review['status'],
                'receipt_path': str(local / 'contract_review.json'),
                'findings_count': len(contract_review['findings']),
            }
        if promoted:
            receipt['promoted_same_named_exports'] = [
                {'source': str(source), 'requested_output': str(output)} for source, output in promoted]
        if readonly_outputs:
            receipt['readonly_outputs'] = [str(p) for p in readonly_outputs]
            receipt['restored_output_paths'] = changed_readonly
        if error:
            receipt['workspace_files'] = [str(path.relative_to(directory)) for path in sorted(directory.rglob('*'))
                if path.is_file() and not path.is_symlink()
                and path.relative_to(directory).parts[0] != 'execution_attempts'][:200]
        (local / 'execution.json').write_text(json.dumps(receipt, indent=2))
        attempts.append(receipt)
        if error is None:
            return {'ok':True, 'plan':plan, 'result':result, 'summary':summary,
                    'exit_code':0, 'attempts':attempts}
        errors.append(error)
        if code:
            last_program = {'previous_code': code, 'previous_plan': plan, 'error': error,
                            'runtime_diagnostics': diagnostics}
        repair = dict(last_program or {'previous_code': '', 'previous_plan': plan,
                                      'error': error, 'runtime_diagnostics': diagnostics})
        if not code and last_program:
            # A rejected edit is not a replacement program. Preserve the last
            # real code, its bound plan and its execution error for the next edit.
            repair['latest_generation_error'] = error
        request = {**payload, 'requested_output_paths':[str(p) for p in required_outputs],
                   'readonly_output_paths': [str(p) for p in readonly_outputs],
                   'execution_contract': execution_contract,
                   'execution_repair':{**repair, 'previous_attempts': list(attempts),
                       'instruction':'Repair the program and create the requested files using only the same declared inputs. Do not invent missing data or change the requested scientific criteria.'}}
    return {'ok':False, 'plan':plan, 'result':{}, 'summary':'', 'exit_code':exit_code,
            'error':errors[-1], 'attempts':attempts}
