"""Execute a generated program and retain structured, value-free error context.

Run as a script so the analysis interpreter needs no application installation.
Local variable values are not recorded; names, types and array shapes suffice
to diagnose scalar/Series and numeric/string boundary errors.
"""
import ast
import json
from pathlib import Path
import runpy
import sys
import traceback
from contextlib import contextmanager


@contextmanager
def structured_excel_cells():
    """Encode structured cells as JSON while preserving ordinary scalar types.

    This is a serialization adapter, not a scientific transformation. Reject
    oversized values before openpyxl can silently truncate the evidence.
    """
    receipt = {"format": "json", "converted_cells": 0, "source_types": {}}
    try:
        from openpyxl.cell.cell import Cell
    except ImportError:
        yield receipt
        return
    original = Cell._bind_value

    def bind(cell, value):
        if isinstance(value, (dict, list, tuple)):
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if len(encoded) > 32767:
                raise ValueError("Structured Excel cell exceeds 32767 characters; preserve the complete value "
                                 "in a separate evidence sheet or file and put a reference in this cell.")
            kind = type(value).__name__
            receipt["converted_cells"] += 1
            receipt["source_types"][kind] = receipt["source_types"].get(kind, 0) + 1
            value = encoded
        return original(cell, value)

    Cell._bind_value = bind
    try:
        yield receipt
    finally:
        Cell._bind_value = original


def describe(value, *, depth=0, seen=None):
    kind = type(value)
    result = {"type": kind.__module__ + "." + kind.__qualname__}
    if kind.__module__.split(".")[0] in {"numpy", "pandas"}:
        shape = getattr(value, "shape", None)
        if shape is not None:
            result["shape"] = [int(n) for n in shape]
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            result["dtype"] = str(dtype)
        if kind.__name__ == "DataFrame":
            result["column_dtypes"] = {str(k): str(v) for k, v in value.dtypes.items()}
    elif kind in {str, list, tuple, dict, set}:
        result["length"] = len(value)
        seen = set() if seen is None else seen
        if id(value) in seen:
            result["recursive"] = True
        elif depth < 2 and kind in {dict, list, tuple}:
            seen = seen | {id(value)}
            if kind is dict:
                # Field names are schema, as with DataFrame column names.
                # Record no leaf values and bound both depth and breadth.
                result["fields"] = {str(key): describe(item, depth=depth+1, seen=seen)
                                    for key, item in list(value.items())[:32] if isinstance(key, (str, int))}
            else:
                result["item_types"] = [describe(item, depth=depth+1, seen=seen) for item in value[:4]]
    return result


def definition_order_context(code_path, exc):
    """Explain an observed missing global whose definition has not been reached.

    This is diagnostic context for the existing repair loop, not a speculative
    execution-order validator. A forward reference in a deferred function body
    is valid when the module finishes binding the helper before calling it.
    Only an actual NameError at a generated source name load is considered.
    """
    if not isinstance(exc, NameError) or isinstance(exc, UnboundLocalError):
        return None
    name = getattr(exc, 'name', None)
    frames = list(traceback.walk_tb(exc.__traceback__))
    if not name or not frames:
        return None
    use_frame, use_line = frames[-1]
    if (Path(use_frame.f_code.co_filename).resolve() != code_path
            or name in use_frame.f_globals):
        return None
    module_lines = [line for frame, line in frames
                    if Path(frame.f_code.co_filename).resolve() == code_path
                    and frame.f_code.co_name == '<module>']
    if not module_lines:
        return None
    try:
        tree = ast.parse(code_path.read_text())
    except (OSError, UnicodeError, SyntaxError):
        return None  # Never obscure the original exception with a diagnostic.
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and node.name == name]
    if len(definitions) != 1 or definitions[0].lineno <= module_lines[-1]:
        return None
    # Explicitly raising a NameError is not evidence of a failed name lookup.
    if not any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
               and node.id == name and node.lineno <= use_line <= node.end_lineno
               for node in ast.walk(tree)):
        return None
    definition = definitions[0]
    return {
        'name': name, 'use_line': use_line,
        'module_execution_line': module_lines[-1],
        'definition_line': definition.lineno,
        'definition_kind': type(definition).__name__,
        'repair_instruction':
            f"The failing global {name!r} is referenced at line {use_line}, while module "
            f"execution is still at line {module_lines[-1]}. Its top-level definition "
            f"at line {definition.lineno} has not been reached. Arrange for the definition "
            "and its dependencies to execute before this module-level invocation. "
            "Changing only the helper body does not repair this execution order. "
            "Preserve the requested calculations, inputs and definition semantics; "
            "do not drop the call or substitute empty results.",
    }


def execute(code_path, diagnostic_path):
    code_path = Path(code_path).resolve()
    # Match direct `python generated_code.py` execution. The diagnostic wrapper's
    # own arguments and module directory must not change the generated program's
    # argument parsing or imports of task-local helper modules.
    sys.argv = [str(code_path)]
    sys.path.insert(0, str(code_path.parent))
    serialization = None
    try:
        with structured_excel_cells() as serialization:
            runpy.run_path(str(code_path), run_name="__main__")
    except Exception as exc:
        frames = []
        for frame, line in traceback.walk_tb(exc.__traceback__):
            if Path(frame.f_code.co_filename).resolve() == code_path:
                frames.append({"line": line, "function": frame.f_code.co_name,
                               "local_types": {name: describe(value) for name, value in frame.f_locals.items()
                                               if not name.startswith("__")}})
        record = {"exception_type": type(exc).__name__, "message": str(exc), "frames": frames}
        order = definition_order_context(code_path, exc)
        if order is not None:
            record['definition_order'] = order
        if isinstance(exc, SyntaxError):
            record["syntax_location"] = {"line": exc.lineno, "offset": exc.offset}
        Path(diagnostic_path).write_text(json.dumps(record, indent=2))
        traceback.print_exc()
        return 1
    finally:
        if serialization and serialization["converted_cells"]:
            Path(diagnostic_path).with_name("serialization.json").write_text(json.dumps(serialization, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(execute(sys.argv[1], sys.argv[2]))
