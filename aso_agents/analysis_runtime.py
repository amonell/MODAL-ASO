"""Execute a generated program and retain structured, value-free error context.

Run as a script so the analysis interpreter needs no application installation.
Local variable values are not recorded; names, types and array shapes suffice
to diagnose scalar/Series and numeric/string boundary errors.
"""
import json
from pathlib import Path
import runpy
import sys
import traceback


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


def execute(code_path, diagnostic_path):
    code_path = Path(code_path).resolve()
    # Match direct `python generated_code.py` execution. The diagnostic wrapper's
    # own arguments and module directory must not change the generated program's
    # argument parsing or imports of task-local helper modules.
    sys.argv = [str(code_path)]
    sys.path.insert(0, str(code_path.parent))
    try:
        runpy.run_path(str(code_path), run_name="__main__")
    except Exception as exc:
        frames = []
        for frame, line in traceback.walk_tb(exc.__traceback__):
            if Path(frame.f_code.co_filename).resolve() == code_path:
                frames.append({"line": line, "function": frame.f_code.co_name,
                               "local_types": {name: describe(value) for name, value in frame.f_locals.items()
                                               if not name.startswith("__")}})
        record = {"exception_type": type(exc).__name__, "message": str(exc), "frames": frames}
        if isinstance(exc, SyntaxError):
            record["syntax_location"] = {"line": exc.lineno, "offset": exc.offset}
        Path(diagnostic_path).write_text(json.dumps(record, indent=2))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(execute(sys.argv[1], sys.argv[2]))
