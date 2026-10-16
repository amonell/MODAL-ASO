"""Check statically known, required paths into the runtime input manifest.

Only straight-line module/with assignments are resolved. Conditional lookups,
dynamic keys, functions and other data files remain runtime responsibilities.
"""
import ast
from pathlib import Path


def declared_input_path_errors(code, directory, manifest):
    root = Path(directory).resolve() / "declared_inputs.json"
    aliases, handles, errors, seen = {}, set(), [], set()

    def source_file(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "Path"}:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                path = Path(node.args[0].value)
                return (path if path.is_absolute() else root.parent / path).resolve() == root
        return False

    def loaded(node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or not node.args:
            return False
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "json":
            return False
        value = node.args[0]
        if node.func.attr == "load":
            return source_file(value) or isinstance(value, ast.Name) and value.id in handles
        return (node.func.attr == "loads" and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute) and value.func.attr == "read_text"
                and source_file(value.func.value))

    def path_for(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            base = path_for(node.value)
            key = node.slice.value
            if base is not None and type(key) in {str, int}:
                return (*base, key)
        return None

    def check(node):
        # Resolve only actual subscripts; optional .get() paths are not required.
        pending = [node]
        while pending:
            part = pending.pop()
            if isinstance(part, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.ListComp,
                                 ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue  # Their subscripts may be guarded or scoped locally.
            pending.extend(ast.iter_child_nodes(part))
            if not isinstance(part, ast.Subscript):
                continue
            path = path_for(part)
            if path is None or path in seen:
                continue
            seen.add(path)
            value = manifest
            for index, key in enumerate(path):
                present = (isinstance(value, dict) and key in value or isinstance(value, list)
                           and type(key) is int and -len(value) <= key < len(value))
                if not present:
                    location = "declared_inputs.json" + "".join(f"[{key!r}]" for key in path[:index + 1])
                    available = sorted(map(str, value)) if isinstance(value, dict) else f"{type(value).__name__}"
                    errors.append(f"line {part.lineno}: required input path {location} is absent; available here: {available}")
                    break
                value = value[key]

    def forget_assignments(statement):
        for part in ast.walk(statement):
            if isinstance(part, ast.Name) and isinstance(part.ctx, (ast.Store, ast.Del)):
                aliases.pop(part.id, None)
                handles.discard(part.id)

    def statements(body):
        for statement in body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                if value is None:
                    continue
                check(value)
                path = () if loaded(value) else path_for(value)
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases.pop(target.id, None)
                        if path is not None:
                            aliases[target.id] = path
            elif isinstance(statement, ast.With):
                old_handles = set(handles)
                for item in statement.items:
                    if isinstance(item.optional_vars, ast.Name):
                        handles.discard(item.optional_vars.id)
                        aliases.pop(item.optional_vars.id, None)
                        if source_file(item.context_expr):
                            handles.add(item.optional_vars.id)
                statements(statement.body)
                handles.clear()
                handles.update(old_handles)
            elif isinstance(statement, ast.Expr):
                # Method calls may mutate a manifest alias. Do not carry a
                # stale assumption past a mutation or arbitrary function call.
                check(statement.value)
                if isinstance(statement.value, ast.Call):
                    aliases.clear()
            elif not isinstance(statement, (ast.Import, ast.ImportFrom)):
                # No false claims about branch guards, loops or user functions.
                forget_assignments(statement)
                if isinstance(statement, (ast.If, ast.Try, ast.For, ast.While, ast.Match)):
                    aliases.clear()

    statements(ast.parse(code).body)
    return list(dict.fromkeys(errors))
