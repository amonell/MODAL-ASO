"""Catch statically unbound globals before executing generated analysis code.

This checks lexical bindings, not whether imports are installed or a program's
results are correct. Dynamic namespaces cannot be resolved this way and retain
the existing execution/error-repair path.
"""
import ast
import builtins
import symtable


_SCRIPT_GLOBALS = {
    "__name__", "__file__", "__package__", "__spec__", "__cached__",
    "__loader__", "__builtins__", "__annotations__",
}


def unbound_global_names(code):
    tree = ast.parse(code, filename="generated_code.py")
    # Imports with a wildcard and deliberate namespace mutation are not closed
    # lexical programs. Do not misreport their dynamically supplied names.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "globals", "locals", "vars"}:
                return None
    module = symtable.symtable(code, "generated_code.py", "exec")
    scopes = []
    pending = [module]
    while pending:
        scope = pending.pop()
        scopes.append(scope)
        pending.extend(scope.get_children())
    bound = set(dir(builtins)) | _SCRIPT_GLOBALS
    # A nested `global` assignment/import can establish a module binding too.
    # We do not attempt to prove execution order or reachability.
    for scope in scopes:
        bound.update(symbol.get_name() for symbol in scope.get_symbols()
                     if symbol.is_global() and (symbol.is_assigned() or symbol.is_imported()))
    referenced = {symbol.get_name() for scope in scopes for symbol in scope.get_symbols()
                  if symbol.is_global() and symbol.is_referenced()}
    return sorted(referenced - bound)


def validate_program_dependencies(code):
    missing = unbound_global_names(code)
    if missing:
        raise ValueError(
            "Generated program references global names with no import or definition: "
            + ", ".join(missing)
            + ". Make the program self-contained using the intended imports or definitions; "
              "preserve the requested calculations and evidence."
        )
