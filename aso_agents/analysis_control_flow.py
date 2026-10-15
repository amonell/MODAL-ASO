"""Reject provably unreachable statements in generated analysis programs."""
import ast


def unreachable_statement_lines(code):
    """Find dead statements after unconditional exits within the same block.

    No data-dependent conditions or output counts are inferred. Loop exits do
    not make code after the loop unreachable, and finally blocks are traversed
    independently so their cleanup remains valid.
    """
    tree = ast.parse(code, filename="generated_code.py")
    dead = []

    def exits(statement):
        if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return True
        if isinstance(statement, ast.If):
            return (any(exits(item) for item in statement.body)
                    and any(exits(item) for item in statement.orelse))
        return False

    def visit(node):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                terminated = False
                for child in value:
                    if not isinstance(child, ast.AST):
                        continue
                    if isinstance(child, ast.stmt):
                        if terminated and not isinstance(child, ast.Pass):
                            dead.append(child.lineno)
                        terminated = terminated or exits(child)
                    visit(child)
            elif isinstance(value, ast.AST):
                visit(value)

    visit(tree)
    return sorted(set(dead))


def validate_program_control_flow(code):
    lines = unreachable_statement_lines(code)
    if lines:
        raise ValueError(
            "Unreachable generated analysis statements at lines "
            + ", ".join(map(str, lines))
            + ". These statements follow an unconditional return, raise, break or continue "
              "in the same block. Check indentation and function/loop boundaries; move "
              "required processing to the reachable intended scope. Preserve the input "
              "records and requested calculations; do not delete required analysis or "
              "claim an empty dataset merely to make execution pass."
        )
