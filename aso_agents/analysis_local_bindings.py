"""Find local helper reads before their only direct definition.

This deliberately narrow check does not infer branch outcomes, execute code,
or reorder definitions. Deferred closures and dynamic namespaces remain the
responsibility of execution validation.
"""
import ast


class _Bindings(ast.NodeVisitor):
    def __init__(self, function):
        self.names = {}
        self.declarations = set()
        for arg in (function.args.posonlyargs + function.args.args
                    + function.args.kwonlyargs):
            self.bind(arg.arg, arg)
        for arg in (function.args.vararg, function.args.kwarg):
            if arg is not None:
                self.bind(arg.arg, arg)

    def bind(self, name, node):
        self.names.setdefault(name, []).append(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id, node)

    def visit_FunctionDef(self, node):
        self.bind(node.name, node)
        # Defaults/decorators execute in this scope; the body does not.
        for value in node.decorator_list + node.args.defaults + node.args.kw_defaults:
            if value is not None:
                self.visit(value)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bind(node.name, node)
        for value in node.decorator_list + node.bases + node.keywords:
            self.visit(value)

    def visit_Import(self, node):
        for alias in node.names:
            self.bind(alias.asname or alias.name.split('.')[0], node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.bind(alias.asname or alias.name, node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bind(node.name, node)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.declarations.update(node.names)

    visit_Nonlocal = visit_Global

    def visit_MatchAs(self, node):
        if node.name:
            self.bind(node.name, node)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self.bind(node.rest, node)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        for value in node.args.defaults + node.args.kw_defaults:
            if value is not None:
                self.visit(value)

    def visit_ListComp(self, node):
        # Comprehension targets have their own scope; assignment expressions
        # may bind the containing function. Over-approximating these bindings
        # excludes a candidate rather than rejecting valid code.
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                self.visit(child.target)

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp


class _ImmediateReads(ast.NodeVisitor):
    def __init__(self):
        self.reads = []

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.reads.append(node)

    def skip(self, node):
        pass

    # No claim about invocation order in another scope or deferred expression.
    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = skip
    visit_Lambda = visit_GeneratorExp = skip
    visit_ListComp = visit_SetComp = visit_DictComp = skip


def premature_local_helper_reads(code):
    tree = ast.parse(code, filename='generated_code.py')
    findings = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Namespace mutation and nonlocal writes defeat the sole-binding
        # proof. Conservatively leave these programs to runtime diagnostics.
        dynamic = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id in {'exec', 'locals', 'vars'}
                      for n in ast.walk(function))
        if dynamic:
            continue
        nonlocal_names = {name for n in ast.walk(function)
                          if isinstance(n, ast.Nonlocal) for name in n.names}
        bindings = _Bindings(function)
        for statement in function.body:
            bindings.visit(statement)
        preceding = _ImmediateReads()
        for statement in function.body:
            if (isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and bindings.names.get(statement.name) == [statement]
                    and statement.name not in bindings.declarations | nonlocal_names):
                for read in preceding.reads:
                    if read.id == statement.name:
                        findings.append({'function': function.name, 'name': read.id,
                                         'read_line': read.lineno,
                                         'definition_line': statement.lineno})
            preceding.visit(statement)
    return sorted(findings, key=lambda row: (row['read_line'], row['name']))
