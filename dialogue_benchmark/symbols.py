"""Conservative Python AST references; no type inference or code execution."""

import ast


def inspect_code(path, content):
    if not path.endswith(".py"):
        return {"status": "unsupported_language", "symbols": [], "references": [], "imports": {}}
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"status": "parse_error", "symbols": [], "references": [], "imports": {}}
    symbols, references, imports = [], [], {}

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def visit_ImportFrom(self, node):
            for alias in node.names:
                imports[alias.asname or alias.name] = {"module": node.module or "",
                                                       "level": node.level, "name": alias.name}

        def visit_ClassDef(self, node):
            self.definition(node, "class")

        def visit_FunctionDef(self, node):
            self.definition(node, "function")

        visit_AsyncFunctionDef = visit_FunctionDef

        def definition(self, node, kind):
            self.scope.append(node.name)
            symbols.append({"name": ".".join(self.scope), "kind": kind,
                            "line": node.lineno, "end_line": node.end_lineno})
            self.generic_visit(node)
            self.scope.pop()

        def visit_Call(self, node):
            target = ast.unparse(node.func)
            references.append({"from": ".".join(self.scope) or "<module>",
                               "expression": target, "line": node.lineno,
                               "simple_name": node.func.id if isinstance(node.func, ast.Name) else None})
            self.generic_visit(node)

    Visitor().visit(tree)
    return {"status": "parsed", "symbols": symbols, "references": references, "imports": imports}
