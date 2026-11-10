"""An AST scan for the places where an error is built from something that is not a literal
(`tests/unit/test_infer_serve_error_messages.py`).

`interpolations(source, path)` returns, for every call of `DataError`, `ConfigError`,
`UpstreamError` or `RequestTextError` (by name or as an attribute), each expression that goes into
its message: the value of an f-string field, an argument of `.format`, the right side of `%`, a
non-literal operand of `+`, or the argument itself when it is a bare expression (`DataError(str(x))`).
`names` says which call names count, for a package whose errors are made by a helper.
Each finding names the file, the enclosing function (`Class.method` inside a class) and the source
of the expression, as `ast.unparse` writes it, so that a reformatting of the code changes nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final

ERROR_NAMES: Final = frozenset(
    {"DataError", "ConfigError", "UpstreamError", "RequestTextError"}
)


@dataclass(frozen=True)
class Finding:
    path: str
    function: str
    expression: str


def interpolations(
    source: str, path: str, *, names: frozenset[str] = ERROR_NAMES
) -> list[Finding]:
    scanner = _Scanner(path, names)
    scanner.visit(ast.parse(source))
    return scanner.found


def _is_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, path: str, names: frozenset[str]) -> None:
        self.path = path
        self._names = names
        self.found: list[Finding] = []
        self._scope: list[str] = []

    def _enter(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node) in self._names:
            for argument in [*node.args, *(k.value for k in node.keywords)]:
                for expression in _expressions(argument):
                    finding = Finding(
                        self.path, ".".join(self._scope) or "<module>", expression
                    )
                    if finding not in self.found:
                        self.found.append(finding)
        self.generic_visit(node)


def _expressions(argument: ast.expr) -> list[str]:
    """What goes into the message `argument` makes, as source."""
    if _is_literal(argument):
        return []
    found: list[str] = []
    bare = True
    for node in ast.walk(argument):
        if isinstance(node, ast.FormattedValue):
            found.append(ast.unparse(node.value))
            bare = False
        elif isinstance(node, ast.JoinedStr):
            bare = False
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and _is_literal(node.func.value)
        ):
            found += [
                ast.unparse(a) for a in [*node.args, *(k.value for k in node.keywords)]
            ]
            bare = False
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if _is_literal(node.left):
                found.append(ast.unparse(node.right))
                bare = False
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if _is_literal(node.left) or _is_literal(node.right):
                found += [
                    ast.unparse(side)
                    for side in (node.left, node.right)
                    if not _is_literal(side) and not isinstance(side, ast.BinOp)
                ]
                bare = False
    if bare:
        found.append(ast.unparse(argument))
    return found
