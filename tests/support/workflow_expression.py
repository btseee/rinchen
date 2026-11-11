"""A small model of the expressions `${{ ... }}` of GitHub Actions, for the static tests of a
workflow (D14, Task 6): the three operators `||`, `&&` and `==`, brackets, names of the context and
quoted strings, and nothing else. The model exists so that a test can say what an expression gives
for a trigger without running the workflow (which is UNMEASURED); an operator or a character it does
not know is a ValueError, never a guess."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

_TOKEN: Final = re.compile(r"\s*(\|\||&&|==|\(|\)|'[^']*'|[A-Za-z_][\w.]*)")


class Expression:
    """An expression `${{ ... }}` over names of the context, given flat by their dotted name.

    `||` gives its first truthy operand (else its last) and `&&` its first falsy operand (else its
    last), as GitHub does; `==` gives a boolean; a name the context does not hold is null; `==`
    binds tighter than `&&`, which binds tighter than `||`.

    Raises:
        ValueError: (on construction or on `value`) the text is not an expression of this grammar.
    """

    def __init__(self, source: str, context: Mapping[str, str]) -> None:
        body = source.strip()
        if not (body.startswith("${{") and body.endswith("}}")):
            raise ValueError("not an expression")
        self._tokens = self._split(body[3:-2].strip())
        self._context = context
        self._position = 0

    @staticmethod
    def _split(body: str) -> list[str]:
        tokens: list[str] = []
        position = 0
        while position < len(body):
            match = _TOKEN.match(body, position)
            if match is None:
                raise ValueError("an unreadable character")
            tokens.append(match.group(1))
            position = match.end()
        return tokens

    def value(self) -> object:
        result = self._or()
        if self._position != len(self._tokens):
            raise ValueError("a token too many")
        return result

    def _or(self) -> object:
        result = self._and()
        while self._take("||"):
            right = self._and()
            result = result or right
        return result

    def _and(self) -> object:
        result = self._equal()
        while self._take("&&"):
            right = self._equal()
            result = result and right
        return result

    def _equal(self) -> object:
        result = self._atom()
        if self._take("=="):
            return result == self._atom()
        return result

    def _atom(self) -> object:
        if self._position == len(self._tokens):
            raise ValueError("an operand is missing")
        token = self._tokens[self._position]
        self._position += 1
        if token == "(":
            result = self._or()
            if not self._take(")"):
                raise ValueError("a bracket is not closed")
            return result
        if token.startswith("'"):
            return token[1:-1]
        return self._context.get(token)

    def _take(self, token: str) -> bool:
        if self._position < len(self._tokens) and self._tokens[self._position] == token:
            self._position += 1
            return True
        return False
