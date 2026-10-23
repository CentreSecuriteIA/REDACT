"""Prose in the jailbreak code: text a technique could write, kept in Python.

A technique's text belongs in a template file under ``configs/jailbreak/``,
where its pool is checked and counted. This lint reads every module under
``src/redact/jailbreak/`` with :mod:`ast` and finds each string of
:data:`MIN_WORDS` or more words that is not a docstring and not part of a
``raise``, a logging call or ``warnings.warn``: prose a technique could put
in its output. An f-string counts as one string, each placeholder as
``{}``. Text built from shorter strings (a word list, a map of
abbreviations, the keys of ``to_json``'s wrapper) is not found.

``fixtures/prose_in_code.json`` allows the strings found when the lint was
written, each by file, enclosing function (or the module-level name it is
assigned to) and word count, never by its text. The list only shrinks: a
string not on it fails, and so does an entry that matches no string, so
moving a text into a JSON template file means taking its entry out. A
failure prints ``file:line`` and word counts, never the text.
"""

import ast
import json
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).parents[2] / "src" / "redact" / "jailbreak"
_FIXTURE = Path(__file__).parents[1] / "fixtures" / "prose_in_code.json"

#: The fewest words a string needs to count as prose.
MIN_WORDS = 8

#: The objects whose calls carry a message, not a technique's text.
_MESSAGE_OWNERS = {"logger", "logging", "warnings"}


def _docstring(node: ast.AST) -> ast.Constant | None:
    """The docstring of a module, class or function ``node``, as its node."""
    body = getattr(node, "body", None)
    if (isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef
                   | ast.AsyncFunctionDef)
            and body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[0].value
    return None


class _Prose(ast.NodeVisitor):
    """The prose strings of one module: ``(line, where, words)`` each."""

    def __init__(self, tree: ast.Module):
        self.found: list[tuple[int, str, int]] = []
        self.scope: list[str] = []
        self.target: str | None = None
        self.docstrings = {id(_docstring(node)) for node in ast.walk(tree)}

    def _where(self) -> str:
        return ".".join(self.scope) or self.target or "<module>"

    def _note(self, node: ast.AST, text: str) -> None:
        if len(text.split()) >= MIN_WORDS:
            self.found.append((node.lineno, self._where(), len(text.split())))

    def _scoped(self, node) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_Assign(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not self.scope and isinstance(targets[0], ast.Name):
            self.target = targets[0].id
        self.generic_visit(node)
        if not self.scope:
            self.target = None

    visit_AnnAssign = visit_Assign

    def visit_Raise(self, node: ast.Raise) -> None:
        """A raised error's message is no technique's text."""

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in _MESSAGE_OWNERS):
            return
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        text = "".join(part.value if isinstance(part, ast.Constant) else "{}"
                       for part in node.values)
        self._note(node, text)
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                self.visit(part.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            self._note(node, node.value)


def prose(path: Path) -> list[tuple[int, str, int]]:
    """The prose strings of the module at ``path``: ``(line, where, words)``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    visitor = _Prose(tree)
    visitor.visit(tree)
    return visitor.found


def found() -> list[tuple[str, int, str, int]]:
    """``(file, line, where, words)`` of every prose string under the
    jailbreak package, ``file`` relative to it."""
    return [(path.relative_to(_SRC).as_posix(), line, where, words)
            for path in sorted(_SRC.rglob("*.py"))
            for line, where, words in prose(path)]


def _allowed() -> Counter:
    """The fixture's entries, as ``(file, function, words)`` counts."""
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return Counter((entry["file"], entry["function"], entry["words"])
                   for entry in fixture["allowed"])


def test_the_lint_finds_prose_but_not_docstrings_or_messages(tmp_path):
    module = tmp_path / "m.py"
    module.write_text(
        '"""alpha beta gamma delta epsilon zeta eta theta."""\n'
        "import logging, warnings\n"
        "logger = logging.getLogger(__name__)\n"
        'GREETING = "alpha beta gamma delta epsilon zeta eta theta"\n'
        "def to_x(prompt):\n"
        '    """alpha beta gamma delta epsilon zeta eta theta."""\n'
        '    logger.warning("alpha beta gamma delta epsilon zeta eta theta")\n'
        '    warnings.warn("alpha beta gamma delta epsilon zeta eta theta")\n'
        "    if not prompt:\n"
        '        raise ValueError("alpha beta gamma delta epsilon zeta eta theta")\n'
        '    short = "alpha beta gamma"\n'
        '    return f"alpha beta gamma {prompt} delta epsilon zeta eta", short\n',
        encoding="utf-8")
    assert prose(module) == [(4, "GREETING", 8), (12, "to_x", 8)]


def test_every_prose_string_is_allowed():
    """A string not on the list: move its text into a template file, or, for
    a message or a docstring the lint cannot tell apart, add its entry."""
    allowed = _allowed()
    unlisted = []
    for file, line, where, words in found():
        if allowed[(file, where, words)]:
            allowed[(file, where, words)] -= 1
        else:
            unlisted.append(f"{file}:{line} ({where}, {words} words)")
    assert unlisted == []


def test_every_allowed_entry_is_still_found():
    """An entry no string matches: its text moved out, so take it off the list."""
    present = Counter((file, where, words) for file, _, where, words in found())
    assert sorted((_allowed() - present).elements()) == []


if __name__ == "__main__":
    for file, line, where, words in found():
        print(f"{file}:{line} {where} {words}")
