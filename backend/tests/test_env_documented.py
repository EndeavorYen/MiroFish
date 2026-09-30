"""Every environment variable the code reads is documented (#68).

The scan walks the AST of ``app/``, ``scripts/`` and ``run.py`` for
``os.environ.get(...)``, ``os.getenv(...)``, ``os.environ[...]``,
``os.environ.setdefault(...)`` and ``"X" in os.environ`` -- also through a
name bound to ``os.environ`` (``env = os.environ if ... else ...``,
``dict(os.environ)``) or a parameter called ``env``/``environ`` -- with the
name given as a string or as a module-level string constant (``ENV_VAR =
"SIM_AGENT_CONTEXT_TOKENS"``). Each name needs its own table row
(`` | `NAME` | ``) in docs/configuration.md. Reads through computed keys
(a loop over a dict of defaults) are out of its reach.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DOC = BACKEND.parent / "docs" / "configuration.md"
_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


_ENV_NAMES = {"environ", "env", "environment"}


def _mentions_environ(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == "environ" for n in ast.walk(node))


def _is_environ(node: ast.AST, aliases: set[str]) -> bool:
    """``os.environ``, a bare ``environ``/``env``, or a name bound to it."""

    if isinstance(node, ast.Attribute) and node.attr == "environ":
        return True
    return isinstance(node, ast.Name) and (node.id in _ENV_NAMES or node.id in aliases)


def _reads(tree: ast.AST) -> set[str]:
    # Module-level constants only: a local of the same name must not win.
    constants = {
        target.id: node.value.value
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    aliases = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _mentions_environ(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def name_of(arg: ast.AST) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            value = arg.value
        elif isinstance(arg, ast.Name):
            value = constants.get(arg.id)
        else:
            return None
        return value if value and _NAME.match(value) else None

    found: set[str] = set()
    for node in ast.walk(tree):
        arg = None
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("get", "setdefault", "pop") and _is_environ(func.value, aliases):
                arg = node.args[0]
            elif isinstance(func, ast.Attribute) and func.attr == "getenv":
                arg = node.args[0]
            elif isinstance(func, ast.Name) and func.id == "getenv":
                arg = node.args[0]
        elif isinstance(node, ast.Subscript) and _is_environ(node.value, aliases):
            arg = node.slice
        elif isinstance(node, ast.Compare) and len(node.comparators) == 1 and _is_environ(node.comparators[0], aliases):
            arg = node.left
        name = name_of(arg) if arg is not None else None
        if name:
            found.add(name)
    return found


def environment_reads() -> dict[str, list[str]]:
    reads: dict[str, list[str]] = {}
    files = [*sorted((BACKEND / "app").rglob("*.py")), *sorted((BACKEND / "scripts").rglob("*.py")), BACKEND / "run.py"]
    for path in files:
        for name in _reads(ast.parse(path.read_text(encoding="utf-8"))):
            reads.setdefault(name, []).append(str(path.relative_to(BACKEND)))
    return reads


def test_the_scan_finds_reads_through_constants_and_aliases():
    tree = ast.parse(
        'import os\nVAR = "SOME_SETTING"\nx = os.environ.get(VAR)\ny = "OTHER_ONE" in os.environ\n'
        'def f(environ=None):\n    e = os.environ if environ is None else environ\n    return e.get("THIRD_ONE")\n'
        'def g(env):\n    VAR = "LOCAL_ONLY"\n    return env.get("FOURTH_ONE")\n'
    )
    assert _reads(tree) == {"SOME_SETTING", "OTHER_ONE", "THIRD_ONE", "FOURTH_ONE"}


def test_every_environment_variable_is_documented():
    # A row of its own, not a mention inside another row.
    documented = set(re.findall(r"^\| `([A-Z][A-Z0-9_]{2,})` \|", DOC.read_text(encoding="utf-8"), re.M))
    missing = {name: files for name, files in environment_reads().items() if name not in documented}
    assert not missing, f"document these in docs/configuration.md: {missing}"
