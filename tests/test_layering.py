"""Boundaries that must hold no matter who is in a hurry.

Three rules, all checked by reading the source rather than by running
it, because a rule that only fails at runtime fails on the day it
matters:

1. Only `data/storage.py` talks to the database. The moment a second
   module opens an engine, "one place knows the schema" stops being
   true and nobody notices until a migration breaks something
   unrelated.
2. No module imports the LLM package. The dependency runs one way, so
   the shared modules stay usable by both tracks and the CatBoost side
   never acquires a language-model dependency.
3. The `cli/` package is the single exception, because assembling every
   track's subcommands is the whole of its job. The exemption is pinned
   to that directory here, so it cannot spread to a module outside it.

Every rule reads the package with `rglob`, so grouping the modules into
subpackages does not quietly narrow what is checked.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path("src") / "facility_prediction"
STORAGE_MODULE = pathlib.Path("data") / "storage.py"
DATABASE_PACKAGES = {"sqlalchemy", "alembic", "psycopg", "psycopg2"}

LLM_PACKAGE = "facility_prediction.llm"

# The composition root is a package, not a module: assembling every
# track's subcommands outgrew one file. The exemption is pinned to that
# directory, so it still cannot spread to a module outside it.
COMPOSITION_ROOT = pathlib.Path("cli")


def _relative(path: pathlib.Path) -> pathlib.Path:
    return path.relative_to(PACKAGE)


def _modules():
    """Every module in the package, at any depth.

    `rglob` rather than `glob`: the package is grouped into
    subpackages, and a rule that only reads the top level would stop
    seeing the modules it exists to constrain.
    """
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _in_composition_root(path: pathlib.Path) -> bool:
    return COMPOSITION_ROOT in _relative(path).parents


def _in_llm_package(path: pathlib.Path) -> bool:
    return pathlib.Path("llm") in _relative(path).parents


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "path", [path for path in _modules() if _relative(path) != STORAGE_MODULE]
)
def test_only_storage_reaches_the_database(path):
    offending = _imported_roots(path) & DATABASE_PACKAGES

    assert not offending, (
        f"{_relative(path)} imports {sorted(offending)}; database access "
        f"belongs in {STORAGE_MODULE}"
    )


def test_storage_is_the_module_that_reaches_the_database():
    roots = _imported_roots(PACKAGE / STORAGE_MODULE)

    assert roots & DATABASE_PACKAGES


def _imports_the_llm_package(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    modules += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    return any(name.startswith(LLM_PACKAGE) for name in modules)


@pytest.mark.parametrize(
    "path",
    [
        path
        for path in _modules()
        if not _in_composition_root(path) and not _in_llm_package(path)
    ],
)
def test_no_module_but_the_cli_imports_the_llm_package(path):
    assert not _imports_the_llm_package(path), (
        f"{_relative(path)} imports {LLM_PACKAGE}; only {COMPOSITION_ROOT}/ "
        "may know about a single track's modules"
    )


def test_the_llm_exemption_never_widens_past_the_cli():
    importers = {
        _relative(path)
        for path in _modules()
        if _imports_the_llm_package(path) and not _in_llm_package(path)
    }

    assert all(COMPOSITION_ROOT in path.parents for path in importers), (
        f"{sorted(map(str, importers))} reach into {LLM_PACKAGE}"
    )
