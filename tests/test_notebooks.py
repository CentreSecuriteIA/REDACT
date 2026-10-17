"""The notebooks call the library as it is, read without running them."""

import ast
import inspect
import json
from pathlib import Path

import pytest

import redact

_NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))
# Names the eval/training split removed or renamed.
_GONE = ("paraphrased.csv", "include_standalone_benign", "standalone_benign_categories")


def _cells(path: Path, kind: str) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == kind]


def _calls(path: Path, name: str | None = None) -> list[ast.Call]:
    """The calls of a function of ``redact``, ``name``'s alone when given."""
    tree = ast.parse("\n".join(_cells(path, "code")))
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in redact.__all__ and name in (None, node.func.id)]


def _mode(call: ast.Call) -> object:
    """The ``mode`` a ``build_dataset`` call passes, else its default."""
    given = [kw.value for kw in call.keywords if kw.arg == "mode"]
    return ast.literal_eval(given[0]) if given else "training"


def _dataset_type(call: ast.Call) -> object:
    """The ``dataset_type`` of the recipe a ``run_pipeline`` call writes out."""
    recipe = call.args[0]
    assert isinstance(recipe, ast.Dict)
    given = [value for key, value in zip(recipe.keys, recipe.values, strict=True)
             if isinstance(key, ast.Constant) and key.value == "dataset_type"]
    return ast.literal_eval(given[0]) if given else None


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda path: path.stem)
def test_each_keyword_a_notebook_passes_is_a_parameter(path):
    for call in _calls(path):
        parameters = inspect.signature(getattr(redact, call.func.id)).parameters
        unknown = [kw.arg for kw in call.keywords if kw.arg not in parameters]
        assert not unknown, f"{call.func.id} takes no {unknown}"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda path: path.stem)
def test_a_notebook_names_nothing_the_split_removed(path):
    text = "\n".join(_cells(path, "code") + _cells(path, "markdown"))
    assert [name for name in _GONE if name in text] == []


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda path: path.stem)
def test_each_recipe_a_notebook_runs_names_its_dataset_type(path):
    for call in _calls(path, "run_pipeline"):
        assert _dataset_type(call) in ("eval", "training")


@pytest.mark.parametrize("mode", ["eval", "training"])
def test_a_pipeline_notebook_builds_and_runs_its_dataset_type(mode):
    path = Path(__file__).parents[1] / "notebooks" / f"{mode}_pipeline.ipynb"
    builds = _calls(path, "build_dataset")
    assert builds
    assert [_mode(call) for call in builds] == [mode] * len(builds)
    assert [_dataset_type(call) for call in _calls(path, "run_pipeline")] == [mode]
