"""The notebooks call the library as it is, read without running them (but
for the eval notebook's first cell, which calls no model)."""

import ast
import inspect
import json
from pathlib import Path

import pytest

import redact
from redact.jailbreak.augmentation import resolve
from redact.recipe import augmentations_of, load_recipe
from redact.stages import STAGES, RunContext

_FOLDER = Path(__file__).parents[1] / "notebooks"
_NOTEBOOKS = sorted(_FOLDER.glob("*.ipynb"))
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
        unknown = [kw.arg for kw in call.keywords
                   if kw.arg is not None and kw.arg not in parameters]
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
    path = _FOLDER / f"{mode}_pipeline.ipynb"
    builds = _calls(path, "build_dataset")
    assert builds
    assert [_mode(call) for call in builds] == [mode] * len(builds)
    assert [_dataset_type(call) for call in _calls(path, "run_pipeline")] == [mode]


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda path: path.stem)
def test_a_notebook_builds_in_the_mode_its_folder_names(path):
    """``./runs/eval`` holds an eval build, ``./runs/training`` a training one."""
    tree = ast.parse("\n".join(_cells(path, "code")))
    (folder,) = [ast.literal_eval(node.value) for node in ast.walk(tree)
                 if isinstance(node, ast.Assign)
                 and [ast.unparse(target) for target in node.targets] == ["DATA_DIR"]]
    builds = _calls(path, "build_dataset")
    if Path(folder).name in ("eval", "training"):
        assert [_mode(call) for call in builds] == [Path(folder).name] * len(builds)


@pytest.mark.parametrize("mode", ["eval", "training"])
def test_a_pipeline_notebooks_cells_make_the_run_its_last_cell_makes(tmp_path, mode):
    """Each step takes the type's params with the notebook's PARAMS over them
    and the run's augmentation section, so the last cell resumes what the
    cells wrote (a resume with other plan settings would raise)."""
    path = _FOLDER / f"{mode}_pipeline.ipynb"
    setup: dict = {}
    exec(_cells(path, "code")[0], setup)    # noqa: S102 - imports and names only
    (run,) = _calls(path, "run_pipeline")
    recipe = {ast.literal_eval(key): value
              for key, value in zip(run.args[0].keys, run.args[0].values, strict=True)}
    given = ({"augmentations": eval(ast.unparse(recipe["augmentations"]), setup)}  # noqa: S307
             if "augmentations" in recipe else {})
    run_section = augmentations_of(load_recipe({"dataset_type": mode, **given}))
    ctx = RunContext(
        data_dir=tmp_path, dataset_type=mode, models={}, prompt_dir=None,
        taxonomy="harmful", taxonomy_dir=None, params=setup["PARAMS"],
        augmentations=run_section, resume=True, verbose=False)
    fronts = {stage.front: stage for stage in STAGES.values() if stage.name != "build"}
    called = []
    for call in _calls(path):
        if call.func.id not in fronts:
            continue
        stage = fronts[call.func.id]
        called.append(stage.name)
        # Each step takes the type's params with PARAMS over them, as the run does.
        assert [ast.unparse(kw.value) for kw in call.keywords if kw.arg is None] == [
            f"params({stage.name!r})"]
        assert setup["params"](stage.name) == stage.params(ctx)
    assert called == ["constitution", "inputs", "outputs", "jailbreaks"]
    assert ast.literal_eval(recipe["stages"]) == [*called, "build"]
    (jailbreaks,) = _calls(path, "generate_jailbreaks")
    (section,) = [kw.value for kw in jailbreaks.keywords if kw.arg == "augmentation"]
    # The front resolves its section over the packaged default alone, the run
    # over the type's: they must come to the same kinds.
    section = eval(ast.unparse(section), setup)     # noqa: S307
    assert {kind: resolve(section, kind) for kind in run_section} == run_section
    assert [ast.unparse(kw.value) for kw in run.keywords if kw.arg == "params"] == [
        "PARAMS"]
