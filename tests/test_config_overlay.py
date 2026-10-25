"""A user's config folder over the packaged configs/, and the folder
``scaffold_setup`` writes."""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from redact import paths, telemetry
from redact.config_overlay import (
    config_sources,
    extra_eval_techniques,
    report_config_sources,
    taxonomy_folder,
    unseen_natural_families,
)
from redact.content_moderation import axes as A
from redact.driver import read_record, run_pipeline
from redact.jailbreak.augmentation import resolve
from redact.recipe import check_dimensions, load_dataset_type, load_params, load_recipe
from redact.stages import STAGES, RunContext
from tests.conftest import MockBackend, make_client

_SIX = ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks", "build"]


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _augmentations(**adversarial) -> dict:
    """The packaged augmentations.json with ``adversarial`` keys over its kind."""
    data = json.loads((paths.jailbreak_configs_dir() / "augmentations.json")
                      .read_text(encoding="utf-8"))
    data["adversarial"].update(adversarial)
    return data


# --- each kind of file is read from the folder first ---------------------------

def test_a_dataset_type_file_in_the_folder_wins_and_the_others_stay_packaged(tmp_path):
    _write(tmp_path / "dataset_types" / "eval.json",
           {"stages": ["inputs"], "params": {}, "augmentations": {}})
    eval_recipe = load_recipe({"dataset_type": "eval", "config_dir": str(tmp_path)})
    assert eval_recipe["stages"] == ["inputs"]
    assert eval_recipe["augmentations"] == {
        kind: resolve(None, kind) for kind in ("adversarial", "natural")}
    assert load_dataset_type("eval", tmp_path)["stages"] == ["inputs"]
    assert load_dataset_type("eval")["stages"] == _SIX
    assert load_recipe({"dataset_type": "training",
                        "config_dir": str(tmp_path)})["stages"] == _SIX


def test_a_dataset_type_of_another_name_in_the_folder_is_refused_and_not_read(
        tmp_path):
    _write(tmp_path / "dataset_types" / "mine.json",
           {"stages": [], "params": {}, "augmentations": {}})
    message = "dataset_type must be \"eval\" or \"training\", got 'mine'."
    with pytest.raises(ValueError) as raised:
        load_recipe({"dataset_type": "mine", "config_dir": str(tmp_path)})
    assert str(raised.value) == message
    with pytest.raises(ValueError) as raised:
        load_dataset_type("mine", tmp_path)
    assert str(raised.value) == message
    assert config_sources(tmp_path)["ignored"] == ["dataset_types/mine.json"]


def test_a_taxonomy_file_in_the_folder_wins_and_the_others_stay_packaged(
        tmp_path, monkeypatch):
    from redact import generate_constitution
    from redact.llms import ModelClient
    from tests.conftest import MockBackend, make_client

    _write(tmp_path / "taxonomy" / "harmful.json",
           {"name": "harmful", "categories": {"Mine": {"description": "d"}}})
    assert taxonomy_folder("harmful", None, tmp_path) == tmp_path / "taxonomy"
    assert taxonomy_folder("benign", None, tmp_path) is None
    assert taxonomy_folder("harmful") is None

    backend = MockBackend("no entry\n[END]")
    monkeypatch.setattr(ModelClient, "create",
                        lambda model, *a, **k: make_client(backend, model))
    generate_constitution(entry_types=["harmful"], model="m", data_dir=tmp_path,
                          config_dir=tmp_path, include_general_benign=True,
                          num_taxonomy_categories=1, verbose=False, read_back=False)
    keys = {json.loads(line)["unit"] for line in (
        paths.constitution_dir(tmp_path) / "constitution.state.jsonl"
    ).read_text(encoding="utf-8").splitlines() if '"unit"' in line}
    benign = (paths.taxonomy_dir() / "benign.json").read_text(encoding="utf-8")
    packaged_benign = next(iter(json.loads(benign)["categories"]))
    assert keys == {"Mine::harmful", f"{packaged_benign}::general_benign"}


def test_taxonomy_dir_wins_over_the_folders_taxonomy(tmp_path):
    _write(tmp_path / "cfg" / "taxonomy" / "harmful.json", {"categories": {}})
    assert taxonomy_folder("harmful", tmp_path / "tax", tmp_path / "cfg") == (
        tmp_path / "tax")
    ctx = RunContext(
        data_dir=tmp_path, dataset_type="training", models={}, prompt_dir=None,
        taxonomy="harmful", taxonomy_dir=None, params={}, augmentations={},
        resume=True, verbose=False, config_dir=tmp_path / "cfg")
    assert STAGES["constitution"].sources(ctx) == [
        tmp_path / "cfg" / "taxonomy" / "harmful.json",
        paths.taxonomy_dir() / "benign.json"]


def test_an_augmentations_file_in_the_folder_is_the_default_of_both_kinds(tmp_path):
    _write(tmp_path / "jailbreak" / "augmentations.json",
           {**_augmentations(max_complexity=3), "natural_only_families": []})
    recipe = load_recipe({"dataset_type": "training", "config_dir": str(tmp_path)})
    adversarial = recipe["augmentations"]["adversarial"]
    assert adversarial["max_complexity"] == 3      # noqa: PLR2004
    assert adversarial["families"]["deny"] == []    # no natural-only family
    assert resolve(None, "adversarial", config_dir=tmp_path) == adversarial
    # Resolved again over the same folder, as the jailbreak front does, it stays.
    assert resolve({"adversarial": adversarial}, "adversarial",
                   config_dir=tmp_path) == adversarial
    assert resolve(None, "adversarial")["max_complexity"] != 3     # noqa: PLR2004


def test_an_unknown_key_in_an_augmentations_copy_is_left_out_with_a_warning(
        tmp_path, caplog):
    """A typo did nothing before, so a copy may hold one: the run warns and
    resolves as without it."""
    data = _augmentations(max_complexty=9)
    data["adversarial"]["layers"] = {"deny": [], "alow": ["rewrite"]}
    path = _write(tmp_path / "jailbreak" / "augmentations.json", data)
    with caplog.at_level("WARNING", logger="redact.jailbreak.augmentation"):
        adversarial = resolve(None, "adversarial", config_dir=tmp_path)
    assert adversarial == resolve(None, "adversarial")
    assert [r.getMessage().split(" which")[0] for r in caplog.records] == [
        f"{path}: adversarial has unknown key(s) ['max_complexty'],",
        f"{path}: adversarial.layers has unknown key(s) ['alow'],"]


@pytest.mark.parametrize(("change", "message"), [
    (lambda d: d["adversarial"].pop("layers"), "adversarial lacks ['layers']"),
    (lambda d: d["natural"].pop("exact_techniques"),
     "natural lacks ['exact_techniques']"),
    (lambda d: d.pop("natural_only_families"), "lacks natural_only_families"),
    (lambda d: d.__setitem__("natural", None),
     "natural must be an object of the kind's keys, got None"),
    (lambda d: d["adversarial"].__setitem__("exact_techniques", 1),
     "adversarial gives exact_techniques, a key of the natural kind"),
], ids=["no layers", "no exact_techniques", "no natural_only_families",
        "a kind null", "a count on the attacks"])
def test_an_augmentations_copy_that_cannot_be_read_raises_naming_the_file(
        tmp_path, change, message):
    """What failed as a bare KeyError, or read the attacks as the natural
    kind, now stops the recipe at load with the file and the key."""
    data = _augmentations()
    change(data)
    path = _write(tmp_path / "jailbreak" / "augmentations.json", data)
    with pytest.raises(ValueError) as raised:
        load_recipe({"dataset_type": "training", "config_dir": str(tmp_path)})
    assert str(raised.value).startswith(str(path))
    assert message in str(raised.value)


def test_the_jailbreak_front_resolves_over_the_folder(tmp_path, monkeypatch):
    import redact.stages as P

    _write(tmp_path / "jailbreak" / "augmentations.json",
           {**_augmentations(), "natural_only_families": []})
    pools = []
    monkeypatch.setattr(P, "run_augmentation",
                        lambda inputs, pool, **k: pools.append(pool) or pd.DataFrame())
    monkeypatch.setattr(P.ModelClient, "create", lambda model: None)
    P.generate_jailbreaks(
        inputs=pd.DataFrame({"sample": ["x"], "entry_type": ["harmful"]}),
        data_dir=tmp_path, config_dir=tmp_path, verbose=False,
        augmentation={"natural": None})
    (pool,) = pools
    assert any("casual_writing" in fn.families for fn in pool)


# --- what the folder holds that is not read ------------------------------------

def test_a_file_the_overlay_does_not_read_is_warned_about(tmp_path, caplog):
    for relative in ("llm/models.json", "jailbreak/combination_spec.json",
                     "jailbreak/languages.json",
                     "jailbreak/hacking/personas.json", "datasets/x.json",
                     "dataset_types/eval.json", "dataset_types/mine.json"):
        _write(tmp_path / relative, {})
    _write(tmp_path / "taxonomy" / "harmful.json", {"categories": {}})
    _write(tmp_path / "taxonomy" / "mine.json", {"categories": {}})
    with caplog.at_level("WARNING", logger="redact.config_overlay"):
        found = report_config_sources(tmp_path, verbose=False)
    assert found == config_sources(tmp_path) == {
        "overridden": ["dataset_types/eval.json", "taxonomy/harmful.json"],
        "added": ["taxonomy/mine.json"],
        "ignored": ["dataset_types/mine.json", "datasets/x.json",
                    "jailbreak/combination_spec.json",
                    "jailbreak/hacking/personas.json", "jailbreak/languages.json",
                    "llm/models.json"]}
    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert "llm/models.json" in record.getMessage()
    assert "dataset_types/mine.json" in record.getMessage()
    assert "eval.json and training.json" in record.getMessage()
    assert "ignored" in record.getMessage()


def _natural_allowing(*families: str) -> dict:
    """The packaged augmentations.json with its natural kind allowing
    ``families`` only: a copy made before the library allowed the others."""
    data = _augmentations()
    data["natural"]["families"]["allow"] = list(families)
    return data


def test_a_copy_whose_natural_kind_misses_a_packaged_family_is_logged(
        tmp_path, caplog):
    _write(tmp_path / "jailbreak" / "augmentations.json", _natural_allowing(
        "typos", "casual_writing", "answer_language", "answer_format"))
    # Spec 1.4's language-mix families are allowed by the packaged natural kind
    # too, so a copy made before them misses them as well.
    assert unseen_natural_families(tmp_path) == [
        "noise", "neutral_container", "asking", "indirect", "smalltalk_prefix",
        "smalltalk_suffix", "benign_role", "transliteration", "code_switching",
        "romanization", "dialect"]
    with caplog.at_level("INFO", logger="redact.config_overlay"):
        report_config_sources(tmp_path, verbose=False)
    (record,) = caplog.records
    assert record.levelname == "INFO"
    assert ("does not allow noise, neutral_container, asking, indirect, "
            "smalltalk_prefix, smalltalk_suffix, benign_role, transliteration, "
            "code_switching, romanization, dialect" in record.getMessage())
    assert "overwrite=True" in record.getMessage()


def test_a_current_copy_or_none_misses_no_natural_family(tmp_path):
    assert unseen_natural_families(tmp_path) == []
    _write(tmp_path / "jailbreak" / "augmentations.json", _augmentations())
    assert unseen_natural_families(tmp_path) == []
    # A natural kind that allows every family, or is off, misses none.
    data = _augmentations()
    data["natural"]["families"]["allow"] = None
    _write(tmp_path / "jailbreak" / "augmentations.json", data)
    assert unseen_natural_families(tmp_path) == []
    _write(tmp_path / "jailbreak" / "augmentations.json", {**data, "natural": None})
    assert unseen_natural_families(tmp_path) == []


def _eval_before_growth() -> dict:
    """The packaged eval.json as a copy made before the library grew would
    hold it: the temporal family allowed whole, and two denies missing."""
    data = json.loads((paths.configs_dir() / "dataset_types" / "eval.json")
                      .read_text(encoding="utf-8"))
    adversarial = data["augmentations"]["adversarial"]
    adversarial["families"]["allow"].append("temporal")
    adversarial["techniques"]["allow"].remove("to_use_past_tense")
    adversarial["techniques"]["deny"].remove("to_hex")
    adversarial["techniques"]["deny"].remove("to_fsh_random_many")
    return data


def test_a_stale_eval_copy_names_what_it_adds_to_the_eval_pool(tmp_path, caplog):
    _write(tmp_path / "dataset_types" / "eval.json", _eval_before_growth())
    extra = ["to_fsh_random_many", "to_hex", "to_use_future_tense",
             "to_use_historical_era"]
    assert extra_eval_techniques(tmp_path) == extra
    with caplog.at_level("WARNING", logger="redact.config_overlay"):
        report_config_sources(tmp_path, verbose=False, dataset_type="eval")
    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert f"draw 4 technique(s) the library's eval type does not: {', '.join(extra)}" \
        in record.getMessage()
    assert "overwrite=True" in record.getMessage()


def test_only_an_eval_run_is_warned_about_its_eval_copy(tmp_path, caplog):
    _write(tmp_path / "dataset_types" / "eval.json", _eval_before_growth())
    with caplog.at_level("WARNING", logger="redact.config_overlay"):
        report_config_sources(tmp_path, verbose=False, dataset_type="training")
        report_config_sources(tmp_path, verbose=False)
    assert caplog.records == []


def test_a_current_eval_copy_or_none_adds_nothing(tmp_path, caplog):
    assert extra_eval_techniques(tmp_path) == []
    _write(tmp_path / "dataset_types" / "eval.json", json.loads(
        (paths.configs_dir() / "dataset_types" / "eval.json").read_text("utf-8")))
    _write(tmp_path / "jailbreak" / "augmentations.json", _augmentations())
    with caplog.at_level("WARNING", logger="redact.config_overlay"):
        report_config_sources(tmp_path, verbose=False, dataset_type="eval")
    assert extra_eval_techniques(tmp_path) == []
    assert caplog.records == []


def test_a_run_records_the_files_it_read_from_the_folder(tmp_path, monkeypatch, caplog):
    import redact

    config = tmp_path / "cfg"
    _write(config / "taxonomy" / "harmful.json", {"categories": {}})
    _write(config / "llm" / "models.json", {})
    monkeypatch.setattr(redact, "generate_outputs", lambda **k: pd.DataFrame())
    recipe = tmp_path / "recipe.json"
    _write(recipe, {"dataset_type": "training", "config_dir": "cfg",
                    "data_dir": str(tmp_path), "stages": ["outputs"]})
    try:
        with caplog.at_level("INFO", logger="redact.config_overlay"):
            run_pipeline(str(recipe), params={}, verbose=True)
    finally:
        telemetry.uninstall()
    record = read_record("outputs", tmp_path)
    assert record["config_dir"] == str(config)
    assert record["config_files"] == ["taxonomy/harmful.json"]
    assert [r.levelname for r in caplog.records
            if r.name == "redact.config_overlay"] == ["WARNING", "INFO"]


def test_config_dir_is_read_relative_to_the_recipe(tmp_path):
    (tmp_path / "sub" / "cfg").mkdir(parents=True)
    recipe = _write(tmp_path / "sub" / "recipe.json",
                    {"dataset_type": "eval", "config_dir": "cfg"})
    assert load_recipe(recipe)["config_dir"] == str(tmp_path / "sub" / "cfg")


def test_a_config_dir_that_is_no_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not a folder"):
        load_recipe({"dataset_type": "eval", "config_dir": str(tmp_path / "none")})


# --- the generation axes and the dimensions ------------------------------------

def _overlaid(config: Path, name: str, change: Callable[[dict], object]) -> Path:
    """Write the packaged ``configs/<name>`` into ``config``, changed by
    ``change``."""
    data = json.loads((paths.configs_dir() / name).read_text(encoding="utf-8"))
    change(data)
    return _write(config / name, data)


def _very_long_only(axes: dict) -> None:
    for value, entry in axes["inputs"]["length"]["values"].items():
        entry["weight"] = 1 if value == "very_long" else 0


def _training_augmentations() -> dict:
    return load_recipe({"dataset_type": "training"})["augmentations"]


def test_an_axes_file_in_the_folder_is_read_and_reported(tmp_path):
    _overlaid(tmp_path, "generation_axes.json", _very_long_only)
    mine = A.resolve("inputs", config_dir=tmp_path)
    assert [value for value, entry in mine["length"]["values"].items()
            if entry["weight"]] == ["very_long"]
    assert A.resolve("inputs") != mine
    assert config_sources(tmp_path)["overridden"] == ["generation_axes.json"]


def test_a_dimensions_file_in_the_folder_is_what_the_check_reads(tmp_path):
    """Here, focus decided apart from the families it was undecided against."""
    _overlaid(tmp_path, "dimensions.json", lambda tags: tags["focus"].pop("undecided"))
    mixed = {"inputs": {"axes": {"focus": {"mixed": 0.25}}}}
    with pytest.raises(ValueError, match="inputs.focus"):
        check_dimensions(mixed, _training_augmentations(), dataset_type="training")
    check_dimensions(mixed, _training_augmentations(), dataset_type="training",
                     config_dir=tmp_path)
    assert config_sources(tmp_path)["overridden"] == ["dimensions.json"]


def test_a_dimensions_file_without_a_tag_a_family_carries_is_refused(tmp_path):
    _overlaid(tmp_path, "dimensions.json", lambda tags: tags.pop("answer_shape"))
    with pytest.raises(ValueError, match=r"answer_format \('answer_shape'\)"):
        check_dimensions({}, _training_augmentations(), dataset_type="training",
                         config_dir=tmp_path)


def test_an_axes_or_dimensions_file_elsewhere_in_the_folder_is_ignored(tmp_path):
    _write(tmp_path / "jailbreak" / "generation_axes.json", {})
    _write(tmp_path / "llm" / "dimensions.json", {})
    _write(tmp_path / "notes.json", {})     # beside the recipes: none of configs/
    assert config_sources(tmp_path) == {
        "overridden": [], "added": [],
        "ignored": ["jailbreak/generation_axes.json", "llm/dimensions.json"]}
    assert A.resolve("inputs", config_dir=tmp_path) == A.resolve("inputs")


def test_the_inputs_front_deals_the_axes_of_the_folder(tmp_path, monkeypatch):
    from redact import generate_inputs
    from redact.llms import ModelClient

    config = tmp_path / "cfg"
    _overlaid(config, "generation_axes.json", _very_long_only)
    backends = {"gen-model": MockBackend(["1. one zq0 sample\n[END]",
                                          "1. another zq0 sample\n[END]"]),
                "check-model": MockBackend("Yes")}
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(backends[m], m))
    constitution = pd.DataFrame([{
        "source_category": "Cyber", "constitution_subcategory": "Sub",
        "sample_description": "what zq0 is about", "entry_type": "harmful"}])
    rows = generate_inputs(
        data_dir=tmp_path, constitution_df=constitution, samples_per_entry=2,
        model="gen-model", check_model="check-model", config_dir=config,
        verbose=False)
    assert rows["length"].tolist() == ["very_long", "very_long"]


def test_the_outputs_front_deals_the_axes_of_the_folder(tmp_path, monkeypatch):
    from redact import generate_outputs
    from redact.llms import ModelClient

    def tables_only(axes: dict) -> None:
        for value, entry in axes["outputs"]["format"]["values"].items():
            entry["weight"] = 1 if value == "table" else 0

    config = tmp_path / "cfg"
    _overlaid(config, "generation_axes.json", tables_only)
    monkeypatch.setattr(ModelClient, "create",
                        lambda m: make_client(MockBackend("an answer"), m))
    rows = generate_outputs(
        inputs=pd.DataFrame({"sample": ["one", "two"], "category": "Cyber",
                             "entry_type": "harmful"}),
        data_dir=tmp_path, model="gen-model", check_outputs=False,
        config_dir=config, verbose=False)
    assert rows["response_format"].tolist() == ["table", "table"]


@pytest.mark.parametrize("stage", ["inputs", "outputs"])
def test_a_run_passes_the_front_its_config_dir(tmp_path, stage):
    ctx = RunContext(
        data_dir=tmp_path, dataset_type="training", models={"gen": None,
                                                            "check": None},
        prompt_dir=None, taxonomy="harmful", taxonomy_dir=None, params={},
        augmentations={}, resume=True, verbose=False, config_dir=tmp_path)
    assert STAGES[stage].recipe_kwargs(ctx)["config_dir"] == tmp_path


# --- scaffold_setup -------------------------------------------------------------

_COPIES = {
    "dataset_types/eval.json": "dataset_types/eval.json",
    "dataset_types/training.json": "dataset_types/training.json",
    "taxonomy/harmful.json": "taxonomy/harmful.json",
    "taxonomy/benign.json": "taxonomy/benign.json",
    "jailbreak/augmentations.json": "jailbreak/augmentations.json",
    "generation_axes.json": "generation_axes.json",
    "dimensions.json": "dimensions.json",
    "eval_params.json": "runs/params_eval_example.json",
    "training_params.json": "runs/params_training_example.json",
}


def test_scaffold_setup_writes_the_packaged_files_as_they_are(tmp_path):
    from redact import scaffold_setup

    written = scaffold_setup(tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == sorted(
        [*_COPIES, "eval_recipe.json", "training_recipe.json", "README.md"])
    for mine, packaged in _COPIES.items():
        assert (tmp_path / mine).read_bytes() == (
            paths.configs_dir() / packaged).read_bytes(), mine
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "scaffold_prompts.py" in readme and "prompt_dir" in readme


@pytest.mark.parametrize(("mine", "packaged"), [
    ("eval_recipe.json", "eval_example.json"),
    ("training_recipe.json", "training_example.json")])
def test_a_scaffolded_recipe_runs_as_the_packaged_example(tmp_path, mine, packaged):
    from redact import scaffold_setup

    scaffold_setup(tmp_path / "setup")
    ours = load_recipe(tmp_path / "setup" / mine)
    theirs = load_recipe(paths.configs_dir() / "runs" / packaged)
    assert ours["config_dir"] == str(tmp_path / "setup")
    assert ours["stages"] == theirs["stages"]
    assert ours["augmentations"] == theirs["augmentations"]
    assert ours["data_dir"] == theirs["data_dir"]

    def effective(recipe):
        ctx = RunContext(
            data_dir=tmp_path, dataset_type=recipe["dataset_type"], models={},
            prompt_dir=None, taxonomy=recipe["taxonomy"], taxonomy_dir=None,
            params=load_params(recipe["params_file"], base_dir=recipe["_base_dir"]),
            augmentations=recipe["augmentations"], resume=True, verbose=False,
            config_dir=recipe.get("config_dir"))
        return {stage: STAGES[stage].params(ctx) for stage in recipe["stages"]}

    assert effective(ours) == effective(theirs)


def test_scaffold_setup_refuses_to_overwrite_unless_asked(tmp_path):
    from redact import scaffold_setup

    scaffold_setup(tmp_path)
    (tmp_path / "taxonomy" / "harmful.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError) as raised:
        scaffold_setup(tmp_path)
    assert "taxonomy/harmful.json" in str(raised.value)
    assert "overwrite=True" in str(raised.value)
    assert (tmp_path / "taxonomy" / "harmful.json").read_text(encoding="utf-8") == "{}"
    scaffold_setup(tmp_path, overwrite=True)
    assert (tmp_path / "taxonomy" / "harmful.json").read_bytes() == (
        paths.taxonomy_dir() / "harmful.json").read_bytes()


def test_the_scaffold_script_writes_the_folder(tmp_path, capsys):
    import importlib.util

    script = Path(__file__).parents[1] / "scripts" / "scaffold_setup.py"
    spec = importlib.util.spec_from_file_location("scaffold_setup_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([str(tmp_path)]) == 0
    assert (tmp_path / "eval_recipe.json").is_file()
    assert module.main([str(tmp_path)]) == 1
    assert "overwrite" in capsys.readouterr().err
    assert module.main([str(tmp_path), "--overwrite"]) == 0
