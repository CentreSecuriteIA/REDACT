"""Tests for the single-source path module (redact/paths.py)."""


import pytest

from redact import paths


def test_root_default_vs_explicit(tmp_path):
    # default root derives from get_output_dir(); explicit root wins.
    assert paths.datasets(tmp_path) == tmp_path / "Datasets"
    assert paths.data_cache(tmp_path) == tmp_path / "Data_cache"
    assert paths.datasets() == paths.datasets(None)  # None -> get_output_dir()


def test_all_artifact_resolvers(tmp_path):
    ds = tmp_path / "Datasets"
    dc = tmp_path / "Data_cache"
    assert paths.jailbreaks_csv(tmp_path) == ds / "jailbreaks.csv"
    assert paths.variations_csv(tmp_path) == ds / "variations.csv"
    assert paths.output_responses_csv(tmp_path) == ds / "output_responses.csv"
    assert paths.complete_dataset_csv(tmp_path) == ds / "complete_dataset.csv"
    assert paths.benign_csv(tmp_path) == dc / "benign" / "benign_samples.csv"
    assert paths.constitution_dir(tmp_path) == dc / "constitution"
    assert paths.constitution_merged_csv(tmp_path) == dc / "constitution" / "merged.csv"
    assert paths.constitution_inputs_dir(tmp_path) == ds / "constitution_inputs"
    assert (paths.constitution_inputs_merged_csv(tmp_path)
            == ds / "constitution_inputs_merged.csv")
    assert paths.paraphrases_inputs_csv(tmp_path) == ds / "paraphrases_inputs.csv"
    assert paths.paraphrases_outputs_csv(tmp_path) == ds / "paraphrases_outputs.csv"
    assert paths.eval_inputs_csv(tmp_path) == ds / "eval_inputs.csv"
    assert paths.eval_outputs_csv(tmp_path) == ds / "eval_outputs.csv"
    assert paths.eval_jailbreaks_csv(tmp_path) == ds / "eval_jailbreaks.csv"
    assert paths.conversations_csv(tmp_path) == ds / "conversations.csv"
    assert paths.conversations_scored_csv(tmp_path) == ds / "conversations_scored.csv"


def test_category_csv_default_and_explicit_root(tmp_path):
    # explicit root wins; the default is the inputs tree, constitution_inputs_dir().
    assert paths.category_csv("Cyber", root=tmp_path) == tmp_path / "Cyber" / "samples.csv"
    assert paths.category_csv("Cyber", root=tmp_path, filename="x.csv") == tmp_path / "Cyber" / "x.csv"
    assert paths.category_csv("Cyber") == (
        paths.constitution_inputs_dir() / "Cyber" / paths.SAMPLES_FILENAME)


@pytest.mark.parametrize("name", ["Home/Garden", "Home\\Garden"])
def test_a_category_name_with_a_path_separator_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="one folder name"):
        paths.category_csv(name, root=tmp_path)


def test_the_output_dir_is_the_env_var_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("REDACT_OUTPUT_DIR", str(tmp_path))
    assert paths.get_output_dir() == tmp_path.resolve()
    assert paths.datasets() == tmp_path.resolve() / "Datasets"


def test_the_package_root_reexports_get_output_dir():
    import redact
    assert redact.get_output_dir is paths.get_output_dir


def test_taxonomy_dir_is_package_relative():
    td = paths.taxonomy_dir()
    assert td.name == "taxonomy" and td.parent.name == "configs"
    assert td.is_absolute()
