"""Tests for the single-source path module (redact/paths.py)."""


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
    assert paths.output_responses_csv(tmp_path) == ds / "output_responses.csv"
    assert paths.complete_dataset_csv(tmp_path) == ds / "complete_dataset.csv"
    assert paths.benign_csv(tmp_path) == dc / "benign" / "benign_samples.csv"
    assert paths.constitution_dir(tmp_path) == dc / "constitution"
    assert paths.constitution_inputs_dir(tmp_path) == ds / "constitution_inputs"
    assert paths.paraphrases_inputs_csv(tmp_path) == ds / "paraphrases_inputs.csv"
    assert paths.paraphrases_outputs_csv(tmp_path) == ds / "paraphrases_outputs.csv"
    assert paths.paraphrased_csv(tmp_path) == ds / "paraphrased.csv"
    assert paths.conversations_csv(tmp_path) == ds / "conversations.csv"


def test_category_csv_default_and_explicit_base(tmp_path):
    # explicit base wins; default base is datasets().
    assert paths.category_csv("Cyber", base=tmp_path) == tmp_path / "Cyber" / "samples.csv"
    assert paths.category_csv("Cyber", base=tmp_path, filename="x.csv") == tmp_path / "Cyber" / "x.csv"
    assert paths.category_csv("Cyber") == paths.datasets() / "Cyber" / paths.SAMPLES_FILENAME


def test_taxonomy_dir_is_package_relative():
    td = paths.taxonomy_dir()
    assert td.name == "taxonomy" and td.parent.name == "configs"
    assert td.is_absolute()
