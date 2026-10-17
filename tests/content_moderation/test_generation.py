"""Tests for content moderation InputPipeline with mock backend."""

from redact import paths
from redact.content_moderation.inputs import InputPipeline
from tests.conftest import MockBackend, make_client, write_inputs


def test_the_pipeline_writes_to_the_inputs_tree_by_default():
    client = make_client(MockBackend(), "m")
    pipeline = InputPipeline(gen=client, check=client)
    assert pipeline.dataset_dir == paths.constitution_inputs_dir()


class TestOutputsOfInputsWithoutASubcategory:
    def test_an_input_without_a_subcategory_gives_an_empty_one(
            self, tmp_path, monkeypatch):
        """Not the text ``nan``."""
        import pandas as pd

        import redact.stages as P
        from redact import generate_outputs

        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(MockBackend("ans"), m))
        write_inputs(paths.constitution_inputs_dir(tmp_path), ["an input"],
                     extra={"entry_type": "harmful", "subcategory": ""})
        generate_outputs(data_dir=tmp_path, check_outputs=False, verbose=False)
        written = pd.read_csv(paths.output_responses_csv(tmp_path), dtype=str,
                              keep_default_na=False)
        assert list(written["subcategory"]) == [""]


class TestGeneralBenignEntries:
    """An entry of the general-benign part is an input of its benign category."""

    _PROMPT = {"system_prompt": "sys", "template": "about {sample_description}"}

    def _run(self, tmp_path, monkeypatch, **entry):
        import pandas as pd

        import redact.content_moderation.inputs as G

        checked = []
        build = G.build_quality_checker
        monkeypatch.setattr(G, "build_quality_checker",
                            lambda **k: checked.append(k) or build(**k))
        pipeline = InputPipeline(
            gen=make_client(MockBackend("1. a question about bread"), "m"),
            check=make_client(MockBackend("Yes"), "m-check"), dataset_dir=tmp_path)
        pipeline.run_from_constitution(
            pd.DataFrame([{"source_category": "Cooking", "sample_description": "desc",
                           "constitution_subcategory": "Bread",
                           "constitution_category": "Baking", "entry_type": "benign",
                           "source_group_tag": "general", **entry}]),
            self._PROMPT, samples_per_entry=1, verbose=False)
        rows = pd.read_csv(tmp_path / "Cooking" / "samples.csv", dtype=str,
                           keep_default_na=False)
        return checked, rows

    def test_it_is_checked_against_its_benign_category_and_keeps_its_taxonomy(
            self, tmp_path, monkeypatch):
        checked, rows = self._run(tmp_path, monkeypatch, source_taxonomy="benign")
        assert checked == [{"category": "Cooking", "entry_type": "benign",
                            "subcategory": "Bread", "prompt_dir": None}]
        assert rows[["category", "entry_type", "subcategory", "source_taxonomy",
                     "accepted"]].to_dict("records") == [
            {"category": "Cooking", "entry_type": "benign", "subcategory": "Bread",
             "source_taxonomy": "benign", "accepted": "True"}]

    def test_a_constitution_without_the_column_gives_inputs_without_it(
            self, tmp_path, monkeypatch):
        _, rows = self._run(tmp_path, monkeypatch)
        assert "source_taxonomy" not in rows.columns
