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
