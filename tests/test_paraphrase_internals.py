"""Internals-capture wiring for generate_paraphrases (case B: provisional id,
renamed to the real sample_id once the paraphrased text is known).

Runs the stage's real units (llm_pipeline.checked on the drive_units pool) on
a ParaphraseBackend, whose checks say Yes, so the actual internals ids and
rename_capture calls the stage makes are exercised, not monkeypatched away.
"""

import pytest

import redact.content_moderation.paraphrase as P
from redact import generate_paraphrases, paths
from redact.dataset import content_id
from redact.llms.backends import ComputeConfig
from tests.conftest import ParaphraseBackend, make_client, write_inputs


@pytest.fixture(autouse=True)
def _no_short_text_skip(monkeypatch):
    """The stub texts are a few words long: the paraphrase stage's short-text
    skip would leave every unit unparaphrased. Its own tests set min_words."""
    monkeypatch.setattr(P, "MIN_WORDS", 0)


class _InternalsBackend(ParaphraseBackend):
    """Paraphrases every text as ``reply`` and tracks rename_capture calls
    too, so the wiring can be asserted without a real
    TransformersIntrospectionBackend (covered in
    tests/llms/test_introspection_backend.py)."""

    def __init__(self, reply="a paraphrase"):
        super().__init__(paraphrase=lambda model, sample: reply)
        self.renames: list[tuple[str, str]] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_internals=True)

    def rename_capture(self, old_internals_id, new_internals_id):
        self.renames.append((old_internals_id, new_internals_id))


def _seed_inputs(tmp_path):
    write_inputs(paths.constitution_inputs_dir(tmp_path), ["alpha prompt"])


def test_no_internals_kwarg_for_default_backend(tmp_path, monkeypatch):
    backend = ParaphraseBackend(paraphrase=lambda model, sample: "a paraphrase")
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
    _seed_inputs(tmp_path)
    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert all("internals_id" not in c for c in backend.calls)


def test_provisional_id_then_renamed_to_sample_id(tmp_path, monkeypatch):
    backend = _InternalsBackend("a paraphrase")
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
    _seed_inputs(tmp_path)
    df = generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)["inputs"]

    base_id = df.iloc[0]["input_id"]
    sample_id = df.iloc[0]["sample_id"]
    assert sample_id == content_id(df.iloc[0]["sample"])

    # generate() was called with the pre-call-known provisional id (bid/k),
    # on the paraphrase only: the check carries none.
    assert backend.calls[0]["internals_id"] == f"{base_id}/paraphrase/attempt_0"
    assert [c.get("internals_id") for c in backend.calls[1:]] == [None]
    # ...then relabeled to the real sample_id once the text came back
    assert backend.renames == [
        (f"{base_id}/paraphrase/attempt_0", f"{base_id}/paraphrase/{sample_id}")
    ]


def test_dropped_paraphrase_is_not_renamed(tmp_path, monkeypatch):
    # A paraphrase that dedupes to a no-op (== original text) is dropped —
    # there's no real sample_id to rename to, so the provisional folder stays.
    backend = _InternalsBackend("alpha prompt")  # identical to the source text
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
    _seed_inputs(tmp_path)
    res = generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)["inputs"]

    assert res.empty  # dropped, nothing written
    assert backend.calls[0]["internals_id"].endswith("/attempt_0")
    assert backend.renames == []


def test_a_paraphrase_of_a_response_is_captured_under_the_response(
        tmp_path, monkeypatch):
    import pandas as pd

    backend = _InternalsBackend("a paraphrase")
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
    outputs = pd.DataFrame({"input_id": ["in-1"], "sample_id": ["out-a"],
                            "output_response": ["an answer"], "accepted": [True]})
    df = generate_paraphrases(outputs=outputs, data_dir=tmp_path, target="outputs",
                              verbose=False)["outputs"]
    sample_id = df.iloc[0]["sample_id"]
    assert backend.calls[0]["internals_id"] == "out-a/paraphrase/attempt_0"
    assert backend.renames == [
        ("out-a/paraphrase/attempt_0", f"out-a/paraphrase/{sample_id}")]
