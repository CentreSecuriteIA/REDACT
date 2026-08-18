"""Internals-capture wiring for generate_paraphrases (case B: provisional id,
renamed to the real sample_id once the paraphrased text is known).

Runs the real paraphrase_batch()/build_messages() path (only batch_check_samples
is stubbed, to keep acceptance deterministic) so the actual internals_ids/
rename_capture calls the pipeline makes are exercised, not just monkeypatched away.
"""

import pandas as pd
import pytest

import redact.pipelines as P
from redact import generate_paraphrases, paths
from redact.dataset.io import append_samples, _hash_text
from tests.conftest import MockBackend


class _InternalsBackend(MockBackend):
    """Tracks rename_capture calls too, so the wiring can be asserted without
    needing a real TransformersIntrospectionBackend (already covered in
    tests/llms/test_introspection_backend.py)."""

    def __init__(self, responses="mock response"):
        super().__init__(responses)
        self.renames: list[tuple[str, str]] = []

    @property
    def supports_internals(self) -> bool:
        return True

    def rename_capture(self, old_internals_id, new_internals_id):
        self.renames.append((old_internals_id, new_internals_id))


def _seed_inputs(tmp_path):
    append_samples(
        ["alpha prompt"], category="Cyber", turn=0,
        accepted=[True], dataset_dir=paths.datasets(tmp_path),
    )


@pytest.fixture()
def accept_all(monkeypatch):
    monkeypatch.setattr(
        P, "batch_check_samples",
        lambda cb, cm, payloads, checker, **kw: [(True, "") for _ in payloads],
    )


def test_no_internals_kwarg_for_default_backend(tmp_path, monkeypatch, accept_all):
    backend = MockBackend("a paraphrase")
    monkeypatch.setattr(P, "get_backend", lambda m: backend)
    _seed_inputs(tmp_path)
    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert all("internals_id" not in c for c in backend.calls)


def test_provisional_id_then_renamed_to_sample_id(tmp_path, monkeypatch, accept_all):
    backend = _InternalsBackend("a paraphrase")
    monkeypatch.setattr(P, "get_backend", lambda m: backend)
    _seed_inputs(tmp_path)
    df = generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)["inputs"]

    base_id = df.iloc[0]["input_id"]
    sample_id = df.iloc[0]["sample_id"]
    assert sample_id == _hash_text(df.iloc[0]["sample"])

    # generate() was called with the pre-call-known provisional id (bid/k)
    assert backend.calls[0]["internals_id"] == f"{base_id}/paraphrase/attempt_0"
    # ...then relabeled to the real sample_id once the text came back
    assert backend.renames == [
        (f"{base_id}/paraphrase/attempt_0", f"{base_id}/paraphrase/{sample_id}")
    ]


def test_dropped_paraphrase_is_not_renamed(tmp_path, monkeypatch, accept_all):
    # A paraphrase that dedupes to a no-op (== original text) is dropped —
    # there's no real sample_id to rename to, so the provisional folder stays.
    backend = _InternalsBackend("alpha prompt")  # identical to the source text
    monkeypatch.setattr(P, "get_backend", lambda m: backend)
    _seed_inputs(tmp_path)
    res = generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)["inputs"]

    assert res.empty  # dropped, nothing written
    assert backend.calls[0]["internals_id"].endswith("/attempt_0")
    assert backend.renames == []
