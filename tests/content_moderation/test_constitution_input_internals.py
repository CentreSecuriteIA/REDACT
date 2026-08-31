"""Internals-capture wiring for constitution-seeded input generation.

Fully offline (MockBackend). Covers the design in
.claude/introspection_backend_plan.md's "input"/"val_in" rows: generation is
gated to samples_per_entry=1 (one forward pass can't be attributed to more than
one resulting sample), the checker is never gated (it's already one call per
already-extracted sample), and both root their internals_id on a pre-call-known
identity — the constitution entry's own composite hash for generation, the
sample's own content-hash (its eventual sample_id) for the checker.
"""

import pytest

import tests.content_moderation.test_constitution_input_ledger as ledger_mod
from redact.content_moderation.generation import InputPipeline
from redact.dataset.io import _hash_text
from redact.llms.backends import ComputeConfig
from tests.conftest import MockBackend, make_client

_PROMPT = ledger_mod._PROMPT
_const_df = ledger_mod._const_df


class _InternalsBackend(MockBackend):
    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_internals=True)


def _pipeline(tmp_path, gen_backend=None, check_backend=None):
    return InputPipeline(
        gen=make_client(gen_backend or MockBackend("1. alpha sample"), "m"),
        check=make_client(check_backend or MockBackend("Yes"), "m"),
        dataset_dir=tmp_path,
    )


def test_no_internals_kwarg_for_default_backends(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert all("internals_id" not in c for c in pipe.gen.backend.calls)
    assert all("internals_id" not in c for c in pipe.check.backend.calls)


def test_gen_raises_when_samples_per_entry_multi(tmp_path):
    gen = _InternalsBackend(["1. a\n2. b"])
    pipe = _pipeline(tmp_path, gen_backend=gen)
    with pytest.raises(ValueError, match="internals capture"):
        pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)


def test_gen_internals_id_is_entry_hash_slash_input(tmp_path):
    gen = _InternalsBackend(["1. alpha sample", "1. beta sample"])
    pipe = _pipeline(tmp_path, gen_backend=gen)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1, verbose=False)

    seen = {c["internals_id"] for c in gen.calls}
    expected = {
        f'{_hash_text("desc one:harmful:")}/input',
        f'{_hash_text("desc two:benign:")}/input',
    }
    assert seen == expected


def test_gen_internals_id_includes_style_suffix(tmp_path):
    gen = _InternalsBackend(["1. alpha sample", "1. beta sample"])
    pipe = _pipeline(tmp_path, gen_backend=gen)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1,
                               style="long", verbose=False)

    seen = {c["internals_id"] for c in gen.calls}
    expected = {
        f'{_hash_text("desc one:harmful:long")}/input_long',
        f'{_hash_text("desc two:benign:long")}/input_long',
    }
    assert seen == expected


def test_val_in_internals_id_is_sample_hash_slash_val_in(tmp_path):
    check = _InternalsBackend("Yes")
    pipe = _pipeline(
        tmp_path,
        gen_backend=MockBackend(["1. alpha sample", "1. beta sample"]),
        check_backend=check,
    )
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1, verbose=False)

    seen = {c["internals_id"] for c in check.calls}
    expected = {
        f'{_hash_text("alpha sample")}/val_in',
        f'{_hash_text("beta sample")}/val_in',
    }
    assert seen == expected


def test_val_in_not_gated_by_multi_sample_per_entry(tmp_path):
    # The checker is called once per already-extracted sample — it's never at
    # risk of the multi-sample-per-call attribution problem, so a val_in-capable
    # backend must NOT raise even when samples_per_entry > 1 (only gen does).
    check = _InternalsBackend("Yes")
    pipe = _pipeline(
        tmp_path,
        gen_backend=MockBackend(["1. a\n2. b", "1. c\n2. d"]),
        check_backend=check,
    )
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert len(check.calls) == 4
    assert all("internals_id" in c and c["internals_id"].endswith("/val_in") for c in check.calls)


def test_val_in_disabled_when_use_checker_false(tmp_path):
    check = _InternalsBackend("Yes")
    pipe = _pipeline(
        tmp_path,
        gen_backend=MockBackend(["1. alpha sample", "1. beta sample"]),
        check_backend=check,
    )
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1,
                               use_checker=False, verbose=False)
    assert check.calls == []
