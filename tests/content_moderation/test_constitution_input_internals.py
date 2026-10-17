"""Internals-capture wiring for constitution-seeded input generation.

Fully offline (MockBackend). Covers the design in
.claude/introspection_backend_plan.md's "input"/"val_in" rows: every call
generates one sample (a unit is one iteration of an entry), so one forward
pass is one sample's at any ``samples_per_entry``; both root their
internals_id on a pre-call-known identity — the constitution entry's own
composite hash and the unit's iteration for generation, the sample's own
content-hash (its eventual sample_id) for the checker.
"""

import pytest

import tests.content_moderation.test_constitution_input_ledger as ledger_mod
from redact.content_moderation.inputs import InputPipeline
from redact.dataset import content_id
from redact.llms.backends import ComputeConfig
from tests.conftest import MockBackend, make_client

_PROMPT = ledger_mod._PROMPT
_const_df = ledger_mod._const_df
one = ledger_mod.one


class _InternalsBackend(MockBackend):
    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_internals=True)


def _pipeline(tmp_path, gen_backend=None, check_backend=None):
    return InputPipeline(
        gen=make_client(gen_backend or MockBackend(one("alpha sample")), "m"),
        check=make_client(check_backend or MockBackend("Yes"), "m-check"),
        dataset_dir=tmp_path,
    )


def test_the_stage_drives_with_the_pool_its_lanes_allow(tmp_path, monkeypatch):
    import redact.content_moderation.inputs as G

    seen = {}

    def stop(source, **kwargs):
        seen.update(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(G, "drive_units", stop)
    monkeypatch.setattr(G, "pool_size", lambda batch_size, *clients: 7 * batch_size)
    pipe = _pipeline(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1,
                                   batch_size=4, verbose=False)
    assert (seen["pool_size"], seen["cap"]) == (28, 4)


def test_no_internals_kwarg_for_default_backends(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert all("internals_id" not in c for c in pipe.gen.backend.calls)
    assert all("internals_id" not in c for c in pipe.check.backend.calls)


def test_gen_captures_each_unit_of_an_entry_apart(tmp_path):
    gen = _InternalsBackend([one(text) for text in "abcd"])
    pipe = _pipeline(tmp_path, gen_backend=gen)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)

    entry = content_id("desc one:harmful:")
    assert {c["internals_id"] for c in gen.calls} >= {
        f"{entry}/0/input", f"{entry}/1/input"}
    assert len({c["internals_id"] for c in gen.calls}) == 4


def test_gen_internals_id_is_entry_hash_slash_iteration_slash_input(tmp_path):
    gen = _InternalsBackend([one("alpha sample"), one("beta sample")])
    pipe = _pipeline(tmp_path, gen_backend=gen)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1, verbose=False)

    seen = {c["internals_id"] for c in gen.calls}
    expected = {
        f'{content_id("desc one:harmful:")}/0/input',
        f'{content_id("desc two:benign:")}/0/input',
    }
    assert seen == expected


def test_gen_internals_id_includes_style_suffix(tmp_path):
    gen = _InternalsBackend([one("alpha sample"), one("beta sample")])
    pipe = _pipeline(tmp_path, gen_backend=gen)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1,
                               style="long", verbose=False)

    seen = {c["internals_id"] for c in gen.calls}
    expected = {
        f'{content_id("desc one:harmful:long")}/0/input_long',
        f'{content_id("desc two:benign:long")}/0/input_long',
    }
    assert seen == expected


def test_val_in_internals_id_is_sample_hash_slash_val_in(tmp_path):
    check = _InternalsBackend("Yes")
    pipe = _pipeline(
        tmp_path,
        gen_backend=MockBackend([one("alpha sample"), one("beta sample")]),
        check_backend=check,
    )
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1, verbose=False)

    seen = {c["internals_id"] for c in check.calls}
    expected = {
        f'{content_id("alpha sample")}/val_in',
        f'{content_id("beta sample")}/val_in',
    }
    assert seen == expected


def test_val_in_at_several_samples_per_entry(tmp_path):
    # The checker is called once per sample, as the generation is.
    check = _InternalsBackend("Yes")
    pipe = _pipeline(
        tmp_path,
        gen_backend=MockBackend([one(text) for text in "abcd"]),
        check_backend=check,
    )
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert len(check.calls) == 4
    assert all("internals_id" in c and c["internals_id"].endswith("/val_in") for c in check.calls)


def test_val_in_disabled_when_use_checker_false(tmp_path):
    check = _InternalsBackend("Yes")
    pipe = _pipeline(
        tmp_path,
        gen_backend=MockBackend([one("alpha sample"), one("beta sample")]),
        check_backend=check,
    )
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=1,
                               use_checker=False, verbose=False)
    assert check.calls == []
