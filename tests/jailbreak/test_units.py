"""A planned unit as the driver runs it: its generator, its row, its resume.

Drives units through ``llm_pipeline.drive_units`` — the path the stage uses —
with content-dispatch backends, the way ``test_pipeline_jailbreaks`` does, since
the old round-by-round engine (``batch_apply_combinations``) is gone.
"""

import random

from redact.dataset import content_id
from redact.jailbreak.chain import combine_techniques
from redact.jailbreak.obfuscation.translation import (
    _resolve_translate_model,
    to_swahili,
)
from redact.jailbreak.units import (
    error_row_of,
    is_pending,
    pending_units,
    row_of,
    unit_gen,
)
from redact.llm_pipeline import LLMRequest, drive_units
from redact.llms.backends import ComputeConfig
from tests.conftest import make_client

GEN = "test-gen"
TRANS = _resolve_translate_model()


# --- fake techniques ---------------------------------------------------------

def upper(text, **kwargs):
    """Pure transform."""
    return text.upper(), "upper"


def two_step(text, *, gen_model=None, **kwargs):
    """Generator with two sequential rounds."""
    a = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    b = yield LLMRequest(gen_model, [{"role": "user", "content": "step2 " + a}])
    return b, "two_step"


def reject_after_one(text, *, gen_model=None, **kwargs):
    """Generator that rejects after one round."""
    yield LLMRequest(gen_model, [{"role": "user", "content": text}])
    return text, "DISCARDED; feedback=nope"


class _Backend:
    """A content-dispatch fake, one per model.

    ``step1/step2`` -> ``S1``/``DONE``; a translator request -> ``TRANSLATED:
    ...`` and a translation check -> ``check_reply``. Records each
    ``generate()`` call's size, reply limit and internals ids.
    """

    def __init__(self, model, *, check_reply="Yes", supports_internals=False):
        self.model = model
        self.check_reply = check_reply
        self._supports = supports_internals
        self.calls: list[int] = []
        self.limits: list = []
        self.internals: list = []
        self.renames: list[tuple[str, str]] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True,
                             supports_internals=self._supports)

    def generate(self, messages_list, *, max_tokens=None, internals_ids=None, **kwargs):
        self.calls.append(len(messages_list))
        self.limits.append(max_tokens)
        self.internals.append(internals_ids)
        out = []
        for msgs in messages_list:
            system, user = msgs[0]["content"].lower(), msgs[-1]["content"]
            if "checking translation" in system:
                out.append(self.check_reply)
            elif "translator" in system:
                out.append("TRANSLATED: " + user[-12:])
            elif user.startswith("step1 "):
                out.append("S1")
            elif user.startswith("step2 "):
                out.append("DONE")
            else:
                out.append("GENERIC")
        return out

    def rename_capture(self, old_id, new_id):
        self.renames.append((old_id, new_id))


def _unit(i, prompt, *techs, iteration=0):
    return {"id": f"id{i}", "prompt": prompt, "iteration": iteration,
            "combination": combine_techniques(*techs, sort_by_hierarchy=False)}


def _drive(units, backends, *, capture=False, progress=None, spec_version=""):
    """Drive ``units`` through ``drive_units`` and return their rows in order."""
    resolve = lambda m: make_client(backends[m], m)  # noqa: E731
    live: dict[int, dict] = {}
    rows: dict[int, dict] = {}

    def source():
        for i, u in enumerate(units):
            live[i] = u
            yield i, unit_gen(u, gen_model=GEN, translate_model=TRANS, capture=capture)

    def on_done(i, value):
        rows[i] = row_of(live.pop(i), value, {}, spec_version,
                         gen_model=GEN, resolve=resolve)

    def on_error(i, exc):
        rows[i] = error_row_of(live.pop(i), exc, {}, spec_version,
                               gen_model=GEN, resolve=resolve)

    n = len(units)
    drive_units(source(), on_done=on_done, on_error=on_error,
                pool_size=max(2 * n, 2), cap=max(n, 1),
                resolve=resolve, progress=progress)
    return [rows[i] for i in range(n) if i in rows]


def _backends(**kwargs):
    return {GEN: _Backend(GEN, **kwargs), TRANS: _Backend(TRANS, **kwargs)}


# --- is_pending / pending_units ----------------------------------------------

class TestPending:
    def test_is_pending_needs_the_input_and_an_unfinished_unit(self):
        pos = {"a": 0}
        assert is_pending({"sample_id": "a", "iteration": 0}, pos, set())
        assert not is_pending({"sample_id": "a", "iteration": 0}, pos, {("a", 0)})
        assert not is_pending({"sample_id": "b", "iteration": 0}, pos, set())

    def test_pending_units_builds_lazily_records_live_and_keys_by_position(self):
        import pandas as pd

        plan = [
            {"sample_id": "a", "iteration": 0, "combination": ["upper"]},
            {"sample_id": "b", "iteration": 0, "combination": ["upper"]},  # no input
            {"sample_id": "a", "iteration": 1, "combination": ["upper"]},
        ]
        prompts = pd.Series(["alpha"])
        live: dict = {}
        built = list(pending_units(
            plan, prompts, {"a": 0}, set(), {"upper": upper}, live,
            gen_model=GEN))
        # Two of the three plan rows are of these inputs, keyed 0 and 1.
        assert [i for i, _ in built] == [0, 1]
        assert set(live) == {0, 1}
        assert live[0]["id"] == "a" and live[0]["iteration"] == 0
        assert live[1]["iteration"] == 1


# --- row_of / error_row_of ---------------------------------------------------

class TestRows:
    def test_row_of_carries_the_input_columns_then_the_jailbreak_fields(self):
        unit = _unit(0, "alpha", upper, iteration=2)
        row = row_of(unit, ("ALPHA", "upper"), {"category": "Cyber", "sample_id": "in"},
                     "1.0", gen_model=GEN)
        assert row == {
            "category": "Cyber",
            "sample_id": content_id("ALPHA"),   # its own, overwriting the input's
            "input_id": "id0",
            "jailbreak": "ALPHA",
            "technique": "upper",
            "technique_info": "upper",
            "complexity": 0,
            "num_techniques": 1,
            "is_noop": False,
            "accepted": True,
            "reasoning": "",
            "iteration": 2,
            "combination_spec_version": "1.0",
        }

    def test_row_of_marks_a_discarded_result_rejected(self):
        unit = _unit(0, "alpha", upper)
        row = row_of(unit, ("alpha", "DISCARDED; feedback=no"), {}, "", gen_model=GEN)
        assert row["accepted"] is False
        assert row["reasoning"] == "DISCARDED; feedback=no"
        assert row["is_noop"] is True   # text unchanged

    def test_error_row_has_its_own_sample_id_not_the_inputs(self):
        unit = _unit(0, "alpha", two_step, iteration=3)
        row = error_row_of(unit, ValueError("boom"), {}, "1.0", gen_model=GEN)
        assert row["input_id"] == "id0"
        assert row["sample_id"] == content_id("id0:3")
        assert row["sample_id"] != content_id("alpha")   # not the input's id
        assert row["jailbreak"] == "alpha"
        assert row["technique_info"] == "ERROR: boom"
        assert row["reasoning"] == "boom"
        assert row["is_noop"] is True and row["accepted"] is False


# --- driving units -----------------------------------------------------------

class TestDrive:
    def test_a_mixed_batch_links_rows_to_their_inputs(self):
        units = [_unit(0, "alpha", upper), _unit(1, "beta", to_swahili),
                 _unit(2, "gamma", two_step)]
        rows = {r["input_id"]: r for r in _drive(units, _backends())}
        assert rows["id0"]["jailbreak"] == "ALPHA" and rows["id0"]["accepted"] is True
        assert rows["id1"]["jailbreak"].startswith("TRANSLATED:")
        assert rows["id2"]["jailbreak"] == "DONE"
        assert rows["id2"]["technique"] == "two_step"

    def test_sample_id_is_the_jailbreaks_own_hash(self):
        units = [_unit(0, "alpha", upper), _unit(0, "alpha", to_swahili)]
        rows = _drive(units, _backends())
        assert all(r["input_id"] == "id0" for r in rows)
        assert len({r["sample_id"] for r in rows}) == len(rows)
        for r in rows:
            assert r["sample_id"] == content_id(r["jailbreak"]) != r["input_id"]

    def test_one_call_per_model_per_round(self):
        units = [_unit(0, "alpha", upper), _unit(1, "beta", to_swahili),
                 _unit(2, "gamma", two_step)]
        backends = _backends()
        _drive(units, backends)
        # two_step: step1, step2 on GEN; translation: translate, check on TRANS.
        assert backends[GEN].calls == [1, 1]
        assert backends[TRANS].calls == [1, 1]

    def test_two_same_model_units_share_a_round(self):
        backends = _backends()
        _drive([_unit(0, "a", two_step), _unit(1, "b", two_step)], backends)
        assert backends[GEN].calls == [2, 2]   # step1 round, step2 round

    def test_a_discarded_unit_exits_without_blocking(self):
        units = [_unit(0, "alpha", reject_after_one), _unit(1, "beta", two_step)]
        rows = {r["input_id"]: r for r in _drive(units, _backends())}
        assert rows["id0"]["accepted"] is False
        assert rows["id0"]["jailbreak"] == "alpha"
        assert rows["id1"]["jailbreak"] == "DONE"

    def test_translation_exhaustion_discards(self):
        rows = _drive([_unit(0, "beta", to_swahili)],
                      _backends(check_reply="No, the tone is wrong"))
        assert rows[0]["accepted"] is False and "Swahili" in rows[0]["reasoning"]

    def test_an_error_becomes_an_error_row(self):
        def boom(text, *, gen_model=None, **kwargs):
            yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
            raise ValueError("chain failed")

        rows = _drive([_unit(0, "a", boom)], _backends())
        assert rows[0]["technique_info"] == "ERROR: chain failed"
        assert rows[0]["sample_id"] == content_id("id0:0")

    def test_a_technique_tagged_with_max_tokens_caps_its_requests(self):
        def capped(text, *, gen_model=None, **kwargs):
            msgs = [{"role": "user", "content": "step1 " + text}]
            reply = yield LLMRequest(gen_model, msgs)
            return reply, "capped"
        capped.max_tokens = 40

        backends = _backends()
        _drive([_unit(0, "a", capped), _unit(1, "b", two_step)], backends)
        # The capped (40) and uncapped (None) requests are separate calls.
        assert sorted(backends[GEN].limits, key=str) == [40, None, None]


# --- internals capture -------------------------------------------------------

class TestCapture:
    def _support(self, monkeypatch, by_model):
        import redact.llms.model_config as model_config

        monkeypatch.setattr(
            model_config, "model_compute_config",
            lambda m, backend_type=None: ComputeConfig(
                supports_internals=by_model.get(m, False)))

    def test_requests_tagged_then_relabeled_to_the_sample_id(self, monkeypatch):
        self._support(monkeypatch, {GEN: True})
        backends = _backends(supports_internals=True)
        unit = _unit(0, "alpha", two_step, iteration=2)
        rows = _drive([unit], backends, capture=True)

        # two_step yields twice, tagged sequentially under the provisional root.
        assert backends[GEN].internals == [
            ["id0/jailbreak/attempt_2/two_step/0"],
            ["id0/jailbreak/attempt_2/two_step/1"],
        ]
        assert backends[GEN].renames == [
            ("id0/jailbreak/attempt_2", f"id0/jailbreak/{rows[0]['sample_id']}")]

    def test_no_capture_tags_nothing(self, monkeypatch):
        self._support(monkeypatch, {GEN: True})
        backends = _backends(supports_internals=True)
        _drive([_unit(0, "alpha", two_step)], backends, capture=False)
        assert backends[GEN].internals == [None, None]
        assert backends[GEN].renames == []

    def test_only_a_supporting_model_is_tagged(self, monkeypatch):
        # GEN supports internals, the translation model does not: to_swahili's
        # requests must never carry an id, or they would trip the client's guard.
        self._support(monkeypatch, {GEN: True, TRANS: False})
        backends = {GEN: _Backend(GEN, supports_internals=True),
                    TRANS: _Backend(TRANS, supports_internals=False)}
        rows = _drive([_unit(0, "alpha", to_swahili)], backends, capture=True)
        assert backends[TRANS].internals == [None, None]
        assert rows[0]["accepted"] is True


def test_module_random_is_untouched():
    """A sanity check that these fakes do not reseed the global RNG."""
    random.seed(0)
    first = random.random()
    random.seed(0)
    assert random.random() == first
