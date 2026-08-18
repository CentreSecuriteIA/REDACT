"""Internals-capture wiring for the batched jailbreak engine.

Case B (independent iterations) + nested per-technique tagging: a sample's
provisional root ({id}/jailbreak/attempt_{iteration}) is used to tag every
LLM-backed technique's requests (only for whichever target model's backend
actually supports capture — checked per request), then relabeled to
{id}/jailbreak/{sample_id} once the final jailbroken text is known.

Reuses the FakeRouter/technique fixtures from test_engine.py; only get_backend
is monkeypatched to control per-model internals support.
"""

from redact.jailbreak.engine import batch_apply_combinations
from redact.jailbreak.utils import combine_techniques
from redact.jailbreak.obfuscation.translation import to_swahili
from redact.llms.translator import DEFAULT_TRANSLATE_MODEL

from tests.jailbreak.test_engine import FakeRouter, upper, two_step, GEN_MODEL


def _sample(i, prompt, *techs, iteration=0):
    return {
        "id": f"id{i}", "prompt": prompt, "iteration": iteration,
        "combination": combine_techniques(*techs, sort_by_hierarchy=False),
    }


class _FakeBackend:
    def __init__(self, supports=True):
        self._supports = supports
        self.renames: list[tuple[str, str]] = []

    @property
    def supports_internals(self):
        return self._supports

    def rename_capture(self, old_id, new_id):
        self.renames.append((old_id, new_id))


def _patch_get_backend(monkeypatch, support_by_model: dict[str, bool]):
    """redact.llms.api.get_backend, monkeypatched for _tag_yields/_rename_capture."""
    import redact.llms.api as api_module
    backends = {m: _FakeBackend(s) for m, s in support_by_model.items()}
    monkeypatch.setattr(api_module, "get_backend", lambda m: backends[m])
    return backends


class TestNoCapture:
    def test_capture_internals_false_never_tags(self, monkeypatch):
        backends = _patch_get_backend(monkeypatch, {GEN_MODEL: True})
        samples = [_sample(0, "alpha", upper)]
        router = FakeRouter()
        batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router,
                                 capture_internals=False)
        assert router.calls == []  # pure technique, no LLM call at all — sanity
        assert backends[GEN_MODEL].renames == []


class TestCaptureTagging:
    def test_llm_requests_tagged_under_provisional_root(self, monkeypatch):
        _patch_get_backend(monkeypatch, {GEN_MODEL: True})
        samples = [_sample(0, "alpha", two_step, iteration=2)]
        router = FakeRouter()
        batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router,
                                 capture_internals=True)

        # two_step yields twice (step1, step2) — sequential tags under the
        # provisional root, since the real sample_id isn't known yet.
        assert router.internals_ids_seen == [
            ["id0/jailbreak/attempt_2/two_step/0"],
            ["id0/jailbreak/attempt_2/two_step/1"],
        ]

    def test_relabeled_to_final_sample_id_on_finalize(self, monkeypatch):
        backends = _patch_get_backend(monkeypatch, {GEN_MODEL: True})
        samples = [_sample(0, "alpha", two_step, iteration=0)]
        router = FakeRouter()
        results = batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router,
                                           capture_internals=True)

        sample_id = results[0]["sample_id"]
        assert backends[GEN_MODEL].renames == [
            ("id0/jailbreak/attempt_0", f"id0/jailbreak/{sample_id}")
        ]

    def test_pure_technique_only_chain_has_nothing_to_rename(self, monkeypatch):
        # No LLM call happens at all (pure transform) -> nothing was ever
        # captured under the provisional root, so relabeling is a no-op
        # (the backend's own rename_capture would just find nothing there;
        # this asserts the pipeline still calls it unconditionally though,
        # since it can't know in advance whether anything was captured).
        backends = _patch_get_backend(monkeypatch, {GEN_MODEL: True})
        samples = [_sample(0, "alpha", upper, iteration=0)]
        router = FakeRouter()
        results = batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router,
                                           capture_internals=True)
        sample_id = results[0]["sample_id"]
        assert backends[GEN_MODEL].renames == [
            ("id0/jailbreak/attempt_0", f"id0/jailbreak/{sample_id}")
        ]


class TestMixedModelSupport:
    def test_only_the_supporting_model_gets_tagged(self, monkeypatch):
        # gen_model supports internals, the translation-role model doesn't —
        # to_swahili's requests must never be tagged (would trip BatchCaller's
        # guard on a non-supporting backend), but nothing about gen_model's
        # own calls should be affected by translate_model's lack of support.
        _patch_get_backend(monkeypatch, {
            GEN_MODEL: True, DEFAULT_TRANSLATE_MODEL: False,
        })
        samples = [_sample(0, "alpha", to_swahili, iteration=0)]
        router = FakeRouter()
        batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router,
                                 capture_internals=True)
        # translate + check rounds, both routed to DEFAULT_TRANSLATE_MODEL —
        # neither should carry internals_ids since that model doesn't support it.
        assert router.internals_ids_seen == [None, None]

    def test_untagged_model_calls_never_trip_the_guard(self, monkeypatch):
        # Regression: previously, tagging every yield unconditionally would
        # pass a real internals_ids list to drive_generators for ANY model,
        # even ones whose backend doesn't support capture — which BatchCaller
        # would reject. This must not raise.
        _patch_get_backend(monkeypatch, {
            GEN_MODEL: True, DEFAULT_TRANSLATE_MODEL: False,
        })
        samples = [_sample(0, "alpha", to_swahili, iteration=0)]
        router = FakeRouter()
        results = batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router,
                                           capture_internals=True)
        assert results[0]["accepted"] is True
