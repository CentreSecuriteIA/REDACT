"""Tests for the batched, generator-driven combination engine."""

from collections import Counter

from redact.jailbreak.protocol import LLMRequest
from redact.jailbreak.utils import combine_techniques
from redact.jailbreak.engine import batch_apply_combinations
from redact.jailbreak.obfuscation.translation import to_swahili
from redact.llms.translator import DEFAULT_TRANSLATE_MODEL


GEN_MODEL = "test-gen"


class FakeRouter:
    """Records each batch_generate call and serves canned responses.

    Distinguishes translation translate-vs-check requests by system prompt so a
    single fake can drive the real translation generator end-to-end.
    """

    def __init__(self, check_reply="Yes"):
        self.calls = []  # list of (model, batch_size)
        self.progress_labels = []  # progress= kwarg seen per call (None if absent)
        self.check_reply = check_reply

    def batch_generate(self, model, messages_list, **kwargs):
        self.calls.append((model, len(messages_list)))
        self.progress_labels.append(kwargs.get("progress"))
        out = []
        for msgs in messages_list:
            system = msgs[0]["content"].lower()
            user = msgs[-1]["content"]
            if "checking translation" in system:
                out.append(self.check_reply)            # translation check verdict
            elif "translator" in system:
                out.append("TRANSLATED: " + user[-12:])  # translation output
            elif user.startswith("step1 "):
                out.append("S1")                          # two-step round 1
            elif user.startswith("step2 "):
                out.append("DONE")                        # two-step round 2
            else:
                out.append("GENERIC")
        return out


# --- fake techniques ---------------------------------------------------------

def upper(text, **kwargs):
    """Pure transform."""
    return text.upper(), "upper"


def two_step(text, *, gen_model=None, **kwargs):
    """LLM technique generator with two sequential rounds."""
    a = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    b = yield LLMRequest(gen_model, [{"role": "user", "content": "step2 " + a}])
    return b, "two_step"


def reject_after_one(text, *, gen_model=None, **kwargs):
    """LLM technique generator that rejects after one round."""
    _ = yield LLMRequest(gen_model, [{"role": "user", "content": text}])
    return text, "DISCARDED; feedback=nope"


def _sample(i, prompt, *techs):
    return {"id": f"id{i}", "prompt": prompt,
            "combination": combine_techniques(*techs, sort_by_hierarchy=False)}


class TestBatchApplyCombinations:
    def test_mixed_batch_links_outputs_to_ids(self):
        samples = [
            _sample(0, "alpha", upper),          # pure only
            _sample(1, "beta", to_swahili),      # multi-round translation
            _sample(2, "gamma", two_step),       # 2-step generator
        ]
        router = FakeRouter()
        results = batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router)

        assert len(results) == 3
        by_id = {r["input_id"]: r for r in results}
        assert by_id["id0"]["jailbreak"] == "ALPHA"        # pure ran inline
        assert by_id["id0"]["accepted"] is True
        assert by_id["id1"]["jailbreak"].startswith("TRANSLATED:")
        assert by_id["id1"]["accepted"] is True
        assert by_id["id2"]["jailbreak"] == "DONE"
        assert by_id["id2"]["technique"] == "two_step"

    def test_one_batch_call_per_model_per_round(self):
        samples = [
            _sample(0, "alpha", upper),      # done at prime, no rounds
            _sample(1, "beta", to_swahili),  # deepseek: translate r1, check r2
            _sample(2, "gamma", two_step),   # test-gen: step1 r1, step2 r2
        ]
        router = FakeRouter()
        batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router)

        # Two rounds, two distinct models each round -> 4 calls, each batch of 1.
        assert len(router.calls) == 4
        models = Counter(m for m, _ in router.calls)
        assert models[DEFAULT_TRANSLATE_MODEL] == 2   # translate + check
        assert models[GEN_MODEL] == 2                 # step1 + step2
        assert all(size == 1 for _, size in router.calls)

    def test_pure_samples_share_a_round(self):
        # Two same-model generator samples in one round -> one batched call.
        samples = [_sample(0, "a", two_step), _sample(1, "b", two_step)]
        router = FakeRouter()
        batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router)
        # Round 1 (step1) and round 2 (step2), each batching both samples.
        assert router.calls == [(GEN_MODEL, 2), (GEN_MODEL, 2)]

    def test_discarded_sample_exits_without_blocking(self):
        samples = [
            _sample(0, "alpha", reject_after_one),  # rejects after round 1
            _sample(1, "beta", two_step),           # completes over 2 rounds
        ]
        router = FakeRouter()
        results = batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router)
        by_id = {r["input_id"]: r for r in results}

        assert by_id["id0"]["accepted"] is False
        assert by_id["id0"]["reasoning"].startswith("DISCARDED")
        assert by_id["id0"]["jailbreak"] == "alpha"   # pre-failure text preserved
        # The other sample still finished.
        assert by_id["id1"]["jailbreak"] == "DONE"
        assert by_id["id1"]["accepted"] is True

    def test_translation_exhaustion_discards(self):
        # Checker always rejects -> translation generator exhausts retries.
        samples = [_sample(0, "beta", to_swahili)]
        router = FakeRouter(check_reply="No, the tone is wrong")
        results = batch_apply_combinations(samples, gen_model=GEN_MODEL, router=router)
        assert results[0]["accepted"] is False
        assert "DISCARDED" in results[0]["reasoning"]
        assert "Swahili" in results[0]["reasoning"]

    def test_verbose_passes_round_progress_labels(self):
        # The engine delegates printing to the router/BatchCaller; its job is to
        # pass a per-round progress label. Actual tick printing is covered in
        # tests/llms/test_progress.py and test_wrappers.py.
        def make():
            return [
                _sample(0, "alpha", upper),       # pure, no rounds
                _sample(1, "gamma", two_step),    # 2 LLM rounds
            ]

        quiet_router = FakeRouter()
        quiet = batch_apply_combinations(
            make(), gen_model=GEN_MODEL, router=quiet_router
        )
        verbose_router = FakeRouter()
        verbose = batch_apply_combinations(
            make(), gen_model=GEN_MODEL, router=verbose_router, verbose=True
        )

        # verbose must not change outcomes.
        assert {r["input_id"]: r["jailbreak"] for r in quiet} == \
               {r["input_id"]: r["jailbreak"] for r in verbose}

        # Quiet: no progress kwarg ever passed (preserves minimal router contract).
        assert all(p is None for p in quiet_router.progress_labels)
        # Verbose: a "round N (model)" label per dispatch.
        assert verbose_router.progress_labels == [
            f"round 1 ({GEN_MODEL})",
            f"round 2 ({GEN_MODEL})",
        ]
