"""Tests for rate limiting, the system-prompt policy, and the batch caller."""

import os
import threading
import time
from unittest.mock import patch

import pytest

from redact.llms.backends import (
    ComputeConfig,
    extract_system_prompt,
    fold_system_into_first_message,
)
from redact.llms.client import ModelClient, clear_client_cache
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    get_model_config,
    register_model,
)
from redact.llms import wrappers
from redact.llms.wrappers import BatchCaller, RateLimiter
from tests.conftest import MockBackend, make_client


def caller_for(backend, model="mock-model", **kw):
    """A BatchCaller over a backend bound to ``model``'s registry entry."""
    return BatchCaller(make_client(backend, model).backend, **kw)


class TestFoldSystemIntoFirstMessage:
    """Lives in backends/base.py: the system-prompt policy is a fact of the
    model, applied by the backend it is bound to."""

    def test_noop_when_no_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        assert fold_system_into_first_message(messages) is messages

    def test_folds_system_into_first_user_message(self):
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ]
        result = fold_system_into_first_message(messages)
        assert result == [{"role": "user", "content": "Be terse.\n\nHello"}]

    def test_multiple_system_messages_merged_in_order(self):
        messages = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hi"},
        ]
        result = fold_system_into_first_message(messages)
        assert result == [{"role": "user", "content": "Rule 1.\n\nRule 2.\n\nHi"}]

    def test_only_system_messages_become_single_user_message(self):
        messages = [{"role": "system", "content": "Just a system prompt."}]
        result = fold_system_into_first_message(messages)
        assert result == [{"role": "user", "content": "Just a system prompt."}]

    def test_preserves_later_messages_untouched(self):
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = fold_system_into_first_message(messages)
        assert result == [
            {"role": "user", "content": "Sys.\n\nQ1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

    def test_does_not_mutate_input(self):
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q"},
        ]
        original_first = messages[1]
        fold_system_into_first_message(messages)
        assert messages[1] is original_first
        assert messages[1]["content"] == "Q"


class TestExtractSystemPrompt:
    """extract_system_prompt: the supports_system_prompt=True counterpart to
    fold_system_into_first_message — splits rather than merges."""

    def test_no_system_message_returns_none_and_same_messages(self):
        messages = [{"role": "user", "content": "hi"}]
        system_prompt, rest = extract_system_prompt(messages)
        assert system_prompt is None
        assert rest == messages

    def test_single_system_message_extracted(self):
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, rest = extract_system_prompt(messages)
        assert system_prompt == "Be terse."
        assert rest == [{"role": "user", "content": "Hello"}]

    def test_multiple_system_messages_joined_in_order(self):
        messages = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hi"},
        ]
        system_prompt, rest = extract_system_prompt(messages)
        assert system_prompt == "Rule 1.\n\nRule 2."
        assert rest == [{"role": "user", "content": "Hi"}]

    def test_only_system_messages_leaves_empty_rest(self):
        messages = [{"role": "system", "content": "Just a system prompt."}]
        system_prompt, rest = extract_system_prompt(messages)
        assert system_prompt == "Just a system prompt."
        assert rest == []

    def test_preserves_non_system_messages_and_order(self):
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        system_prompt, rest = extract_system_prompt(messages)
        assert system_prompt == "Sys."
        assert rest == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]


class TestBoundParamResolution:
    """Defaults and the system-prompt extract-vs-fold branch are settled when
    the backend is built, so they show up in what actually reaches the
    transport — there is no separate per-call resolution step any more.
    """

    def _call(self, model, messages_list, **kw):
        backend = MockBackend("ok")
        make_client(backend, model).generate(messages_list, **kw)
        return backend.calls

    def test_registered_model_uses_its_max_tokens_default(self):
        assert self._call("claude-opus-4-6",
                          [[{"role": "user", "content": "hi"}]])[0]["max_tokens"] == 300

    def test_explicit_max_tokens_overrides_default(self):
        calls = self._call("claude-opus-4-6", [[{"role": "user", "content": "hi"}]],
                           max_tokens=1000)
        assert calls[0]["max_tokens"] == 1000

    def test_explicit_temperature_overrides_default(self):
        calls = self._call("venice-uncensored", [[{"role": "user", "content": "hi"}]],
                           temperature=0.3)
        assert calls[0]["temperature"] == 0.3

    def test_supports_system_prompt_true_extracts(self):
        calls = self._call("venice-uncensored", [[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "hi"},
        ]])
        assert calls[0]["system_prompt"] == "Sys."
        assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_system_message_after_the_user_turn_reaches_the_transport(self):
        calls = self._call("venice-uncensored", [[
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Sys."},
        ]])
        assert calls[0]["system_prompt"] == "Sys."
        assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_several_system_messages_are_merged_in_order(self):
        calls = self._call("venice-uncensored", [[
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "hi"},
        ]])
        assert calls[0]["system_prompt"] == "First.\n\nSecond."

    def test_empty_system_message_counts_as_none(self):
        calls = self._call("venice-uncensored", [[
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ]])
        assert "system_prompt" not in calls[0]
        assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_supports_system_prompt_false_folds(self):
        name = "_test_no_system_prompt_model"
        try:
            register_model(
                name,
                backend_type="api",
                supports_system_prompt=False,
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                              base_url="https://test.example/v1", rpm=5),
            )
            calls = self._call(name, [[
                {"role": "system", "content": "Sys."},
                {"role": "user", "content": "hi"},
            ]])
            assert "system_prompt" not in calls[0]
            assert calls[0]["messages"] == [{"role": "user", "content": "Sys.\n\nhi"}]
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_several_system_messages_are_folded_in_order(self):
        name = "_test_no_system_prompt_model_several"
        try:
            register_model(
                name,
                backend_type="api",
                supports_system_prompt=False,
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                              base_url="https://test.example/v1", rpm=5),
            )
            calls = self._call(name, [[
                {"role": "system", "content": "First."},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Second."},
            ]])
            assert "system_prompt" not in calls[0]
            assert calls[0]["messages"] == [
                {"role": "user", "content": "First.\n\nSecond.\n\nhi"}
            ]
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_batch_of_multiple_items_resolved_independently(self):
        calls = self._call("venice-uncensored", [
            [{"role": "system", "content": "Sys A."}, {"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ])
        assert calls[0]["system_prompt"] == "Sys A."
        assert "system_prompt" not in calls[1]

    def test_bound_extra_body_never_reaches_a_local_binding(self):
        """The dual-setup entry is the only one that can express this: binding
        venice-uncensored's .vllm setup must not inherit the endpoint's
        provider params (they would land in SamplingParams and raise)."""
        from redact.llms.backends import backend_for
        from redact.llms.model_config import get_model_config

        api_backend = backend_for(get_model_config("venice-uncensored"), "api")
        assert api_backend._extra_body is not None
        assert not hasattr(api_backend, "hf_model_id")


class TestRateLimiter:
    def test_allows_under_rpm(self):
        limiter = RateLimiter()
        start = time.time()
        limiter.wait_if_needed(make_client(model="rl-under-rpm").backend)
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_local_backend_is_not_rate_limited(self):
        # A backend built from a model's local setup carries rpm=None, so the
        # limiter must not track it at all — even though its registry entry
        # also describes a rate-limited hosted endpoint.
        from redact.llms.backends import LLMBackend

        class _LocalT(LLMBackend):
            def generate(self, messages_list, **kw):
                return [""] * len(messages_list)

            @property
            def backend_name(self):
                return "vllm"

        backend = make_client(_LocalT("x"), "venice-uncensored", backend_type="vllm").backend
        assert backend.rpm is None
        limiter = RateLimiter()
        limiter.wait_if_needed(backend)
        assert limiter._timestamps == []

    def test_one_instance_is_one_window(self):
        """The key lives in shared_limiter(), not inside the limiter. Two
        backends handed to the same instance deliberately share its window —
        keeping them apart is the caller's job, done by asking for two
        instances."""
        limiter = RateLimiter()
        limiter.wait_if_needed(make_client(model="model-a").backend)
        limiter.wait_if_needed(make_client(model="model-b").backend)
        assert len(limiter._timestamps) == 2

    def test_rechecks_capacity_after_sleeping(self):
        """Regression: the sleep path was a single `if` with an unconditional
        append, so a waiter that woke to a *still-full* window issued its
        request anyway and overshot the cap. Deterministic here: the first
        sleep frees nothing, so a correct limiter must sleep again rather than
        proceed.
        """
        rpm = 3
        backend = make_client(model="rl-recheck").backend
        backend.rpm = rpm
        limiter = RateLimiter()
        for _ in range(rpm):
            limiter.wait_if_needed(backend)
        assert len(limiter._timestamps) == rpm

        calls = []

        def _sleep(_seconds):
            calls.append(_seconds)
            if len(calls) == 2:        # only the SECOND sleep frees a slot
                limiter._timestamps[0] -= 120.0

        with patch.object(wrappers.time, "sleep", side_effect=_sleep):
            limiter.wait_if_needed(backend)

        assert len(calls) == 2, "woke to a full window and proceeded anyway"
        assert len(limiter._timestamps) == rpm, "cap exceeded"

    def test_tracks_timestamps(self):
        limiter = RateLimiter()
        backend = make_client(model="rl-timestamps").backend
        limiter.wait_if_needed(backend)
        limiter.wait_if_needed(backend)
        assert len(limiter._timestamps) == 2

    def test_a_wall_clock_jump_does_not_empty_the_window(self):
        """The window runs on the monotonic clock, so a full window stays
        full when the system clock is set forward."""
        rpm = 2
        backend = make_client(model="rl-monotonic").backend
        backend.rpm = rpm
        limiter = RateLimiter()
        for _ in range(rpm):
            limiter.wait_if_needed(backend)

        slept = []

        def _sleep(seconds):
            slept.append(seconds)
            limiter._timestamps.clear()      # the window has passed

        wall_clock = time.time
        with patch.object(wrappers.time, "time", lambda: wall_clock() + 3600.0), \
             patch.object(wrappers.time, "sleep", side_effect=_sleep):
            limiter.wait_if_needed(backend)

        assert len(slept) == 1, "a clock jump let a request through a full window"


class TestBatchCaller:
    def test_sequential(self):
        backend = MockBackend(["r0", "r1", "r2"])
        caller = caller_for(backend)
        messages_list = [
            [{"role": "user", "content": f"msg{i}"}]
            for i in range(3)
        ]
        results = caller.run(messages_list)
        assert results == ["r0", "r1", "r2"]

    def test_preserves_order(self):
        backend = MockBackend(["a", "b", "c"])
        caller = caller_for(backend, max_workers=1)
        messages_list = [
            [{"role": "user", "content": str(i)}]
            for i in range(3)
        ]
        results = caller.run(messages_list)
        assert results == ["a", "b", "c"]

    def test_callback_called(self):
        backend = MockBackend(["x", "y"])
        caller = caller_for(backend)
        completed = []
        caller.run([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            on_complete=lambda i, r: completed.append((i, r)),
        )
        assert len(completed) == 2
        assert (0, "x") in completed
        assert (1, "y") in completed

    def test_concurrent(self):
        backend = MockBackend(["a", "b", "c"])
        caller = caller_for(backend, max_workers=2)
        messages_list = [
            [{"role": "user", "content": str(i)}]
            for i in range(3)
        ]
        results = caller.run(messages_list)
        assert sorted(results) == ["a", "b", "c"]

    def test_with_rate_limiter(self):
        backend = MockBackend("ok")
        limiter = RateLimiter()
        caller = caller_for(backend, "rl-test-model", rate_limiter=limiter)
        results = caller.run([[{"role": "user", "content": "hi"}]])
        assert results == ["ok"]
        assert len(limiter._timestamps) == 1

    def test_passes_kwargs(self):
        backend = MockBackend("ok")
        caller = caller_for(backend)
        caller.run([[{"role": "user", "content": "hi"}]],
            max_tokens=50,
        )
        assert backend.calls[0]["max_tokens"] == 50

    def test_a_failed_item_cancels_the_items_not_yet_started(self):
        started = []

        class _FailsOnTheFirstItem(MockBackend):
            def generate(self, messages_list, **kwargs):
                text = messages_list[0][0]["content"]
                started.append(text)
                if text == "0":
                    raise RuntimeError("boom")
                time.sleep(0.05)
                return ["ok"]

        caller = caller_for(_FailsOnTheFirstItem(), max_workers=2)
        batch = [[{"role": "user", "content": str(i)}] for i in range(30)]
        with pytest.raises(RuntimeError, match="boom"):
            caller.run(batch)
        assert len(started) < len(batch)


class _InternalsBackend(MockBackend):
    """MockBackend variant that declares internals support."""

    compute_config = ComputeConfig(supports_internals=True)


class TestBatchCallerInternalsIds:
    """internals_ids/internals_id: capability-gated, never leaks to unsupported backends."""

    def test_run_raises_when_backend_does_not_support_internals(self):
        backend = MockBackend("ok")  # supports_internals=False (base default)
        caller = caller_for(backend)
        with pytest.raises(ValueError, match="does not support internals"):
            caller.run([[{"role": "user", "content": "hi"}]],
                internals_ids=["id-0"],
            )

    def test_no_error_and_no_kwarg_leak_when_internals_ids_is_none(self):
        # The common case: internals_ids omitted entirely — must be a total no-op,
        # not just "no crash" (confirms nothing reaches backend.generate as a kwarg).
        backend = MockBackend("ok")
        caller = caller_for(backend)
        caller.run([[{"role": "user", "content": "hi"}]])
        assert "internals_id" not in backend.calls[0]

    def test_run_unzips_internals_ids_per_call(self):
        backend = _InternalsBackend(["r0", "r1"])
        caller = caller_for(backend)
        caller.run([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            internals_ids=["id-0", "id-1"],
        )
        assert backend.calls[0]["internals_id"] == "id-0"
        assert backend.calls[1]["internals_id"] == "id-1"

    def test_length_mismatch_raises(self):
        backend = _InternalsBackend(["r0", "r1"])
        caller = caller_for(backend)
        with pytest.raises(ValueError, match="same length"):
            caller.run([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
                internals_ids=["only-one"],
            )


class TestAssertSingleSamplePerCall:
    """Guard for pipelines shaped like run_from_constitution (N samples/call)."""

    def test_noop_when_backend_does_not_support_internals(self):
        from redact.llms.router import assert_single_sample_per_call
        backend = MockBackend("ok")  # supports_internals=False
        assert_single_sample_per_call(make_client(backend), samples_per_call=5)  # no raise

    def test_noop_when_samples_per_call_is_one(self):
        from redact.llms.router import assert_single_sample_per_call
        assert_single_sample_per_call(make_client(_InternalsBackend("ok")), samples_per_call=1)  # no raise

    def test_raises_when_internals_and_multi_sample(self):
        from redact.llms.router import assert_single_sample_per_call
        with pytest.raises(ValueError, match="internals capture"):
            assert_single_sample_per_call(make_client(_InternalsBackend("ok")), samples_per_call=3)

class TestRateLimitScope:
    """Whose budget `rpm` describes: this model, or the whole account.

    Venice meters per model, so three models on one key hold three independent
    windows. Anthropic meters the account, so N models each taking their own
    full budget would collectively blow it — two entries at rpm=5 issuing
    10/min against a 5/min account.
    """

    @staticmethod
    def _register(name, *, scope, rpm, key_env="TEST_KEY", base_url="https://t/v1"):
        register_model(
            name, backend_type="api",
            api=APIConfig(backend_type="openai", api_key_env=key_env,
                          base_url=base_url, rpm=rpm, rate_limit_scope=scope),
        )
        return name

    @pytest.fixture(autouse=True)
    def _clean(self):
        yield
        for n in list(MODEL_REGISTRY):
            if n.startswith("_test_scope_"):
                MODEL_REGISTRY.pop(n, None)
        clear_client_cache()

    def test_defaults_to_per_model(self):
        assert get_model_config("venice-uncensored").api.rate_limit_scope == "model"

    def test_per_model_scope_gives_each_model_its_own_window(self):
        """Two Venice models on one key must not throttle each other."""
        a = self._register("_test_scope_a", scope="model", rpm=10)
        b = self._register("_test_scope_b", scope="model", rpm=20)
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            with patch.dict(os.environ, {"TEST_KEY": "k"}):
                ca, cb = ModelClient.create(a), ModelClient.create(b)
        assert ca._rate_limiter is not cb._rate_limiter
        assert (ca._limit_key, cb._limit_key) == (a, b)

    def test_endpoint_scope_shares_one_limiter_and_one_key(self):
        """Both halves matter: a shared instance keyed per model would still
        hand each model its own window, which is the bug this prevents."""
        a = self._register("_test_scope_acct_a", scope="endpoint", rpm=5)
        b = self._register("_test_scope_acct_b", scope="endpoint", rpm=5)
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            with patch.dict(os.environ, {"TEST_KEY": "k"}):
                ca, cb = ModelClient.create(a), ModelClient.create(b)
        assert ca._rate_limiter is cb._rate_limiter
        assert ca._limit_key == cb._limit_key
        assert ca._limit_key == get_model_config(a).api.endpoint_id

    def test_endpoint_scope_does_not_leak_across_endpoints(self):
        a = self._register("_test_scope_e1", scope="endpoint", rpm=5,
                           base_url="https://one/v1")
        b = self._register("_test_scope_e2", scope="endpoint", rpm=5,
                           base_url="https://two/v1")
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            with patch.dict(os.environ, {"TEST_KEY": "k"}):
                ca, cb = ModelClient.create(a), ModelClient.create(b)
        assert ca._rate_limiter is not cb._rate_limiter
        assert ca._limit_key != cb._limit_key

    def test_shared_window_actually_counts_both_models(self):
        """The point of the whole feature, asserted on the limiter itself."""
        a = self._register("_test_scope_shared_a", scope="endpoint", rpm=5)
        b = self._register("_test_scope_shared_b", scope="endpoint", rpm=5)
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            with patch.dict(os.environ, {"TEST_KEY": "k"}):
                ca, cb = ModelClient.create(a), ModelClient.create(b)
        ca._rate_limiter.wait_if_needed(ca.backend)
        cb._rate_limiter.wait_if_needed(cb.backend)
        assert len(ca._rate_limiter._timestamps) == 2,             "both models must count against one window"

    def test_disagreeing_rpm_on_one_endpoint_is_rejected(self):
        """One window cannot honour two budgets; picking either silently would
        over- or under-spend with nothing to point at."""
        a = self._register("_test_scope_bad_a", scope="endpoint", rpm=5)
        b = self._register("_test_scope_bad_b", scope="endpoint", rpm=50)
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            with patch.dict(os.environ, {"TEST_KEY": "k"}):
                ModelClient.create(a)
                with pytest.raises(ValueError, match="different rpm"):
                    ModelClient.create(b)

    def test_a_model_registered_again_refreshes_its_own_claim(self):
        a = self._register("_test_scope_again_a", scope="endpoint", rpm=5)
        b = self._register("_test_scope_again_b", scope="endpoint", rpm=5)
        raised = 50
        with patch("redact.llms.backends.openai.openai.OpenAI"), \
             patch.dict(os.environ, {"TEST_KEY": "k"}):
            ModelClient.create(a)
            self._register(a, scope="endpoint", rpm=raised)
            assert ModelClient.create(a).backend.rpm == raised
            # The refreshed claim still binds the other models.
            with pytest.raises(ValueError, match="different rpm"):
                ModelClient.create(b)

    def test_local_setup_never_takes_a_shared_limiter(self):
        """A local binding carries rpm=None, so the limiter is inert anyway —
        but it must not be handed the endpoint's window regardless. Resolved
        directly: building the backend would load real weights."""
        from redact.llms.client import _resolve_rate_limit

        config = get_model_config("llama-3.2-3b-debug")
        assert _resolve_rate_limit("llama-3.2-3b-debug", config, "vllm") == (None, None)

    def test_rejects_an_unknown_scope(self):
        with pytest.raises(ValueError, match="rate_limit_scope"):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=1, rate_limit_scope="account")

    def test_shipped_anthropic_entry_is_endpoint_scoped(self):
        """Anthropic meters the account — the case this was built for."""
        assert get_model_config("claude-opus-4-6").api.rate_limit_scope == "endpoint"


