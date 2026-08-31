"""Tests for router.py's caller-facing helpers and for the client that
executes them — dispatch strategy, chunking, and the check loop.
"""

import logging
import os
from unittest.mock import patch

import pytest

from redact.llms import ModelClient
from redact.llms.backends import ComputeConfig, backend_for, clear_transport_caches
from redact.llms.client import clear_client_cache
from redact.llms.model_config import get_model_config
from redact.llms.router import (
    batch_check_samples,
    batch_generate_samples,
    check_sample,
    generate_sample,
)
from redact.llms.wrappers import BatchCaller
from tests.conftest import MockBackend, make_client


class TestGenerateSample:
    def test_calls_backend(self, mock_backend):
        messages = [{"role": "user", "content": "hi"}]
        result = generate_sample(make_client(mock_backend, "test-model"), messages)
        assert result == "mock response"
        assert len(mock_backend.calls) == 1
        assert mock_backend.calls[0]["model"] == "test-model"

    def test_passes_kwargs(self, mock_backend):
        messages = [{"role": "user", "content": "hi"}]
        generate_sample(make_client(mock_backend), messages, max_tokens=100)
        assert mock_backend.calls[0]["max_tokens"] == 100


class TestCheckSample:
    def _check(self, backend, response_text):
        b = MockBackend(response_text)
        return check_sample(make_client(b), "sample text",
            build_check_messages=lambda o, s: [{"role": "user", "content": f"Check: {s}"}],
        )

    def test_accepted_yes(self, mock_backend):
        assert self._check(mock_backend, "Yes, this looks good")[0] is True

    def test_accepted_ok(self, mock_backend):
        assert self._check(mock_backend, "ok")[0] is True

    def test_accepted_accept(self, mock_backend):
        assert self._check(mock_backend, "ACCEPT")[0] is True

    def test_accepted_pass(self, mock_backend):
        assert self._check(mock_backend, "Pass - looks fine")[0] is True

    def test_rejected(self, mock_backend):
        accepted, reasoning = self._check(mock_backend, "No, this is off-topic")
        assert accepted is False
        assert "off-topic" in reasoning

    def test_reasoning_empty_on_accept(self, mock_backend):
        _, reasoning = self._check(mock_backend, "yes")
        assert reasoning == ""


class _CountingBackend(MockBackend):
    """MockBackend that also counts real generate() invocations — the
    fixed shared MockBackend records one call-record per *batch item*
    regardless of how many real generate() invocations produced them, so
    telling apart "one native call with N items" from "N calls with one
    item each" (BatchCaller's per-item fan-out) needs its own counter.
    """

    def __init__(self, *args, native: bool = False, **kwargs):
        # Set before super(): LLMBackend.__init__ reads compute_config to
        # clamp max_workers, and this class's is derived from _native.
        self._native = native
        self.generate_call_count = 0
        super().__init__(*args, **kwargs)

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=self._native, supports_internals=True)

    def generate(self, messages_list, **kwargs):
        self.generate_call_count += 1
        return super().generate(messages_list, **kwargs)


class TestDispatchBatch:
    """ModelClient is the one place ``supports_native_batching`` is read —
    everything here tests that decision directly, plus the progress and
    internals-id behavior layered on top of it.
    """

    def _messages(self, n):
        return [[{"role": "user", "content": str(i)}] for i in range(n)]

    def test_empty_messages_list_returns_empty(self):
        backend = _CountingBackend([])
        assert make_client(backend).generate([]) == []
        assert backend.generate_call_count == 0

    def test_native_batching_dispatches_in_one_call(self):
        backend = _CountingBackend(["a", "b", "c"], native=True)
        results = make_client(backend).generate(self._messages(3))
        assert results == ["a", "b", "c"]
        assert backend.generate_call_count == 1  # one native engine pass
        assert len(backend.calls) == 3

    def test_non_native_dispatches_one_call_per_item(self):
        # BatchCaller.run() (default max_workers=1) fans out via _call_one,
        # which always wraps a single item — confirms BatchCaller is
        # actually doing per-item dispatch, not secretly batching.
        backend = _CountingBackend(["a", "b", "c"], native=False)
        results = make_client(backend).generate(self._messages(3))
        assert results == ["a", "b", "c"]
        assert backend.generate_call_count == 3

    def test_internals_ids_forwarded_through_native_batch(self):
        backend = _CountingBackend(["r0", "r1"], native=True)
        results = make_client(backend).generate(self._messages(2), internals_ids=["id-0", "id-1"],
        )
        assert results == ["r0", "r1"]
        assert backend.calls[0]["internals_id"] == "id-0"
        assert backend.calls[1]["internals_id"] == "id-1"

    def test_internals_ids_raise_when_backend_does_not_support_them(self):
        backend = MockBackend("ok")  # supports_internals=False (base default)
        with pytest.raises(ValueError, match="does not support internals"):
            make_client(backend).generate(self._messages(1), internals_ids=["id-0"])

    def test_progress_label_emits_ticks_without_changing_results(self, caplog):
        backend = _CountingBackend(["x", "y", "z"])
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            results = make_client(backend).generate(self._messages(3), progress="gen chunk 1/1")
        assert results == ["x", "y", "z"]
        assert "gen chunk 1/1: 0/3 ..." in caplog.text
        assert "gen chunk 1/1: 3/3 done" in caplog.text

    def test_progress_chains_with_on_complete(self, caplog):
        backend = _CountingBackend(["a", "b"])
        seen = []
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            results = make_client(backend).generate(self._messages(2),
                on_complete=lambda i, r: seen.append((i, r)),
                progress="lbl",
            )
        assert results == ["a", "b"]
        assert sorted(seen) == [(0, "a"), (1, "b")]
        assert "lbl: 2/2 done" in caplog.text

    def test_uses_batch_caller_for_non_native(self):
        # Sanity: the non-native path really does go through BatchCaller,
        # not some parallel hand-rolled fan-out.
        backend = _CountingBackend("ok", native=False)
        caller = BatchCaller(make_client(backend, "m").backend)
        assert caller.max_workers == 1


class TestBatchChunkingWithInternalsIds:
    """internals_ids must be sliced per chunk, not forwarded whole.

    Every real caller today passes batch_size=len(messages_list), so a
    mis-sliced list would never surface in production — these drive the
    multi-chunk path directly.
    """

    def _messages(self, n):
        return [[{"role": "user", "content": str(i)}] for i in range(n)]

    def test_generate_slices_internals_ids_across_chunks(self):
        backend = _CountingBackend([f"r{i}" for i in range(5)])
        results = batch_generate_samples(make_client(backend), self._messages(5),
            batch_size=2,
            internals_ids=[f"id-{i}" for i in range(5)],
        )
        assert results == ["r0", "r1", "r2", "r3", "r4"]
        # Each item must see its OWN id, not chunk 0's ids repeated.
        assert [c["internals_id"] for c in backend.calls] == [
            "id-0", "id-1", "id-2", "id-3", "id-4",
        ]

    def test_check_slices_internals_ids_across_chunks(self):
        backend = _CountingBackend(["yes"] * 5)
        results = batch_check_samples(make_client(backend), [f"s{i}" for i in range(5)],
            build_check_messages=lambda o, s: [{"role": "user", "content": s}],
            batch_size=2,
            internals_ids=[f"id-{i}" for i in range(5)],
        )
        assert [accepted for accepted, _ in results] == [True] * 5
        assert [c["internals_id"] for c in backend.calls] == [
            "id-0", "id-1", "id-2", "id-3", "id-4",
        ]

    def test_length_mismatch_raises(self):
        backend = _CountingBackend(["a", "b"])
        with pytest.raises(ValueError, match="internals_ids"):
            batch_generate_samples(make_client(backend), self._messages(2), internals_ids=["only-one"],
            )


class TestBatchCallerConcurrencyGuard:
    """max_workers>1 on a series-only backend must raise, not silently degrade."""

    class _SeriesOnly(MockBackend):
        compute_config = ComputeConfig(supports_parallel_calls=False)

    def test_run_raises_on_explicit_parallel_override(self):
        backend = make_client(self._SeriesOnly("ok")).backend
        caller = BatchCaller(backend, max_workers=4)
        with pytest.raises(ValueError, match="requires series calls"):
            caller.run([[{"role": "user", "content": "hi"}]])

    def test_backend_construction_clamps_instead_of_raising(self):
        # The clamp happens once, when the backend binds its budget — so a
        # BatchCaller built the normal way can never be over-concurrent.
        backend = make_client(self._SeriesOnly("ok"), "m").backend
        assert backend.max_workers == 1
        assert BatchCaller(backend).max_workers == 1


class TestClientIsTheExecutor:
    """The strategy is bound at construction and never re-decided per call."""

    def _messages(self, n):
        return [[{"role": "user", "content": str(i)}] for i in range(n)]

    def test_native_transport_takes_the_whole_batch_in_one_call(self):
        backend = _CountingBackend(["a", "b", "c"], native=True)
        assert make_client(backend).generate(self._messages(3)) == ["a", "b", "c"]
        assert backend.generate_call_count == 1

    def test_non_native_transport_gets_one_item_per_call(self):
        backend = _CountingBackend(["a", "b", "c"], native=False)
        assert make_client(backend).generate(self._messages(3)) == ["a", "b", "c"]
        assert backend.generate_call_count == 3

    def test_series_only_transport_is_clamped_to_one_worker(self):
        # Introspection's hard constraint: one sample per real generate() call.
        class _SeriesOnly(_CountingBackend):
            compute_config = ComputeConfig(
                supports_parallel_calls=False, supports_internals=True
            )

        client = make_client(_SeriesOnly(["a", "b"]))
        assert client._caller.max_workers == 1
        assert client.generate(self._messages(2)) == ["a", "b"]
        assert client.backend.generate_call_count == 2

    def test_strategy_is_fixed_at_construction(self):
        # Flipping the flag afterwards must not change how the client runs —
        # the decision was made once, not re-read per call.
        backend = _CountingBackend(["a", "b"], native=False)
        client = make_client(backend)
        backend._native = True
        assert client.generate(self._messages(2)) == ["a", "b"]
        assert backend.generate_call_count == 2  # still fanned out

    def test_no_limiter_or_dispatch_mode_passed_by_the_caller(self):
        # Every client is called the same way: messages in, replies out.
        import inspect
        params = inspect.signature(ModelClient.generate).parameters
        assert "rate_limiter" not in params
        assert "backend" not in params


class TestCreateFactory:
    def test_caches_per_model_and_setup(self):
        clear_client_cache()
        clear_transport_caches()
        with patch.dict(os.environ, {"VENICE_API_KEY": "test"}):
            a = ModelClient.create("venice-uncensored")
            b = ModelClient.create("venice-uncensored")
        assert a is b

    def test_local_binding_carries_no_api_budget(self):
        """One entry, two setups: binding the local one must not inherit the
        hosted endpoint's RPM or worker budget."""
        cfg = get_model_config("venice-uncensored")
        with patch.dict(os.environ, {"VENICE_API_KEY": "test"}):
            api_backend = backend_for(cfg, "api")
        assert api_backend.rpm == 75
        assert api_backend.max_workers == 3

        class _LocalT(MockBackend):
            @property
            def backend_name(self):
                return "vllm"

        local = make_client(_LocalT("x"), "venice-uncensored", backend_type="vllm")
        assert local.backend.rpm is None
        assert local.backend.max_workers == 1

    def test_local_binding_sends_no_extra_body(self):
        """Regression: the endpoint's provider params must not reach a local
        engine — they were forwarded into SamplingParams and raised TypeError.
        """
        cfg = get_model_config("venice-uncensored")
        assert cfg.api.default_extra_body  # the entry really does declare one

        class _LocalT(MockBackend):
            @property
            def backend_name(self):
                return "vllm"

        backend = _LocalT("ok")
        make_client(backend, "venice-uncensored", backend_type="vllm").generate(
            [[{"role": "user", "content": "hi"}]]
        )
        assert "extra_body" not in backend.calls[0]

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            ModelClient.create("no-such-model-anywhere")

    def test_setup_the_entry_does_not_have_raises(self):
        with pytest.raises(ValueError, match="no introspect setup"):
            ModelClient.create("venice-uncensored", backend_type="introspect")
