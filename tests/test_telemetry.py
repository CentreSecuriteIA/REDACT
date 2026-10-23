"""Telemetry: the emit hook, event granularity, sinks, cost, and the logger contract."""

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from redact import telemetry
from redact.llms import observe
from redact.llms.backends import ComputeConfig
from redact.llms.resources import measure
from tests.conftest import MockBackend, first_cut_warning, make_client


@pytest.fixture(autouse=True)
def _clean():
    yield
    telemetry.uninstall()


def _messages(n):
    return [[{"role": "user", "content": str(i)}] for i in range(n)]


class TestObserveHook:
    """llms/ emits and stays dependency-free — that is the whole contract."""

    def test_record_is_a_noop_with_no_emitter(self):
        # The default state for anyone who just `import redact.llms`. Must not
        # raise, must not need a sink, must not pull in a vendor SDK.
        observe.set_emitter(None)
        observe.record({"ev": "call"})  # no exception

    def test_a_failing_sink_never_breaks_a_run(self, caplog):
        def boom(event):
            raise RuntimeError("sink is down")

        observe.set_emitter(boom)
        with caplog.at_level(logging.DEBUG, logger="redact.llms.observe"):
            observe.record({"ev": "call"})  # logged at DEBUG, not raised
        observe.set_emitter(None)
        assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
            ("DEBUG", "Telemetry emitter failed (RuntimeError: sink is down)"),
        ]

    def test_stage_label_is_attached_to_events(self):
        seen = []
        observe.set_emitter(seen.append)
        with telemetry.stage("inputs"):
            observe.record({"ev": "call"})
        assert seen[0]["stage"] == "inputs"
        # ...and restored afterwards, so a later event isn't mislabelled.
        observe.record({"ev": "call"})
        assert seen[-1]["stage"] is None


class TestEventGranularity:
    """One event per real TRANSPORT call, never per batch item.

    This is why telemetry cannot hang off ``on_complete``: a native engine pass
    over N prompts fires that callback N times but is one call.
    """

    class _Native(MockBackend):
        compute_config = ComputeConfig(supports_native_batching=True)

    def test_native_batch_is_one_event_covering_all_items(self):
        col = telemetry.install(sink="off")
        seen = []
        observe.set_emitter(seen.append)
        make_client(self._Native(["a", "b", "c"])).generate(_messages(3))
        calls = [e for e in seen if e["ev"] == "call"]
        assert len(calls) == 1
        assert calls[0]["n_items"] == 3
        assert col is not None

    def test_non_native_batch_is_one_event_per_item(self):
        telemetry.install(sink="off")
        seen = []
        observe.set_emitter(seen.append)
        make_client(MockBackend(["a", "b", "c"])).generate(_messages(3))
        calls = [e for e in seen if e["ev"] == "call"]
        assert len(calls) == 3
        assert [c["n_items"] for c in calls] == [1, 1, 1]


class TestUsageCapture:
    """Token counts come off the provider response, and must degrade gracefully."""

    def _backend(self, usage):
        from redact.llms.backends import OpenAIBackend

        OpenAIBackend.clear_cache()
        b = OpenAIBackend("venice-uncensored", api_key="k", base_url="https://t/v1")
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "hi"
        resp.usage = usage
        client.chat.completions.create.return_value = resp
        b._client = client
        return b

    def test_usage_is_recorded_when_the_provider_reports_it(self):
        telemetry.install(sink="off")
        seen = []
        observe.set_emitter(seen.append)
        usage = MagicMock(prompt_tokens=812, completion_tokens=1904)
        self._backend(usage).generate([[{"role": "user", "content": "hi"}]])
        assert (seen[0]["in_tok"], seen[0]["out_tok"], seen[0]["out_tok_max"]) == (812, 1904, 1904)

    def test_missing_usage_records_nulls_rather_than_raising(self):
        # usage is optional in the OpenAI-compatible spec; plenty of endpoints
        # omit it, and a trace is never worth failing a generation call over.
        telemetry.install(sink="off")
        seen = []
        observe.set_emitter(seen.append)
        result = self._backend(None).generate([[{"role": "user", "content": "hi"}]])
        assert result == ["hi"]
        assert seen[0]["in_tok"] is None and seen[0]["out_tok"] is None
        assert seen[0]["out_tok_max"] is None

    def test_a_failed_call_is_recorded_then_re_raised(self):
        telemetry.install(sink="off")
        seen = []
        observe.set_emitter(seen.append)
        b = self._backend(None)
        b._client.chat.completions.create.side_effect = RuntimeError("502")
        with pytest.raises(RuntimeError):
            b.generate([[{"role": "user", "content": "hi"}]])
        assert "502" in seen[0]["error"]


class TestJsonlSink:
    def test_writes_a_trace_beside_the_other_sidecars(self, tmp_path):
        col = telemetry.install(data_dir=tmp_path)
        with telemetry.stage("inputs"):
            make_client(MockBackend("ok"), "venice-uncensored").generate(_messages(2))
        path = col.trace_path
        assert path.parent.name == "Datasets"       # same home as manifest/state
        assert path.name.endswith(".trace.jsonl")
        events = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        calls = [e for e in events if e["ev"] == "call"]
        assert len(calls) == 2
        assert all(e["stage"] == "inputs" for e in calls)
        assert all("ts" in e for e in events)

    def test_off_writes_nothing_and_installs_no_emitter(self, tmp_path):
        telemetry.install(data_dir=tmp_path, sink="off")
        make_client(MockBackend("ok")).generate(_messages(1))
        assert list(tmp_path.rglob("*.trace.jsonl")) == []

    def test_unknown_sink_falls_back_to_the_default(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="redact.telemetry"):
            col = telemetry.install(data_dir=tmp_path, sink="telepathy")
        assert "telepathy" in caplog.text
        make_client(MockBackend("ok")).generate(_messages(1))
        assert col.trace_path.exists()          # fell back to jsonl, not to off

    def test_two_runs_in_one_second_write_two_traces(self, tmp_path):
        """The run id is to the second, so a later install takes the next
        free name rather than appending to the earlier run's trace."""
        from datetime import UTC, datetime

        with patch("redact.telemetry.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)
            paths = [telemetry.install(data_dir=tmp_path).trace_path for _ in range(3)]
        assert [p.name for p in paths] == [
            "run-20261017-120000.trace.jsonl", "run-20261017-120000-2.trace.jsonl",
            "run-20261017-120000-3.trace.jsonl"]
        assert all(p.exists() for p in paths)    # claimed before any event


class TestLoggerContract:
    """The library attaches no handler and sets no level — telemetry must not
    change that. A trace that hijacks the root logger would silently override
    whatever scripts/run.py configured."""

    def test_installing_adds_no_handler_and_sets_no_level(self, tmp_path):
        root = logging.getLogger("redact")
        before_level = root.level
        before_handlers = list(root.handlers)

        telemetry.install(data_dir=tmp_path)
        make_client(MockBackend("ok")).generate(_messages(1))

        assert root.level == before_level == logging.NOTSET
        assert root.handlers == before_handlers

    def test_the_jsonl_sink_is_not_a_logging_handler(self, tmp_path):
        col = telemetry.install(data_dir=tmp_path)
        assert not any(isinstance(s, logging.Handler) for s in col._sinks)

    def test_summary_narrates_through_the_standard_logger(self, tmp_path, caplog):
        telemetry.install(data_dir=tmp_path)
        make_client(MockBackend("ok"), "venice-uncensored").generate(_messages(1))
        with caplog.at_level(logging.INFO, logger="redact.telemetry"):
            telemetry.summary()
        assert "venice-uncensored" in caplog.text


class TestCost:
    def test_api_cost_uses_the_models_declared_prices(self, tmp_path):
        from redact.llms.model_config import MODEL_REGISTRY, APIConfig, register_model

        name = "_test_priced_model"
        try:
            register_model(
                name, backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                              base_url="https://t/v1", rpm=100,
                              price_per_1m_input=3.0, price_per_1m_output=15.0),
            )
            telemetry.install(data_dir=tmp_path)
            observe.record({"ev": "call", "model": name, "n_items": 1,
                            "in_tok": 1_000_000, "out_tok": 200_000, "ms": 10})
            s = telemetry.summary()
            # 1M in @ $3 + 0.2M out @ $15 = 3.00 + 3.00
            assert s["models"][name]["cost_usd"] == pytest.approx(6.0)
            assert s["api_cost_usd"] == pytest.approx(6.0)
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_a_model_served_locally_is_not_priced_by_its_tokens(self, tmp_path):
        """An entry with a hosted and a local setup: only its hosted calls cost tokens."""
        from redact.llms.model_config import (
            MODEL_REGISTRY, APIConfig, VLLMConfig, register_model,
        )

        name = "_test_two_setups"
        try:
            register_model(
                name, backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                              base_url="https://t/v1", rpm=100,
                              price_per_1m_input=3.0, price_per_1m_output=15.0),
                vllm=VLLMConfig(hf_model_id="org/x"),
            )
            telemetry.install(data_dir=tmp_path)
            for backend in ("openai", "vllm"):
                observe.record({"ev": "call", "model": name, "backend": backend,
                                "n_items": 1, "in_tok": 1_000_000, "out_tok": 0,
                                "ms": 10})
            s = telemetry.summary()
            assert s["models"][name]["in"] == 2_000_000
            assert s["models"][name]["cost_usd"] == pytest.approx(3.0)
            assert s["api_cost_usd"] == pytest.approx(3.0)
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_a_model_only_served_locally_has_no_token_cost(self, tmp_path):
        from redact.llms.model_config import (
            MODEL_REGISTRY, APIConfig, VLLMConfig, register_model,
        )

        name = "_test_two_setups_local"
        try:
            register_model(
                name, backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                              base_url="https://t/v1", rpm=100,
                              price_per_1m_input=3.0, price_per_1m_output=15.0),
                vllm=VLLMConfig(hf_model_id="org/x"),
            )
            telemetry.install(data_dir=tmp_path)
            observe.record({"ev": "call", "model": name, "backend": "vllm",
                            "n_items": 1, "in_tok": 1_000_000, "out_tok": 0, "ms": 10})
            s = telemetry.summary()
            assert "cost_usd" not in s["models"][name]
            assert s["api_cost_usd"] == 0.0
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_unpriced_model_reports_tokens_but_no_dollars(self, tmp_path):
        """Registers its own unpriced model rather than borrowing a shipped one.

        This used to point at venice-uncensored, which worked only while every
        shipped entry happened to be unpriced — i.e. while the cost roll-up was
        dark. Pricing them turned that accident into a failure, so the model
        under test now carries the property deliberately.
        """
        from redact.llms.model_config import MODEL_REGISTRY, APIConfig, register_model

        name = "_test_unpriced_model"
        try:
            register_model(
                name, backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                              base_url="https://t/v1", rpm=100),
            )
            telemetry.install(data_dir=tmp_path)
            observe.record({"ev": "call", "model": name, "n_items": 1,
                            "in_tok": 500, "out_tok": 100, "ms": 5})
            s = telemetry.summary()
            assert s["models"][name]["in"] == 500
            assert "cost_usd" not in s["models"][name]
            assert s["api_cost_usd"] is None
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_engine_lifetime_accumulates_per_checkpoint(self, tmp_path):
        # The GPU is leased per engine, so two models sharing a checkpoint share
        # one meter — the cost centre is the checkpoint, not the model name.
        telemetry.install(data_dir=tmp_path)
        observe.record({"ev": "local", "phase": "release",
                        "hf_model_id": "org/big", "held_s": 600.0})
        observe.record({"ev": "local", "phase": "release",
                        "hf_model_id": "org/big", "held_s": 300.0})
        assert telemetry.summary()["local_s"]["org/big"] == 900.0


class TestLiveEngineTime:
    """Engines still loaded when the summary is built have sent no release."""

    @pytest.fixture()
    def clock(self):
        import redact.llms.backends.introspection as intro
        import redact.llms.backends.vllm as vllm_module

        vllm_module.VLLMBackend.clear_cache()
        intro.TransformersIntrospectionBackend.clear_cache()
        now = [1000.0]
        with patch("time.perf_counter", lambda: now[0]):
            yield vllm_module, intro, now
        vllm_module._engine_loaded_at.clear()
        intro._models_loaded_at.clear()

    def test_summary_counts_engines_that_are_still_loaded(self, tmp_path, clock):
        vllm_module, intro, now = clock
        telemetry.install(data_dir=tmp_path)
        vllm_module._engine_loaded_at[("org/live", None, "[]")] = 400.0
        intro._models_loaded_at[("org/tf", "auto", "auto", "[]")] = 900.0
        s = telemetry.summary()
        assert s["local_s"] == {"org/live": 600.0, "org/tf": 100.0}
        with patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 2)), \
             patch("redact.telemetry.gpu_hourly_rate", return_value=3.0):
            # The two overlap, so 600 s are billed: on 2 rented cards at $3/h.
            s = telemetry.summary()
        assert s["local_billed_s"] == 600.0
        assert s["local_cost_usd"] == pytest.approx(1.0)

    def test_a_released_engine_and_a_live_one_that_overlapped_are_billed_once(
        self, tmp_path, clock
    ):
        vllm_module, _, _ = clock
        telemetry.install(data_dir=tmp_path)
        observe.record({"ev": "local", "phase": "release",     # held 400..1000
                        "hf_model_id": "org/a", "held_s": 600.0})
        vllm_module._engine_loaded_at[("org/b", None, "[]")] = 700.0
        s = telemetry.summary()
        assert s["local_s"] == {"org/a": 600.0, "org/b": 300.0}
        assert s["local_billed_s"] == 600.0

    def test_engines_held_one_after_the_other_add_up(self, tmp_path, clock):
        _, _, now = clock
        telemetry.install(data_dir=tmp_path)
        observe.record({"ev": "local", "phase": "release",     # held 700..1000
                        "hf_model_id": "org/a", "held_s": 300.0})
        now[0] = 1500.0
        observe.record({"ev": "local", "phase": "release",     # held 1100..1500
                        "hf_model_id": "org/b", "held_s": 400.0})
        assert telemetry.summary()["local_billed_s"] == 700.0

    def test_a_later_release_is_not_counted_twice(self, tmp_path, clock):
        vllm_module, _, now = clock
        telemetry.install(data_dir=tmp_path)
        vllm_module._engine_loaded_at[("org/live", None, "[]")] = 400.0
        assert telemetry.summary()["local_s"] == {"org/live": 600.0}
        assert telemetry.summary()["local_s"] == {"org/live": 600.0}   # not accumulated
        now[0] = 1100.0
        vllm_module.VLLMBackend.clear_cache()         # emits held_s=700
        assert telemetry.summary()["local_s"] == {"org/live": 700.0}

    def test_released_and_live_time_of_one_checkpoint_add_up(self, tmp_path, clock):
        vllm_module, _, _ = clock
        telemetry.install(data_dir=tmp_path)
        observe.record({"ev": "local", "phase": "release",
                        "hf_model_id": "org/live", "held_s": 50.0})
        vllm_module._engine_loaded_at[("org/live", None, "[]")] = 900.0
        assert telemetry.summary()["local_s"] == {"org/live": 150.0}

    def test_the_backends_do_not_import_telemetry(self):
        import redact.llms.backends.introspection as intro
        import redact.llms.backends.vllm as vllm_module

        for module in (vllm_module, intro):
            assert "telemetry" not in vars(module)


class TestGpuPricing:
    def test_local_provider_is_free_not_unknown(self):
        # 0.0 is a real rate meaning "your own machine", distinct from None
        # meaning "not in the table" — a truthiness check conflates them.
        assert telemetry.gpu_hourly_rate("any card", provider="local") == 0.0

    def test_rate_looked_up_by_provider_and_gpu_name(self):
        assert telemetry.gpu_hourly_rate(
            "NVIDIA H100 80GB HBM3", provider="runpod") == 2.4

    def test_unknown_gpu_reports_no_rate_rather_than_guessing(self):
        assert telemetry.gpu_hourly_rate(
            "NVIDIA Fictional 9000", provider="runpod") is None

    def test_unknown_provider_reports_no_rate(self):
        assert telemetry.gpu_hourly_rate("NVIDIA H100 80GB HBM3",
                                         provider="not-a-cloud") is None

    def test_detect_gpus_parses_one_line_per_device(self):
        out = "NVIDIA A100-SXM4-80GB\nNVIDIA A100-SXM4-80GB\n"
        with patch("redact.llms.resources.measure.shutil.which",
                   return_value="/usr/bin/nvidia-smi"):
            with patch("redact.llms.resources.measure.subprocess.run",
                       return_value=MagicMock(stdout=out)):
                assert measure.detect_gpus() == ("NVIDIA A100-SXM4-80GB", 2)

    def test_no_nvidia_smi_reports_no_gpus(self):
        with patch("redact.llms.resources.measure.shutil.which", return_value=None):
            assert measure.detect_gpus() == (None, 0)

    def test_billed_on_rented_gpus_with_used_reported_separately(self, tmp_path):
        # You rent the pod, so the bill is rate x rented, not rate x occupied.
        # Surfacing both is what makes idle capacity visible.
        telemetry.install(data_dir=tmp_path)
        observe.record({"ev": "local", "phase": "release",
                        "hf_model_id": "org/big", "held_s": 3600.0,
                        "tensor_parallel_size": 1})
        with patch("redact.llms.resources.measure.detect_gpus",
                   return_value=("NVIDIA H100 80GB HBM3", 4)):
            with patch("redact.telemetry.gpu_hourly_rate", return_value=2.4):
                s = telemetry.summary()
        assert s["gpu"]["rented"] == 4
        assert s["local_cost_usd"] == pytest.approx(9.6)   # 1h x 4 cards x $2.40



def _reply(backend, cut, max_tokens=64):
    """Record one call of ``backend``, cut at the token limit or not."""
    backend._record_call(n_items=1, started=time.perf_counter(),
                         max_tokens=max_tokens, cut=cut)


class TestCutReplies:
    """A model's first cut reply in a stage warns; the stage's end logs how
    many were cut, and the summary holds the count."""

    def test_the_first_cut_warns_and_the_stage_end_counts(self, tmp_path, caplog):
        telemetry.install(data_dir=tmp_path)
        gen, check = MockBackend(model="gen"), MockBackend(model="check")
        with caplog.at_level(logging.WARNING, logger="redact"):
            with telemetry.stage("outputs"):
                for cut in (True, False, True, True):
                    _reply(gen, cut)
                _reply(check, True, max_tokens=32)
            with telemetry.stage("jailbreaks"):
                _reply(gen, True)
        assert [r.getMessage() for r in caplog.records] == [
            first_cut_warning("gen", 64),
            first_cut_warning("check", 32),
            "[outputs] gen: 3 replies were cut at the token limit and are incomplete.",
            "[outputs] check: 1 reply was cut at the token limit and is incomplete.",
            first_cut_warning("gen", 64),
            "[jailbreaks] gen: 1 reply was cut at the token limit and is incomplete.",
        ]
        assert telemetry.summary()["cut"] == {
            "outputs": {"gen": 3, "check": 1}, "jailbreaks": {"gen": 1}}

    def test_a_stage_run_again_warns_again(self, tmp_path, caplog):
        """A second run in a notebook: its first cut reply warns at once."""
        gen = MockBackend(model="gen")
        with caplog.at_level(logging.WARNING, logger="redact"):
            for _ in range(2):
                telemetry.install(data_dir=tmp_path)
                with telemetry.stage("outputs"):
                    _reply(gen, True)
                    _reply(gen, True)
                telemetry.uninstall()
        assert [r.getMessage() for r in caplog.records
                if r.name.startswith("redact.llms")] == [first_cut_warning("gen", 64)] * 2

    def test_the_call_event_says_the_reply_was_cut(self):
        seen = []
        observe.set_emitter(seen.append)
        gen = MockBackend(model="gen")
        _reply(gen, True)
        _reply(gen, False)
        assert [e.get("cut") for e in seen] == [True, None]

    def test_without_telemetry_the_first_cut_still_warns(self, caplog):
        telemetry.install(sink="off")
        gen = MockBackend(model="gen")
        with caplog.at_level(logging.WARNING, logger="redact"):
            _reply(gen, True)
            _reply(gen, True)
        assert [r.getMessage() for r in caplog.records] == [
            first_cut_warning("gen", 64)]
        assert telemetry.summary() == {}
