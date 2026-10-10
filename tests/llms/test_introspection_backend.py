"""Tests for the transformers-introspection backend and its registry wiring.

The model-load path (torch/transformers + real weights) is GPU/heavy-dependency
territory and is not exercised here — these tests cover the parts that matter
without a real model: capability flags, backend_for() routing, and the
registry error messages, all offline.
"""

import contextlib
import json
import logging
import types

import pytest

from redact.llms.backends import backend_for, clear_transport_caches
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    IntrospectConfig,
    ModelConfig,
    get_model_config,
    register_model,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    yield
    clear_transport_caches()


class TestModelConfigIntrospectFields:
    def test_introspect_fields_present(self):
        from redact.llms.model_config import IntrospectConfig
        config = ModelConfig(
            name="test", backend_type="introspect",
            introspect=IntrospectConfig(hf_model_id="org/model", log_dir="/tmp/x"),
        )
        assert config.introspect.hf_model_id == "org/model"
        assert config.introspect.log_dir == "/tmp/x"

    def test_introspect_default_none(self):
        config = ModelConfig(name="test", api=APIConfig(
            backend_type="openai", api_key_env="TEST_API_KEY",
            base_url="https://test.example/v1", rpm=10))
        assert config.introspect is None

    def test_register_model_passes_introspect_fields(self):
        name = "_test_register_introspect"
        try:
            register_model(
                name,
                backend_type="introspect",
                introspect=IntrospectConfig(hf_model_id="org/model", log_dir="/tmp/x", capture={"attention": True}),
            )
            config = get_model_config(name)
            assert config.backend_type == "introspect"
            assert config.introspect.hf_model_id == "org/model"
            assert config.introspect.log_dir == "/tmp/x"
            assert config.introspect.capture == {"attention": True}
        finally:
            MODEL_REGISTRY.pop(name, None)


class TestBackendForRoutingIntrospect:
    def test_raises_when_the_entry_has_no_introspect_setup(self):
        # IntrospectConfig requires hf_model_id AND log_dir at construction,
        # so an incomplete setup can't reach here — only a missing one.
        name = "_test_introspect_absent"
        try:
            register_model(name, api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))
            with pytest.raises(ValueError, match="no introspect setup"):
                backend_for(get_model_config(name), "introspect")
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_load_cache_is_private_to_this_backend(self):
        # Same hf_model_id under vLLM and under transformers is two different
        # runtimes and two independent loads — the caches must not be shared.
        import redact.llms.backends.introspection as intro_module
        import redact.llms.backends.vllm as vllm_module

        clear_transport_caches()
        vllm_module._engines[("org/model", None, "[]")] = "a vllm engine"
        assert intro_module._models == {}
        clear_transport_caches()
        assert vllm_module._engines == {}


class TestSupportsInternalsCapabilityFlag:
    def test_default_backend_does_not_support_internals(self):
        from redact.llms.backends import LLMBackend

        class _Dummy(LLMBackend):
            def generate(self, messages_list, **kwargs):
                return ["x"] * len(messages_list)

            @property
            def backend_name(self):
                return "dummy"

        assert _Dummy("dummy-model").compute_config.supports_internals is False

    def test_introspection_backend_declares_support_without_loading_model(self):
        # compute_config is a CLASS attribute, so its full profile is readable
        # straight off the class — no instance, no torch/transformers import,
        # no weights. That is what lets registration validate concurrency for
        # this backend type without touching a transport.
        from redact.llms.backends import TransformersIntrospectionBackend

        cc = TransformersIntrospectionBackend.compute_config
        assert cc.supports_internals is True
        assert cc.supports_native_batching is False
        assert cc.supports_parallel_calls is False


class TestMetaJson:
    """_save_meta/_save_capture's always-written companion JSON.

    Doesn't need torch/transformers installed: constructs the backend via
    object.__new__ (skipping __init__'s model load) and drives _save_capture
    with capture flags all off, so no tensor-saving branch (the only code that
    touches self._torch) executes — only the meta.json write does.
    """

    def _backend(self, tmp_path, capture=None):
        from redact.llms.backends import TransformersIntrospectionBackend

        b = object.__new__(TransformersIntrospectionBackend)
        b._log_dir = tmp_path
        b.model = "test-model"
        b.hf_model_id = "org/model"
        b._capture = capture or {"logprobs": False, "hidden_states": False, "attention": False}
        b._torch = None  # unused when all capture flags are off; just needs to exist
        return b

    def test_save_capture_writes_meta_json(self, tmp_path):
        import types

        b = self._backend(tmp_path)
        outputs = types.SimpleNamespace(
            sequences=[[]], scores=None, hidden_states=None, attentions=None,
        )
        meta = {
            "model": "org/model",
            "settings": {"max_tokens": 50, "temperature": 0.7, "top_p": 0.85, "capture": b._capture},
            "messages": [{"role": "user", "content": "hi"}],
            "output": "hello there",
        }
        b._save_capture(
            "abc123/output", outputs, prompt_len=0, meta=meta, tokenizer=None)

        meta_path = tmp_path / "abc123" / "output" / "meta.json"
        assert meta_path.exists()
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        assert loaded == meta

    def test_a_capture_replaces_the_files_of_an_earlier_one(self, tmp_path, caplog):
        # A re-run under the same id with less captured: the old tensor file
        # must not stay beside a meta.json that says it was not captured.
        b = self._backend(tmp_path)
        outputs = types.SimpleNamespace(
            sequences=[[]], logits=None, hidden_states=None, attentions=None,
        )
        folder = b._capture_dir("abc123/output")
        (folder / "attention.pt").write_text("stale")
        nested = b._capture_dir("abc123/output/val")
        (nested / "meta.json").write_text("another capture")

        logger = "redact.llms.backends.introspection"
        with caplog.at_level(logging.DEBUG, logger=logger):
            b._save_capture(
                "abc123/output", outputs, prompt_len=0, meta={}, tokenizer=None)

        assert sorted(p.name for p in folder.iterdir()) == ["meta.json", "val"]
        assert (nested / "meta.json").read_text() == "another capture"
        assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
            ("DEBUG", f"[internals] saved meta.json -> {folder}"),
        ]

    @pytest.mark.parametrize("bad", ["", "..", "abc/../..", "ABSOLUTE"])
    def test_an_id_outside_the_log_dir_is_refused(self, tmp_path, bad):
        # A capture replaces and moves files, so its id must stay in log_dir.
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "keep.txt").write_text("x")
        (tmp_path / "log").mkdir()
        (tmp_path / "log" / "notes.txt").write_text("x")
        b = self._backend(tmp_path / "log")
        bad = str(outside) if bad == "ABSOLUTE" else bad
        outputs = types.SimpleNamespace(
            sequences=[[]], logits=None, hidden_states=None, attentions=None,
        )
        with pytest.raises(ValueError, match="internals_id"):
            b._save_capture(bad, outputs, prompt_len=0, meta={}, tokenizer=None)
        with pytest.raises(ValueError, match="internals_id"):
            b.rename_capture(bad, "ok/id")
        with pytest.raises(ValueError, match="internals_id"):
            b.rename_capture("ok/id", bad)
        assert (outside / "keep.txt").read_text() == "x"
        assert (tmp_path / "log" / "notes.txt").read_text() == "x"

    def test_capture_dir_nests_on_embedded_slash(self, tmp_path):
        # internals_id = "{input_id}/{subfolder}" must nest correctly — no
        # backend/wrapper code change needed for the folder-per-input_id design,
        # since pathlib splits on "/" and mkdir(parents=True) creates both levels.
        b = self._backend(tmp_path)
        d = b._capture_dir("abc123/jailbreak_translate/deadbeef")
        assert d == tmp_path / "abc123" / "jailbreak_translate" / "deadbeef"
        assert d.is_dir()


class TestRenameCapture:
    """rename_capture: relabels a provisional-id capture once the real id is
    known (paraphrase/jailbreak's output text isn't known until after the call
    returns, but internals_id has to be supplied before it)."""

    def _backend(self, tmp_path):
        from redact.llms.backends import TransformersIntrospectionBackend
        b = object.__new__(TransformersIntrospectionBackend)
        b._log_dir = tmp_path
        return b

    def test_moves_folder_to_new_id(self, tmp_path):
        b = self._backend(tmp_path)
        b._capture_dir("base1/paraphrase/attempt_0")
        (tmp_path / "base1" / "paraphrase" / "attempt_0" / "meta.json").write_text("{}")

        b.rename_capture("base1/paraphrase/attempt_0", "base1/paraphrase/deadbeef")

        assert not (tmp_path / "base1" / "paraphrase" / "attempt_0").exists()
        assert (tmp_path / "base1" / "paraphrase" / "deadbeef" / "meta.json").exists()

    def test_noop_when_old_id_was_never_captured(self, tmp_path):
        b = self._backend(tmp_path)
        b.rename_capture("never/existed", "also/never")  # must not raise
        assert not (tmp_path / "also").exists()

    def test_creates_intermediate_dirs_for_new_id(self, tmp_path):
        b = self._backend(tmp_path)
        b._capture_dir("base1/attempt_0")
        b.rename_capture("base1/attempt_0", "base1/nested/deeper/final")
        assert (tmp_path / "base1" / "nested" / "deeper" / "final").is_dir()

    def test_replaces_an_existing_capture_under_the_new_id(self, tmp_path):
        # A re-run that produces the same text lands on the same sample_id.
        b = self._backend(tmp_path)
        (b._capture_dir("base1/paraphrase/deadbeef") / "meta.json").write_text("old")
        (b._capture_dir("base1/paraphrase/attempt_0") / "meta.json").write_text("new")

        b.rename_capture("base1/paraphrase/attempt_0", "base1/paraphrase/deadbeef")

        final = tmp_path / "base1" / "paraphrase" / "deadbeef"
        assert (final / "meta.json").read_text() == "new"
        assert not (tmp_path / "base1" / "paraphrase" / "attempt_0").exists()

    def test_same_old_and_new_id_keeps_the_capture(self, tmp_path):
        b = self._backend(tmp_path)
        (b._capture_dir("base1/output") / "meta.json").write_text("{}")
        b.rename_capture("base1/output", "base1/output")
        assert (tmp_path / "base1" / "output" / "meta.json").exists()


class _FakeIds(list):
    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, key):
        out = list.__getitem__(self, key)
        return _FakeIds(out) if isinstance(key, slice) else out


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<s>PROMPT"

    def __call__(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return _FakeInputs(input_ids=types.SimpleNamespace(shape=(1, 3)))

    def decode(self, ids, skip_special_tokens):
        return " reply "


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            sequences=[_FakeIds([1, 2, 3, 4, 5])],
            logits=None, hidden_states=None, attentions=None,
        )


def _generate_backend(log_dir, temperature=0.7, model=None):
    import redact.llms.backends.introspection as intro_module
    from redact.llms.backends import LLMBackend, TransformersIntrospectionBackend

    b = object.__new__(TransformersIntrospectionBackend)
    LLMBackend.__init__(b, "test-model", default_temperature=temperature)
    b.hf_model_id, b._log_dir = "org/model", log_dir
    b._capture = {"logprobs": True, "hidden_states": False, "attention": False}
    b._sampling = {}
    b._torch = types.SimpleNamespace(no_grad=contextlib.nullcontext)
    # The backend reads its tokenizer and model from the cache, by key.
    b._key = intro_module._model_key("org/model", "auto", "auto", {})
    intro_module._models[b._key] = (_FakeTokenizer(), model or _FakeModel())
    return b


_MSG = [[{"role": "user", "content": "hi"}]]


class TestGenerate:
    """generate() against a fake tokenizer and model (no torch needed)."""

    def test_chat_template_output_gets_no_second_set_of_special_tokens(self, tmp_path):
        b = _generate_backend(tmp_path)
        assert b.generate(_MSG) == ["reply"]
        assert b._loaded()[0].calls == [
            {"return_tensors": "pt", "add_special_tokens": False}
        ]

    def test_logprobs_are_requested_as_raw_logits(self, tmp_path):
        b = _generate_backend(tmp_path)
        b.generate(_MSG, internals_ids=["u1/output"])
        sent = b._loaded()[1].calls[0]
        assert sent["output_logits"] is True
        assert "output_scores" not in sent
        assert (tmp_path / "u1" / "output" / "meta.json").exists()

    def test_zero_temperature_is_greedy_and_an_integer_is_accepted(self, tmp_path):
        b = _generate_backend(tmp_path, temperature=0.0)
        b.generate(_MSG)
        b.generate(_MSG, temperature=1)
        calls = b._loaded()[1].calls
        assert [c["do_sample"] for c in calls] == [False, True]
        assert all(isinstance(c["temperature"], float) for c in calls)

    def test_a_failed_model_call_is_recorded_and_raised(self, tmp_path):
        from redact.llms import observe

        class _FailingModel(_FakeModel):
            def generate(self, **kwargs):
                raise RuntimeError("CUDA out of memory")

        events = []
        observe.set_emitter(events.append)
        try:
            b = _generate_backend(tmp_path, model=_FailingModel())
            with pytest.raises(RuntimeError, match="out of memory"):
                b.generate(_MSG, max_tokens=5)
        finally:
            observe.set_emitter(None)
        assert [(e["ev"], e["model"], e["n_items"], e["max_tokens"], e["error"])
                for e in events] == [
            ("call", "test-model", 1, 5, "RuntimeError: CUDA out of memory"),
        ]

    def test_a_short_internals_ids_list_costs_no_forward_pass(self, tmp_path):
        b = _generate_backend(tmp_path)
        with pytest.raises(ValueError, match=r"same length as messages_list \(1 != 2\)"):
            b.generate(_MSG * 2, internals_ids=["only/one"])
        assert b._loaded()[1].calls == []

    def test_the_cache_is_read_once_per_generate(self, tmp_path):
        b = _generate_backend(tmp_path)
        loaded = b._loaded()
        reads = []
        b._loaded = lambda: reads.append(1) or loaded
        b.generate(_MSG * 3, internals_ids=["a/output", None, "c/output"])
        assert reads == [1]

    def test_a_released_model_raises_on_generate(self, tmp_path):
        from redact.llms.backends import TransformersIntrospectionBackend

        b = _generate_backend(tmp_path)
        assert b.generate(_MSG) == ["reply"]
        TransformersIntrospectionBackend.clear_cache()
        with pytest.raises(RuntimeError, match="was released"):
            b.generate(_MSG)


def _introspect(**kwargs):
    return IntrospectConfig(hf_model_id="org/model", log_dir="/tmp/x", **kwargs)


class TestEagerAttention:
    """Attention weights come back only under the eager implementation, so a
    setup that captures attention loads with it. Run against fakes: a real
    load needs a GPU session."""

    @pytest.mark.parametrize("config, expected", [
        (_introspect(capture={"attention": True}), {"attn_implementation": "eager"}),
        (_introspect(capture={"attention": True},
                     extra_kwargs={"attn_implementation": "sdpa"}),
         {"attn_implementation": "sdpa"}),
        (_introspect(capture={"attention": False}), {}),
        (_introspect(extra_kwargs={"trust_remote_code": True}),
         {"trust_remote_code": True}),
    ], ids=["attention", "own-implementation-wins", "no-attention", "no-capture"])
    def test_load_kwargs(self, config, expected):
        assert config.load_kwargs == expected

    def test_load_kwargs_does_not_change_the_setup(self):
        config = _introspect(capture={"attention": True}, extra_kwargs={"a": 1})
        config.load_kwargs["b"] = 2
        assert config.extra_kwargs == {"a": 1}

    def test_the_model_load_receives_it_and_the_tokenizer_does_not(self, monkeypatch):
        import sys

        import redact.llms.backends.introspection as intro_module

        seen = {}

        class _Auto:
            def __init__(self, kind):
                self.kind = kind

            def from_pretrained(self, hf_model_id, **kwargs):
                seen[self.kind] = kwargs
                return types.SimpleNamespace(eval=lambda: None)

        monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
            __version__="4.50.0",
            AutoTokenizer=_Auto("tokenizer"), AutoModelForCausalLM=_Auto("model"),
        ))
        config = _introspect(capture={"attention": True},
                             extra_kwargs={"trust_remote_code": True})
        intro_module._load("org/model", "auto", "auto", config.load_kwargs)

        assert seen["tokenizer"] == {"trust_remote_code": True}
        assert seen["model"] == {
            "device_map": "auto", "torch_dtype": "auto",
            "trust_remote_code": True, "attn_implementation": "eager",
        }

    @pytest.mark.parametrize("capture", [{"attention": True}, None])
    def test_the_planner_and_the_backend_share_a_load_key(self, monkeypatch, capture):
        import sys

        import redact.llms.backends.introspection as intro_module
        from redact.llms.backends import TransformersIntrospectionBackend
        from redact.llms.resources import residency

        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
        loads = []
        monkeypatch.setattr(intro_module, "_load", lambda *args: loads.append(args))
        config = ModelConfig(name="m", introspect=_introspect(
            capture=capture, extra_kwargs={"trust_remote_code": True}))

        backend = TransformersIntrospectionBackend.from_config(config)

        assert backend._key == residency._engine_id(config.introspect, "introspect")[1:]
        assert backend._key == intro_module._model_key(*loads[0])
        assert ("attn_implementation" in backend._key[-1]) is bool(capture)
