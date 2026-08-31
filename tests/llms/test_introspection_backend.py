"""Tests for the transformers-introspection backend and its registry wiring.

The model-load path (torch/transformers + real weights) is GPU/heavy-dependency
territory and is not exercised here — these tests cover the parts that matter
without a real model: capability flags, backend_for() routing, and the
registry error messages, all offline.
"""

import json

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
        config = ModelConfig(name="test")
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
        b._save_capture("abc123/output", outputs, prompt_len=0, meta=meta)

        meta_path = tmp_path / "abc123" / "output" / "meta.json"
        assert meta_path.exists()
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        assert loaded == meta

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
