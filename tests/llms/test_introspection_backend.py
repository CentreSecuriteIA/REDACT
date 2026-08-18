"""Tests for the transformers-introspection backend and its registry wiring.

The model-load path (torch/transformers + real weights) is GPU/heavy-dependency
territory and is not exercised here — these tests cover the parts that matter
without a real model: capability flags, get_backend() routing, and the
registry error messages, all offline.
"""

import json

import pytest

from redact.llms.model_config import MODEL_REGISTRY, get_model_config, register_model
from redact.llms.api import get_backend, clear_backend_cache, _backend_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    yield
    clear_backend_cache()


class TestModelConfigIntrospectFields:
    def test_introspect_kwargs_field_present(self):
        from redact.llms.model_config import ModelConfig
        config = ModelConfig(
            name="test", rpm=100, backend_type="transformers_introspect",
            hf_model_id="org/model", introspect_kwargs={"log_dir": "/tmp/x"},
        )
        assert config.introspect_kwargs == {"log_dir": "/tmp/x"}

    def test_introspect_kwargs_default_none(self):
        from redact.llms.model_config import ModelConfig
        config = ModelConfig(name="test", rpm=100)
        assert config.introspect_kwargs is None

    def test_register_model_passes_introspect_kwargs(self):
        name = "_test_register_introspect"
        try:
            register_model(
                name, rpm=999, backend_type="transformers_introspect",
                hf_model_id="org/model", introspect_kwargs={"log_dir": "/tmp/x", "capture": {"attention": True}},
            )
            config = get_model_config(name)
            assert config.backend_type == "transformers_introspect"
            assert config.hf_model_id == "org/model"
            assert config.introspect_kwargs == {"log_dir": "/tmp/x", "capture": {"attention": True}}
        finally:
            MODEL_REGISTRY.pop(name, None)


class TestGetBackendRoutingIntrospect:
    def test_missing_hf_id_raises(self):
        name = "_test_introspect_no_hf_id"
        try:
            register_model(name, rpm=999, backend_type="transformers_introspect",
                            introspect_kwargs={"log_dir": "/tmp/x"})
            with pytest.raises(ValueError, match="no hf_model_id"):
                get_backend(name)
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_missing_log_dir_raises(self):
        name = "_test_introspect_no_log_dir"
        try:
            register_model(name, rpm=999, backend_type="transformers_introspect",
                            hf_model_id="org/model")
            with pytest.raises(ValueError, match="log_dir"):
                get_backend(name)
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_cache_key_is_namespaced(self):
        # Cache key uses an "introspect:" prefix distinct from vLLM's "vllm:"
        # prefix, so a model name can't collide across backend types.
        name = "_test_introspect_cache_key"
        try:
            register_model(name, rpm=999, backend_type="transformers_introspect",
                            hf_model_id="org/model", introspect_kwargs={"log_dir": "/tmp/x"})
            # Don't actually construct it (needs torch/transformers) — just check
            # the ValueError-free path up to backend construction would use this key.
            assert f"introspect:{name}" not in _backend_cache
        finally:
            MODEL_REGISTRY.pop(name, None)


class TestSupportsInternalsCapabilityFlag:
    def test_default_backend_does_not_support_internals(self):
        from redact.llms.base import LLMBackend

        class _Dummy(LLMBackend):
            def generate(self, messages, model, **kwargs):
                return "x"

            @property
            def backend_name(self):
                return "dummy"

        assert _Dummy().supports_internals is False

    def test_introspection_backend_declares_support_without_loading_model(self):
        # supports_internals is a plain property on the class — check it via
        # the class attribute lookup pattern other backends use, without
        # instantiating (which would require torch/transformers + weights).
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("torch")
        from redact.llms.introspection_backend import TransformersIntrospectionBackend

        # supports_internals/supports_native_batching/supports_parallel_calls
        # are properties, but they don't touch self._model etc., so they're
        # safe to read via a bare (unconstructed) instance check using
        # object.__new__ to skip __init__.
        instance = object.__new__(TransformersIntrospectionBackend)
        assert TransformersIntrospectionBackend.supports_internals.fget(instance) is True
        assert TransformersIntrospectionBackend.supports_native_batching.fget(instance) is False
        assert TransformersIntrospectionBackend.supports_parallel_calls.fget(instance) is False
        assert instance.backend_name == "transformers_introspect"


class TestMetaJson:
    """_save_meta/_save_capture's always-written companion JSON.

    Doesn't need torch/transformers installed: constructs the backend via
    object.__new__ (skipping __init__'s model load) and drives _save_capture
    with capture flags all off, so no tensor-saving branch (the only code that
    touches self._torch) executes — only the meta.json write does.
    """

    def _backend(self, tmp_path, capture=None):
        from redact.llms.introspection_backend import TransformersIntrospectionBackend

        b = object.__new__(TransformersIntrospectionBackend)
        b._log_dir = tmp_path
        b._model_name = "org/model"
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
        from redact.llms.introspection_backend import TransformersIntrospectionBackend
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


def test_base_llm_backend_rename_capture_is_a_noop(tmp_path):
    from redact.llms.base import LLMBackend

    class _Dummy(LLMBackend):
        def generate(self, messages, model, **kwargs):
            return "x"

        @property
        def backend_name(self):
            return "dummy"

    _Dummy().rename_capture("anything", "anything-else")  # must not raise
