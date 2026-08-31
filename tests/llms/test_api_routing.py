"""Tests for setup resolution and backend construction (capabilities.py)."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from redact.llms.backends import (
    backend_for,
    clear_transport_caches,
    resolve_setup,
    transport_for,
)
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    ModelConfig,
    VLLMConfig,
    register_model,
)


class TestResolveSetup:
    """With no declared preference, the setup is derived from what exists."""

    def test_single_setup_needs_no_declaration(self):
        cfg = ModelConfig(name="m", vllm=VLLMConfig(hf_model_id="org/m"))
        assert resolve_setup(cfg) == "vllm"

    def test_api_only_entry(self):
        cfg = ModelConfig(name="m", api=APIConfig(
            backend_type="openai", api_key_env="K",
            base_url="https://x/v1", rpm=10))
        assert resolve_setup(cfg) == "api"

    def test_ambiguous_entry_must_declare(self):
        # Two setups and no preference is a real question, not something to
        # guess from the model name.
        cfg = ModelConfig(
            name="m",
            api=APIConfig(backend_type="openai", api_key_env="K",
                          base_url="https://x/v1", rpm=10),
            vllm=VLLMConfig(hf_model_id="org/m"),
        )
        with pytest.raises(ValueError, match="no backend_type"):
            resolve_setup(cfg)

    def test_entry_with_no_setup_at_all(self):
        with pytest.raises(ValueError, match="no setup at all"):
            resolve_setup(ModelConfig(name="m"))

    def test_explicit_request_wins_over_the_entry_default(self):
        cfg = ModelConfig(
            name="m", backend_type="api",
            api=APIConfig(backend_type="openai", api_key_env="K",
                          base_url="https://x/v1", rpm=10),
            vllm=VLLMConfig(hf_model_id="org/m"),
        )
        assert resolve_setup(cfg, "vllm") == "vllm"

    def test_requesting_a_setup_the_entry_lacks_raises(self):
        cfg = ModelConfig(name="m", vllm=VLLMConfig(hf_model_id="org/m"))
        with pytest.raises(ValueError, match="no api setup"):
            resolve_setup(cfg, "api")

    def test_unknown_setup_name_raises(self):
        cfg = ModelConfig(name="m", vllm=VLLMConfig(hf_model_id="org/m"))
        with pytest.raises(ValueError, match="unknown backend_type"):
            resolve_setup(cfg, "telepathy")


class TestTransportFor:
    """The setup names the field; the API provider names the transport."""

    def test_api_setup_defers_to_its_own_provider(self):
        cfg = ModelConfig(name="m", api=APIConfig(
            backend_type="anthropic", api_key_env="K", rpm=5))
        assert transport_for(cfg) == "anthropic"

    def test_local_setup_names_itself(self):
        cfg = ModelConfig(name="m", vllm=VLLMConfig(hf_model_id="org/m"))
        assert transport_for(cfg) == "vllm"


class TestBackendFor:
    def setup_method(self):
        clear_transport_caches()

    def teardown_method(self):
        clear_transport_caches()

    def _cfg(self, name):
        return MODEL_REGISTRY[name]

    def test_openai_backend(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.backends.openai.openai.OpenAI"):
                backend = backend_for(self._cfg("venice-uncensored"))
        assert backend.backend_name == "openai"
        assert backend.model == "venice-uncensored"

    def test_anthropic_backend(self):
        mock_anthropic = MagicMock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                backend = backend_for(self._cfg("claude-opus-4-6"))
        assert backend.backend_name == "anthropic"
        assert backend.model == "claude-opus-4-6"

    def test_backend_is_per_model_but_shares_one_sdk_client(self):
        # Two models on one endpoint are two configured backends — with one
        # HTTP connection pool underneath them.
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.backends.openai.openai.OpenAI"):
                b1 = backend_for(self._cfg("venice-uncensored"))
                b2 = backend_for(self._cfg("deepseek-v3.2"))
        assert b1 is not b2
        assert b1.model != b2.model
        assert b1._client is b2._client

    def test_budget_comes_from_the_bound_setup(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.backends.openai.openai.OpenAI"):
                backend = backend_for(self._cfg("venice-uncensored"))
        assert (backend.rpm, backend.max_workers) == (75, 3)

    def test_series_only_provider_clamps_workers_at_construction(self):
        name = "test-routing-series-only"
        try:
            register_model(
                name, backend_type="api",
                api=APIConfig(backend_type="anthropic", api_key_env="TEST_API_KEY",
                              rpm=5, recommended_max_workers=1),
            )
            mock_anthropic = MagicMock()
            with patch.dict(os.environ, {"TEST_API_KEY": "k"}):
                with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                    backend = backend_for(self._cfg(name))
            assert backend.max_workers == 1
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_raises_when_the_entry_lacks_the_requested_setup(self):
        # "present but incomplete" is unconstructable — VLLMConfig requires
        # hf_model_id — so the only reachable gap is "not there at all".
        name = "test-routing-no-vllm"
        try:
            register_model(name, api=APIConfig(
                backend_type="openai", api_key_env="TEST_API_KEY",
                base_url="https://test.example/v1", rpm=10))
            with pytest.raises(ValueError, match="no vllm setup"):
                backend_for(self._cfg(name), "vllm")
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_backend_type_override_selects_non_default_setup(self):
        # One registry row with both .api and .vllm populated (the shape the
        # venice-uncensored entry uses) — the override picks a specific setup
        # off it without a second row.
        name = "test-routing-dual-setup"
        MODEL_REGISTRY[name] = ModelConfig(
            name=name, backend_type="api",
            api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                          base_url="https://test.example/v1", rpm=10),
            vllm=VLLMConfig(hf_model_id="org/local-weights"),
        )
        try:
            with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}):
                with patch("redact.llms.backends.openai.openai.OpenAI"):
                    # No override: resolves to the row's own default setup.
                    assert backend_for(self._cfg(name)).backend_name == "openai"
            # Override: same row, explicitly requesting its other setup. Stub
            # the engine loader so this asserts the *routing* (which setup was
            # selected, with which weights) without loading a real engine.
            import redact.llms.backends.vllm as vllm_module

            built = {}

            def _stub_engine(hf_model_id, quantization, vllm_kwargs):
                built["hf_model_id"] = hf_model_id
                return MagicMock()

            with patch.object(vllm_module, "_engine", _stub_engine):
                backend = backend_for(self._cfg(name), "vllm")
            assert backend.backend_name == "vllm"
            # ...and it used the .vllm setup's weights, not the API default.
            assert built["hf_model_id"] == "org/local-weights"
            # A local binding carries no network budget.
            assert backend.rpm is None
        finally:
            MODEL_REGISTRY.pop(name, None)


class TestEngineCacheIsPerBackendClass:
    """Same checkpoint, two runtimes: never one shared load."""

    def teardown_method(self):
        clear_transport_caches()

    def test_two_models_on_one_checkpoint_share_a_vllm_engine(self):
        # venice-uncensored's .vllm setup and venice-paraphraser name the
        # same HF weights; loading them twice would double GPU memory.
        assert (MODEL_REGISTRY["venice-uncensored"].vllm.hf_model_id
                == MODEL_REGISTRY["venice-paraphraser"].vllm.hf_model_id)
        clear_transport_caches()
        loads = []

        class _FakeLLM:
            def __init__(self, **kw):
                loads.append(kw["model"])

        fake_vllm = MagicMock()
        fake_vllm.LLM = _FakeLLM
        with patch("redact.llms.backends.vllm._prepare_environment"):
            with patch.dict(sys.modules, {"vllm": fake_vllm}):
                a = backend_for(MODEL_REGISTRY["venice-uncensored"], "vllm")
                b = backend_for(MODEL_REGISTRY["venice-paraphraser"], "vllm")

        assert a is not b                 # two configured models
        assert a.model != b.model
        assert a._llm is b._llm           # one loaded engine
        assert loads == [MODEL_REGISTRY["venice-uncensored"].vllm.hf_model_id]

    def test_vllm_and_introspection_never_share_a_load(self):
        import redact.llms.backends.introspection as intro_module
        import redact.llms.backends.vllm as vllm_module

        clear_transport_caches()
        assert vllm_module._engines == {}
        assert intro_module._models == {}
        vllm_module._engines[("org/x", None, "[]")] = "engine"
        # The introspection cache is a different dict entirely, so the same
        # checkpoint cannot be served out of vLLM's.
        assert intro_module._models == {}
        clear_transport_caches()
        assert vllm_module._engines == {}
