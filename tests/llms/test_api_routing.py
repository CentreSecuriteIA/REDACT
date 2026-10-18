"""Tests for setup resolution and backend construction (capabilities.py)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from redact.llms.backends import (
    backend_for,
    capabilities,
    clear_transport_caches,
    resolve_setup,
    transport_for,
)
from redact.llms.client import ModelClient
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    ModelConfig,
    VLLMConfig,
    model_compute_config,
    register_model,
)
from redact.residency import footprint
from tests.conftest import fake_vllm_modules


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
        # ModelConfig rejects this at construction, so the setup is removed after.
        cfg = ModelConfig(name="m", vllm=VLLMConfig(hf_model_id="org/m"))
        cfg.vllm = None
        with pytest.raises(ValueError, match="no setup at all"):
            resolve_setup(cfg)

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
        # venice-uncensored-fp8 and venice-paraphraser name the same HF
        # weights at FP8; loading them twice would double GPU memory.
        assert (MODEL_REGISTRY["venice-uncensored-fp8"].vllm.hf_model_id
                == MODEL_REGISTRY["venice-paraphraser"].vllm.hf_model_id)
        clear_transport_caches()
        loads = []

        class _FakeLLM:
            scheduler_config = SimpleNamespace(max_num_seqs=256)

            def __init__(self, **kw):
                loads.append(kw["model"])

        # The load reads the HF config for the default context length.
        with patch("redact.llms.backends.vllm._prepare_environment"),              patch("redact.llms.resources.estimate._load_config", return_value=None):
            with patch.dict(sys.modules, fake_vllm_modules(_FakeLLM)):
                a = backend_for(MODEL_REGISTRY["venice-uncensored-fp8"], "vllm")
                b = backend_for(MODEL_REGISTRY["venice-paraphraser"], "vllm")

        assert a is not b                 # two configured models
        assert a.model != b.model
        assert a._llm is b._llm           # one loaded engine
        assert loads == [MODEL_REGISTRY["venice-uncensored-fp8"].vllm.hf_model_id]

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


class TestSetupReachesTheBackend:
    def test_min_gpus_reaches_the_engine_as_tensor_parallel_size(self):
        import redact.llms.backends.vllm as vllm_module
        seen = []

        class _StubLLM:
            scheduler_config = SimpleNamespace(max_num_seqs=256)

            def __init__(self, **kw):
                seen.append(kw)

        clear_transport_caches()
        try:
            with patch.dict(sys.modules, fake_vllm_modules(_StubLLM)), \
                 patch.object(vllm_module, "_prepare_environment"),                  patch("redact.llms.resources.estimate._load_config", return_value=None):
                backend_for(ModelConfig(name="m", backend_type="vllm", vllm=VLLMConfig(
                    hf_model_id="org/big", min_gpus=2)))
                backend_for(ModelConfig(name="m2", backend_type="vllm", vllm=VLLMConfig(
                    hf_model_id="org/big2", min_gpus=4,
                    vllm_kwargs={"tensor_parallel_size": 4})))
                backend_for(ModelConfig(name="m3", backend_type="vllm", vllm=VLLMConfig(
                    hf_model_id="org/small")))
        finally:
            clear_transport_caches()
        assert seen[0]["tensor_parallel_size"] == 2
        assert seen[1]["tensor_parallel_size"] == 4       # stated twice, in agreement
        assert "tensor_parallel_size" not in seen[2]

    def test_a_missing_api_key_names_the_model_and_the_variable(self):
        config = ModelConfig(name="m", api=APIConfig(
            backend_type="openai", api_key_env="_REDACT_TEST_MISSING_KEY",
            base_url="https://test.example/v1", rpm=10))
        with pytest.raises(KeyError, match="'m' needs an API key.*_REDACT_TEST_MISSING_KEY"):
            backend_for(config)

    def test_an_empty_api_key_counts_as_missing(self):
        config = ModelConfig(name="m", api=APIConfig(
            backend_type="openai", api_key_env="_REDACT_TEST_EMPTY_KEY",
            base_url="https://test.example/v1", rpm=10))
        with patch.dict(os.environ, {"_REDACT_TEST_EMPTY_KEY": ""}):
            with pytest.raises(KeyError, match="_REDACT_TEST_EMPTY_KEY"):
                backend_for(config)


class TestRunServing:
    """A run's choice for a model is read wherever no setup is asked for."""

    @pytest.fixture()
    def dual(self):
        """A model with a hosted and a local setup, the hosted one its default.

        Nothing is loaded and nothing is read from the network: the SDK
        client, the engine and the HF config are stubbed.
        """
        name = "test-routing-run-serving"
        MODEL_REGISTRY[name] = ModelConfig(
            name=name, backend_type="api",
            api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                          base_url="https://test.example/v1", rpm=10,
                          recommended_max_workers=3),
            vllm=VLLMConfig(hf_model_id="org/local-weights", vram_gb=5.0),
        )
        engine = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=256))
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}), \
             patch("redact.llms.backends.openai.openai.OpenAI"), \
             patch("redact.llms.backends.vllm._engine", return_value=engine), \
             patch("redact.llms.resources.estimate._load_config", return_value=None):
            yield name
        MODEL_REGISTRY.pop(name, None)
        clear_transport_caches()

    def test_the_runs_setup_is_picked_until_the_run_ends(self, dual):
        config = MODEL_REGISTRY[dual]
        capabilities.set_run_serving({dual: {"backend_type": "vllm"}})
        assert resolve_setup(config) == "vllm"
        assert transport_for(config) == "vllm"
        assert model_compute_config(dual).supports_native_batching is True
        assert footprint(dual).setup == "vllm"
        local = ModelClient.create(dual).backend
        assert (local.backend_name, local.hf_model_id, local.rpm) == (
            "vllm", "org/local-weights", None)

        capabilities.set_run_serving({})
        assert resolve_setup(config) == "api"
        assert transport_for(config) == "openai"
        assert model_compute_config(dual).supports_native_batching is False
        assert footprint(dual) is None
        hosted = ModelClient.create(dual).backend
        assert (hosted.backend_name, hosted.rpm) == ("openai", 10)

    def test_an_explicit_backend_type_wins_over_the_run(self, dual):
        capabilities.set_run_serving({dual: {"backend_type": "vllm"}})
        assert resolve_setup(MODEL_REGISTRY[dual], "api") == "api"
        assert model_compute_config(dual, "api").supports_native_batching is False
        assert footprint(dual, "api") is None
        hosted = ModelClient.create(dual, backend_type="api").backend
        assert hosted.backend_name == "openai"

    def test_a_model_the_run_does_not_name_keeps_its_default(self, dual):
        capabilities.set_run_serving({"another-model": {"backend_type": "vllm"}})
        assert resolve_setup(MODEL_REGISTRY[dual]) == "api"

    def test_the_runs_workers_replace_the_registrys_on_an_api_backend(self, dual):
        capabilities.set_run_serving({dual: {"workers": 2}})
        assert backend_for(MODEL_REGISTRY[dual]).max_workers == 2
        assert ModelClient.create(dual).backend.max_workers == 2
        # The registry entry is not edited.
        assert MODEL_REGISTRY[dual].api.recommended_max_workers == 3

        capabilities.set_run_serving({})
        assert ModelClient.create(dual).backend.max_workers == 3

    def test_the_runs_workers_do_not_reach_a_local_backend(self, dual):
        """A local model's requests at once stay the engine's."""
        capabilities.set_run_serving({dual: {"workers": 2}})
        assert ModelClient.create(dual, backend_type="vllm").backend.max_workers == 256

    def test_workers_a_series_only_provider_cannot_use_are_refused(self):
        """Raised, not clamped to 1 without a word."""
        capabilities.set_run_serving({"claude-opus-4-6": {"workers": 2}})
        with pytest.raises(ValueError, match="doesn't support parallel calls"):
            backend_for(MODEL_REGISTRY["claude-opus-4-6"])
