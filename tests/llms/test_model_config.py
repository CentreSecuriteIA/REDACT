"""Tests for model registry and configuration."""

import pytest

from redact.llms.backends import (
    API_BACKEND_TYPES,
    BACKEND_TYPES,
    LLMBackend,
    compute_config_for,
    validate_concurrency,
)
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    IntrospectConfig,
    ModelConfig,
    VLLMConfig,
    get_model_config,
    register_model,
)


class TestModelConfig:
    def test_dataclass_fields(self):
        config = ModelConfig(name="test", api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))
        assert config.name == "test"
        assert config.api.rpm == 10
        assert config.default_max_tokens == 2000
        assert config.default_temperature is None
        assert config.backend_type is None
        assert config.vllm is None
        assert config.introspect is None

    def test_identity_fields_have_no_backend_setup_by_default(self):
        config = ModelConfig(name="bare")
        assert config.api is None
        assert config.vllm is None
        assert config.introspect is None

    def test_vllm_fields(self):
        config = ModelConfig(
            name="local",
            backend_type="vllm",
            vllm=VLLMConfig(
                hf_model_id="org/model",
                quantization="gptq",
                vllm_kwargs={"gpu_memory_utilization": 0.9},
            ),
        )
        assert config.vllm.hf_model_id == "org/model"
        assert config.vllm.quantization == "gptq"
        assert config.vllm.vllm_kwargs["gpu_memory_utilization"] == 0.9

    def test_introspect_fields(self):
        config = ModelConfig(
            name="local-introspect",
            backend_type="introspect",
            introspect=IntrospectConfig(
                hf_model_id="org/model",
                log_dir="/tmp/x",
                capture={"attention": True},
            ),
        )
        assert config.introspect.hf_model_id == "org/model"
        assert config.introspect.log_dir == "/tmp/x"
        assert config.introspect.capture == {"attention": True}


class TestModelRegistry:
    def test_known_models_present(self):
        expected = {"venice-uncensored", "deepseek-v3.2", "claude-opus-4-6"}
        assert expected.issubset(set(MODEL_REGISTRY.keys()))

    def test_venice_has_extra_body(self):
        config = MODEL_REGISTRY["venice-uncensored"]
        assert config.api is not None
        assert config.api.default_extra_body is not None
        assert "venice_parameters" in config.api.default_extra_body

    def test_claude_backend_type(self):
        config = MODEL_REGISTRY["claude-opus-4-6"]
        assert config.backend_type == "api"          # which setup
        assert config.api.backend_type == "anthropic"  # which provider

    def test_one_entry_carries_both_setups(self):
        # A model that exists as a hosted endpoint AND as local weights is one
        # model: one entry with both setups, not two near-duplicate rows.
        config = MODEL_REGISTRY["venice-uncensored"]
        assert config.backend_type == "api"           # the default setup
        assert config.api.backend_type == "openai"    # the provider
        assert config.api is not None
        assert config.vllm is not None
        assert config.vllm.hf_model_id is not None

    def test_no_separate_vllm_twin_row(self):
        # Deliberately removed: the "-vllm" suffix row duplicated an identity
        # that backend_type already selects between.
        assert "venice-uncensored-vllm" not in MODEL_REGISTRY

    def test_local_only_model_has_no_api_config(self):
        # A genuinely local-only model isn't rate-limited at all.
        assert MODEL_REGISTRY["llama-3.2-3b-debug"].api is None

    def test_local_selection_does_not_inherit_endpoint_rpm(self):
        # The whole point of one-entry-two-setups: binding the local setup
        # must NOT pick up the hosted endpoint's rate limit. The budget lives
        # on the backend, resolved from the setup it was actually built from.
        from redact.llms import ModelClient

        api_client = ModelClient.create("venice-uncensored")
        assert api_client.backend.rpm == 75

        cfg = MODEL_REGISTRY["venice-uncensored"]
        assert cfg.api is not None and cfg.api.rpm == 75  # entry still declares it

        from redact.llms.backends import LLMBackend
        from tests.conftest import make_client

        class _LocalT(LLMBackend):
            def generate(self, messages_list, **kw):
                return [""] * len(messages_list)

            @property
            def backend_name(self):
                return "vllm"

        local_client = make_client(_LocalT("x"), "venice-uncensored", backend_type="vllm")
        assert local_client.backend.rpm is None

    def test_model_compute_config_reads_flags_without_building(self):
        # Answers the capability question off the backend *class* — building
        # a local transport to ask would load model weights.
        from redact.llms.model_config import model_compute_config

        assert model_compute_config("venice-uncensored").supports_native_batching is False
        assert model_compute_config("venice-uncensored", "vllm").supports_native_batching is True
        assert model_compute_config("claude-opus-4-6").supports_parallel_calls is False


class TestGetModelConfig:
    def test_known_model(self):
        config = get_model_config("venice-uncensored")
        assert config.name == "venice-uncensored"
        assert config.api.rpm == 75

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="totally-unknown-model"):
            get_model_config("totally-unknown-model")


class TestRegisterModel:
    def test_register_new(self):
        register_model(
                "test-new-model",
                backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=42),
            )
        config = get_model_config("test-new-model")
        assert config.api.rpm == 42
        assert config.backend_type == "api"
        # Cleanup
        del MODEL_REGISTRY["test-new-model"]

    def test_register_overwrites(self):
        register_model(
                "test-overwrite",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10),
            )
        register_model(
                "test-overwrite",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=99),
            )
        assert get_model_config("test-overwrite").api.rpm == 99
        del MODEL_REGISTRY["test-overwrite"]

    def test_register_vllm_fields(self):
        register_model(
                "test-vllm",
                backend_type="vllm",
                vllm=VLLMConfig(hf_model_id="org/model", quantization="awq", vllm_kwargs={"max_model_len": 4096}),
            )
        config = get_model_config("test-vllm")
        assert config.vllm.hf_model_id == "org/model"
        assert config.vllm.quantization == "awq"
        assert config.vllm.vllm_kwargs["max_model_len"] == 4096
        assert config.api is None  # no rpm passed -> no API config
        del MODEL_REGISTRY["test-vllm"]

    def test_register_without_rpm_has_no_api_config(self):
        register_model(
                "test-no-rpm",
                backend_type="vllm",
                vllm=VLLMConfig(hf_model_id="org/model"),
            )
        assert get_model_config("test-no-rpm").api is None
        del MODEL_REGISTRY["test-no-rpm"]

    def test_register_introspect_fields(self):
        register_model(
                "test-introspect",
                backend_type="introspect",
                introspect=IntrospectConfig(hf_model_id="org/model", log_dir="/tmp/x", capture={"logprobs": True}),
            )
        config = get_model_config("test-introspect")
        assert config.introspect.hf_model_id == "org/model"
        assert config.introspect.log_dir == "/tmp/x"
        assert config.introspect.capture == {"logprobs": True}
        assert config.vllm is None
        del MODEL_REGISTRY["test-introspect"]

    def test_series_only_backend_with_max_workers_raises(self):
        with pytest.raises(ValueError, match="recommended_max_workers"):
            register_model(
                "test-bad-workers",
                backend_type="api",
                api=APIConfig(backend_type="anthropic",
                              api_key_env="ANTHROPIC_API_KEY",
                              rpm=5, recommended_max_workers=2),
            )
        assert "test-bad-workers" not in MODEL_REGISTRY

    def test_workers_are_checked_against_the_api_provider(self):
        # The budget governs the API transport, so a vllm-preferring entry
        # with an anthropic API setup is still rejected for workers>1.
        with pytest.raises(ValueError, match="recommended_max_workers"):
            register_model(
                "test-bad-vllm-workers",
                backend_type="vllm",
                api=APIConfig(backend_type="anthropic",
                              api_key_env="ANTHROPIC_API_KEY",
                              rpm=20, recommended_max_workers=2),
                vllm=VLLMConfig(hf_model_id="org/model"),
            )
        assert "test-bad-vllm-workers" not in MODEL_REGISTRY

    def test_parallel_backend_with_max_workers_is_fine(self):
        register_model(
                "test-ok-workers",
                backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10, recommended_max_workers=3),
            )
        assert get_model_config("test-ok-workers").api.recommended_max_workers == 3
        del MODEL_REGISTRY["test-ok-workers"]


class TestValidateConcurrency:
    """Direct tests of the registration-time guard, independent of
    register_model(
                ,
            )'s other side effects.
    """

    def test_noop_when_backend_type_unknown(self):
        validate_concurrency("m", None, recommended_max_workers=10)  # no raise

    def test_noop_when_max_workers_is_one(self):
        validate_concurrency("m", "anthropic", recommended_max_workers=1)  # no raise

    def test_noop_for_unrecognized_backend_type_string(self):
        # Best-effort: only the known profiles table is checked.
        validate_concurrency("m", "some-future-backend", recommended_max_workers=5)

    def test_raises_for_series_only_backend_type(self):
        with pytest.raises(ValueError, match="anthropic"):
            validate_concurrency("claude-like", "anthropic", recommended_max_workers=2)

    def test_raises_for_native_batching_backend_type(self):
        with pytest.raises(ValueError, match="vllm"):
            validate_concurrency("local-model", "vllm", recommended_max_workers=2)

    def test_allows_parallel_backend_type(self):
        validate_concurrency("api-model", "openai", recommended_max_workers=5)  # no raise


class TestBackendCapabilities:
    """backends/capabilities.py is the single source for backend-type facts."""

    def test_every_backend_type_maps_to_a_real_backend(self):
        for backend_type, cls in BACKEND_TYPES.items():
            assert issubclass(cls, LLMBackend), backend_type

    def test_capabilities_read_off_the_class_without_constructing(self):
        # The whole point: constructing a vLLM/introspection backend loads
        # weights, so registration-time validation must not need an instance.
        for backend_type, cls in BACKEND_TYPES.items():
            assert compute_config_for(backend_type) is cls.compute_config

    def test_every_backend_declares_all_three_flags_explicitly(self):
        # No silent inheritance: reading one class tells you its whole profile.
        import inspect
        for cls in BACKEND_TYPES.values():
            src = inspect.getsource(cls)
            assert "compute_config: ClassVar[ComputeConfig]" in src, cls.__name__
            for flag in ("supports_native_batching", "supports_parallel_calls",
                         "supports_internals"):
                assert flag in src, f"{cls.__name__} leaves {flag} implicit"

    def test_setup_types_are_named_after_their_config_field(self):
        """The convention that removes the need for a lookup table.

        Every non-API backend type must be spelled the same as the
        ModelConfig field holding its setup ("vllm" -> .vllm). If a new
        backend type breaks that, ModelConfig's getattr would raise at
        registration instead of a clear message — so fail here, loudly.
        """
        import dataclasses

        from redact.llms.backends import SETUP_TYPES
        from redact.llms.model_config import ModelConfig

        fields = {f.name for f in dataclasses.fields(ModelConfig)}
        for setup in SETUP_TYPES:
            assert setup in fields, (
                f"setup {setup!r} has no matching ModelConfig field; either "
                f"rename it or reintroduce an explicit mapping"
            )

    def test_api_backend_types_are_a_subset_of_all_types(self):
        assert API_BACKEND_TYPES < set(BACKEND_TYPES)

    def test_unknown_and_absent_backend_types_are_not_errors(self):
        # A registry entry may legitimately leave backend_type unset (inferred
        # later from the model name), so this must not raise.
        assert compute_config_for(None) is None
        assert compute_config_for("some-future-backend") is None

    def test_transport_and_setup_namespaces_stay_distinct(self):
        # BACKEND_TYPES names transports; SETUP_TYPES names the fields a
        # ModelConfig can prefer. "api" covers both providers, which is why
        # the two sets are not the same thing.
        from redact.llms.backends import SETUP_TYPES

        assert set(BACKEND_TYPES) == {"openai", "anthropic", "vllm", "introspect"}
        assert set(SETUP_TYPES) == {"api", "vllm", "introspect"}
        assert API_BACKEND_TYPES < set(BACKEND_TYPES)


class TestConcurrencyCheckerGuardsUserRegistrations:
    """The reason the checker exists: someone registering their own model."""

    def test_rejects_parallel_workers_on_a_series_only_backend(self):
        name = "_user_model_series_only"
        try:
            with pytest.raises(ValueError, match="doesn't support parallel calls"):
                register_model(
                name,
                backend_type="api",
                api=APIConfig(backend_type="anthropic",
                              api_key_env="ANTHROPIC_API_KEY",
                              rpm=5, recommended_max_workers=4),
            )
            assert name not in MODEL_REGISTRY  # rejected, not half-registered
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_accepts_parallel_workers_on_an_api_backend(self):
        name = "_user_model_parallel_ok"
        try:
            register_model(
                name,
                backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=60, recommended_max_workers=4),
            )
            assert get_model_config(name).api.recommended_max_workers == 4
        finally:
            MODEL_REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------
# Each setup validates itself, so these test the config in isolation — no
# registry, no model name, no backend. That separation is the point: a bad
# rpm is an APIConfig problem, not a register_model problem.
# ---------------------------------------------------------------------------


class TestAPIConfigValidatesItself:
    def test_accepts_a_valid_setup(self):
        cfg = APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=60, recommended_max_workers=3,
                        default_extra_body={"k": "v"})
        assert (cfg.rpm, cfg.recommended_max_workers) == (60, 3)

    def test_rpm_is_required(self):
        with pytest.raises(TypeError):
            APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", )

    @pytest.mark.parametrize("rpm", [0, -1])
    def test_rejects_out_of_range_rpm(self, rpm):
        with pytest.raises(ValueError, match="rpm"):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=rpm)

    @pytest.mark.parametrize("rpm", ["60", 1.5, True])
    def test_rejects_wrong_typed_rpm(self, rpm):
        # bool included: it subclasses int, so True would pass as 1.
        with pytest.raises(TypeError, match="rpm"):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=rpm)

    @pytest.mark.parametrize("workers", [0, -2])
    def test_rejects_out_of_range_worker_count(self, workers):
        with pytest.raises(ValueError, match="recommended_max_workers"):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=10,
                      recommended_max_workers=workers)

    @pytest.mark.parametrize("workers", ["3", True])
    def test_rejects_wrong_typed_worker_count(self, workers):
        with pytest.raises(TypeError, match="recommended_max_workers"):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=10,
                      recommended_max_workers=workers)

    def test_rejects_non_dict_extra_body(self):
        with pytest.raises(TypeError, match="default_extra_body"):
            APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10, default_extra_body=["not", "a", "dict"])


class TestApiModelIdAndPricing:
    """The registry name is the stable identity; the slug is what goes upstream."""

    def test_every_shipped_api_model_is_priced(self):
        """An unpriced model reports tokens but no cost — the roll-up goes half
        dark, which is what motivated filling these in.

        Reads models.json rather than MODEL_REGISTRY: the live registry is
        mutable and other tests register throwaway models into it (some of them
        deliberately unpriced), so asserting over it would be asserting on test
        order. The claim is about what *ships*.
        """
        import json

        from redact import paths

        shipped = json.loads(paths.models_json().read_text(encoding="utf-8"))
        unpriced = [
            name for name, entry in shipped.items()
            if "api" in entry and not (
                entry["api"].get("price_per_1m_input") is not None
                and entry["api"].get("price_per_1m_output") is not None
            )
        ]
        assert not unpriced, f"shipped API models with no price: {unpriced}"

    def test_venice_uncensored_sends_the_versioned_slug(self):
        api = get_model_config("venice-uncensored").api
        assert api.api_model_id == "venice-uncensored-1-2"

    def test_registry_name_is_unchanged_by_the_slug(self):
        """Renaming the entry instead would orphan traces and break pricing:
        telemetry emits self.model and summary() looks *that* up in the
        registry, so the two must stay the registry name."""
        cfg = get_model_config("venice-uncensored")
        assert cfg.name == "venice-uncensored"
        assert cfg.api.price_per_1m_input == 0.2

    def test_slug_reaches_the_request_and_registry_name_reaches_telemetry(self):
        """The whole point of the split, asserted end to end on one call."""
        import os
        from unittest.mock import MagicMock, patch

        from redact.llms import observe
        from redact.llms.backends import OpenAIBackend

        events = []
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))], usage=None,
        )
        OpenAIBackend.clear_cache()
        with patch.dict(os.environ, {"VENICE_API_KEY": "k"}):
            with patch("redact.llms.backends.openai.openai.OpenAI", return_value=sdk):
                backend = OpenAIBackend.from_config(get_model_config("venice-uncensored"))
        observe.set_emitter(events.append)
        try:
            backend.generate([[{"role": "user", "content": "hi"}]])
        finally:
            observe.set_emitter(None)
            OpenAIBackend.clear_cache()

        sent = sdk.chat.completions.create.call_args.kwargs["model"]
        assert sent == "venice-uncensored-1-2"          # provider's slug
        call = next(e for e in events if e.get("ev") == "call")
        assert call["model"] == "venice-uncensored"     # registry identity
        assert MODEL_REGISTRY[call["model"]].api.price_per_1m_input == 0.2

    def test_defaults_to_the_registry_name_when_no_slug_is_set(self):
        cfg = get_model_config("deepseek-v3.2")
        assert cfg.api.api_model_id is None

    def test_rejects_a_non_string_slug(self):
        with pytest.raises(TypeError):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=1, api_model_id=7)

    def test_rejects_an_empty_slug(self):
        with pytest.raises(ValueError):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=1, api_model_id="  ")

    def test_rejects_a_negative_price(self):
        with pytest.raises(ValueError):
            APIConfig(backend_type="openai", api_key_env="K",
                      base_url="https://x/v1", rpm=1, price_per_1m_input=-1.0)


class TestVLLMConfigValidatesItself:
    def test_accepts_a_valid_setup(self):
        cfg = VLLMConfig(hf_model_id="org/m", quantization="awq",
                         vllm_kwargs={"trust_remote_code": True})
        assert cfg.hf_model_id == "org/m"

    def test_weights_are_required_not_validated_later(self):
        # hf_model_id has no default: "a vLLM setup without weights" is
        # unrepresentable rather than something to check downstream.
        with pytest.raises(TypeError):
            VLLMConfig()

    @pytest.mark.parametrize("hf_id", ["", "   "])
    def test_rejects_empty_hf_model_id(self, hf_id):
        with pytest.raises(ValueError, match="hf_model_id"):
            VLLMConfig(hf_model_id=hf_id)

    @pytest.mark.parametrize("hf_id", [None, 42])
    def test_rejects_wrong_typed_hf_model_id(self, hf_id):
        with pytest.raises(TypeError, match="hf_model_id"):
            VLLMConfig(hf_model_id=hf_id)

    def test_rejects_non_dict_vllm_kwargs(self):
        with pytest.raises(TypeError, match="vllm_kwargs"):
            VLLMConfig(hf_model_id="org/m", vllm_kwargs=[1])

    def test_rejects_non_string_quantization(self):
        with pytest.raises(TypeError, match="quantization"):
            VLLMConfig(hf_model_id="org/m", quantization=8)


class TestIntrospectConfigValidatesItself:
    def test_accepts_a_valid_setup(self):
        cfg = IntrospectConfig(hf_model_id="org/m", log_dir="/tmp/x",
                               capture={"logprobs": True})
        assert (cfg.hf_model_id, cfg.log_dir) == ("org/m", "/tmp/x")

    def test_both_weights_and_log_dir_are_required(self):
        with pytest.raises(TypeError):
            IntrospectConfig(hf_model_id="org/m")     # nowhere to write
        with pytest.raises(TypeError):
            IntrospectConfig(log_dir="/tmp/x")        # nothing to run

    @pytest.mark.parametrize("log_dir", ["", "  "])
    def test_rejects_empty_log_dir(self, log_dir):
        with pytest.raises(ValueError, match="log_dir"):
            IntrospectConfig(hf_model_id="org/m", log_dir=log_dir)

    @pytest.mark.parametrize("log_dir", [None, 7])
    def test_rejects_wrong_typed_log_dir(self, log_dir):
        with pytest.raises(TypeError, match="log_dir"):
            IntrospectConfig(hf_model_id="org/m", log_dir=log_dir)

    def test_rejects_non_dict_capture(self):
        with pytest.raises(TypeError, match="capture"):
            IntrospectConfig(hf_model_id="org/m", log_dir="/tmp/x", capture="all")


class TestModelConfigValidatesIdentityAndComposition:
    """What only makes sense once the setups are assembled."""

    @pytest.mark.parametrize("name", ["", "   "])
    def test_rejects_empty_name(self, name):
        with pytest.raises(ValueError, match="name"):
            ModelConfig(name=name)

    def test_rejects_wrong_typed_name(self):
        with pytest.raises(TypeError, match="name"):
            ModelConfig(name=None)

    def test_rejects_blank_role(self):
        with pytest.raises(ValueError, match="role"):
            ModelConfig(name="m", roles=["   "])

    def test_rejects_unknown_role(self):
        # Roles must be declared in configs/llm/roles.json, so a typo fails
        # here rather than surfacing later as "no model registered for role".
        with pytest.raises(ValueError, match="unknown role"):
            ModelConfig(name="m", roles=["uncensored_genn"])

    def test_rejects_non_list_roles(self):
        with pytest.raises(TypeError, match="roles"):
            ModelConfig(name="m", roles="uncensored_gen")

    @pytest.mark.parametrize("temp", [-0.1, 2.5])
    def test_rejects_temperature_out_of_range(self, temp):
        with pytest.raises(ValueError, match="default_temperature"):
            ModelConfig(name="m", default_temperature=temp)

    def test_rejects_wrong_typed_temperature(self):
        with pytest.raises(TypeError, match="default_temperature"):
            ModelConfig(name="m", default_temperature="hot")

    @pytest.mark.parametrize("tokens", [0, -5])
    def test_rejects_out_of_range_max_tokens(self, tokens):
        with pytest.raises(ValueError, match="default_max_tokens"):
            ModelConfig(name="m", default_max_tokens=tokens)

    def test_rejects_wrong_typed_max_tokens(self):
        with pytest.raises(TypeError, match="default_max_tokens"):
            ModelConfig(name="m", default_max_tokens="many")

    def test_rejects_non_bool_flags(self):
        with pytest.raises(TypeError, match="is_uncensored"):
            ModelConfig(name="m", is_uncensored="yes")

    def test_rejects_unknown_backend_type(self):
        with pytest.raises(ValueError, match="unknown backend_type"):
            ModelConfig(name="m", backend_type="gpt5")

    def test_rejects_a_setup_object_of_the_wrong_type(self):
        with pytest.raises(TypeError, match="vllm must be a VLLMConfig"):
            ModelConfig(name="m", vllm=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))

    def test_default_backend_type_must_have_its_setup(self):
        with pytest.raises(ValueError, match="needs a vllm config"):
            ModelConfig(name="m", backend_type="vllm")
        with pytest.raises(ValueError, match="needs a introspect config"):
            ModelConfig(name="m", backend_type="introspect")

    def test_rejects_parallel_workers_on_a_series_only_provider(self):
        with pytest.raises(ValueError, match="doesn't support parallel calls"):
            ModelConfig(name="m", backend_type="api",
                        api=APIConfig(backend_type="anthropic",
                                      api_key_env="ANTHROPIC_API_KEY",
                                      rpm=5, recommended_max_workers=4))

    def test_workers_follow_the_api_provider_not_the_preferred_setup(self):
        # An entry can prefer its local setup and still carry a real API
        # budget: the workers govern the API transport, which the APIConfig
        # names itself, so this is legitimate rather than contradictory.
        cfg = ModelConfig(
            name="m", backend_type="vllm",
            vllm=VLLMConfig(hf_model_id="org/m"),
            api=APIConfig(backend_type="openai", api_key_env="K",
                          base_url="https://x/v1", rpm=10,
                          recommended_max_workers=3),
        )
        assert cfg.api.recommended_max_workers == 3

    def test_backend_type_none_defers_cross_checks(self):
        # Inferred from the name later, so there is nothing to check yet.
        cfg = ModelConfig(name="m", api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))
        assert cfg.backend_type is None


class TestRegisterModelComposesSetups:
    """register_model only assembles — the pieces are already valid."""

    def test_registers_a_single_setup(self):
        name = "_test_single_setup"
        try:
            register_model(name, backend_type="vllm",
                           vllm=VLLMConfig(hf_model_id="org/m", quantization="awq"))
            cfg = get_model_config(name)
            assert cfg.vllm.quantization == "awq"
            assert cfg.api is None and cfg.introspect is None
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_registers_both_setups_on_one_entry(self):
        # The capability the flat-kwargs signature could not express.
        name = "_test_dual_setup"
        try:
            register_model(
                name,
                backend_type="api",                       # the default
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=60, recommended_max_workers=3),
                vllm=VLLMConfig(hf_model_id="org/my-model"),  # also reachable
            )
            cfg = get_model_config(name)
            assert cfg.backend_type == "api"
            assert cfg.api.backend_type == "openai"
            assert cfg.api.rpm == 60
            assert cfg.vllm.hf_model_id == "org/my-model"
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_a_user_can_reproduce_the_builtin_dual_entry(self):
        builtin = get_model_config("venice-uncensored")
        name = "_test_like_venice"
        try:
            register_model(
                name,
                backend_type="api",
                api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=builtin.api.rpm,
                              recommended_max_workers=builtin.api.recommended_max_workers),
                vllm=VLLMConfig(hf_model_id=builtin.vllm.hf_model_id),
            )
            mine = get_model_config(name)
            assert (mine.api.rpm, mine.vllm.hf_model_id) == (
                builtin.api.rpm, builtin.vllm.hf_model_id)
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_a_rejected_registration_writes_nothing(self):
        name = "_test_rejected"
        try:
            with pytest.raises(ValueError):
                register_model(name, backend_type="vllm")   # no vllm setup
            assert name not in MODEL_REGISTRY
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_overwrites_an_existing_entry(self):
        name = "_test_overwrite"
        try:
            register_model(name, api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))
            register_model(name, api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=99))
            assert get_model_config(name).api.rpm == 99
        finally:
            MODEL_REGISTRY.pop(name, None)
