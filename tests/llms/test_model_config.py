"""Tests for model registry and configuration."""

from redact.llms.model_config import (
    ModelConfig,
    MODEL_REGISTRY,
    DEFAULT_RPM,
    get_model_config,
    register_model,
)


class TestModelConfig:
    def test_dataclass_fields(self):
        config = ModelConfig(name="test", rpm=10)
        assert config.name == "test"
        assert config.rpm == 10
        assert config.default_max_tokens == 2000
        assert config.default_temperature is None
        assert config.default_extra_body is None
        assert config.backend_type is None
        assert config.hf_model_id is None
        assert config.quantization is None
        assert config.vllm_kwargs is None

    def test_vllm_fields(self):
        config = ModelConfig(
            name="local", rpm=999,
            backend_type="vllm",
            hf_model_id="org/model",
            quantization="gptq",
            vllm_kwargs={"gpu_memory_utilization": 0.9},
        )
        assert config.hf_model_id == "org/model"
        assert config.quantization == "gptq"
        assert config.vllm_kwargs["gpu_memory_utilization"] == 0.9


class TestModelRegistry:
    def test_known_models_present(self):
        expected = {"venice-uncensored", "deepseek-v3.2", "claude-opus-4-6", "venice-uncensored-vllm"}
        assert expected.issubset(set(MODEL_REGISTRY.keys()))

    def test_venice_has_extra_body(self):
        config = MODEL_REGISTRY["venice-uncensored"]
        assert config.default_extra_body is not None
        assert "venice_parameters" in config.default_extra_body

    def test_claude_backend_type(self):
        config = MODEL_REGISTRY["claude-opus-4-6"]
        assert config.backend_type == "anthropic"

    def test_vllm_has_hf_model_id(self):
        config = MODEL_REGISTRY["venice-uncensored-vllm"]
        assert config.backend_type == "vllm"
        assert config.hf_model_id is not None


class TestGetModelConfig:
    def test_known_model(self):
        config = get_model_config("venice-uncensored")
        assert config.name == "venice-uncensored"
        assert config.rpm == 75

    def test_unknown_model_returns_default(self):
        config = get_model_config("totally-unknown-model")
        assert config.rpm == DEFAULT_RPM
        assert config.default_extra_body is None


class TestRegisterModel:
    def test_register_new(self):
        register_model("test-new-model", rpm=42, backend_type="venice")
        config = get_model_config("test-new-model")
        assert config.rpm == 42
        assert config.backend_type == "venice"
        # Cleanup
        del MODEL_REGISTRY["test-new-model"]

    def test_register_overwrites(self):
        register_model("test-overwrite", rpm=10)
        register_model("test-overwrite", rpm=99)
        assert get_model_config("test-overwrite").rpm == 99
        del MODEL_REGISTRY["test-overwrite"]

    def test_register_vllm_fields(self):
        register_model(
            "test-vllm", rpm=999,
            backend_type="vllm",
            hf_model_id="org/model",
            quantization="awq",
            vllm_kwargs={"max_model_len": 4096},
        )
        config = get_model_config("test-vllm")
        assert config.hf_model_id == "org/model"
        assert config.quantization == "awq"
        assert config.vllm_kwargs["max_model_len"] == 4096
        del MODEL_REGISTRY["test-vllm"]
