"""vLLM backend integration tests.

Requires: GPU with >= 8GB VRAM, vllm installed.
Run all:      pytest tests/test_vllm_backend.py -v
Run non-GPU:  pytest tests/test_vllm_backend.py -v -m "not gpu"
"""

import pytest

from redact.llms.model_config import (
    ModelConfig,
    MODEL_REGISTRY,
    get_model_config,
    register_model,
)
from redact.llms.api import get_backend, clear_backend_cache, _backend_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_MODEL_NAME = "test-gemma-2b"
TEST_HF_ID = "VibeStudio/Nidum-Gemma-2B-Uncensored"


def _vllm_available() -> bool:
    """Check if vLLM can run: requires Linux + CUDA GPU."""
    import sys
    if sys.platform != "linux":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _vllm_available(), reason="Requires Linux + CUDA GPU (vLLM)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vllm_model():
    """Register a small vLLM model for testing; clean up after."""
    register_model(
        TEST_MODEL_NAME,
        rpm=999,
        default_max_tokens=100,
        default_temperature=0.7,
        backend_type="vllm",
        hf_model_id=TEST_HF_ID,
    )
    yield TEST_MODEL_NAME
    MODEL_REGISTRY.pop(TEST_MODEL_NAME, None)
    _backend_cache.pop(f"vllm:{TEST_MODEL_NAME}", None)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure backend cache is clean between tests."""
    yield
    clear_backend_cache()


# ---------------------------------------------------------------------------
# Non-GPU tests (config and routing logic)
# ---------------------------------------------------------------------------

class TestModelConfigVLLMFields:
    """Verify ModelConfig accepts and stores vLLM-specific fields."""

    def test_fields_present(self):
        config = ModelConfig(
            name="test",
            rpm=100,
            backend_type="vllm",
            hf_model_id="org/model",
            quantization="gptq",
            vllm_kwargs={"gpu_memory_utilization": 0.9},
        )
        assert config.hf_model_id == "org/model"
        assert config.quantization == "gptq"
        assert config.vllm_kwargs == {"gpu_memory_utilization": 0.9}

    def test_fields_default_none(self):
        config = ModelConfig(name="test", rpm=100)
        assert config.hf_model_id is None
        assert config.quantization is None
        assert config.vllm_kwargs is None

    def test_register_model_passes_vllm_fields(self):
        name = "_test_register_vllm"
        try:
            register_model(
                name,
                rpm=999,
                backend_type="vllm",
                hf_model_id="org/model",
                quantization="awq",
                vllm_kwargs={"trust_remote_code": True},
            )
            config = get_model_config(name)
            assert config.backend_type == "vllm"
            assert config.hf_model_id == "org/model"
            assert config.quantization == "awq"
            assert config.vllm_kwargs == {"trust_remote_code": True}
        finally:
            MODEL_REGISTRY.pop(name, None)


class TestGetBackendRouting:
    """Verify get_backend() routing logic without needing a GPU."""

    def test_vllm_missing_hf_id_raises(self):
        name = "_test_no_hf_id"
        try:
            register_model(name, rpm=999, backend_type="vllm")
            with pytest.raises(ValueError, match="no hf_model_id"):
                get_backend(name)
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_venice_registry_entry_unchanged(self):
        config = get_model_config("venice-uncensored")
        assert config.backend_type == "venice"

    def test_venice_uncensored_vllm_registered(self):
        config = get_model_config("venice-uncensored-vllm")
        assert config.backend_type == "vllm"
        assert config.hf_model_id == "dphn/Dolphin-Mistral-24B-Venice-Edition"


# ---------------------------------------------------------------------------
# GPU tests (require vllm + CUDA)
# ---------------------------------------------------------------------------

@gpu
class TestVLLMBackendGPU:
    """Integration tests that load a real model on GPU."""

    @pytest.fixture(scope="class")
    def backend(self):
        """Load the 2B model once for all tests in this class."""
        from redact.llms.vllm_backend import VLLMBackend
        return VLLMBackend(model=TEST_HF_ID)

    def test_get_backend_routes_vllm(self, vllm_model):
        from redact.llms.vllm_backend import VLLMBackend
        backend = get_backend(vllm_model)
        assert isinstance(backend, VLLMBackend)

    def test_get_backend_caches_vllm(self, vllm_model):
        b1 = get_backend(vllm_model)
        b2 = get_backend(vllm_model)
        assert b1 is b2

    def test_generate(self, backend):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ]
        result = backend.generate(messages, model=TEST_MODEL_NAME)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_generate(self, backend):
        prompts = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
            [{"role": "user", "content": "What is 4+4?"}],
        ]
        results = backend.batch_generate(prompts, model=TEST_MODEL_NAME)
        assert len(results) == 3
        assert all(isinstance(r, str) and len(r) > 0 for r in results)
