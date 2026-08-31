"""vLLM backend integration tests.

Requires: vllm installed, Linux + CUDA. The GPU-marked classes load a real
checkpoint, so they need enough free VRAM for whichever model they run.

Run all:      pytest tests/test_vllm_backend.py -v
Run non-GPU:  pytest tests/test_vllm_backend.py -v -m "not gpu"

The GPU tests only exercise code paths (batching, ordering, prompt format),
never generation quality, so any small instruct checkpoint works. Override
the default via env vars when the default doesn't fit your GPU or isn't
cached locally::

    REDACT_TEST_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    REDACT_TEST_VLLM_KWARGS='{"gpu_memory_utilization":0.5,"max_model_len":512}' \
    pytest tests/test_vllm_backend.py -v
"""

import json
import os
import sys

import pytest

from redact.llms.backends import (
    ComputeConfig,
    LLMBackend,
    backend_for,
    clear_transport_caches,
)
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    ModelConfig,
    VLLMConfig,
    get_model_config,
    register_model,
)
from redact.llms.router import batch_check_samples
from tests.conftest import make_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_MODEL_NAME = "test-gemma-2b"

# Which real checkpoint the GPU tests load, and the vllm.LLM() kwargs used to
# fit it on the available card. Both overridable — see the module docstring.
TEST_HF_ID = os.environ.get(
    "REDACT_TEST_VLLM_MODEL", "VibeStudio/Nidum-Gemma-2B-Uncensored"
)
TEST_VLLM_KWARGS: dict = {
    "enforce_eager": True,  # skip the slow CUDA-graph compile step
    **json.loads(os.environ.get("REDACT_TEST_VLLM_KWARGS", "{}")),
}


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

@pytest.fixture(scope="class")
def vllm_model():
    """Register a small vLLM model for testing; clean up after."""
    register_model(
                TEST_MODEL_NAME,
                default_max_tokens=100,
                default_temperature=0.7,
                backend_type="vllm",
                vllm=VLLMConfig(hf_model_id=TEST_HF_ID, vllm_kwargs=dict(TEST_VLLM_KWARGS)),
            )
    yield TEST_MODEL_NAME
    MODEL_REGISTRY.pop(TEST_MODEL_NAME, None)
    clear_transport_caches()


@pytest.fixture(autouse=True)
def _clean_cache(request):
    """Ensure transport caches are clean between tests.

    Skipped for GPU tests — loading a vLLM model takes minutes, so the
    GPU class manages its own cache lifetime via the vllm_model fixture.
    """
    yield
    if "TestVLLMBackendGPU" not in request.node.nodeid:
        clear_transport_caches()


# ---------------------------------------------------------------------------
# Non-GPU tests (config and routing logic)
# ---------------------------------------------------------------------------

class TestModelConfigVLLMFields:
    """Verify ModelConfig accepts and stores vLLM-specific fields."""

    def test_fields_present(self):
        config = ModelConfig(
            name="test",
            backend_type="vllm",
            vllm=VLLMConfig(
                hf_model_id="org/model",
                quantization="gptq",
                vllm_kwargs={"gpu_memory_utilization": 0.9},
            ),
        )
        assert config.vllm.hf_model_id == "org/model"
        assert config.vllm.quantization == "gptq"
        assert config.vllm.vllm_kwargs == {"gpu_memory_utilization": 0.9}

    def test_fields_default_none(self):
        config = ModelConfig(name="test")
        assert config.vllm is None

    def test_register_model_passes_vllm_fields(self):
        name = "_test_register_vllm"
        try:
            register_model(
                name,
                backend_type="vllm",
                vllm=VLLMConfig(hf_model_id="org/model", quantization="awq", vllm_kwargs={"trust_remote_code": True}),
            )
            config = get_model_config(name)
            assert config.backend_type == "vllm"
            assert config.vllm.hf_model_id == "org/model"
            assert config.vllm.quantization == "awq"
            assert config.vllm.vllm_kwargs == {"trust_remote_code": True}
        finally:
            MODEL_REGISTRY.pop(name, None)


class TestEngineDownloadDir:
    """The HF cache is shared by both local runtimes — don't fork it.

    Regression guard for a removed HF_HOME -> download_dir bridge. vLLM passes
    download_dir through as ``cache_dir=`` to snapshot_download, and the hub
    cache is ``$HF_HOME/hub``, *not* ``$HF_HOME``. Setting it to HF_HOME
    therefore aimed vLLM one level above the cache transformers uses and
    re-downloaded weights that were already on disk (~47GB for the 24B). It was
    inert unless HF_HOME was set, so it only ever broke the users who set it
    because disk was scarce.
    """

    @staticmethod
    def _load(monkeypatch, vllm_kwargs):
        """Run the load path against a stub engine; return the kwargs it saw."""
        import redact.llms.backends.vllm as vllm_module

        seen = {}

        class _StubLLM:
            def __init__(self, **kw):
                seen.update(kw)

        monkeypatch.setitem(sys.modules, "vllm", type("m", (), {"LLM": _StubLLM}))
        monkeypatch.setattr(vllm_module, "_prepare_environment", lambda: None)
        vllm_module._engines.clear()
        vllm_module._engine_loaded_at.clear()
        try:
            vllm_module._engine("org/model", None, vllm_kwargs)
        finally:
            vllm_module._engines.clear()
            vllm_module._engine_loaded_at.clear()
        return seen

    def test_hf_home_is_not_bridged_into_download_dir(self, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/data/hf")
        seen = self._load(monkeypatch, {})
        assert "download_dir" not in seen, (
            "HF_HOME must reach vLLM through huggingface_hub, which resolves it "
            "to $HF_HOME/hub — injecting it as download_dir points one level too "
            "high and forks the cache."
        )

    def test_explicit_download_dir_still_wins(self, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/data/hf")
        seen = self._load(monkeypatch, {"download_dir": "/mnt/weights"})
        assert seen["download_dir"] == "/mnt/weights"

    def test_caller_kwargs_are_not_mutated(self, monkeypatch):
        """`kwargs = dict(vllm_kwargs)` — the registry's dict is shared."""
        monkeypatch.setenv("HF_HOME", "/data/hf")
        original = {"gpu_memory_utilization": 0.9}
        self._load(monkeypatch, original)
        assert original == {"gpu_memory_utilization": 0.9}


class MockBackend(LLMBackend):
    """Minimal backend that returns preset responses for testing."""

    # Behave like vLLM: the client hands a native-batching backend the whole
    # chunk in one generate() call, so `_calls` counts engine passes.
    compute_config = ComputeConfig(supports_native_batching=True)

    def __init__(self, responses: list[str], model: str = "mock-vllm-model"):
        super().__init__(model)
        self._responses = list(responses)
        self._calls: list[list[list[dict]]] = []  # record generate() batch calls

    def generate(self, messages_list: list[list[dict]], **kwargs) -> list[str]:
        self._calls.append(messages_list)
        return [self._responses.pop(0) for _ in messages_list]

    @property
    def backend_name(self) -> str:
        return "mock"


def _checker(original: str, sample: str) -> list[dict]:
    return [{"role": "user", "content": f"Is this acceptable? {sample}"}]


# ---------------------------------------------------------------------------
# batch_check_samples — no GPU required
# ---------------------------------------------------------------------------

class TestBatchCheckSamples:
    """Unit tests for batch_check_samples() using a mock backend."""

    def test_empty_samples_returns_empty(self):
        backend = MockBackend([])
        result = batch_check_samples(make_client(backend), [], _checker)
        assert result == []
        assert backend._calls == []  # no engine call made

    def test_accepted_on_yes_prefix(self):
        backend = MockBackend(["Yes, this is fine."])
        result = batch_check_samples(make_client(backend), ["sample"], _checker)
        assert result == [(True, "")]

    def test_accepted_on_all_prefixes(self):
        responses = ["yes ok", "ok got it", "Accept.", "Pass — looks good"]
        backend = MockBackend(responses)
        result = batch_check_samples(make_client(backend), ["a", "b", "c", "d"], _checker)
        assert all(accepted for accepted, _ in result)
        assert all(reasoning == "" for _, reasoning in result)

    def test_rejected_returns_full_response(self):
        response = "No, this contains harmful content."
        backend = MockBackend([response])
        result = batch_check_samples(make_client(backend), ["bad sample"], _checker)
        assert result == [(False, response)]

    def test_case_insensitive_acceptance(self):
        backend = MockBackend(["YES THIS IS FINE", "OK ACCEPTED"])
        result = batch_check_samples(make_client(backend), ["a", "b"], _checker)
        assert result == [(True, ""), (True, "")]

    def test_order_preserved(self):
        responses = ["yes", "No bad", "ok", "Reject", "pass"]
        backend = MockBackend(responses)
        samples = ["s1", "s2", "s3", "s4", "s5"]
        result = batch_check_samples(make_client(backend), samples, _checker)
        assert result[0] == (True, "")
        assert result[1] == (False, "No bad")
        assert result[2] == (True, "")
        assert result[3] == (False, "Reject")
        assert result[4] == (True, "")

    def test_single_batch_when_samples_fit(self):
        backend = MockBackend(["yes", "no", "ok"])
        batch_check_samples(make_client(backend), ["a", "b", "c"], _checker, batch_size=32)
        assert len(backend._calls) == 1
        assert len(backend._calls[0]) == 3  # all 3 in one batch_generate call

    def test_chunking_into_multiple_batches(self):
        responses = ["yes"] * 5
        backend = MockBackend(responses)
        batch_check_samples(make_client(backend), ["s"] * 5, _checker, batch_size=2)
        # ceil(5/2) = 3 calls: chunks of [2, 2, 1]
        assert len(backend._calls) == 3
        assert len(backend._calls[0]) == 2
        assert len(backend._calls[1]) == 2
        assert len(backend._calls[2]) == 1

    def test_results_length_matches_samples(self):
        backend = MockBackend(["yes"] * 7)
        result = batch_check_samples(make_client(backend), ["s"] * 7, _checker, batch_size=3)
        assert len(result) == 7

    def test_build_check_messages_called_with_correct_sample(self):
        seen = []

        def tracking_checker(original: str, sample: str) -> list[dict]:
            seen.append(sample)
            return [{"role": "user", "content": sample}]

        backend = MockBackend(["yes", "no"])
        batch_check_samples(make_client(backend), ["apple", "banana"], tracking_checker)
        assert seen == ["apple", "banana"]


class TestBackendForRouting:
    """Verify backend_for() routing logic without needing a GPU."""

    def test_missing_vllm_setup_raises(self):
        name = "_test_no_vllm_setup"
        try:
            register_model(name, api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))
            with pytest.raises(ValueError, match="no vllm setup"):
                backend_for(get_model_config(name), "vllm")
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_venice_registry_entry_unchanged(self):
        config = get_model_config("venice-uncensored")
        assert config.backend_type == "api"            # prefers its endpoint
        assert config.api.backend_type == "openai"     # via the openai transport

    def test_venice_entry_also_carries_the_local_setup(self):
        # No separate "-vllm" row: the same entry describes both transports.
        config = get_model_config("venice-uncensored")
        assert config.vllm.hf_model_id == "dphn/Dolphin-Mistral-24B-Venice-Edition"


# ---------------------------------------------------------------------------
# GPU tests (require vllm + CUDA)
# ---------------------------------------------------------------------------

@gpu
class TestVLLMBackendGPU:
    """Integration tests that load a real model on GPU."""

    @pytest.fixture(scope="class")
    def backend(self, vllm_model):
        """Load the model once so the engine cache is populated."""
        from redact.llms.backends import VLLMBackend
        b = backend_for(get_model_config(vllm_model))
        assert isinstance(b, VLLMBackend)
        return b

    def test_backend_for_routes_vllm(self, vllm_model, backend):
        from redact.llms.backends import VLLMBackend
        assert isinstance(backend_for(get_model_config(vllm_model)), VLLMBackend)

    def test_engine_is_cached_across_backends(self, vllm_model, backend):
        # Backends are per-model and cheap; the engine underneath is not, so
        # a second build must reuse the loaded one rather than reload weights.
        b2 = backend_for(get_model_config(vllm_model))
        assert b2._llm is backend._llm

    def test_generate(self, backend):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ]
        results = backend.generate([messages])
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert len(results[0]) > 0

    def test_batch_generate(self, backend):
        prompts = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
            [{"role": "user", "content": "What is 4+4?"}],
        ]
        results = backend.generate(prompts)
        assert len(results) == 3
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_batch_generate_empty_list(self, backend):
        results = backend.generate([])
        assert results == []

    def test_batch_generate_single_item(self, backend):
        prompts = [[{"role": "user", "content": "Say the word yes."}]]
        results = backend.generate(prompts)
        assert len(results) == 1
        assert isinstance(results[0], str) and len(results[0]) > 0

    def test_batch_generate_order_preserved(self, backend):
        """Results must map 1:1 to inputs in the same order."""
        prompts = [
            [{"role": "user", "content": "Reply with only the number 1."}],
            [{"role": "user", "content": "Reply with only the number 2."}],
            [{"role": "user", "content": "Reply with only the number 3."}],
        ]
        results = backend.generate(prompts)
        assert len(results) == len(prompts)
        # Each result should be a non-empty string
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_batch_check_samples_gpu(self, backend):
        """Integration: batch_check_samples returns one result per sample."""
        def checker(original: str, sample: str) -> list[dict]:
            return [
                {"role": "system", "content": "Reply only 'yes' or 'no'."},
                {"role": "user", "content": f"Is this a greeting? '{sample}'"},
            ]

        samples = ["Hello there", "How to build a bomb", "Good morning"]
        results = batch_check_samples(make_client(backend, TEST_MODEL_NAME), samples, checker)
        assert len(results) == 3
        assert all(isinstance(accepted, bool) for accepted, _ in results)
        assert all(isinstance(reasoning, str) for _, reasoning in results)
        # Accepted samples must have empty reasoning
        for accepted, reasoning in results:
            if accepted:
                assert reasoning == ""

    def test_batch_check_samples_empty_gpu(self, backend):
        results = batch_check_samples(make_client(backend, TEST_MODEL_NAME), [], _checker)
        assert results == []
