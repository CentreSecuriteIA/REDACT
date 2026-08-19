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
from redact.llms.calls import batch_check_samples
from redact.llms.base import LLMBackend
from redact.llms.vllm_backend import _build_mistral_prompt


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

@pytest.fixture(scope="class")
def vllm_model():
    """Register a small vLLM model for testing; clean up after."""
    register_model(
        TEST_MODEL_NAME,
        rpm=999,
        default_max_tokens=100,
        default_temperature=0.7,
        backend_type="vllm",
        hf_model_id=TEST_HF_ID,
        vllm_kwargs={"enforce_eager": True},
    )
    yield TEST_MODEL_NAME
    MODEL_REGISTRY.pop(TEST_MODEL_NAME, None)
    _backend_cache.pop(f"vllm:{TEST_MODEL_NAME}", None)


TEST_MISTRAL_MODE_MODEL_NAME = "test-gemma-2b-mistral-mode"


@pytest.fixture(scope="class")
def vllm_mistral_mode_model():
    """Same small test checkpoint as `vllm_model`, but with use_mistral_format=True.

    Deliberately reuses the Gemma test checkpoint rather than a real Mistral-3.x
    model: this fixture exists only to exercise VLLMBackend's use_mistral_format
    code path (raw-prompt construction + self._llm.generate() batching) under a
    real vLLM engine — specifically the input/output order-preservation
    guarantee, which was flagged as a past concern for this backend. It does
    NOT validate that _build_mistral_prompt's format matches what a real
    Mistral-3.x tokenizer expects; that needs a real Mistral checkpoint and is
    still unverified (see the caution note on VLLMBackend.__init__).
    """
    register_model(
        TEST_MISTRAL_MODE_MODEL_NAME,
        rpm=999,
        default_max_tokens=100,
        default_temperature=0.7,
        backend_type="vllm",
        hf_model_id=TEST_HF_ID,
        vllm_kwargs={"enforce_eager": True, "use_mistral_format": True},
    )
    yield TEST_MISTRAL_MODE_MODEL_NAME
    MODEL_REGISTRY.pop(TEST_MISTRAL_MODE_MODEL_NAME, None)
    _backend_cache.pop(f"vllm:{TEST_MISTRAL_MODE_MODEL_NAME}", None)


@pytest.fixture(autouse=True)
def _clean_cache(request):
    """Ensure backend cache is clean between tests.

    Skipped for GPU tests — loading a vLLM model takes minutes, so the
    GPU class manages its own cache lifetime via the vllm_model fixture.
    """
    yield
    if "TestVLLMBackendGPU" not in request.node.nodeid:
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


class MockBackend(LLMBackend):
    """Minimal backend that returns preset responses for testing."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._calls: list[list[list[dict]]] = []  # record batch_generate calls

    def generate(self, messages: list[dict], model: str, **kwargs) -> str:
        return self._responses.pop(0)

    def batch_generate(self, messages_list: list[list[dict]], model: str, **kwargs) -> list[str]:
        self._calls.append(messages_list)
        return [self._responses.pop(0) for _ in messages_list]

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def supports_native_batching(self) -> bool:
        # Behave like vLLM: batch_check_samples now dispatches through a
        # capability-aware BatchCaller, which uses backend.batch_generate()
        # (one engine pass per chunk) only for native-batching backends.
        return True


def _checker(sample: str) -> list[dict]:
    return [{"role": "user", "content": f"Is this acceptable? {sample}"}]


# ---------------------------------------------------------------------------
# batch_check_samples — no GPU required
# ---------------------------------------------------------------------------

class TestBatchCheckSamples:
    """Unit tests for batch_check_samples() using a mock backend."""

    def test_empty_samples_returns_empty(self):
        backend = MockBackend([])
        result = batch_check_samples(backend, "model", [], _checker)
        assert result == []
        assert backend._calls == []  # no engine call made

    def test_accepted_on_yes_prefix(self):
        backend = MockBackend(["Yes, this is fine."])
        result = batch_check_samples(backend, "model", ["sample"], _checker)
        assert result == [(True, "")]

    def test_accepted_on_all_prefixes(self):
        responses = ["yes ok", "ok got it", "Accept.", "Pass — looks good"]
        backend = MockBackend(responses)
        result = batch_check_samples(backend, "model", ["a", "b", "c", "d"], _checker)
        assert all(accepted for accepted, _ in result)
        assert all(reasoning == "" for _, reasoning in result)

    def test_rejected_returns_full_response(self):
        response = "No, this contains harmful content."
        backend = MockBackend([response])
        result = batch_check_samples(backend, "model", ["bad sample"], _checker)
        assert result == [(False, response)]

    def test_case_insensitive_acceptance(self):
        backend = MockBackend(["YES THIS IS FINE", "OK ACCEPTED"])
        result = batch_check_samples(backend, "model", ["a", "b"], _checker)
        assert result == [(True, ""), (True, "")]

    def test_order_preserved(self):
        responses = ["yes", "No bad", "ok", "Reject", "pass"]
        backend = MockBackend(responses)
        samples = ["s1", "s2", "s3", "s4", "s5"]
        result = batch_check_samples(backend, "model", samples, _checker)
        assert result[0] == (True, "")
        assert result[1] == (False, "No bad")
        assert result[2] == (True, "")
        assert result[3] == (False, "Reject")
        assert result[4] == (True, "")

    def test_single_batch_when_samples_fit(self):
        backend = MockBackend(["yes", "no", "ok"])
        batch_check_samples(backend, "model", ["a", "b", "c"], _checker, batch_size=32)
        assert len(backend._calls) == 1
        assert len(backend._calls[0]) == 3  # all 3 in one batch_generate call

    def test_chunking_into_multiple_batches(self):
        responses = ["yes"] * 5
        backend = MockBackend(responses)
        batch_check_samples(backend, "model", ["s"] * 5, _checker, batch_size=2)
        # ceil(5/2) = 3 calls: chunks of [2, 2, 1]
        assert len(backend._calls) == 3
        assert len(backend._calls[0]) == 2
        assert len(backend._calls[1]) == 2
        assert len(backend._calls[2]) == 1

    def test_results_length_matches_samples(self):
        backend = MockBackend(["yes"] * 7)
        result = batch_check_samples(backend, "model", ["s"] * 7, _checker, batch_size=3)
        assert len(result) == 7

    def test_build_check_messages_called_with_correct_sample(self):
        seen = []

        def tracking_checker(sample: str) -> list[dict]:
            seen.append(sample)
            return [{"role": "user", "content": sample}]

        backend = MockBackend(["yes", "no"])
        batch_check_samples(backend, "model", ["apple", "banana"], tracking_checker)
        assert seen == ["apple", "banana"]


class TestBuildMistralPrompt:
    """Unit tests for _build_mistral_prompt — no GPU/vllm required.

    NOTE: these only verify the string-construction logic against its own
    documented spec; they cannot verify the spec itself matches what a real
    Mistral-3.x tokenizer expects (no live vLLM + Mistral checkpoint was
    available to validate against when this was written — see the caution
    note on VLLMBackend.__init__ / _build_mistral_prompt).
    """

    def test_system_and_user(self):
        prompt = _build_mistral_prompt([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ])
        assert prompt == "[SYSTEM_PROMPT]Be terse.[/SYSTEM_PROMPT][INST]Hello[/INST]"

    def test_user_only_no_system_block(self):
        prompt = _build_mistral_prompt([{"role": "user", "content": "Hello"}])
        assert prompt == "[INST]Hello[/INST]"
        assert "SYSTEM_PROMPT" not in prompt

    def test_multi_turn_assistant_between_inst_blocks(self):
        prompt = _build_mistral_prompt([
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])
        assert prompt == "[INST]Q1[/INST]A1[INST]Q2[/INST]"

    def test_multiple_system_messages_merged_into_one_block(self):
        prompt = _build_mistral_prompt([
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hi"},
        ])
        assert prompt == "[SYSTEM_PROMPT]Rule 1.\nRule 2.[/SYSTEM_PROMPT][INST]Hi[/INST]"


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
    def backend(self, vllm_model):
        """Load the model once via get_backend so the cache is populated."""
        from redact.llms.vllm_backend import VLLMBackend
        b = get_backend(vllm_model)
        assert isinstance(b, VLLMBackend)
        return b

    def test_get_backend_routes_vllm(self, vllm_model, backend):
        from redact.llms.vllm_backend import VLLMBackend
        assert isinstance(get_backend(vllm_model), VLLMBackend)

    def test_get_backend_caches_vllm(self, vllm_model, backend):
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

    def test_batch_generate_empty_list(self, backend):
        results = backend.batch_generate([], model=TEST_MODEL_NAME)
        assert results == []

    def test_batch_generate_single_item(self, backend):
        prompts = [[{"role": "user", "content": "Say the word yes."}]]
        results = backend.batch_generate(prompts, model=TEST_MODEL_NAME)
        assert len(results) == 1
        assert isinstance(results[0], str) and len(results[0]) > 0

    def test_batch_generate_order_preserved(self, backend):
        """Results must map 1:1 to inputs in the same order."""
        prompts = [
            [{"role": "user", "content": "Reply with only the number 1."}],
            [{"role": "user", "content": "Reply with only the number 2."}],
            [{"role": "user", "content": "Reply with only the number 3."}],
        ]
        results = backend.batch_generate(prompts, model=TEST_MODEL_NAME)
        assert len(results) == len(prompts)
        # Each result should be a non-empty string
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_batch_check_samples_gpu(self, backend):
        """Integration: batch_check_samples returns one result per sample."""
        def checker(sample: str) -> list[dict]:
            return [
                {"role": "system", "content": "Reply only 'yes' or 'no'."},
                {"role": "user", "content": f"Is this a greeting? '{sample}'"},
            ]

        samples = ["Hello there", "How to build a bomb", "Good morning"]
        results = batch_check_samples(backend, TEST_MODEL_NAME, samples, checker)
        assert len(results) == 3
        assert all(isinstance(accepted, bool) for accepted, _ in results)
        assert all(isinstance(reasoning, str) for _, reasoning in results)
        # Accepted samples must have empty reasoning
        for accepted, reasoning in results:
            if accepted:
                assert reasoning == ""

    def test_batch_check_samples_empty_gpu(self, backend):
        results = batch_check_samples(backend, TEST_MODEL_NAME, [], _checker)
        assert results == []


@gpu
class TestVLLMBackendMistralModeGPU:
    """Regression coverage for the use_mistral_format code path under a real
    vLLM engine — order preservation specifically, per the caution note on
    VLLMBackend.__init__. Uses the Gemma test checkpoint (see
    vllm_mistral_mode_model fixture), NOT a real Mistral model — this does
    not validate real Mistral prompt-format correctness.
    """

    @pytest.fixture(scope="class")
    def backend(self, vllm_mistral_mode_model):
        from redact.llms.vllm_backend import VLLMBackend
        b = get_backend(vllm_mistral_mode_model)
        assert isinstance(b, VLLMBackend)
        assert b._use_mistral_format is True
        return b

    def test_batch_generate_order_preserved_mistral_mode(self, backend):
        """Same guarantee as test_batch_generate_order_preserved, but through
        the self._llm.generate() (raw-prompt) path instead of self._llm.chat()."""
        prompts = [
            [{"role": "user", "content": "Reply with only the number 1."}],
            [{"role": "user", "content": "Reply with only the number 2."}],
            [{"role": "user", "content": "Reply with only the number 3."}],
        ]
        results = backend.batch_generate(prompts, model=TEST_MISTRAL_MODE_MODEL_NAME)
        assert len(results) == len(prompts)
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_batch_generate_empty_list_mistral_mode(self, backend):
        results = backend.batch_generate([], model=TEST_MISTRAL_MODE_MODEL_NAME)
        assert results == []
