"""Model registry with rate limits and provider-specific defaults.

Centralizes model metadata that was previously hardcoded across modules
(e.g. MODEL_RPMS in the jailbreak library's _RateLimitedClient).

Each model entry defines its RPM limit, default generation parameters,
and provider-specific extra_body parameters so callers don't need to
repeat them on every call.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    name: str
    rpm: int
    default_max_tokens: int = 2000
    default_temperature: float | None = None  # None = use provider default
    default_extra_body: dict | None = None
    # Backend type for auto-routing: "venice", "anthropic", "vllm", or None.
    # None triggers name-based inference in get_backend().
    backend_type: str | None = None
    # vLLM-specific fields (ignored by API backends).
    hf_model_id: str | None = None  # HuggingFace model ID or local path
    quantization: str | None = None  # e.g. "gptq", "awq"
    vllm_kwargs: dict | None = None  # Extra kwargs for vllm.LLM()
    # Capability flags consumed by the router + BatchCaller. These describe
    # what the model can do, independent of which backend implements it.
    is_uncensored: bool = False
    supports_system_prompt: bool = True
    supports_parallel_calls: bool = True   # False => series-only (Claude)
    supports_native_batching: bool = False  # True => single engine pass (vLLM)
    recommended_max_workers: int = 1
    # Logical role this model fills. Used by router.for_role() and by
    # default_model_for_role() so callers can ask for "translation" instead
    # of hardcoding "deepseek-v3.2".
    role: str | None = None


DEFAULT_RPM = 20

# Pre-populated with known models from the reference implementations.
# Users can add more via register_model().
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Venice AI models (OpenAI-compatible API).
    # NOTE: include_venice_system_prompt is not always disabled — some
    # pipelines may need the Venice system prompt active. Review per use case.
    "venice-uncensored": ModelConfig(
        name="venice-uncensored",
        rpm=75,
        backend_type="venice",
        default_extra_body={
            "venice_parameters": {
                "disable_thinking": True,
                "include_venice_system_prompt": False,
            }
        },
        is_uncensored=True,
        supports_parallel_calls=True,
        supports_native_batching=False,
        recommended_max_workers=3,
        role="uncensored_gen",
    ),
    "deepseek-v3.2": ModelConfig(
        name="deepseek-v3.2",
        rpm=20,
        backend_type="venice",
        default_extra_body={
            "venice_parameters": {
                "disable_thinking": True,
                "include_venice_system_prompt": False,
            }
        },
        is_uncensored=True,
        supports_parallel_calls=True,
        supports_native_batching=False,
        recommended_max_workers=2,
        role="translation",
    ),
    "olafangensan-glm-4.7-flash-heretic": ModelConfig(
        name="olafangensan-glm-4.7-flash-heretic",
        rpm=20,
        backend_type="venice",
        default_extra_body={
            "venice_parameters": {
                "disable_thinking": True,
                "include_venice_system_prompt": False,
            }
        },
        is_uncensored=True,
        supports_parallel_calls=True,
        supports_native_batching=False,
        recommended_max_workers=2,
        role="uncensored_gen",
    ),
    # Local vLLM models — use model name with "-vllm" suffix to distinguish
    # from API variants. Auto-routed by get_backend() when backend_type="vllm".
    "venice-uncensored-vllm": ModelConfig(
        name="venice-uncensored-vllm",
        rpm=999,
        default_max_tokens=2000,
        default_temperature=0.7,
        backend_type="vllm",
        hf_model_id="dphn/Dolphin-Mistral-24B-Venice-Edition",
        is_uncensored=True,
        supports_parallel_calls=False,   # thread-level parallel => GPU contention
        supports_native_batching=True,    # use backend.batch_generate() instead
        recommended_max_workers=1,
        role="uncensored_local",
    ),
    # Anthropic Claude — strict rate limits (typically 5 RPM on free tier).
    # Series-only enforced by the router; BatchCaller raises ValueError if a
    # caller tries to use max_workers>1.
    "claude-opus-4-6": ModelConfig(
        name="claude-opus-4-6",
        rpm=5,
        default_max_tokens=300,
        backend_type="anthropic",
        is_uncensored=False,
        supports_parallel_calls=False,
        supports_native_batching=False,
        recommended_max_workers=1,
        role="constitution_gen",
    ),
    # Paraphraser role - a DEDICATED model with its OWN identity (not an alias). The
    # real defingerprinting model is trained separately and unavailable here, so as a
    # stand-in this entry LOADS THE SAME HF WEIGHTS as venice-uncensored-vllm
    # (Dolphin-Mistral-24B-Venice-Edition) via vLLM - it is its own model that happens
    # to load the same weights, not the same registry entry. Swap hf_model_id, or
    # register the real paraphraser with role="paraphraser", when it lands.
    "venice-paraphraser": ModelConfig(
        name="venice-paraphraser",
        rpm=999,
        default_temperature=0.8,
        backend_type="vllm",
        hf_model_id="dphn/Dolphin-Mistral-24B-Venice-Edition",
        is_uncensored=True,
        supports_parallel_calls=False,
        supports_native_batching=True,
        recommended_max_workers=1,
        role="paraphraser",
    ),
}


def get_model_config(model: str) -> ModelConfig:
    """Look up model config from registry.

    Returns default config (DEFAULT_RPM, no extra params) for unknown models.
    """
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    return ModelConfig(name=model, rpm=DEFAULT_RPM)


def register_model(
    name: str,
    rpm: int,
    default_max_tokens: int = 2000,
    default_temperature: float | None = None,
    default_extra_body: dict | None = None,
    backend_type: str | None = None,
    hf_model_id: str | None = None,
    quantization: str | None = None,
    vllm_kwargs: dict | None = None,
    is_uncensored: bool = False,
    supports_system_prompt: bool = True,
    supports_parallel_calls: bool = True,
    supports_native_batching: bool = False,
    recommended_max_workers: int = 1,
    role: str | None = None,
) -> None:
    """Add or update a model in the registry at runtime.

    Args:
        name: Model identifier.
        rpm: Requests per minute limit.
        default_max_tokens: Default max tokens for generation.
        default_temperature: Default sampling temperature (None = provider default).
        default_extra_body: Provider-specific parameters merged into every call.
        backend_type: Backend for auto-routing ("venice", "anthropic", "vllm").
            None triggers name-based inference in get_backend().
        hf_model_id: HuggingFace model ID or local path (vLLM only).
        quantization: Quantization method, e.g. "gptq", "awq" (vLLM only).
        vllm_kwargs: Extra kwargs passed to vllm.LLM() (vLLM only).
        is_uncensored: True for uncensored generation models.
        supports_system_prompt: False if the model ignores system messages.
        supports_parallel_calls: False forces series-only execution (Claude).
        supports_native_batching: True if backend has a true batch path (vLLM).
        recommended_max_workers: Default concurrency the router applies.
        role: Logical role for router.for_role() lookup.
    """
    MODEL_REGISTRY[name] = ModelConfig(
        name=name,
        rpm=rpm,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
        default_extra_body=default_extra_body,
        backend_type=backend_type,
        hf_model_id=hf_model_id,
        quantization=quantization,
        vllm_kwargs=vllm_kwargs,
        is_uncensored=is_uncensored,
        supports_system_prompt=supports_system_prompt,
        supports_parallel_calls=supports_parallel_calls,
        supports_native_batching=supports_native_batching,
        recommended_max_workers=recommended_max_workers,
        role=role,
    )


def get_models_by_role(role: str) -> list[ModelConfig]:
    """Return all registered models whose ``role`` matches.

    Order follows registration order (insertion order of MODEL_REGISTRY).
    Returns an empty list if no model has that role.
    """
    return [cfg for cfg in MODEL_REGISTRY.values() if cfg.role == role]


def default_model_for_role(role: str) -> str:
    """Return the first registered model name for a role.

    Raises:
        KeyError: If no model in the registry has this role.
    """
    matches = get_models_by_role(role)
    if not matches:
        raise KeyError(f"No model registered for role {role!r}")
    return matches[0].name
