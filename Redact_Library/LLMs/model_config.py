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


DEFAULT_RPM = 20

# Pre-populated with known models from the reference implementations.
# Users can add more via register_model().
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # NOTE: include_venice_system_prompt is not always disabled — some
    # pipelines may need the Venice system prompt active. Review per use case.
    "venice-uncensored": ModelConfig(
        name="venice-uncensored",
        rpm=75,
        default_extra_body={
            "venice_parameters": {
                "disable_thinking": True,
                "include_venice_system_prompt": False,
            }
        },
    ),
    "deepseek-v3.2": ModelConfig(
        name="deepseek-v3.2",
        rpm=20,
        default_extra_body={
            "venice_parameters": {
                "disable_thinking": True,
                "include_venice_system_prompt": False,
            }
        },
    ),
    "olafangensan-glm-4.7-flash-heretic": ModelConfig(
        name="olafangensan-glm-4.7-flash-heretic",
        rpm=20,
        default_extra_body={
            "venice_parameters": {
                "disable_thinking": True,
                "include_venice_system_prompt": False,
            }
        },
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
) -> None:
    """Add or update a model in the registry at runtime."""
    MODEL_REGISTRY[name] = ModelConfig(
        name=name,
        rpm=rpm,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
        default_extra_body=default_extra_body,
    )
