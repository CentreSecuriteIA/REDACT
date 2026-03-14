"""LLM backend router — auto-selects backend from model name.

Given a model name, the router looks up its ``backend_type`` in the model
registry and returns the appropriate backend instance. Backends are cached
by type (not model name) since they are stateless API wrappers.

Usage::

    from redact.llms import get_backend, generate_sample

    backend = get_backend("venice-uncensored")   # → VeniceBackend
    backend = get_backend("claude-opus-4-6")      # → AnthropicBackend

For vLLM (local inference), create the backend manually::

    from redact.llms import VLLMBackend

    backend = VLLMBackend(model="path/to/weights", quantization="gptq")
    # Then pass it directly to pipeline functions:
    generate_inputs(model="venice-uncensored", backend=backend)

NOTE: Some models (e.g. venice-uncensored) can run both via API and locally
via vLLM for faster generation. To use local inference, create a VLLMBackend
instance and pass it as the ``backend`` parameter to pipeline functions.
The router only handles API backends (Venice, Anthropic) — vLLM requires
manual init with model path and GPU config.
"""

from .base import LLMBackend
from .model_config import get_model_config
from .venice_backend import VeniceBackend

# Backward-compat alias: existing code that imports APIBackend still works.
APIBackend = VeniceBackend

# Cache keyed by backend type ("venice", "anthropic"), not model name.
# One instance per type serves all models of that type.
_backend_cache: dict[str, LLMBackend] = {}


def _infer_backend_type(model: str) -> str:
    """Infer backend type from model name when not in the registry."""
    if model.startswith("claude-"):
        return "anthropic"
    return "venice"


def get_backend(model: str) -> LLMBackend:
    """Return a backend instance appropriate for the given model.

    Looks up ``backend_type`` from the model registry. If the model is
    not registered, infers the backend from the model name
    (``claude-*`` → Anthropic, everything else → Venice).

    Backends are cached by type — repeated calls for different models
    of the same type return the same backend instance.

    Args:
        model: Model identifier (e.g. "venice-uncensored", "claude-opus-4-6").

    Returns:
        A cached LLMBackend instance for the model's provider.

    Raises:
        ValueError: If the model's backend_type is "vllm" (requires manual init).
        KeyError: If the required API key env var is not set.
    """
    config = get_model_config(model)
    backend_type = config.backend_type or _infer_backend_type(model)

    if backend_type == "vllm":
        raise ValueError(
            "vLLM backends require manual init with model path and GPU config. "
            "Use VLLMBackend(model=...) directly and pass it as the `backend` "
            "parameter to pipeline functions."
        )

    if backend_type in _backend_cache:
        return _backend_cache[backend_type]

    if backend_type == "anthropic":
        from .anthropic_backend import AnthropicBackend

        backend: LLMBackend = AnthropicBackend.from_env()
    else:
        backend = VeniceBackend.from_env()

    _backend_cache[backend_type] = backend
    return backend


def clear_backend_cache() -> None:
    """Clear the backend instance cache.

    Useful for testing or when environment variables change mid-session.
    """
    _backend_cache.clear()
