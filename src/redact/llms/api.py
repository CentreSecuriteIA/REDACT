"""LLM backend router — auto-selects backend from model name.

Given a model name, the router looks up its ``backend_type`` in the model
registry and returns the appropriate backend instance. API backends are
cached by type (stateless wrappers), vLLM backends are cached by model
name (each loads unique weights).

Usage::

    from redact.llms import get_backend, generate_sample

    backend = get_backend("venice-uncensored")       # → VeniceBackend (API)
    backend = get_backend("claude-opus-4-6")          # → AnthropicBackend
    backend = get_backend("venice-uncensored-vllm")   # → VLLMBackend (local)

Register new vLLM models at runtime::

    from redact.llms import register_model, get_backend

    register_model("my-llama", rpm=999, backend_type="vllm",
                   hf_model_id="meta-llama/Llama-3-70B-GPTQ", quantization="gptq")
    backend = get_backend("my-llama")  # → VLLMBackend
"""

from .base import LLMBackend
from .model_config import get_model_config
from .venice_backend import VeniceBackend

# Backward-compat alias: existing code that imports APIBackend still works.
APIBackend = VeniceBackend

# Cache keyed by backend type for API backends ("venice", "anthropic")
# and by "vllm:{model_name}" for vLLM backends (per-model instances).
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

    API backends are cached by type — repeated calls for different models
    of the same type return the same backend instance. vLLM backends are
    cached by model name since each loads unique weights.

    Args:
        model: Model identifier (e.g. "venice-uncensored",
            "claude-opus-4-6", "venice-uncensored-vllm").

    Returns:
        A cached LLMBackend instance for the model's provider.

    Raises:
        ValueError: If backend_type is "vllm" but no ``hf_model_id`` is set.
        KeyError: If the required API key env var is not set.
    """
    config = get_model_config(model)
    backend_type = config.backend_type or _infer_backend_type(model)

    if backend_type == "vllm":
        cache_key = f"vllm:{model}"
        if cache_key in _backend_cache:
            return _backend_cache[cache_key]

        if not config.hf_model_id:
            raise ValueError(
                f"Model {model!r} has backend_type='vllm' but no hf_model_id. "
                f"Register it with register_model(..., hf_model_id='...') or "
                f"create VLLMBackend manually."
            )

        from .vllm_backend import VLLMBackend

        backend: LLMBackend = VLLMBackend(
            model=config.hf_model_id,
            quantization=config.quantization,
            **(config.vllm_kwargs or {}),
        )
        _backend_cache[cache_key] = backend
        return backend

    if backend_type == "transformers_introspect":
        cache_key = f"introspect:{model}"
        if cache_key in _backend_cache:
            return _backend_cache[cache_key]

        if not config.hf_model_id:
            raise ValueError(
                f"Model {model!r} has backend_type='transformers_introspect' but "
                f"no hf_model_id. Register it with register_model(..., "
                f"hf_model_id='...', introspect_kwargs={{'log_dir': ...}})."
            )
        introspect_kwargs = dict(config.introspect_kwargs or {})
        if "log_dir" not in introspect_kwargs:
            raise ValueError(
                f"Model {model!r} has backend_type='transformers_introspect' but "
                f"introspect_kwargs has no 'log_dir'. Register it with "
                f"register_model(..., introspect_kwargs={{'log_dir': ...}})."
            )

        from .introspection_backend import TransformersIntrospectionBackend

        backend = TransformersIntrospectionBackend(
            model=config.hf_model_id, **introspect_kwargs,
        )
        _backend_cache[cache_key] = backend
        return backend

    if backend_type in _backend_cache:
        return _backend_cache[backend_type]

    if backend_type == "anthropic":
        from .anthropic_backend import AnthropicBackend

        backend = AnthropicBackend.from_env()
    else:
        backend = VeniceBackend.from_env()

    _backend_cache[backend_type] = backend
    return backend


def clear_backend_cache() -> None:
    """Clear the backend instance cache.

    Useful for testing or when environment variables change mid-session.
    """
    _backend_cache.clear()
