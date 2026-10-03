"""LLM backends, one module per provider, each implementing ``LLMBackend``.

To add a provider, implement ``LLMBackend`` (``base.py``), including
``from_config()``, in a new module here and add it to
``capabilities.BACKEND_TYPES``. A hosted API also goes in
``capabilities.API_BACKEND_TYPES``.
"""

from .anthropic import AnthropicBackend
from .base import (
    ComputeConfig,
    LLMBackend,
    extract_system_prompt,
    fold_system_into_first_message,
)
from .capabilities import (
    API_BACKEND_TYPES,
    BACKEND_TYPES,
    SETUP_TYPES,
    backend_for,
    clear_transport_caches,
    compute_config_for,
    resolve_setup,
    transport_for,
    validate_concurrency,
)
from .introspection import TransformersIntrospectionBackend
from .openai import OpenAIBackend
from .vllm import VLLMBackend

__all__ = [
    "API_BACKEND_TYPES",
    "SETUP_TYPES",
    "BACKEND_TYPES",
    "AnthropicBackend",
    "ComputeConfig",
    "LLMBackend",
    "OpenAIBackend",
    "TransformersIntrospectionBackend",
    "VLLMBackend",
    "backend_for",
    "clear_transport_caches",
    "compute_config_for",
    "extract_system_prompt",
    "fold_system_into_first_message",
    "resolve_setup",
    "transport_for",
    "validate_concurrency",
]
