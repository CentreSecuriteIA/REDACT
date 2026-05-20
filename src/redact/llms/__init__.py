"""LLM abstraction layer — model-agnostic, backend-agnostic.

Quick start (auto-routing)::

    from redact.llms import get_backend, RateLimiter, generate_sample

    backend = get_backend("venice-uncensored")  # auto-selects VeniceBackend
    result = generate_sample(backend, "venice-uncensored", messages)

    backend = get_backend("claude-opus-4-6")    # auto-selects AnthropicBackend
    result = generate_sample(backend, "claude-opus-4-6", messages)

Direct instantiation::

    from redact.llms import VeniceBackend
    backend = VeniceBackend.from_env("VENICE_API_KEY")

Local inference via vLLM::

    from redact.llms import VLLMBackend
    backend = VLLMBackend(model="path/to/weights")
"""

# Model registry
from .model_config import (
    ModelConfig,
    MODEL_REGISTRY,
    DEFAULT_RPM,
    get_model_config,
    register_model,
    get_models_by_role,
    default_model_for_role,
)

# Abstract base
from .base import LLMBackend

# Backends
from .venice_backend import VeniceBackend
from .api import get_backend, clear_backend_cache, APIBackend  # APIBackend = compat alias

# vLLM is imported lazily to avoid hard dependency
try:
    from .vllm_backend import VLLMBackend
except ImportError:
    pass

# Anthropic is imported lazily to avoid hard dependency
try:
    from .anthropic_backend import AnthropicBackend
except ImportError:
    pass

# Wrappers
from .wrappers import (
    RateLimiter,
    with_retries,
    with_feedback_retries,
    BatchCaller,
)

# Router (process-wide LLM access surface)
from .router import ModelRouter, get_router, clear_router

# High-level calls
from .calls import generate_sample, check_sample, batch_check_samples, generate_with_check

# Translation
from .translator import translate, check_translation, translate_with_check

# Prompt loading
from .prompts import load_prompt, render_template, build_messages

# Extraction utilities
from .extraction import (
    get_format_instruction,
    extract_numbered_list,
    extract_structured_qa,
    extract_delimited,
    clean_sample,
    extract_and_clean,
    ConstitutionEntry,
    parse_constitution,
    extract_bold_prompt_answer,
)
