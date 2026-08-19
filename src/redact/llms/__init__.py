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

import contextlib

from .api import (  # APIBackend = compat alias
    APIBackend,
    clear_backend_cache,
    get_backend,
)

# Abstract base
from .base import LLMBackend

# Model registry
from .model_config import (
    DEFAULT_RPM,
    MODEL_REGISTRY,
    ModelConfig,
    default_model_for_role,
    get_model_config,
    get_models_by_role,
    register_model,
)

# Backends
from .venice_backend import VeniceBackend

# vLLM is imported lazily to avoid hard dependency
with contextlib.suppress(ImportError):
    from .vllm_backend import VLLMBackend

# Anthropic is imported lazily to avoid hard dependency
with contextlib.suppress(ImportError):
    from .anthropic_backend import AnthropicBackend

# transformers/torch introspection backend is imported lazily to avoid hard dependency
with contextlib.suppress(ImportError):
    from .introspection_backend import TransformersIntrospectionBackend

# Wrappers
# High-level calls
from .calls import (
    batch_check_samples,
    check_sample,
    generate_sample,
    generate_with_check,
    is_accepted,
)

# Extraction utilities
from .extraction import (
    ConstitutionEntry,
    clean_sample,
    extract_and_clean,
    extract_bold_prompt_answer,
    extract_delimited,
    extract_numbered_list,
    extract_structured_qa,
    get_format_instruction,
    parse_constitution,
)

# Progress reporting (shared across all batched generation)
from .progress import ProgressReporter

# Prompt loading
from .prompts import build_messages, load_prompt, render_template

# Router (process-wide LLM access surface)
from .router import ModelRouter, clear_router, get_router

# Translation
from .translator import check_translation, translate, translate_with_check
from .wrappers import (
    BatchCaller,
    RateLimiter,
    assert_single_sample_per_call,
    with_feedback_retries,
    with_retries,
)
