"""LLM abstraction layer — model-agnostic, backend-agnostic.

Quick start:
    from Redact_Library.LLMs import APIBackend, RateLimiter, generate_sample

    backend = APIBackend.from_env("VENICE_API_KEY")
    limiter = RateLimiter()
    result = generate_sample(backend, "venice-uncensored", messages, rate_limiter=limiter)
"""

# Model registry
from .model_config import (
    ModelConfig,
    MODEL_REGISTRY,
    DEFAULT_RPM,
    get_model_config,
    register_model,
)

# Abstract base
from .base import LLMBackend

# Backends
from .api import APIBackend

# vLLM is imported lazily to avoid hard dependency
try:
    from .vllm_backend import VLLMBackend
except ImportError:
    pass

# Wrappers
from .wrappers import (
    RateLimiter,
    with_retries,
    with_feedback_retries,
    BatchCaller,
)

# High-level calls
from .calls import generate_sample, check_sample, generate_with_check

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
)
