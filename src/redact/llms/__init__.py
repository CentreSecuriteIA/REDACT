"""LLM abstraction layer — model-agnostic, backend-agnostic.

Quick start (auto-routing)::

    from redact.llms import ModelClient

    venice = ModelClient.create("venice-uncensored")   # Venice API
    replies = venice.generate(messages_list)           # batch in -> batch out

    local = ModelClient.create("venice-uncensored", backend_type="vllm")
    replies = local.generate(messages_list)            # same call, local engine

A :class:`ModelClient` *is* the model: transport, rate limiter and batch
strategy are wired at construction, so every client is called the same way and
no call site threads a limiter or picks a dispatch mode — see ``client.py``.

Direct instantiation (when you need a transport the registry doesn't describe).
Prefer ``register_model()`` — it validates the setup and gives you
``ModelClient.create()`` — since anything you don't pass here falls back to
:class:`~redact.llms.backends.base.LLMBackend`'s defaults, including
``rpm=None`` (no rate limiting)::

    import os
    from redact.llms import ModelClient, OpenAIBackend

    backend = OpenAIBackend("venice-uncensored", api_key=os.environ["VENICE_API_KEY"],
                            base_url="https://api.venice.ai/api/v1", rpm=75)
    client = ModelClient(backend)
"""

# Backends — importing the classes never requires their optional heavy
# dependency (vllm/anthropic/torch+transformers) to be installed; each
# backend's own __init__ lazily imports and guards that, only at
# instantiation time. See backends/__init__.py.
from .backends import (
    AnthropicBackend,
    ComputeConfig,
    LLMBackend,
    OpenAIBackend,
    TransformersIntrospectionBackend,
    VLLMBackend,
    backend_for,
    clear_transport_caches,
    resolve_setup,
)
from .client import ModelClient, clear_client_cache

# Extraction utilities
from .extraction import (
    EXTRACTION_STYLES,
    ConstitutionEntry,
    clean_sample,
    extract_and_clean,
    extract_delimited,
    extract_numbered_list,
    extract_structured_qa,
    get_format_instruction,
    parse_constitution,
)

# Model registry
from .model_config import (
    DEFAULT_RPM,
    MODEL_REGISTRY,
    APIConfig,
    IntrospectConfig,
    ModelConfig,
    VLLMConfig,
    available_roles,
    default_model_for_role,
    get_model_config,
    get_models_by_role,
    register_model,
)

# Progress reporting (shared across all batched generation)
from .progress import ProgressReporter

# Prompt loading
from .prompts import PromptTemplate, build_messages, load_prompt, render_template

# Caller-facing helpers over a client: single-sample, chunking, check loop.
from .router import (
    batch_check_samples,
    batch_generate_samples,
    check_sample,
    generate_sample,
    is_accepted,
)
from .wrappers import BatchCaller, RateLimiter, assert_single_sample_per_call
