"""LLM layer: the model registry, the backends that serve it, and prompt helpers.

Usage::

    from redact.llms import ModelClient

    venice = ModelClient.create("venice-uncensored")   # Venice API
    replies = venice.generate(messages_list)           # batch in -> batch out

    local = ModelClient.create("venice-uncensored", backend_type="vllm")
    replies = local.generate(messages_list)            # same call, local engine

For a model the registry does not have, call ``register_model()`` and then
``ModelClient.create()``. A backend built by hand and wrapped in
``ModelClient(backend)`` uses ``LLMBackend``'s defaults for anything not
passed, including ``rpm=None`` (no rate limiting).
"""

# vllm, torch and transformers are optional extras and are imported lazily, so
# importing this package does not need them. anthropic is a required
# dependency that is imported lazily too.
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

# Extraction
from .prompting import (
    EXTRACTION_STYLES,
    ConstitutionEntry,
    clean_sample,
    extract_and_clean,
    extract_numbered_list,
    extract_structured_qa,
    get_format_instruction,
    parse_constitution,
)

# Model registry
from .model_config import (
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

# Progress reporting
from .progress import ProgressReporter

# Prompt loading
from .prompting import PromptTemplate, build_messages, load_prompt

# Helpers over a client
from .router import (
    assert_single_sample_per_call,
    batch_generate_samples,
    generate_sample,
)
from .wrappers import BatchCaller, RateLimiter
