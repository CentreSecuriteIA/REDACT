"""Which backend type a ``backend_type`` string means, what it can do, and
how to build one.

The one place mapping the registry's ``backend_type`` strings to the classes
that implement them — and, on top of that mapping,
:func:`resolve_setup`/:func:`backend_for`, which turn a registry entry into a
finished backend. There is no separate resolver module: picking the setup is
three lines over facts the entry already holds, and the class that implements
each transport reads its own config in ``from_config()``. Because :attr:`LLMBackend.compute_config` is a *class*
attribute, capabilities can be read straight off the class — no transport is
constructed, which matters because constructing a vLLM or introspection
backend loads model weights.

That is what lets :func:`validate_concurrency` run at ``register_model()``
time: a user registering their own model gets told immediately that, say,
``backend_type="anthropic"`` can't take ``recommended_max_workers=4``, instead
of finding out on the first dispatch of a long run.
"""

from .anthropic import AnthropicBackend
from .base import ComputeConfig, LLMBackend
from .introspection import TransformersIntrospectionBackend
from .openai import OpenAIBackend
from .vllm import VLLMBackend

#: ``backend_type`` string -> the class implementing it. Adding a provider
#: means adding it here; everything that dispatches on backend type reads
#: capabilities through this mapping rather than hardcoding a second copy.
BACKEND_TYPES: dict[str, type[LLMBackend]] = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "vllm": VLLMBackend,
    "introspect": TransformersIntrospectionBackend,
}


#: The subset of :data:`BACKEND_TYPES` that reaches a model over the network.
#: These are the only types an ``APIConfig``'s rpm/worker budget describes —
#: a local transport is never rate-limited. Kept beside the mapping it filters
#: so adding a provider is one file, not a hunt for parallel copies.
API_BACKEND_TYPES = frozenset({"openai", "anthropic"})

#: The *setup* names a ``ModelConfig`` can prefer, each matching the field
#: holding it (``.api`` / ``.vllm`` / ``.introspect``). Distinct from
#: :data:`BACKEND_TYPES`, which names *transports*: "api" covers both API
#: providers, and which one is in play is the ``APIConfig``'s own business.
SETUP_TYPES = frozenset({"api", "vllm", "introspect"})


def compute_config_for(backend_type: str | None) -> ComputeConfig | None:
    """Capabilities of a backend type, without constructing one.

    Args:
        backend_type: A key of :data:`BACKEND_TYPES`.

    Returns:
        That backend class's :class:`ComputeConfig`, or ``None`` for an
        unknown or absent type — callers decide whether that's an error, since
        a registry entry may legitimately leave ``backend_type`` unset and have
        it inferred later from the model name.
    """
    if backend_type is None:
        return None
    backend_cls = BACKEND_TYPES.get(backend_type)
    return backend_cls.compute_config if backend_cls is not None else None


def validate_concurrency(
    name: str, backend_type: str | None, recommended_max_workers: int
) -> None:
    """Reject a model registered with more workers than its backend can take.

    ``BatchCaller.run()`` raises the same class of error, but only once the
    model is actually dispatched. This catches it at registration — which is
    the useful moment for someone registering their own model, since the
    mistake is in the call they just wrote.

    A no-op when there's nothing to check: ``backend_type=None`` (inferred
    later from the model name, so no class to ask yet), an unrecognized type,
    or ``recommended_max_workers <= 1``.

    Args:
        name: Model name, for the error message.
        backend_type: The type whose capabilities to check.
        recommended_max_workers: Concurrency the registry entry declares.

    Raises:
        ValueError: If that backend type can't make parallel calls.
    """
    if recommended_max_workers <= 1:
        return
    profile = compute_config_for(backend_type)
    if profile is not None and not profile.supports_parallel_calls:
        raise ValueError(
            f"Model {name!r} (backend_type={backend_type!r}) is registered "
            f"with recommended_max_workers={recommended_max_workers}, but "
            f"this backend type doesn't support parallel calls — "
            f"BatchCaller would raise the same error on first dispatch. "
            f"Set recommended_max_workers=1."
        )


def resolve_setup(config, requested: str | None = None) -> str:
    """Decide which setup on a registry entry to bind.

    Explicit request wins, then the entry's own ``backend_type``, then — when
    exactly one setup is populated — that one, since there is nothing to
    choose between. More than one with no declared preference is genuinely
    ambiguous, so the entry has to say.

    Args:
        config: The model's ``ModelConfig``.
        requested: Optional override, one of :data:`SETUP_TYPES`.

    Returns:
        One of ``"api"`` / ``"vllm"`` / ``"introspect"``.

    Raises:
        ValueError: If the name isn't a known setup, the entry has no such
            setup, it has none at all, or it has several and no preference.
    """
    setup = requested or config.backend_type
    if setup is None:
        present = [s for s in sorted(SETUP_TYPES) if getattr(config, s) is not None]
        if not present:
            raise ValueError(
                f"Model {config.name!r} has no setup at all — register it with "
                f"api=/vllm=/introspect=."
            )
        if len(present) > 1:
            raise ValueError(
                f"Model {config.name!r} has setups {present} but no backend_type "
                f"saying which to prefer. Set backend_type= on the entry, or pass "
                f"it at call time."
            )
        return present[0]

    if setup not in SETUP_TYPES:
        raise ValueError(
            f"Model {config.name!r}: unknown backend_type {setup!r}. "
            f"Expected one of {sorted(SETUP_TYPES)}."
        )
    if getattr(config, setup) is None:
        raise ValueError(
            f"Model {config.name!r} has no {setup} setup. Register it with "
            f"register_model(..., {setup}=...)."
        )
    return setup


def transport_for(config, setup: str | None = None) -> str:
    """Which transport class serves this entry, as a :data:`BACKEND_TYPES` key.

    Distinct from the *setup* name: ``"api"`` covers both API providers, and
    which one is in play is the ``APIConfig``'s own ``backend_type``. Kept as
    its own function because it is answerable without building anything — see
    :func:`redact.llms.model_config.model_compute_config`, which uses it to
    read capabilities off the class rather than constructing a transport.

    Args:
        config: The model's ``ModelConfig``.
        setup: Optional setup override; resolved via :func:`resolve_setup`.

    Returns:
        A key of :data:`BACKEND_TYPES`.

    Raises:
        ValueError: If the setup can't be resolved, or names no known transport.
    """
    resolved = resolve_setup(config, setup)
    transport = config.api.backend_type if resolved == "api" else resolved
    if transport not in BACKEND_TYPES:
        raise ValueError(
            f"Model {config.name!r}: unknown backend type {transport!r}. "
            f"Expected one of {sorted(BACKEND_TYPES)}."
        )
    return transport


def backend_for(config, setup: str | None = None) -> LLMBackend:
    """Build the finished backend for one model on one of its setups.

    The whole of model→transport resolution: pick the setup, map it to the
    class that implements it, and let that class read its own config off the
    entry. Which API *provider* an ``.api`` setup uses is that config's own
    ``backend_type``, so endpoint identity stays out of the entry-level
    selector.

    Args:
        config: The model's ``ModelConfig``.
        setup: Optional override selecting which setup to bind on an entry
            with more than one, e.g. ``"vllm"`` to run locally a model that
            defaults to its hosted endpoint.

    Returns:
        A fully-configured :class:`LLMBackend` — model name, generation
        defaults, provider params, rpm and worker budget all bound.

    Raises:
        ValueError: If the setup can't be resolved or names no known transport.
        KeyError: If the setup's API-key env var is not set.
    """
    return BACKEND_TYPES[transport_for(config, setup)].from_config(config)


def clear_transport_caches() -> None:
    """Reset every backend class's shared-resource cache.

    For tests, or after changing environment variables. Each class clears its
    own — see :meth:`LLMBackend.clear_cache`.
    """
    for backend_cls in BACKEND_TYPES.values():
        backend_cls.clear_cache()
