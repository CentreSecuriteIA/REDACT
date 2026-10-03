"""Map ``backend_type`` strings to backend classes, and build backends.

:func:`resolve_setup` picks which setup of a registry entry to use and
:func:`backend_for` builds the backend for it. Capability checks read
``compute_config`` from the backend class, so they never construct a backend
(which for a local one would load model weights).
"""

from typing import TYPE_CHECKING

from .anthropic import AnthropicBackend
from .base import ComputeConfig, LLMBackend
from .introspection import TransformersIntrospectionBackend
from .openai import OpenAIBackend
from .vllm import VLLMBackend

if TYPE_CHECKING:
    from ..model_config import ModelConfig

#: ``backend_type`` string -> the class implementing it. Add a new provider
#: here.
BACKEND_TYPES: dict[str, type[LLMBackend]] = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "vllm": VLLMBackend,
    "introspect": TransformersIntrospectionBackend,
}


#: The backend types reached over the network. These are the valid values of
#: ``APIConfig.backend_type``.
API_BACKEND_TYPES = frozenset({"openai", "anthropic"})

#: Setup names a ``ModelConfig`` can default to. Each is also the name of the
#: field holding that setup. "api" covers every API provider.
SETUP_TYPES = frozenset({"api", "vllm", "introspect"})


def compute_config_for(backend_type: str | None) -> ComputeConfig | None:
    """Capability flags of a backend type, without constructing a backend.

    Args:
        backend_type: A key of :data:`BACKEND_TYPES`.

    Returns:
        That backend class's :class:`ComputeConfig`, or ``None`` for an
        unknown or missing type.
    """
    if backend_type is None:
        return None
    backend_cls = BACKEND_TYPES.get(backend_type)
    return backend_cls.compute_config if backend_cls is not None else None


def validate_concurrency(
    name: str, backend_type: str | None, recommended_max_workers: int
) -> None:
    """Reject a model registered with more workers than its backend can use.

    Runs at registration. Without it the mismatch would go unnoticed, because
    the backend clamps ``max_workers`` to 1 at construction.

    Does nothing when ``backend_type`` is ``None`` or unknown, or when
    ``recommended_max_workers <= 1``.

    Args:
        name: Model name, for the error message.
        backend_type: The backend type to check.
        recommended_max_workers: Concurrency the registry entry declares.

    Raises:
        ValueError: The backend type cannot make parallel calls.
    """
    if recommended_max_workers <= 1:
        return
    profile = compute_config_for(backend_type)
    if profile is not None and not profile.supports_parallel_calls:
        raise ValueError(
            f"Model {name!r} (backend_type={backend_type!r}) is registered "
            f"with recommended_max_workers={recommended_max_workers}, but "
            f"this backend type doesn't support parallel calls — "
            f"the backend would silently clamp it to 1. "
            f"Set recommended_max_workers=1."
        )


def resolve_setup(config: "ModelConfig", requested: str | None = None) -> str:
    """Pick which setup of a registry entry to use.

    An explicit request wins, then the entry's ``backend_type``, then the
    only populated setup.

    Args:
        config: The model's ``ModelConfig``.
        requested: Optional override, one of :data:`SETUP_TYPES`.

    Returns:
        ``"api"``, ``"vllm"`` or ``"introspect"``.

    Raises:
        ValueError: The name is not a known setup, the entry lacks that
            setup, it has no setup, or it has several and no default.
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


def transport_for(config: "ModelConfig", setup: str | None = None) -> str:
    """The :data:`BACKEND_TYPES` key of the class that serves this entry.

    For the ``"api"`` setup this is the ``APIConfig``'s own ``backend_type``;
    for a local setup it is the setup name.

    Args:
        config: The model's ``ModelConfig``.
        setup: Optional setup override, resolved by :func:`resolve_setup`.

    Raises:
        ValueError: The setup cannot be resolved, or names no known backend
            type.
    """
    resolved = resolve_setup(config, setup)
    transport = config.api.backend_type if resolved == "api" else resolved
    if transport not in BACKEND_TYPES:
        raise ValueError(
            f"Model {config.name!r}: unknown backend type {transport!r}. "
            f"Expected one of {sorted(BACKEND_TYPES)}."
        )
    return transport


def backend_for(config: "ModelConfig", setup: str | None = None) -> LLMBackend:
    """Build the backend for one model on one of its setups.

    Args:
        config: The model's ``ModelConfig``.
        setup: Which setup to use when the entry has more than one, e.g.
            ``"vllm"``. Defaults to the entry's own.

    Raises:
        ValueError: The setup cannot be resolved, or names no known backend
            type.
        KeyError: The setup's API-key env var is not set.
    """
    return BACKEND_TYPES[transport_for(config, setup)].from_config(config)


def clear_transport_caches() -> None:
    """Clear every backend class's cache of shared resources.

    For tests, after changing environment variables, or to free GPU memory
    between local models. This drops the caches' references only: an engine
    stays loaded while a client or backend still refers to it.
    """
    for backend_cls in BACKEND_TYPES.values():
        backend_cls.clear_cache()
