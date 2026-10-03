"""Base class for LLM backends.

A backend is one configured model. Its name, generation defaults,
system-prompt policy, rate budget and provider parameters are bound at
construction, and ``generate()`` takes only what varies per call.

Build backends with :func:`redact.llms.backends.capabilities.backend_for`. A
backend constructed directly uses this class's defaults for anything not
passed, including ``rpm=None`` (no rate limiting).
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .. import observe

if TYPE_CHECKING:
    from ..model_config import ModelConfig


@dataclass(frozen=True)
class ComputeConfig:
    """Static capability flags of a backend class.

    Callers read these flags instead of branching on the backend type. The
    defaults describe a plain API backend.

    Attributes:
        supports_native_batching: ``generate()`` handles the whole batch in
            one engine pass (vLLM). When False, the caller fans the batch out.
        supports_parallel_calls: ``generate()`` may be called concurrently.
            False for Anthropic (rate limits) and for local backends (GPU
            contention).
        supports_internals: The backend can save hidden states, attention
            and logprobs under caller-supplied ``internals_ids``.
    """

    supports_native_batching: bool = False
    supports_parallel_calls: bool = True
    supports_internals: bool = False


# ---------------------------------------------------------------------------
# System-prompt helpers
# ---------------------------------------------------------------------------


def extract_system_prompt(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split system-role content out of a message list.

    Example: ``[{"role": "system", "content": "A"}, {"role": "user", "content": "B"}]``
    becomes ``("A", [{"role": "user", "content": "B"}])``.

    Args:
        messages: Chat messages, possibly with system-role entries.

    Returns:
        ``(system_prompt, rest)``. ``system_prompt`` is the content of every
        system message joined with a blank line, or ``None`` if there is
        none. ``rest`` is the non-system messages, in order.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, rest


def fold_system_into_first_message(messages: list[dict]) -> list[dict]:
    """Fold system-role content into the first non-system message.

    For models registered with ``supports_system_prompt=False``.

    Example: ``[{"role": "system", "content": "A"}, {"role": "user", "content": "B"}]``
    becomes ``[{"role": "user", "content": "A\\n\\nB"}]``.

    Args:
        messages: Chat messages, possibly with system-role entries.

    Returns:
        The messages with the system content prepended to the first
        non-system message, separated by a blank line. If every message is a
        system message, a single user message. ``messages`` itself when there
        is no system content.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    if not system_parts:
        return messages
    rest = [m for m in messages if m.get("role") != "system"]
    prefix = "\n\n".join(system_parts)
    if not rest:
        return [{"role": "user", "content": prefix}]
    folded_first = dict(rest[0])
    folded_first["content"] = f"{prefix}\n\n{folded_first['content']}"
    return [folded_first] + rest[1:]


class LLMBackend(ABC):
    """One registered model on one backend, fully configured."""

    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,
        supports_parallel_calls=True,
        supports_internals=False,
    )
    """Capability flags of this backend class.

    A class attribute, so the flags can be read without constructing a
    backend, which for a local one would load model weights. Subclasses
    override it by assignment.
    """

    def __init__(
        self,
        model: str,
        *,
        default_max_tokens: int = 2000,
        default_temperature: float | None = None,
        supports_system_prompt: bool = True,
        rpm: int | None = None,
        max_workers: int = 1,
    ):
        """Bind the model-level settings that stay fixed between calls.

        Args:
            model: Registry name of the model. An API backend sends its
                ``api_model_id`` to the provider when that differs.
            default_max_tokens: Used when ``generate()`` gets no override.
            default_temperature: Used when ``generate()`` gets no override.
                ``None`` leaves it to the backend or provider.
            supports_system_prompt: False folds system content into the first
                user message.
            rpm: Requests-per-minute budget, or ``None`` for no rate limiting.
            max_workers: Requested concurrency. Clamped to 1 when the class
                has ``supports_parallel_calls=False``.
        """
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.supports_system_prompt = supports_system_prompt
        self.rpm = rpm
        self.max_workers = (
            max_workers if self.compute_config.supports_parallel_calls else 1
        )

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "LLMBackend":
        """Build a backend from a registry entry.

        Each subclass reads its own setup (``.api``, ``.vllm`` or
        ``.introspect``) and the shared identity fields from ``config``.
        """
        raise NotImplementedError(
            f"{cls.__name__} has no from_config(); construct it directly."
        )

    @staticmethod
    def _identity(config: "ModelConfig") -> dict:
        """``__init__`` kwargs for the identity fields every backend shares."""
        return {
            "model": config.name,
            "default_max_tokens": config.default_max_tokens,
            "default_temperature": config.default_temperature,
            "supports_system_prompt": config.supports_system_prompt,
        }

    @staticmethod
    def _api_key(config: "ModelConfig") -> str:
        """The API key for ``config``'s ``.api`` setup, read from its env var.

        Raises:
            KeyError: The variable is not set, or is empty.
        """
        env = config.api.api_key_env
        key = os.environ.get(env)
        if not key:
            raise KeyError(
                f"Model {config.name!r} needs an API key, but the environment "
                f"variable {env} is not set. Set it in your shell or in a .env file."
            )
        return key

    def _prepare(
        self,
        messages_list: list[list[dict]],
        system_prompts: str | list[str | None] | None,
    ) -> tuple[list[str | None], list[list[dict]]]:
        """Apply the model's system-prompt policy to a batch.

        An explicit system prompt is merged with any system-role message in
        the item (explicit first). The result is returned separately when the
        model supports a system role, and folded into the first user message
        otherwise.

        Returns:
            ``(system_prompts, messages_list)``, both aligned with the input.
            The prompts are all ``None`` for a model without a system role.
        """
        n = len(messages_list)
        if system_prompts is None:
            explicit: list[str | None] = [None] * n
        elif isinstance(system_prompts, str):
            explicit = [system_prompts] * n
        else:
            explicit = list(system_prompts)
            if len(explicit) != n:
                raise ValueError(
                    f"system_prompts must be the same length as messages_list "
                    f"({len(explicit)} != {n})."
                )

        if self.supports_system_prompt:
            prompts: list[str | None] = []
            resolved: list[list[dict]] = []
            for messages, given in zip(messages_list, explicit):
                inline, rest = extract_system_prompt(messages)
                parts = [p for p in (given, inline) if p]
                prompts.append("\n\n".join(parts) if parts else None)
                resolved.append(rest)
            return prompts, resolved

        folded = []
        for messages, given in zip(messages_list, explicit):
            merged = (
                [{"role": "system", "content": given}, *messages] if given else messages
            )
            folded.append(fold_system_into_first_message(merged))
        return [None] * n, folded

    def _record_call(
        self,
        *,
        n_items: int,
        started: float,
        in_tok: int | None = None,
        out_tok: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        error: str | None = None,
    ) -> None:
        """Emit one ``call`` telemetry event for one transport call.

        One event per call, not per item: a native pass over N prompts emits
        a single event with ``n_items=N``, and a per-item loop emits one
        event per iteration. ``started`` is ``time.perf_counter()`` taken
        just before the call.
        """
        event = {
            "ev": "call",
            "model": self.model,
            "backend": self.backend_name,
            "n_items": n_items,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "in_tok": in_tok,
            "out_tok": out_tok,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if error is not None:
            event["error"] = error
        observe.record(event)

    def _resolve(
        self, max_tokens: int | None, temperature: float | None
    ) -> tuple[int, float | None]:
        """Fill unset overrides from the model's defaults."""
        return (
            max_tokens if max_tokens is not None else self.default_max_tokens,
            temperature if temperature is not None else self.default_temperature,
        )

    @abstractmethod
    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        system_prompts: str | list[str | None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate one reply per message list.

        Always batch-shaped: a single sample is a batch of one. vLLM runs one
        engine pass over all items; the other backends loop over them.

        Args:
            messages_list: One chat message list per item.
            system_prompts: One string for the whole batch, or one (or
                ``None``) per item. Merged with any system-role message
                already in the item.
            max_tokens: Overrides the model's ``default_max_tokens``.
            temperature: Overrides the model's ``default_temperature``.
            internals_ids: One capture id (or ``None``) per item. Used only
                by a backend with ``supports_internals``.
            **kwargs: Backend-specific extras for the SDK or sampling call.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """

    def rename_capture(self, old_internals_id: str, new_internals_id: str) -> None:
        """Rename a captured-internals folder once its final id is known.

        Does nothing here. A backend that captures internals overrides it.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for the backend type (e.g. 'openai', 'vllm')."""

    @classmethod
    def clear_cache(cls) -> None:
        """Drop this backend class's cache of shared resources.

        Each class caches its own connection pools or loaded weights. Use
        :func:`redact.llms.backends.capabilities.clear_transport_caches` to
        clear every class at once.
        """
