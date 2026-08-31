"""Abstract base class for LLM backends.

A backend **is a configured model**, not a bare transport. Everything fixed
about one registered model — its name, its generation defaults, its
system-prompt policy, its rate budget, its provider params — is resolved once
in ``from_config()``/``__init__`` and stored. ``generate()`` then takes only
what genuinely varies per call: the messages, an optional system prompt, and
the two overrides that have real callers (``max_tokens``/``temperature``).

That is what lets :class:`~redact.llms.client.ModelClient` be pure routing.
It receives a finished backend and decides only *how* a batch runs — one
native engine pass, or fan-out through ``BatchCaller`` — reading the class's
:attr:`LLMBackend.compute_config` and the instance's ``rpm``/``max_workers``.
It never looks at the registry.

Construct backends through
:func:`redact.llms.backends.capabilities.backend_for`, which resolves the setup
and calls ``from_config()``. A transport the registry doesn't describe is
registered first (:func:`redact.llms.model_config.register_model`) so it is
validated like every other model; constructing one directly is the escape hatch
below that, and then every field it doesn't pass falls back to this class's
defaults — notably ``rpm=None``, which means no rate limiting at all.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .. import observe


@dataclass(frozen=True)
class ComputeConfig:
    """Static dispatch facts about a backend class.

    Every layer above reads these flags instead of branching on backend type,
    which is what keeps it backend-agnostic. No two backend types share a
    profile, so dispatch can key off the flags alone.

    Attributes:
        supports_native_batching: ``generate()`` is one engine pass over the
            whole batch (vLLM). False means a caller fans the batch out.
        supports_parallel_calls: ``generate()`` is safe to call concurrently.
            False for tight API limits (Anthropic) or GPU contention (local).
        supports_internals: The backend can persist hidden states / attention
            / logprobs, keyed on caller-supplied ``internals_ids``.
    """

    supports_native_batching: bool = False
    supports_parallel_calls: bool = True
    supports_internals: bool = False


# ---------------------------------------------------------------------------
# System-prompt policy — the two halves of supports_system_prompt
# ---------------------------------------------------------------------------


def extract_system_prompt(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split system-role content out of a message list.

    Args:
        messages: Chat messages, possibly containing system-role entries.

    Returns:
        ``(system_prompt, rest)`` — ``system_prompt`` is ``None`` when there
        was none, otherwise every system-role message's content joined with a
        blank line (same convention as :func:`fold_system_into_first_message`).
        ``rest`` is every non-system message, in order.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, rest


def fold_system_into_first_message(messages: list[dict]) -> list[dict]:
    """Fold system-role message content into the first non-system message.

    For models registered with ``ModelConfig.supports_system_prompt=False``,
    which ignore a dedicated system role.

    Args:
        messages: Chat messages, possibly containing system-role entries.

    Returns:
        A message list with system content prepended to the first non-system
        message. If every message is a system message, a single user message.
        Returns ``messages`` unchanged when there's nothing to fold, so it's
        safe to call unconditionally.
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
    """One registered model, on one transport, fully configured."""

    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,
        supports_parallel_calls=True,
        supports_internals=False,
    )
    """This backend's dispatch-relevant capability flags.

    A **class** attribute, not an instance property: these are static facts
    about the backend type, so ``model_config``'s registration-time checker can
    read them off the class without constructing a transport — which for vLLM
    or introspection would mean loading model weights. Override by assignment
    in a subclass; the default is a plain API-style backend (no native
    batching, safe to parallelize, no internals capture).
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
        """Bind the model-level facts that never change between calls.

        Args:
            model: Registry name of the model this backend serves. For a
                multi-model transport it is also the name sent upstream; for a
                single-checkpoint one it is just the row's identity.
            default_max_tokens: Applied when ``generate()`` gets no override.
            default_temperature: Applied when ``generate()`` gets no override.
                ``None`` means "omit, let the provider decide".
            supports_system_prompt: False folds system content into the first
                user message instead of passing it separately.
            rpm: Requests-per-minute budget, or ``None`` for an unmetered
                (local) transport. Read by ``RateLimiter``.
            max_workers: Requested concurrency. **Clamped here** to 1 when this
                class reports ``supports_parallel_calls=False``, so the
                setup's ``recommended_max_workers`` and the transport's own
                parallelism facts are reconciled exactly once, at construction
                — nothing downstream re-decides it.
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
    def from_config(cls, config) -> "LLMBackend":
        """Build a finished backend from a registry entry.

        Each subclass reads *its own* setup off ``config`` (``.api`` /
        ``.vllm`` / ``.introspect``) plus the shared identity fields, so the
        knowledge of what a setup means to a transport lives with that
        transport. Called through
        :func:`redact.llms.backends.capabilities.backend_for`.

        Args:
            config: The model's :class:`~redact.llms.model_config.ModelConfig`.
        """
        raise NotImplementedError(
            f"{cls.__name__} has no from_config(); construct it directly."
        )

    @staticmethod
    def _identity(config) -> dict:
        """The backend-agnostic ``__init__`` kwargs every subclass passes up.

        Args:
            config: The model's ``ModelConfig``.

        Returns:
            Kwargs for :meth:`LLMBackend.__init__` covering the identity
            fields — the per-setup ones (``rpm``, ``max_workers``, engine,
            credentials) are the subclass's own business.
        """
        return {
            "model": config.name,
            "default_max_tokens": config.default_max_tokens,
            "default_temperature": config.default_temperature,
            "supports_system_prompt": config.supports_system_prompt,
        }

    def _prepare(
        self,
        messages_list: list[list[dict]],
        system_prompts: str | list[str | None] | None,
    ) -> tuple[list[str | None], list[list[dict]]]:
        """Apply this model's fixed system-prompt policy to a batch.

        Merges an explicitly-passed system prompt with any system-role message
        already inside ``messages_list`` (explicit first), then either splits
        the result out as a separate prompt or folds it into the first user
        message, according to ``supports_system_prompt``.

        Args:
            messages_list: One chat message list per item.
            system_prompts: ``None``, one string applied to every item, or one
                string (or ``None``) per item.

        Returns:
            ``(system_prompts, messages_list)`` — both aligned with and the
            same length as the input. The first is all-``None`` for a model
            that doesn't support a system role.

        Raises:
            ValueError: If a per-item list's length doesn't match the batch.
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
        """Emit one ``call`` telemetry event for a real transport call.

        Called from inside each backend's ``generate()``, because token counts
        and call latency exist nowhere else — the client above never sees the
        provider's response object, and ``on_complete`` fires per *item*.

        **One call, not one item.** A native-batching pass over N prompts emits
        a single event with ``n_items=N``; a per-item API loop emits one event
        per iteration. That distinction is the whole reason this lives here.

        Args:
            n_items: Batch items covered by this one transport call.
            started: ``time.perf_counter()`` taken immediately before the call.
            in_tok / out_tok: Prompt and completion tokens, or ``None`` when the
                provider didn't report them.
            max_tokens / temperature: Resolved values actually sent.
            error: Exception summary when the call failed, else ``None``.
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
        """Fill per-call overrides in from this model's stored defaults."""
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
        """Generate responses for a batch of message lists.

        Always batch-shaped — a single sample is just a batch of one, there is
        no separate singular method. For backends with no real batching of
        their own (API backends), this loops internally; real concurrency for
        those comes from ``BatchCaller`` calling this multiple times in
        parallel with a batch of one each, not from this method itself. For
        vLLM, this is a genuine single engine pass over however many prompts
        arrive.

        Everything not listed here — the model name, provider params,
        ``top_p`` and other sampling defaults, the system-prompt policy — is
        already bound to this instance and is not a call argument.

        Args:
            messages_list: One chat message list per batch item. System-role
                entries are allowed and handled by :meth:`_prepare`.
            system_prompts: Optional system prompt — one string for the whole
                batch, or one (or ``None``) per item. Merged with any
                system-role message already present.
            max_tokens: Overrides this model's ``default_max_tokens``.
            temperature: Overrides this model's ``default_temperature``.
            internals_ids: One internals-capture id (or ``None``) per batch
                item, aligned with ``messages_list`` — only meaningful when
                ``compute_config.supports_internals`` is True; backends that
                don't support it never receive this (``ModelClient`` and
                ``BatchCaller`` both enforce that before dispatch).
            **kwargs: Backend-specific extras, forwarded to the underlying
                SDK/sampling call.

        Returns:
            Generated text, one per batch item, same order as ``messages_list``.
        """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for the backend type (e.g. 'openai', 'vllm')."""

    @classmethod
    def clear_cache(cls) -> None:
        """Drop this backend class's shared-resource cache.

        Backends per-model are cheap, but what they *hold* may not be — an
        HTTP connection pool, or loaded model weights — so each class caches
        that resource itself, keyed on the resource's own identity. Caches are
        strictly **per class**: a vLLM engine and a ``transformers`` model are
        different runtimes, so the same ``hf_model_id`` means two independent
        loads that must never share an entry.

        A no-op for backends holding nothing shareable. Call
        :func:`redact.llms.backends.capabilities.clear_transport_caches` to
        reset all of them at once.
        """
