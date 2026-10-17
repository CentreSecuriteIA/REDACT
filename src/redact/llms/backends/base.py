"""Base class for LLM backends.

A backend is one configured model. Its name, generation defaults,
system-prompt policy, rate budget and provider parameters are bound at
construction, and ``generate()`` takes only what varies per call.

Build backends with :func:`redact.llms.backends.capabilities.backend_for`. A
backend constructed directly uses this class's defaults for anything not
passed, including ``rpm=None`` (no rate limiting; enforced in ``ModelClient``).
"""

import gc
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from .. import observe

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

# (stage, model) of each model whose first cut reply in a stage was warned about.
_cut_warned: set[tuple[str | None, str]] = set()
_cut_lock = threading.Lock()


@dataclass(frozen=True)
class ComputeConfig:
    """Static capability flags of a backend class.

    Callers read these flags instead of branching on the backend type. The
    defaults describe a plain API backend.

    Attributes:
        supports_native_batching: ``generate()`` takes the whole batch and
            has its items in flight together (vLLM). When False, the caller
            fans the batch out.
        supports_parallel_calls: ``generate()`` may be called concurrently.
            False for Anthropic (rate limits) and for the introspection
            backend (GPU contention).
        supports_internals: The backend can save hidden states, attention
            and logprobs under caller-supplied ``internals_ids``.
        supports_async_calls: The backend has ``agenerate()``, a coroutine
            for one item that runs on the shared loop (``llms/aio.py``).
    """

    supports_native_batching: bool = False
    supports_parallel_calls: bool = True
    supports_internals: bool = False
    supports_async_calls: bool = False


# ---------------------------------------------------------------------------
# System-prompt helpers
# ---------------------------------------------------------------------------
# These live here, not in prompting/, because split-vs-fold depends on the model.


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
    """Fold system-role content into the first user message.

    For models registered with ``supports_system_prompt=False``.

    Example: ``[{"role": "system", "content": "A"}, {"role": "user", "content": "B"}]``
    becomes ``[{"role": "user", "content": "A\\n\\nB"}]``.

    Args:
        messages: Chat messages, possibly with system-role entries.

    Returns:
        The non-system messages, with the system content prepended to the
        first user message (the first message when none is the user's),
        separated by a blank line. Empty system messages are dropped. If
        every message is a system message, a single user message.
        ``messages`` itself when there is no system message.
    """
    if not any(m.get("role") == "system" for m in messages):
        return messages
    rest = [m for m in messages if m.get("role") != "system"]
    system_parts = [m["content"] for m in messages
                    if m.get("role") == "system" and m["content"]]
    if not system_parts:
        return rest
    prefix = "\n\n".join(system_parts)
    if not rest:
        return [{"role": "user", "content": prefix}]
    target = next((i for i, m in enumerate(rest) if m.get("role") == "user"), 0)
    folded = dict(rest[target])
    folded["content"] = f"{prefix}\n\n{folded['content']}"
    return [*rest[:target], folded, *rest[target + 1:]]


def free_memory() -> None:
    """Collect garbage, then empty torch's CUDA cache. Never raises.

    A local backend calls this once it has dropped a released model, so the
    memory is back before the next load.
    """
    gc.collect()
    # Looked up, not imported: a process that never loaded torch has no cache.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        # is_initialized(), not is_available(), which starts CUDA in this process.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 — cleanup must not fail a release
        logger.debug("empty_cache failed (%s: %s)", type(exc).__name__, exc)


class LLMBackend(ABC):
    """One registered model on one backend, fully configured."""

    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,
        supports_parallel_calls=True,
        supports_internals=False,
        supports_async_calls=False,
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
        self, messages_list: list[list[dict]]
    ) -> tuple[list[str | None], list[list[dict]]]:
        """Apply the model's system-prompt policy to a batch.

        The system-role messages of an item are returned separately when the
        model supports a system role, and folded into the first user message
        otherwise.

        Returns:
            ``(system_prompts, messages_list)``, both aligned with the input.
            The prompts are all ``None`` for a model without a system role.
        """
        prompts: list[str | None] = []
        resolved: list[list[dict]] = []
        for messages in messages_list:
            if self.supports_system_prompt:
                prompt, messages = extract_system_prompt(messages)
            else:
                prompt, messages = None, fold_system_into_first_message(messages)
            # An empty system prompt counts as none.
            prompts.append(prompt or None)
            resolved.append(messages)
        return prompts, resolved

    def _record_call(
        self,
        *,
        n_items: int,
        started: float,
        in_tok: int | None = None,
        out_tok: int | None = None,
        out_tok_max: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        error: str | None = None,
        finish_reason: str | None = None,
        cut: bool = False,
    ) -> None:
        """Emit one ``call`` telemetry event for one transport call.

        One event per call, with ``n_items`` the items it carried. Every
        backend makes one call per item: a per-item loop emits one event per
        iteration, and the vLLM engine takes each item as its own request.
        ``started`` is ``time.perf_counter()`` taken just before the call.
        ``out_tok_max`` is the longest reply of the call, in tokens:
        ``out_tok`` itself for a one-item call. ``finish_reason`` is the
        provider's reason the reply ended, when it reports one. ``cut`` says
        the reply stopped at the token limit, as each backend reads its
        provider's reason: the event then carries ``cut``, which telemetry
        counts per stage, and the model's first cut reply in a stage warns.
        A cut reply is returned as it is.
        """
        event = {
            "ev": "call",
            "model": self.model,
            "backend": self.backend_name,
            "n_items": n_items,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "in_tok": in_tok,
            "out_tok": out_tok,
            "out_tok_max": out_tok_max,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if error is not None:
            event["error"] = error
        if finish_reason is not None:
            event["finish_reason"] = finish_reason
        if cut:
            event["cut"] = True
        observe.record(event)
        if cut:
            key = (observe.current_stage(), self.model)
            with _cut_lock:
                first = key not in _cut_warned
                _cut_warned.add(key)
            if first:
                logger.warning(
                    "%s: a reply was cut at the token limit (max_tokens=%s) and "
                    "is incomplete; raise max_tokens or the context length "
                    "(max_model_len), or ask for shorter replies. Later ones are "
                    "only counted: see the stage's end and the telemetry summary.",
                    self.model, max_tokens,
                )

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
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate one reply per message list.

        Always batch-shaped: a single sample is a batch of one. vLLM has all
        items in flight together; the other backends loop over them.

        Args:
            messages_list: One chat message list per item.
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
