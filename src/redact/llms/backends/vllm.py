"""Local vLLM backend.

The first backend built for a checkpoint loads its weights onto the GPU. The
engine is cached per checkpoint (``hf_model_id``, quantization and engine
kwargs), so two registry entries that agree on all three share one load.
``generate()`` runs the whole batch in a single engine pass, and prompts go
through vLLM's chat template (``llm.chat()``).
"""

import asyncio
import atexit
import contextlib
import itertools
import logging
import os
import threading
import time
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, ClassVar

from .. import aio, observe
from ..resources import estimate
from .base import ComputeConfig, LLMBackend, free_memory

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

# Backend defaults for sampling values that have no ModelConfig field.
# Override them per model with VLLMConfig.sampling.
_DEFAULT_SAMPLING = {"top_p": 0.85}
_DEFAULT_TEMPERATURE = 0.7

# Loaded engines, keyed by _engine_key(). This holds the only reference to
# each: a backend keeps the key, so a released engine can be collected.
_engines: dict[tuple, object] = {}
# Held during a load, so two threads cannot load one checkpoint twice.
_engines_lock = threading.Lock()
# When each engine's load started. Lifetime is timed per checkpoint, not per
# backend instance: the GPU is held for as long as the engine exists.
_engine_loaded_at: dict[tuple, float] = {}
# gpu_memory_utilization per engine key, set by the residency plan. Applied
# at load and kept out of the key, so it never splits a shared engine.
_planned_utilization: dict[tuple, float] = {}
# Set once the exit hook is registered, at the start of the first load (see _engine).
_exit_hook_registered = False
# Numbers the requests: an engine needs each id unique in the process.
_request_ids = itertools.count()


def set_planned_utilization(planned: dict[tuple, float], replace: bool = True) -> None:
    """Set the ``gpu_memory_utilization`` each engine key loads with.

    ``replace=False`` keeps the values already set.
    """
    global _planned_utilization
    # One assignment, so a load never reads a half-filled plan.
    _planned_utilization = {**planned, **({} if replace else _planned_utilization)}


def _engine_key(hf_model_id: str, quantization: str | None, vllm_kwargs: dict) -> tuple:
    return (hf_model_id, quantization, repr(sorted(vllm_kwargs.items())))


def _prepare_environment() -> None:
    """Set up the process environment before vLLM is constructed."""
    # CUDA fails in a forked child when the parent already initialised it.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # WSL2 workaround: vLLM's V2 model runner fails every load there with
    # "UVA is not available" (vllm-project/vllm#47387, fix proposed in
    # #47579), so default to the V1 runner. An explicit
    # VLLM_USE_V2_MODEL_RUNNER still wins. Remove once the fix ships.
    with contextlib.suppress(OSError), open("/proc/version") as f:
        if "microsoft" in f.read().lower():
            os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


def _engine(hf_model_id: str, quantization: str | None, vllm_kwargs: dict):
    """Get or load the vLLM engine for one checkpoint."""
    global _exit_hook_registered
    key = _engine_key(hf_model_id, quantization, vllm_kwargs)
    # One read: clear_cache() may drop the key between a test and a lookup.
    engine = _engines.get(key)
    if engine is not None:
        return engine

    with _engines_lock:
        # Re-check: another thread may have finished the load while we waited.
        if key in _engines:
            return _engines[key]

        _prepare_environment()
        try:
            # lazy: vllm is heavy and optional
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.usage.usage_lib import UsageContext
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:
            raise ImportError(
                "The 'vllm' package is required for VLLMBackend. "
                "Install it with: pip install redact[vllm]"
            )

        if not _exit_hook_registered:
            # Before the load, under the lock: an exit during a load then waits
            # on the lock until the load is done and shuts the engine down.
            # atexit runs hooks last-registered first, and multiprocessing's,
            # registered when multiprocessing.util is first imported, joins
            # the engine's child process and never returns while the engine
            # runs. Import it here so ours is registered after it: left to
            # vllm, it may first be imported while the engine is built, after
            # this point.
            import multiprocessing.util  # noqa: F401
            atexit.register(VLLMBackend.clear_cache)
            _exit_hook_registered = True

        # No download_dir is passed: the hub cache is $HF_HOME/hub, so HF_HOME would download again.
        kwargs = dict(vllm_kwargs)
        # The plan's share, else the planner's whole-card one: vLLM's own
        # default is more than the plan assumes.
        kwargs.setdefault("gpu_memory_utilization", _planned_utilization.get(
            key, estimate.DEFAULT_GPU_MEMORY_UTILIZATION))
        # After the key is built, so the default never splits a shared engine.
        if not kwargs.get("max_model_len"):
            kwargs["max_model_len"] = estimate.effective_max_model_len(hf_model_id)
        # vllm.LLM's default. Left unset, the engine logs a stats line every
        # 10 seconds.
        kwargs.setdefault("disable_log_stats", True)

        started = time.perf_counter()
        # Not measured: vLLM preallocates its memory grant in a separate
        # process, so a measurement would restate the setting.
        _engines[key] = AsyncLLM.from_engine_args(
            AsyncEngineArgs(model=hf_model_id, quantization=quantization, **kwargs),
            # vllm.LLM's context: vLLM's defaults for running sequences and
            # batched tokens depend on it, and the engine's own has lower ones.
            usage_context=UsageContext.LLM_CLASS,
        )
        # Held from the start of the load; set after it, so a failed load
        # leaves no timer.
        _engine_loaded_at[key] = started

        observe.record({
            "ev": "local",
            "phase": "load",
            "backend": "vllm",
            "hf_model_id": hf_model_id,
            "quantization": quantization,
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization"),
            "load_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return _engines[key]


def _shut_down(engine) -> None:
    """Call the engine's ``shutdown()``.

    On the caller's thread, not the shared loop: the call is synchronous and
    safe from any thread. Never raises. An engine without the hook gives its
    memory back only when the object is collected.
    """
    hook = getattr(engine, "shutdown", None)
    if not callable(hook):
        logger.warning(
            "[vllm] %s has no shutdown hook; left to garbage collection",
            type(engine).__name__,
        )
        return
    try:
        hook()
    except Exception as exc:  # noqa: BLE001 — a release must not fail
        logger.debug("[vllm] shutdown() failed (%s: %s)", type(exc).__name__, exc)
        return
    logger.debug("[vllm] engine shut down")


def _release_engines(keep: Collection[tuple] = ()) -> None:
    """Emit a lifetime event per loaded engine, then forget the timers.

    Called from :meth:`VLLMBackend.clear_cache`, at exit too, so a process
    that just exits still reports how long each engine was held. Engines
    whose key is in ``keep`` stay timed.
    """
    now = time.perf_counter()
    for key, loaded_at in list(_engine_loaded_at.items()):
        if key in keep:
            continue
        del _engine_loaded_at[key]
        observe.record({
            "ev": "local",
            "phase": "release",
            "backend": "vllm",
            "hf_model_id": key[0],
            "quantization": key[1],
            "held_s": round(now - loaded_at, 1),
        })


def held_seconds() -> dict[str, float]:
    """Seconds each loaded engine has been held so far, per ``hf_model_id``."""
    now = time.perf_counter()
    held: dict[str, float] = {}
    for key, loaded_at in list(_engine_loaded_at.items()):
        held[key[0]] = held.get(key[0], 0.0) + now - loaded_at
    return held


class VLLMBackend(LLMBackend):
    """One model served by a locally-loaded vLLM engine."""

    # generate() takes the whole list, so the client hands it over undivided.
    # Every item is its own request to one engine on one loop, which makes
    # concurrent calls safe: the engine batches whatever is in flight.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=True,
        supports_parallel_calls=True,
        supports_internals=False,
        supports_async_calls=True,
    )

    def __init__(
        self,
        model: str,
        hf_model_id: str,
        *,
        quantization: str | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
        **identity,
    ):
        """Bind one registered model to a vLLM engine, loading it if needed.

        Args:
            model: Registry name of the model.
            hf_model_id: HuggingFace model ID or local path to load.
            quantization: Quantization method (e.g. "gptq", "awq").
            vllm_kwargs: Passed to ``vllm.LLM()`` (e.g. ``max_model_len``,
                ``tokenizer_mode``).
            sampling: ``SamplingParams`` defaults for this model (``top_p``,
                ``top_k``, ...), merged over :data:`_DEFAULT_SAMPLING`.
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
                ``max_workers`` defaults to the number of sequences the
                engine runs at once.
        """
        engine = _engine(hf_model_id, quantization, vllm_kwargs or {})
        # What the async lane keeps in flight for this model. The engine
        # queues anything above its own limit, so more would only wait there.
        identity.setdefault("max_workers", engine.scheduler_config.max_num_seqs)
        super().__init__(model, **identity)
        # SamplingParams needs a number, so fill in a default when the model
        # has none.
        if self.default_temperature is None:
            self.default_temperature = _DEFAULT_TEMPERATURE
        self.hf_model_id = hf_model_id
        self._sampling = {**_DEFAULT_SAMPLING, **(sampling or {})}
        self._key = _engine_key(hf_model_id, quantization, vllm_kwargs or {})

    @property
    def _llm(self):
        """This model's engine, read from the cache on each use.

        Raises:
            RuntimeError: The engine was released and not loaded again.
        """
        engine = _engines.get(self._key)
        if engine is None:
            raise RuntimeError(
                f"The vLLM engine of {self.model!r} ({self.hf_model_id}) was "
                f"released. Build a new client with ModelClient.create()."
            )
        return engine

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "VLLMBackend":
        """Build from a registry entry's ``.vllm`` setup.

        Raises:
            ValueError: The entry has no ``.vllm`` setup.
        """
        vllm = config.vllm
        if vllm is None:
            raise ValueError(
                f"Model {config.name!r} has no vllm setup. Register it with "
                f"register_model(..., vllm=VLLMConfig(hf_model_id='...'))."
            )
        return cls(
            hf_model_id=vllm.hf_model_id,
            quantization=vllm.quantization,
            vllm_kwargs=vllm.engine_kwargs,
            sampling=vllm.sampling,
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls, keep: Collection[tuple] = ()) -> None:
        """Release the loaded engines and free their memory.

        Each is shut down and its hold time reported. Its planned utilization
        stays, so a reload gets the same share. A backend built on a released
        engine raises on ``generate()`` until the checkpoint is loaded again.
        A call in flight on a released engine is not waited for.
        Also the exit hook, registered when the first load starts: an exit
        during a load waits for it here (the load holds the lock), and an
        engine's child process would otherwise keep the interpreter from exiting.

        Args:
            keep: Engine keys (:func:`_engine_key`) to leave loaded and timed.
                A collection of keys, not one key.
        """
        if any(not isinstance(key, tuple) for key in keep):
            raise TypeError(f"keep takes a collection of engine keys, got {keep!r}.")
        # Locked: a preload may be inserting an engine, and the next load
        # must wait until this memory is back.
        with _engines_lock:
            released = [k for k in _engines if k not in keep]
            for key in released:
                _shut_down(_engines.pop(key))
            if released:
                free_memory()
            # After the teardown, so held_s covers it.
            _release_engines(keep)

    def _clean_output(self, text: str) -> str:
        """Strip whitespace and a leading ``|>`` tokenizer artifact."""
        text = text.strip()
        if text.startswith("|>"):
            text = text[2:].lstrip()
        return text

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate replies for the whole batch, one engine request per item.

        Blocks until every reply is in. The requests run on the shared loop
        (``llms/aio.py``), and the engine batches them with whatever else it
        has in flight. When one fails, the others are cancelled.

        Args:
            messages_list: One chat message list per item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: Ignored. This backend does not capture internals.
            **kwargs: ``SamplingParams`` overrides for this call (e.g.
                ``top_k``, ``presence_penalty``), merged over the model's
                sampling defaults.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """
        if not messages_list:
            return []
        max_tok, temp = self._resolve(max_tokens, temperature)
        replies = aio.run(self._many(messages_list, max_tok, temp, kwargs))
        self._warn_cut(
            sum(reason == "length" for _, reason in replies),
            len(messages_list), max_tok,
        )
        return [text for text, _ in replies]

    async def agenerate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_id: str | None = None,
        **kwargs,
    ) -> str:
        """Generate one reply with one engine request, without blocking the loop.

        Runs on the shared loop (``llms/aio.py``).

        Args:
            messages: Chat messages of the one item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_id: Ignored. This backend does not capture internals.
            **kwargs: ``SamplingParams`` overrides for this call, merged over
                the model's sampling defaults.

        Returns:
            The generated text.
        """
        max_tok, temp = self._resolve(max_tokens, temperature)
        text, reason = await self._one(messages, max_tok, temp, kwargs)
        self._warn_cut(int(reason == "length"), 1, max_tok)
        return text

    async def _many(
        self, messages_list: list[list[dict]], max_tok: int, temp: float, extra: dict
    ) -> list[tuple[str, str | None]]:
        """Send each item as its own request and await them all.

        Returns:
            One :meth:`_one` result per item, in the order of ``messages_list``.
        """
        tasks = [asyncio.ensure_future(self._one(messages, max_tok, temp, extra))
                 for messages in messages_list]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # A failed batch must not keep the engine working: a cancelled
            # request is aborted there.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _one(
        self, messages: list[dict], max_tok: int, temp: float, extra: dict
    ) -> tuple[str, str | None]:
        """Send one request to the engine and await its reply.

        Runs on the shared loop, as every await on an engine must: an engine
        hands its outputs to the loop of its first request.

        Returns:
            ``(text, finish_reason)``. vLLM says ``"length"`` for both caps,
            ``max_tokens`` and ``max_model_len``.

        Raises:
            RuntimeError: The engine was released, or sent no output.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        engine = self._llm  # raises before any work if the engine was released
        (system_prompt,), (chat,) = self._prepare([messages])
        if system_prompt:
            chat = [{"role": "system", "content": system_prompt}, *chat]
        # The engine takes token ids, so the chat template is applied here.
        ids = engine.get_tokenizer().apply_chat_template(
            chat, add_generation_prompt=True, tokenize=True)
        # Some tokenizers return the ids, some a mapping that holds them.
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        params = SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            # One output per request, once the reply is complete.
            output_kind=RequestOutputKind.FINAL_ONLY,
            **{**self._sampling, **extra},
        )
        request_id = f"redact-{next(_request_ids)}"

        # One telemetry event per request: the engine batches continuously,
        # so the request is the transport call.
        started = time.perf_counter()
        try:
            final = None
            async for output in engine.generate(
                    {"prompt_token_ids": ids}, params, request_id):
                final = output
            if final is None:
                raise RuntimeError(
                    f"The vLLM engine of {self.model!r} sent no output for "
                    f"request {request_id}."
                )
        except Exception as exc:
            self._record_call(
                n_items=1, started=started, max_tokens=max_tok,
                temperature=temp, error=f"{type(exc).__name__}: {exc}",
            )
            raise
        reply = final.outputs[0]
        out_tok = len(reply.token_ids)
        self._record_call(
            n_items=1, started=started,
            in_tok=len(final.prompt_token_ids), out_tok=out_tok, out_tok_max=out_tok,
            max_tokens=max_tok, temperature=temp, finish_reason=reply.finish_reason,
        )
        return self._clean_output(reply.text), reply.finish_reason

    @property
    def backend_name(self) -> str:
        return "vllm"

    def __repr__(self) -> str:
        return f"VLLMBackend(model={self.model!r}, hf_model_id={self.hf_model_id!r})"
