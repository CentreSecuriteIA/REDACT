"""Local vLLM backend.

The first backend built for a checkpoint loads its weights onto the GPU. The
engine is cached per checkpoint (``hf_model_id``, quantization and engine
kwargs), so two registry entries that agree on all three share one load.
``generate()`` runs the whole batch in a single engine pass, and prompts go
through vLLM's chat template (``llm.chat()``).
"""

import atexit
import contextlib
import logging
import os
import threading
import time
from collections.abc import Collection
from typing import TYPE_CHECKING, ClassVar

from .. import observe
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
# Load time of each engine. Engine lifetime is timed per checkpoint, not per
# backend instance: the GPU is held for as long as the engine exists.
_engine_loaded_at: dict[tuple, float] = {}
# gpu_memory_utilization per engine key, set by the residency plan. Applied
# at load and kept out of the key, so it never splits a shared engine.
_planned_utilization: dict[tuple, float] = {}


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
            from vllm import LLM  # lazy: vllm is heavy and optional
        except ImportError:
            raise ImportError(
                "The 'vllm' package is required for VLLMBackend. "
                "Install it with: pip install redact[vllm]"
            )

        # No download_dir is passed: the hub cache is $HF_HOME/hub, so HF_HOME would download again.
        kwargs = dict(vllm_kwargs)
        planned = _planned_utilization.get(key)
        if planned is not None:
            kwargs.setdefault("gpu_memory_utilization", planned)
        # After the key is built, so the default never splits a shared engine.
        if not kwargs.get("max_model_len"):
            kwargs["max_model_len"] = estimate.effective_max_model_len(hf_model_id)

        started = time.perf_counter()
        # Not measured: vLLM preallocates its memory grant in a separate
        # process, so a measurement would restate the setting.
        _engines[key] = LLM(model=hf_model_id, quantization=quantization, **kwargs)
        _engine_loaded_at[key] = time.perf_counter()

        observe.record({
            "ev": "local",
            "phase": "load",
            "backend": "vllm",
            "hf_model_id": hf_model_id,
            "quantization": quantization,
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization"),
            "load_ms": round((_engine_loaded_at[key] - started) * 1000, 1),
        })
        return _engines[key]


# Where a vllm.LLM keeps its shutdown hook, outermost first. The place
# differs between vLLM versions, and some have none.
_SHUTDOWN_HOOKS = (
    "shutdown",
    "llm_engine.shutdown",
    "llm_engine.engine_core.shutdown",
    "llm_engine.model_executor.shutdown",
)


def _shut_down(engine) -> None:
    """Call the engine's shutdown hook, if its vLLM version has one.

    Never raises. Without a hook the engine's memory comes back only when
    the object is collected.
    """
    for path in _SHUTDOWN_HOOKS:
        try:
            hook = engine
            for name in path.split("."):
                hook = getattr(hook, name, None)
            if not callable(hook):
                continue
            hook()
        except Exception as exc:  # noqa: BLE001 — try the next, inner hook
            logger.debug("[vllm] %s() failed (%s: %s)", path, type(exc).__name__, exc)
            continue
        logger.debug("[vllm] engine shut down through %s()", path)
        return
    logger.debug(
        "[vllm] %s has no shutdown hook (tried %s); left to garbage collection",
        type(engine).__name__, ", ".join(_SHUTDOWN_HOOKS),
    )


def _release_engines(keep: Collection[tuple] = ()) -> None:
    """Emit a lifetime event per loaded engine, then forget the timers.

    Called from :meth:`VLLMBackend.clear_cache` and at exit, so a process
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


# At exit only the hold time is recorded: a teardown could block on the lock
# a preload thread holds, and process exit returns the card anyway.
atexit.register(_release_engines)


def held_seconds() -> dict[str, float]:
    """Seconds each loaded engine has been held so far, per ``hf_model_id``."""
    now = time.perf_counter()
    held: dict[str, float] = {}
    for key, loaded_at in list(_engine_loaded_at.items()):
        held[key[0]] = held.get(key[0], 0.0) + now - loaded_at
    return held


class VLLMBackend(LLMBackend):
    """One model served by a locally-loaded vLLM engine."""

    # generate() runs one engine pass over the whole list. Concurrent calls
    # would compete for GPU memory, so parallel calls are off.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=True,
        supports_parallel_calls=False,
        supports_internals=False,
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
        """
        super().__init__(model, **identity)
        # SamplingParams needs a number, so fill in a default when the model
        # has none.
        if self.default_temperature is None:
            self.default_temperature = _DEFAULT_TEMPERATURE
        self.hf_model_id = hf_model_id
        self._sampling = {**_DEFAULT_SAMPLING, **(sampling or {})}
        self._key = _engine_key(hf_model_id, quantization, vllm_kwargs or {})
        _engine(hf_model_id, quantization, vllm_kwargs or {})

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
        """Generate replies for the whole batch in a single vLLM pass.

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
        llm = self._llm  # raises before any work if the engine was released
        prompts, resolved = self._prepare(messages_list)
        max_tok, temp = self._resolve(max_tokens, temperature)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            **{**self._sampling, **kwargs},
        )

        chat_inputs = [
            [{"role": "system", "content": sp}, *m] if sp else m
            for m, sp in zip(resolved, prompts)
        ]
        # One telemetry event for the whole pass, with n_items = batch size.
        started = time.perf_counter()
        try:
            outputs = llm.chat(chat_inputs, sampling_params)
        except Exception as exc:
            self._record_call(
                n_items=len(messages_list), started=started, max_tokens=max_tok,
                temperature=temp, error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record_call(
            n_items=len(messages_list), started=started,
            in_tok=sum(len(getattr(o, "prompt_token_ids", None) or ()) for o in outputs),
            out_tok=sum(len(getattr(o.outputs[0], "token_ids", None) or ()) for o in outputs),
            max_tokens=max_tok, temperature=temp,
        )
        return [self._clean_output(out.outputs[0].text) for out in outputs]

    @property
    def backend_name(self) -> str:
        return "vllm"

    def __repr__(self) -> str:
        return f"VLLMBackend(model={self.model!r}, hf_model_id={self.hf_model_id!r})"
