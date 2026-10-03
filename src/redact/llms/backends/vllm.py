"""Local vLLM backend.

The first backend built for a checkpoint loads its weights onto the GPU. The
engine is cached per checkpoint (``hf_model_id``, quantization and engine
kwargs), so two registry entries naming the same checkpoint share one load.
``generate()`` runs the whole batch in a single engine pass, and prompts go
through vLLM's chat template (``llm.chat()``).
"""

import atexit
import contextlib
import multiprocessing
import os
import threading
import time
from typing import TYPE_CHECKING, ClassVar

from .. import observe
from ..resources import measure
from .base import ComputeConfig, LLMBackend

if TYPE_CHECKING:
    from ..model_config import ModelConfig

# Backend defaults for sampling values that have no ModelConfig field.
# Override them per model with VLLMConfig.sampling.
_DEFAULT_SAMPLING = {"top_p": 0.85}
_DEFAULT_TEMPERATURE = 0.7

# Loaded engines, keyed by _engine_key().
_engines: dict[tuple, object] = {}
# Held during a load, so a preload thread and a real call cannot load the same
# checkpoint twice.
_engines_lock = threading.Lock()
# Load time of each engine. Engine lifetime is timed per checkpoint, not per
# backend instance: the GPU is held for as long as the engine exists.
_engine_loaded_at: dict[tuple, float] = {}


def _engine_key(hf_model_id: str, quantization: str | None, vllm_kwargs: dict) -> tuple:
    return (hf_model_id, quantization, repr(sorted(vllm_kwargs.items())))


def _prepare_environment() -> None:
    """Set up the process environment before vLLM is constructed."""
    # vLLM starts engine subprocesses. With Linux's default 'fork' start
    # method, CUDA fails in the child if the parent already initialised it
    # (e.g. in a notebook), so use 'spawn'.
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

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
    if key in _engines:
        return _engines[key]

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

        # Do not pass HF_HOME as vLLM's download_dir. vLLM hands download_dir
        # to snapshot_download as cache_dir, and the shared cache is
        # $HF_HOME/hub, so HF_HOME points one level above it and the weights
        # are downloaded again. An explicit vllm_kwargs={"download_dir": ...}
        # still works.
        kwargs = dict(vllm_kwargs)

        started = time.perf_counter()
        with measure.Measurement() as measured:
            _engines[key] = LLM(model=hf_model_id, quantization=quantization, **kwargs)
        _engine_loaded_at[key] = time.perf_counter()

        # The measurement is for telemetry only. vLLM preallocates
        # gpu_memory_utilization x card, so claimed_gib reflects that setting,
        # and weights_gib is usually None because the engine runs in a
        # separate process.
        observe.record({
            "ev": "engine",
            "phase": "load",
            "backend": "vllm",
            "hf_model_id": hf_model_id,
            "quantization": quantization,
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "load_ms": round((_engine_loaded_at[key] - started) * 1000, 1),
            "claimed_gib": measured.claimed_gib,
            "weights_gib": measured.weights_gib,
        })
        return _engines[key]


def _release_engines() -> None:
    """Emit a lifetime event per loaded engine, then forget the timers.

    Called from :meth:`VLLMBackend.clear_cache` and at exit, so a process
    that just exits still reports how long each engine was held.
    """
    now = time.perf_counter()
    for key, loaded_at in list(_engine_loaded_at.items()):
        observe.record({
            "ev": "engine",
            "phase": "release",
            "backend": "vllm",
            "hf_model_id": key[0],
            "quantization": key[1],
            "held_s": round(now - loaded_at, 1),
        })
    _engine_loaded_at.clear()


atexit.register(_release_engines)


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
            vllm_kwargs: Passed to ``vllm.LLM()`` (e.g.
                ``gpu_memory_utilization``, ``tokenizer_mode``).
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
        self._llm = _engine(hf_model_id, quantization, vllm_kwargs or {})

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
    def clear_cache(cls) -> None:
        """Drop the loaded engines, after reporting how long each was held."""
        _release_engines()
        _engines.clear()

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
        system_prompts: str | list[str | None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate replies for the whole batch in a single vLLM pass.

        Args:
            messages_list: One chat message list per item.
            system_prompts: Inserted as the leading message of each item.
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
        prompts, resolved = self._prepare(messages_list, system_prompts)
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
            outputs = self._llm.chat(chat_inputs, sampling_params)
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
