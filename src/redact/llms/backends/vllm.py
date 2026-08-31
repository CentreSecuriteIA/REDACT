"""Local vLLM backend for self-hosted inference.

Heavy init — loads model weights into GPU memory once. After that, generate()
is fast, and especially efficient for a real batch because vLLM processes all
prompts in a single engine pass.

One instance **is one registered model**: sampling defaults and the engine are
bound in :meth:`VLLMBackend.from_config`, not re-passed per call. The *engine*
is the expensive part, so it is cached on its own identity
(``hf_model_id`` + quantization + engine kwargs) rather than on the model name
— two registry rows pointing at the same checkpoint share one load. That cache
is strictly private to this class: ``introspection.py`` loads the same weights
through ``transformers`` instead, which is a different runtime and must never
share an entry.

Prompts go through vLLM's own chat template (``llm.chat()``), so any
checkpoint whose tokenizer ships a usable ``chat_template`` works as-is.
"""

import atexit
import contextlib
import multiprocessing
import os
import threading
import time
from typing import ClassVar

from .. import observe
from . import vram
from .base import ComputeConfig, LLMBackend

# Sampling values with no ModelConfig field of their own — a backend default,
# not a registry lookup. Overridable per model via VLLMConfig.sampling.
_DEFAULT_SAMPLING = {"top_p": 0.85}
_DEFAULT_TEMPERATURE = 0.7

# Loaded engines, keyed on what actually determines the load. Two registry
# rows naming the same checkpoint (venice-uncensored's .vllm setup and
# venice-paraphraser) share one engine instead of loading the weights twice.
_engines: dict[tuple, object] = {}
# Guards the load path. Preloading (runconfig) happens on a background thread,
# so two threads can miss the cache together — without this, both construct an
# LLM() and two copies of the same checkpoint land on one card.
_engines_lock = threading.Lock()
# When each engine was loaded, keyed the same way. The GPU is *leased* for as
# long as an engine is alive — not just while it generates — so cost is
# wall-clock lifetime, and that clock belongs to the checkpoint (the cache key),
# never to a backend instance. Two registry rows sharing one load share one
# meter, which is exactly right: there is one card being held.
_engine_loaded_at: dict[tuple, float] = {}


def _engine_key(hf_model_id: str, quantization: str | None, vllm_kwargs: dict) -> tuple:
    return (hf_model_id, quantization, repr(sorted(vllm_kwargs.items())))


def _prepare_environment() -> None:
    """Environment fixes that must happen before vLLM is constructed."""
    # vLLM v1 spawns engine core subprocesses. On Linux the default
    # multiprocessing start method is 'fork', which causes CUDA to fail
    # if it was already initialised in the parent (e.g. in a notebook).
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

    # WSL2-specific workaround, verified by actually hitting this on a
    # real WSL2 + RTX 2080 Super setup: vLLM's V2 GPU model runner
    # unconditionally allocates a UVA (Unified Virtual Addressing)
    # buffer, but is_pin_memory_available() defaults to False on WSL2,
    # so UVA allocation raises `RuntimeError: UVA is not available` on
    # every load. This is an open, unmerged upstream bug as of vLLM
    # 0.27.1 (vllm-project/vllm#47387, fix proposed in #47579) — forcing
    # the older V1 runner sidesteps it entirely (no UVA dependency).
    # setdefault so an explicit VLLM_USE_V2_MODEL_RUNNER from the caller
    # still wins; remove this block once the upstream fix ships.
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
            from vllm import LLM  # Lazy import — vllm is heavy and optional
        except ImportError:
            raise ImportError(
                "The 'vllm' package is required for VLLMBackend. "
                "Install it with: pip install redact[vllm]"
            )

        # NOTE: do not bridge HF_HOME into vLLM's `download_dir` here. Both
        # runtimes already honour HF_HOME through huggingface_hub, and the
        # cache root is $HF_HOME/**hub**, not $HF_HOME — vLLM passes
        # download_dir straight through as `cache_dir=` to snapshot_download.
        # Setting it to HF_HOME therefore points vLLM one level above the
        # shared cache and downloads a second copy of weights transformers
        # already has (~47GB for the 24B). Inert when HF_HOME is unset, which
        # is exactly why it survived: it only breaks for the users who set
        # HF_HOME *because* disk is scarce. An explicit
        # vllm_kwargs={"download_dir": ...} still works and still wins.
        kwargs = dict(vllm_kwargs)

        started = time.perf_counter()
        with vram.Measurement() as measured:
            _engines[key] = LLM(model=hf_model_id, quantization=quantization, **kwargs)
        _engine_loaded_at[key] = time.perf_counter()

        # What this load took, recorded for the residency planner. CAUTION:
        # `claimed_gb` here is essentially gpu_memory_utilization x card, since
        # vLLM preallocates weights AND KV cache — it is NOT the model's need.
        # vram.planning_gb() is what reads this back correctly.
        settings = {
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization"),
            "max_model_len": kwargs.get("max_model_len"),
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "quantization": quantization,
            "dtype": kwargs.get("dtype"),
        }
        from ... import paths  # local import: paths is above llms in the tree

        # The variable half of the footprint. Not measurable from the delta —
        # vLLM's pool is sized by gpu_memory_utilization, not by need — so it
        # is derived from the checkpoint's own shapes instead. Without it
        # planning_gb() silently reduces to weights-only and the planner
        # under-books every local model.
        measured.record(
            paths.vram_cache_json(), f"vllm:{hf_model_id}", settings, backend="vllm",
            kv_gb_est=vram.estimate_kv_gb(
                hf_model_id,
                max_model_len=settings["max_model_len"],
                dtype=settings["dtype"],
                tensor_parallel_size=settings["tensor_parallel_size"] or 1,
            ),
        )
        observe.record({
            "ev": "engine",
            "phase": "load",
            "backend": "vllm",
            "hf_model_id": hf_model_id,
            "quantization": quantization,
            "tensor_parallel_size": settings["tensor_parallel_size"],
            "load_ms": round((_engine_loaded_at[key] - started) * 1000, 1),
            "claimed_gb": measured.claimed_gb,
            "weights_gb": measured.weights_gb,
        })
        return _engines[key]


def _release_engines() -> None:
    """Emit a lifetime event per loaded engine, then forget the timers.

    Called from :meth:`VLLMBackend.clear_cache` and from an ``atexit`` hook —
    without the latter, a process that simply exits never reports ``held_s``
    and the local cost of the whole run goes unrecorded.
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

    # generate() runs a single engine pass over the whole list, so
    # native_batching=True. Concurrent generate() calls from threads
    # compete for GPU memory, so parallel_calls=False — the whole batch
    # goes to one call instead.
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
        """Bind one registered model to a vLLM engine.

        Args:
            model: Registry name of the model.
            hf_model_id: HuggingFace model ID or local path to load.
            quantization: Quantization method (e.g. "gptq", "awq").
            vllm_kwargs: Passed to ``vllm.LLM()`` (e.g. revision,
                trust_remote_code, gpu_memory_utilization, tokenizer_mode).
            sampling: Extra ``SamplingParams`` defaults for this model
                (``top_p``, ``top_k``, ...), merged over
                :data:`_DEFAULT_SAMPLING`.
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
        """
        super().__init__(model, **identity)
        # SamplingParams needs a real number; ModelConfig.default_temperature
        # is optional, so a backend default fills in here — once, at
        # construction, rather than inside generate().
        if self.default_temperature is None:
            self.default_temperature = _DEFAULT_TEMPERATURE
        self.hf_model_id = hf_model_id
        self._sampling = {**_DEFAULT_SAMPLING, **(sampling or {})}
        self._llm = _engine(hf_model_id, quantization, vllm_kwargs or {})

    @classmethod
    def from_config(cls, config) -> "VLLMBackend":
        """Build from a registry entry's ``.vllm`` setup.

        Raises:
            ValueError: If the entry has no ``.vllm`` setup.
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
            vllm_kwargs=vllm.vllm_kwargs,
            sampling=vllm.sampling,
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the loaded engines — frees GPU memory between local models.

        Reports each engine's lifetime before forgetting it, so unloading to
        make room for the next model still bills the time the card was held.
        """
        _release_engines()
        _engines.clear()

    def _clean_output(self, text: str) -> str:
        """Strip known tokenizer artifacts from raw vLLM output.

        Belt-and-suspenders fallback. With llm.chat() the model's own
        tokenizer handles stop tokens, so this should rarely fire.
        """
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
        """Generate responses for a batch of message lists in a single vLLM pass.

        A true single engine pass over the whole batch — vLLM processes all
        prompts at once. A single sample is just a batch of one, same code
        path. ``internals_ids`` is accepted for signature parity but never
        used — this backend never declares ``compute_config.supports_internals``.

        Args:
            messages_list: One chat message list per batch item.
            system_prompts: Optional system prompt(s) — re-inserted as the
                leading message before vLLM applies its chat template.
            max_tokens: Overrides this model's default.
            temperature: Overrides this model's default.
            internals_ids: Unused — see above.
            **kwargs: Extra ``SamplingParams`` overrides for this call
                (e.g. top_k, presence_penalty), merged over this model's
                stored sampling defaults.

        Returns:
            Generated text, one per batch item, same order as ``messages_list``.
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
        # ONE event for the whole list: this is a single engine pass, so
        # n_items is the batch size, not a count of separate calls.
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
