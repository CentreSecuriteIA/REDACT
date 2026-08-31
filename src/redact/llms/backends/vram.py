"""Measuring what a local model actually took on the GPU.

Shared by ``vllm.py`` and ``introspection.py``, but the number means something
**different** for each, and confusing the two is the trap this module exists to
prevent:

- **vLLM preallocates** ``gpu_memory_utilization x total_VRAM`` for weights
  *and* KV cache at load. So a naive before/after delta comes back as roughly
  that fraction *regardless of how large the model is*. Measure the 3B debug
  model at ``gpu_memory_utilization=0.86`` and you record "this 3B needs 86% of
  the card". Feed that back as a footprint and the planner will refuse to
  co-locate anything, forever.
- **transformers preallocates nothing**, so there the delta really is the need.

So a measurement records three numbers, not one:

===============  =================================================
``claimed_gb``   what the runtime reserved — a function of the
                 *setting* for vLLM, of the model for transformers
``weights_gb``   the real floor (torch's own allocator high-water)
``kv_gb_est``    the variable part, derived from ``max_model_len``
===============  =================================================

Planning uses ``weights_gb + kv_gb_est``. ``claimed_gb`` is for telemetry,
where "what did this engine hold" is exactly the right question.

Every entry is keyed by the settings it was measured under, so a measurement is
only reused when they still match.
"""

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def _torch():
    """Return ``torch`` if it is importable and CUDA is usable, else ``None``."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001 — a broken CUDA install must not raise here
        return None
    return torch


def free_total_gb() -> tuple[float, float] | None:
    """``(free_gb, total_gb)`` for the current device, or ``None`` without CUDA."""
    torch = _torch()
    if torch is None:
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.debug("mem_get_info failed (%s: %s)", type(exc).__name__, exc)
        return None
    return free / 1e9, total / 1e9


class Measurement:
    """Brackets a local model load and reports what it took.

    Use as a context manager around the load. Everything degrades to ``None``
    without CUDA, so the calling backend needs no guard of its own::

        with Measurement() as m:
            engine = LLM(...)
        m.record(path, key, settings, backend="vllm")
    """

    def __init__(self):
        self._torch = _torch()
        self.before_free_gb: float | None = None
        self.claimed_gb: float | None = None
        self.weights_gb: float | None = None
        self.total_gb: float | None = None

    def __enter__(self) -> "Measurement":
        ft = free_total_gb()
        if ft is not None:
            self.before_free_gb, self.total_gb = ft
            with _suppress():
                self._torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> bool:
        ft = free_total_gb()
        if ft is not None and self.before_free_gb is not None:
            free_after, _ = ft
            # Everything the runtime took off the card, however it took it.
            self.claimed_gb = round(max(0.0, self.before_free_gb - free_after), 2)
            with _suppress():
                # torch's own allocator only sees tensors torch allocated —
                # i.e. the weights. vLLM's KV pool is outside it, which is
                # precisely what makes this the floor rather than the total.
                peak = self._torch.cuda.max_memory_allocated() / 1e9
                self.weights_gb = round(peak, 2) if peak > 0 else None
        return False  # never swallow a load failure

    def record(
        self,
        path: Path,
        key: str,
        settings: dict,
        *,
        backend: str,
        kv_gb_est: float | None = None,
    ) -> None:
        """Append this measurement to the on-disk cache.

        Args:
            path: ``Data_cache/vram.json``.
            key: Stable identity of the load — same components as the backend's
                own cache key.
            settings: Everything that moves the number
                (``gpu_memory_utilization``, ``max_model_len``,
                ``tensor_parallel_size``, ``quantization``, dtype). A cached
                measurement is only reused when these all match.
            backend: ``"vllm"`` or ``"introspect"`` — the same delta means
                different things per runtime, so the reader must know which.
            kv_gb_est: Estimated KV-cache need, when derivable.

        A no-op when nothing was measured (no CUDA). Never raises: failing to
        write a planning hint must not fail a load that just succeeded.
        """
        if self.claimed_gb is None:
            return
        entry = {
            "backend": backend,
            "claimed_gb": self.claimed_gb,
            "weights_gb": self.weights_gb,
            "kv_gb_est": kv_gb_est,
            "total_gb": self.total_gb,
            "settings": settings,
        }
        try:
            with _write_lock:
                data = {}
                if path.exists():
                    with _suppress():
                        data = json.loads(path.read_text(encoding="utf-8"))
                data[key] = entry
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
                )
        except OSError as exc:
            logger.debug("Could not write VRAM cache %s (%s)", path, exc)


_DTYPE_BYTES = {
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
    "float8": 1, "fp8": 1,
}


# TODO: a *runtime* KV estimate, alongside this planning floor.
#
# What this function returns is the feasibility floor: one sequence at
# max_model_len. What a run actually holds is
#
#     kv_gb x concurrent_sequences,  where a sequence's own length is
#     (input_tokens + output_tokens), not just the output budget
#
# so the two missing inputs are batch size and, more importantly, a bound on
# *input* length. `max_tokens` only ever bounds the completion; nothing in the
# library bounds the prompt (verified: no truncation or length check anywhere
# outside vLLM's own max_model_len). That gap is not theoretical — ASCII-art
# obfuscation inflates a one-line request into thousands of tokens, and has
# OOM'd models in practice. Encoding techniques (base64, morse, braille,
# tokenbreak) multiply length the same way, and FSH/DAP stack whole few-shot
# blocks on top.
#
# So this wants two pieces, in order:
#   1. Gate max input somewhere the whole library goes through — most likely
#      LLMBackend._prepare(), which every transport already calls, so no
#      pipeline has to remember to do it. Reject-or-truncate is a policy call:
#      truncating an obfuscated jailbreak silently changes what was tested,
#      so rejecting and recording the drop is probably right.
#   2. Once a max-input bound exists, runtime KV becomes derivable as
#      (max_input + max_output) x batch, and the planner can distinguish
#      "these models fit on the card" from "these models fit *and can run the
#      configured batch size*" — which is the question that actually predicts
#      an OOM mid-run.
def estimate_kv_gb(
    hf_model_id: str,
    *,
    max_model_len: int | None = None,
    dtype: str | None = None,
    tensor_parallel_size: int = 1,
) -> float | None:
    """Per-GPU KV-cache need for **one** sequence at full context, in GB.

    The variable half of a local footprint. Weights are fixed once a checkpoint
    is chosen; KV scales with the context length you configure, which is why it
    cannot be folded into ``weights_gb`` and is recorded separately.

    Read straight off the checkpoint's HF config — shapes only, no weights are
    downloaded or loaded::

        2 (K and V) x layers x kv_heads x head_dim x tokens x dtype_bytes

    One sequence is deliberately the basis. vLLM's actual pool is *elastic* —
    it fills ``gpu_memory_utilization x card`` minus weights, however much that
    is — so there is no "true" KV size to measure. What a planner needs is the
    floor: the point below which the engine cannot serve even a single request
    at ``max_model_len``. Concurrency beyond that buys throughput, not
    feasibility.

    Args:
        hf_model_id: Checkpoint whose config to read.
        max_model_len: Context length the engine is configured for. ``None``
            falls back to the model's own ``max_position_embeddings``.
        dtype: KV dtype name; ``None`` assumes 2 bytes (bf16/fp16), which is
            what vLLM uses unless told otherwise.
        tensor_parallel_size: KV is sharded across TP ranks, so the per-GPU
            need is the total divided by this.

    Returns:
        GB per GPU, or ``None`` if the config can't be read or lacks the shape
        fields. A planning hint is best-effort by definition: never raises, and
        a missing estimate degrades to "weights only", never to a wrong number.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        return None
    try:
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 — a planning hint must not fail a load
        logger.debug("Could not read HF config for %s (%s)", hf_model_id, exc)
        return None

    # Multimodal/composite configs nest the language model's shapes.
    cfg = getattr(cfg, "text_config", None) or cfg
    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_attention_heads", None)
    # GQA/MQA: only the KV heads are cached, which for a modern checkpoint is
    # several times fewer than the attention heads. Falling back to
    # num_attention_heads (i.e. MHA) is the conservative direction.
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = getattr(cfg, "head_dim", None) or (
        hidden // heads if hidden and heads else None
    )
    tokens = max_model_len or getattr(cfg, "max_position_embeddings", None)
    if not all((layers, kv_heads, head_dim, tokens)):
        return None

    dtype_bytes = _DTYPE_BYTES.get(str(dtype).lower().replace("torch.", ""), 2)
    kv_bytes = 2 * layers * kv_heads * head_dim * tokens * dtype_bytes
    return round(kv_bytes / 1e9 / max(1, tensor_parallel_size), 2)


def load_cache(path: Path) -> dict:
    """Read ``Data_cache/vram.json``; ``{}`` when absent or unreadable."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read VRAM cache %s (%s)", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def planning_gb(entry: dict) -> float | None:
    """The number a planner should use: the floor plus the variable part.

    **Never ``claimed_gb``** — for vLLM that is driven by
    ``gpu_memory_utilization``, not by the model, so planning from it would
    make every model look like it needs most of the card.

    Falls back to ``claimed_gb`` only for ``introspect``, which preallocates
    nothing and where the delta genuinely is the need.
    """
    weights = entry.get("weights_gb")
    if weights is not None:
        return round(weights + (entry.get("kv_gb_est") or 0.0), 2)
    if entry.get("backend") == "introspect":
        return entry.get("claimed_gb")
    return None


class _suppress:
    """Swallow anything a best-effort CUDA probe throws."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_) -> bool:
        return exc_type is not None
