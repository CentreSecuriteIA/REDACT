"""Estimate what a local model needs on the GPU from its HF config.

Only the config is read (shapes, no weights), so a footprint is available
before anything is loaded. Measuring is no substitute for vLLM: it
preallocates ``gpu_memory_utilization x card`` at load, so memory measured
after a load describes that setting and not the model.

Where the config leaves a choice open, the larger figure is taken: an
estimate that is too high costs a GPU, one that is too low ends in an OOM.

All sizes are GiB (bytes / 1024**3).
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3

_DTYPE_BYTES = {
    "float64": 8, "double": 8,
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
}

#: Weight bit width per vLLM quantization method. Any other method is sized
#: unquantized, which errs high.
_QUANT_BITS = {"awq": 4, "gptq": 4, "fp8": 8}

#: Extra bits per quantized weight for group scales and zero points.
_QUANT_OVERHEAD_BITS = 0.5

#: Widest ``quantization_config.bits`` taken at its word.
_MAX_QUANT_BITS = 16

#: Architectures whose MLP is two matrices. Every other one is sized as a
#: gated MLP (three), since the activation name does not tell them apart:
#: Gemma is gated and uses GELU.
_UNGATED_MODEL_TYPES = frozenset({
    "bert", "gpt_neo", "gpt_neox", "phi", "roberta",
})

#: Multiplier over (weights + KV) for activations and allocator fragmentation.
#: A guess. Raise it if loads run out of memory just above the estimate.
DEFAULT_FRAGMENTATION = 1.1

#: Fixed per-process CUDA context overhead, in GiB. It does not scale with
#: the model, so it is added after the multiplier.
DEFAULT_CUDA_CONTEXT_GIB = 0.6

#TODO: make this a run-level setting, chosen at startup and capped per checkpoint.
#: Context length of a vLLM engine whose setup sets no ``max_model_len``.
DEFAULT_MAX_MODEL_LEN = 10_000

#: Default for ``config=``: the caller has not read the config yet.
_UNREAD = object()

_reported: set[tuple[str, str]] = set()


def _report_once(hf_model_id: str, reason: str) -> None:
    """Log at INFO why a checkpoint has no estimate, once per process."""
    if (hf_model_id, reason) not in _reported:
        _reported.add((hf_model_id, reason))
        logger.info("No VRAM estimate for %s: %s", hf_model_id, reason)


def _load_config(hf_model_id: str):
    """The checkpoint's HF config, or ``None`` if it cannot be read. Never raises."""
    try:
        from transformers import AutoConfig
    except ImportError:
        _report_once(hf_model_id, "transformers is not installed")
        return None
    try:
        return AutoConfig.from_pretrained(hf_model_id, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 — a planning hint must not fail a load
        _report_once(hf_model_id,
                     f"its HF config could not be read ({type(exc).__name__}: {exc})")
        return None


def _language_config(full):
    """The config holding the language model's shapes.

    Multimodal configs nest it as ``text_config``. Their vision tower and
    projector are not counted.
    """
    return getattr(full, "text_config", None) or full


def _resolve_dtype(
    dtype: str | None, full=None, float32_as_half: bool = False
) -> tuple[str, float]:
    """The dtype in effect, as ``(label, bytes per value)``.

    The first name set among ``dtype`` and then the config's dtype fields
    decides; ``None`` and ``"auto"`` count as not set. If that name is not
    recognised, or none is set, float32 is assumed, so an unrecognised
    ``dtype`` does not fall back to the config. With ``float32_as_half``, a
    float32 result counts as 16-bit unless ``dtype`` names a dtype itself,
    recognised or not.
    """
    names = [dtype]
    # A multimodal checkpoint may declare its dtype on either config.
    for cfg in (full, _language_config(full)):
        names += [getattr(cfg, "torch_dtype", None), getattr(cfg, "dtype", None)]
    label, size = "float32 assumed", _DTYPE_BYTES["float32"]
    for name in names:
        key = str(name).lower().replace("torch.", "")
        if name is None or key == "auto":
            continue
        if key in _DTYPE_BYTES:
            label, size = key, _DTYPE_BYTES[key]
        break
    explicit = dtype is not None and str(dtype).lower() != "auto"
    if float32_as_half and not explicit and size == _DTYPE_BYTES["float32"]:
        return "float16, cast from float32", _DTYPE_BYTES["float16"]
    return label, size


def _quant_bits(quantization: str | None, full) -> int | None:
    """Weight bit width under ``quantization``, or ``None`` if not modelled.

    The config's ``quantization_config.bits`` is used when present, else the
    method's usual width.
    """
    usual = _QUANT_BITS.get(str(quantization or "").lower())
    if usual is None:
        return None
    qc = getattr(full, "quantization_config", None)
    bits = qc.get("bits") if isinstance(qc, dict) else getattr(qc, "bits", None)
    if isinstance(bits, bool) or not isinstance(bits, int):
        return usual
    return bits if 0 < bits <= _MAX_QUANT_BITS else usual


def _expert_count(cfg) -> int:
    """MLP experts per layer of an MoE config, shared ones included; else 0."""
    for name in ("num_local_experts", "num_experts", "n_routed_experts"):
        experts = getattr(cfg, name, None)
        if experts:
            return experts + (getattr(cfg, "n_shared_experts", None) or 0)
    return 0


def estimate_weights_gib(
    hf_model_id: str,
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    tensor_parallel_size: int | None = 1,
    float32_as_half: bool = False,
    config=_UNREAD,
) -> float | None:
    """Per-GPU weight footprint, in GiB.

    The parameter count is derived from the config's shapes, for a
    decoder-only transformer::

        embeddings  = vocab x hidden           (x2 when lm_head is untied)
        attention   = layers x hidden x (q_proj + k_proj + v_proj + o_proj)
        mlp         = layers x hidden x intermediate x n_mats x experts

    ``n_mats`` is 3 (gate/up/down, as in Llama, Mistral or Gemma) unless the
    architecture is known to use a plain two-matrix MLP. ``experts`` is the
    config's expert count for an MoE checkpoint and 1 otherwise. An MoE
    config's ``shared_expert_intermediate_size`` (Qwen2-MoE) adds one more
    MLP of that width per layer. Biases and norms are ignored (under 0.01% of
    the total).

    Args:
        hf_model_id: Checkpoint whose config to read.
        dtype: Weight dtype name. ``None`` or ``"auto"`` uses the dtype the
            checkpoint's config declares, and float32 if it declares none.
        quantization: ``"awq"``/``"gptq"`` (4-bit) or ``"fp8"`` (8-bit),
            unless the config's ``quantization_config.bits`` says otherwise.
            Applies to the attention and MLP matrices only; embeddings and
            lm_head keep ``dtype``. Any other method is sized unquantized.
        tensor_parallel_size: Weights are split across this many GPUs, so
            the per-GPU figure is the total divided by it. ``None`` is 1.
        float32_as_half: Size a float32 checkpoint at 16-bit unless ``dtype``
            names a dtype. Set for vLLM, whose default dtype loads it so.
        config: The checkpoint's config if already read; ``None`` if that
            read failed.

    Returns:
        GiB per GPU, or ``None`` when the config cannot be read or lacks the
        shape fields.
    """
    full = _load_config(hf_model_id) if config is _UNREAD else config
    if full is None:
        return None
    cfg = _language_config(full)

    layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None)
    intermediate = getattr(cfg, "intermediate_size", None)
    heads = getattr(cfg, "num_attention_heads", None)
    if not all((layers, hidden, vocab, intermediate, heads)):
        _report_once(hf_model_id, "its config lacks the shape fields")
        return None

    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads)

    # q and o are full width; k and v are narrower under GQA (kv_heads < heads).
    attn_per_layer = (
        hidden * heads * head_dim          # q_proj
        + 2 * hidden * kv_heads * head_dim  # k_proj + v_proj
        + heads * head_dim * hidden        # o_proj
    )

    n_mats = 2 if getattr(cfg, "model_type", None) in _UNGATED_MODEL_TYPES else 3
    experts = _expert_count(cfg)
    if experts:
        width = getattr(cfg, "moe_intermediate_size", None) or intermediate
        # Each expert is a full MLP; the router adds hidden x experts.
        mlp_per_layer = experts * n_mats * hidden * width + hidden * experts
        # Qwen2-MoE: one always-on shared expert per layer, of its own width.
        shared = getattr(cfg, "shared_expert_intermediate_size", None)
        if shared:
            mlp_per_layer += n_mats * hidden * shared
    else:
        mlp_per_layer = n_mats * hidden * intermediate

    linear = layers * (attn_per_layer + mlp_per_layer)
    embed = vocab * hidden
    if not getattr(cfg, "tie_word_embeddings", False):
        embed *= 2  # separate lm_head

    dtype_bytes = _resolve_dtype(dtype, full, float32_as_half)[1]
    bits = _quant_bits(quantization, full)
    linear_bytes = dtype_bytes
    if bits:
        linear_bytes = min(dtype_bytes, (bits + _QUANT_OVERHEAD_BITS) / 8)

    total = linear * linear_bytes + embed * dtype_bytes
    return round(total / _BYTES_PER_GIB / max(1, tensor_parallel_size or 1), 2)


def effective_max_model_len(
    hf_model_id: str, max_model_len: int | None = None, config=_UNREAD
) -> int:
    """The context length a vLLM engine runs with.

    An explicit ``max_model_len`` wins. Otherwise it is
    :data:`DEFAULT_MAX_MODEL_LEN`, capped at the config's
    ``max_position_embeddings`` when that can be read.
    """
    if max_model_len:
        return max_model_len
    full = _load_config(hf_model_id) if config is _UNREAD else config
    limit = getattr(_language_config(full), "max_position_embeddings", None)
    return min(DEFAULT_MAX_MODEL_LEN, limit) if limit else DEFAULT_MAX_MODEL_LEN


def estimate_kv_gib(
    hf_model_id: str,
    *,
    max_model_len: int | None = None,
    dtype: str | None = None,
    tensor_parallel_size: int | None = 1,
    float32_as_half: bool = False,
    config=_UNREAD,
) -> float | None:
    """Per-GPU KV-cache need, in GiB.

    Computed as::

        2 (K and V) x layers x kv_heads x head_dim x context length x bytes

    This is one sequence of the engine's context length: the least KV cache
    vLLM needs to start, not what it will hold. vLLM gives the cache whatever
    its memory grant has left.

    Args:
        hf_model_id: Checkpoint whose config to read.
        max_model_len: The engine's configured context length. ``None`` uses
            :func:`effective_max_model_len`.
        dtype: KV dtype, resolved as in :func:`estimate_weights_gib`.
        tensor_parallel_size: The KV cache is split across this many GPUs.
            ``None`` is 1.
        float32_as_half: As in :func:`estimate_weights_gib`.
        config: As in :func:`estimate_weights_gib`.

    Returns:
        GiB per GPU, or ``None`` when the config cannot be read or lacks the
        shape fields.
    """
    full = _load_config(hf_model_id) if config is _UNREAD else config
    if full is None:
        return None
    cfg = _language_config(full)

    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_attention_heads", None)
    # Under GQA/MQA only the KV heads are cached. Falling back to all heads
    # errs high.
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = getattr(cfg, "head_dim", None) or (
        hidden // heads if hidden and heads else None
    )
    if not all((layers, kv_heads, head_dim)):
        return None
    tokens = effective_max_model_len(hf_model_id, max_model_len, config=full)

    kv_bytes = (
        2 * layers * kv_heads * head_dim * tokens
        * _resolve_dtype(dtype, full, float32_as_half)[1]
    )
    return round(kv_bytes / _BYTES_PER_GIB / max(1, tensor_parallel_size or 1), 2)


@dataclass(frozen=True)
class Estimate:
    """What one checkpoint is estimated to need under given engine settings.

    Attributes:
        weights_gib: Weights per GPU.
        kv_gib: KV cache per GPU, or ``None`` when not sized.
        per_gpu_gib: Planning figure per GPU (:func:`planning_gib`).
        total_gib: ``per_gpu_gib`` over all ``tensor_parallel_size`` GPUs.
        tensor_parallel_size: GPUs the model is split across.
        attention_heads: The config's head count, which a tensor-parallel
            size must divide. ``None`` when unknown.
        description: One line deriving the figure, naming what drives it.
    """

    weights_gib: float
    kv_gib: float | None
    per_gpu_gib: float
    total_gib: float
    tensor_parallel_size: int = 1
    attention_heads: int | None = None
    description: str = ""


def _drivers(
    full, dtype: str | None, quantization: str | None, float32_as_half: bool
) -> list[str]:
    """What sets the size besides the shapes: dtype, quantization, experts."""
    if full is None:
        return []
    out = [_resolve_dtype(dtype, full, float32_as_half)[0]]
    if quantization:
        bits = _quant_bits(quantization, full)
        out.append(f"{quantization} {bits}-bit" if bits
                   else f"{quantization} sized unquantized")
    experts = _expert_count(_language_config(full))
    if experts:
        out.append(f"{experts} experts")
    return out


def estimate(
    hf_model_id: str,
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    max_model_len: int | None = None,
    tensor_parallel_size: int | None = 1,
    kv: bool = True,
    float32_as_half: bool = False,
) -> Estimate | None:
    """Everything a planner needs for one checkpoint, from one config read.

    Arguments are as in :func:`estimate_weights_gib` and
    :func:`estimate_kv_gib`; ``dtype`` is also the KV dtype.

    Args:
        tensor_parallel_size: GPUs the model is split across. ``None`` is 1.
        kv: ``False`` leaves the KV cache out, for a runtime that does not
            preallocate one.

    Returns:
        The estimate, or ``None`` when the weights cannot be estimated.
    """
    tp = max(1, tensor_parallel_size or 1)
    full = _load_config(hf_model_id)
    weights = estimate_weights_gib(
        hf_model_id, dtype=dtype, quantization=quantization,
        tensor_parallel_size=tp, float32_as_half=float32_as_half, config=full)
    if weights is None:
        return None
    kv_gib = None
    if kv:
        kv_gib = estimate_kv_gib(
            hf_model_id, max_model_len=max_model_len, dtype=dtype,
            tensor_parallel_size=tp, float32_as_half=float32_as_half,
            config=full)
    per_gpu = planning_gib(weights, kv_gib)

    terms = f"{weights:.1f} weights"
    if kv_gib:
        terms += f" + {kv_gib:.2f} KV"
    text = f"{terms}, x{DEFAULT_FRAGMENTATION} +{DEFAULT_CUDA_CONTEXT_GIB}"
    if tp > 1:
        text += f", per GPU of {tp}"
    drivers = _drivers(full, dtype, quantization, float32_as_half)
    if drivers:
        text += f" [{', '.join(drivers)}]"
    heads = getattr(_language_config(full), "num_attention_heads", None)
    return Estimate(weights, kv_gib, per_gpu, per_gpu * tp, tp, heads, text)


def padded_gib(need_gib: float) -> float:
    """The padded need: weights plus KV with headroom, unrounded.

    A vLLM grant must cover it.
    """
    return need_gib * DEFAULT_FRAGMENTATION + DEFAULT_CUDA_CONTEXT_GIB


def planning_gib(weights_gib: float | None, kv_gib: float | None) -> float | None:
    """The planning figure per GPU: the padded need rounded up to 0.5 GiB.

    ``kv_gib=None`` counts as 0. Returns ``None`` when ``weights_gib`` is
    ``None``.
    """
    if weights_gib is None:
        return None
    return math.ceil(padded_gib(weights_gib + (kv_gib or 0.0)) * 2) / 2
