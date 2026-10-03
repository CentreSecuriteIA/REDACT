"""Estimate what a local model needs on the GPU from its HF config.

Only the config is read (shapes, no weights), so a footprint is available
before anything is loaded.

Footprints are estimated because measuring does not work for vLLM. It
preallocates ``gpu_memory_utilization x card`` at load, so memory measured
after a load describes that setting and not the model.

All sizes are GiB (bytes / 1024**3).
"""

import logging

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3

_DTYPE_BYTES = {
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
    "float8": 1, "fp8": 1,
    "int8": 1, "awq": 0.5, "gptq": 0.5,
}

#: Multiplier over (weights + KV) for activations and allocator fragmentation.
#: A guess. Raise it if loads run out of memory just above the estimate.
DEFAULT_FRAGMENTATION = 1.1

#: Fixed per-process CUDA context overhead, in GiB. It does not scale with
#: the model, so it is added after the multiplier.
DEFAULT_CUDA_CONTEXT_GIB = 0.6


def _load_config(hf_model_id: str):
    """The checkpoint's HF config, or ``None`` if it cannot be read.

    Never raises.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        logger.debug("transformers not installed; cannot estimate for %s", hf_model_id)
        return None
    try:
        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 — a planning hint must not fail a load
        logger.debug("Could not read HF config for %s (%s)", hf_model_id, exc)
        return None
    # Multimodal/composite configs nest the language model's shapes.
    return getattr(cfg, "text_config", None) or cfg


def _dtype_bytes(dtype: str | None, quantization: str | None = None) -> float:
    """Bytes per parameter. Quantization wins over dtype when both are given."""
    for candidate in (quantization, dtype):
        if candidate:
            key = str(candidate).lower().replace("torch.", "")
            if key in _DTYPE_BYTES:
                return _DTYPE_BYTES[key]
    return 2  # default: 16-bit floats (bf16/fp16)


def estimate_weights_gib(
    hf_model_id: str,
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    tensor_parallel_size: int = 1,
) -> float | None:
    """Per-GPU weight footprint, in GiB.

    The parameter count is derived from the config's shapes, for a
    decoder-only transformer::

        embeddings  = vocab x hidden           (x2 when lm_head is untied)
        attention   = layers x hidden x (q_proj + k_proj + v_proj + o_proj)
        mlp         = layers x hidden x intermediate x n_mats

    ``n_mats`` is 3 for a gated MLP (gate/up/down, as in Llama or Mistral)
    and 2 otherwise. A gated MLP is assumed when the activation name contains
    "silu" or "glu".

    MoE checkpoints are not modelled, and the estimate comes out too low for
    them. Set ``vram_gb`` for those.

    Args:
        hf_model_id: Checkpoint whose config to read.
        dtype: Weight dtype name. ``None`` assumes 2 bytes per parameter.
        quantization: Overrides ``dtype`` when set (``"awq"``/``"gptq"``
            count 0.5 bytes per parameter, ``"int8"`` 1).
        tensor_parallel_size: Weights are split across this many GPUs, so
            the per-GPU figure is the total divided by it.

    Returns:
        GiB per GPU, or ``None`` when the config cannot be read or lacks the
        shape fields.
    """
    cfg = _load_config(hf_model_id)
    if cfg is None:
        return None

    layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None)
    intermediate = getattr(cfg, "intermediate_size", None)
    heads = getattr(cfg, "num_attention_heads", None)
    if not all((layers, hidden, vocab, intermediate, heads)):
        logger.debug("Config for %s lacks shape fields; no weight estimate",
                     hf_model_id)
        return None

    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads)

    # q and o are full width; k and v are narrower under GQA (kv_heads < heads).
    attn_per_layer = (
        hidden * heads * head_dim          # q_proj
        + 2 * hidden * kv_heads * head_dim  # k_proj + v_proj
        + heads * head_dim * hidden        # o_proj
    )

    act = str(getattr(cfg, "hidden_act", "") or getattr(cfg, "hidden_activation", ""))
    n_mats = 3 if "glu" in act.lower() or "silu" in act.lower() else 2
    mlp_per_layer = n_mats * hidden * intermediate

    params = layers * (attn_per_layer + mlp_per_layer) + vocab * hidden
    if not getattr(cfg, "tie_word_embeddings", False):
        params += vocab * hidden  # separate lm_head

    total = params * _dtype_bytes(dtype, quantization)
    return round(total / _BYTES_PER_GIB / max(1, tensor_parallel_size), 2)


def estimate_kv_gib(
    hf_model_id: str,
    *,
    seq_len: int | None = None,
    max_model_len: int | None = None,
    dtype: str | None = None,
    tensor_parallel_size: int = 1,
    batch: int = 1,
) -> float | None:
    """Per-GPU KV-cache need, in GiB.

    Computed as::

        2 (K and V) x layers x kv_heads x head_dim x seq_len x batch x bytes

    A sequence's length is input plus output tokens, and ``max_tokens``
    bounds only the output. ``batch=1`` tells whether the model can run at
    all; the pipeline's chunk size tells whether it can run that batch.

    Args:
        hf_model_id: Checkpoint whose config to read.
        seq_len: Tokens held per sequence (input + output). Falls back to
            ``max_model_len``, then the config's ``max_position_embeddings``.
        max_model_len: The engine's configured context length. ``seq_len``
            is capped at it.
        dtype: KV dtype. ``None`` assumes 2 bytes per value.
        tensor_parallel_size: The KV cache is split across this many GPUs.
        batch: Concurrent sequences held at once.

    Returns:
        GiB per GPU, or ``None`` when the config cannot be read or lacks the
        shape fields.
    """
    cfg = _load_config(hf_model_id)
    if cfg is None:
        return None

    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_attention_heads", None)
    # Under GQA/MQA only the KV heads are cached. Falling back to all heads
    # errs high.
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = getattr(cfg, "head_dim", None) or (
        hidden // heads if hidden and heads else None
    )

    tokens = seq_len or max_model_len or getattr(cfg, "max_position_embeddings", None)
    if tokens and max_model_len:
        tokens = min(tokens, max_model_len)
    if not all((layers, kv_heads, head_dim, tokens)):
        return None

    kv_bytes = (
        2 * layers * kv_heads * head_dim * tokens * max(1, batch)
        * _dtype_bytes(dtype)
    )
    return round(kv_bytes / _BYTES_PER_GIB / max(1, tensor_parallel_size), 2)


def planning_gib(
    weights_gib: float | None,
    kv_gib: float | None,
    *,
    fragmentation: float = DEFAULT_FRAGMENTATION,
    cuda_context_gib: float = DEFAULT_CUDA_CONTEXT_GIB,
) -> float | None:
    """The footprint a planner should use: weights and KV plus headroom.

    Computed as ``(weights + kv) * fragmentation + cuda_context`` and rounded
    up to the next 0.5 GiB.

    Args:
        weights_gib: From :func:`estimate_weights_gib`. ``None`` returns
            ``None``.
        kv_gib: From :func:`estimate_kv_gib`. ``None`` counts as 0.
        fragmentation: Multiplier for activations and fragmentation.
        cuda_context_gib: Fixed per-process overhead, added after the
            multiplier.

    Returns:
        GiB per GPU, or ``None`` when ``weights_gib`` is ``None``.
    """
    if weights_gib is None:
        return None
    total = (weights_gib + (kv_gib or 0.0)) * fragmentation + cuda_context_gib
    import math

    return round(math.ceil(total * 2) / 2, 2)
