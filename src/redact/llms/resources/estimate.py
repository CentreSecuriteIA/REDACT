"""What a local model will need on the GPU, computed rather than measured.

Everything here reads the checkpoint's HF **config** — shapes only, no weights
downloaded and nothing loaded — so a footprint is available before the first
run, on any machine, with no cache to go stale.

**Why estimated and not measured.** The obvious alternative is to load once and
record the delta. It does not work for the backend that matters:

- vLLM preallocates ``gpu_memory_utilization x card`` for weights *and* KV at
  load, so a before/after delta on the card returns that fraction regardless of
  model size. It describes the setting, not the model.
- Torch's own allocator would separate the weights out, but vLLM V1 runs its
  engine core in a **separate process**, so the parent's allocator never sees
  them.
- vLLM does know the split internally, but only through effectively-private
  API that moves between versions.

Meanwhile the number is computable: parameters x dtype bytes, from the same
config this module already reads for KV. So the split is:

===================  ======================================================
``weights_gib``      fixed once a checkpoint and dtype are chosen
``kv_gib``           scales with sequence length x batch — the variable part
``planning_gib``     the two plus **explicit** headroom
===================  ======================================================

Everything is **GiB** (bytes / 1024**3), which is what ``nvidia-smi`` and GPU
spec sheets mean by "GB". The old code mixed decimal GB here with binary GiB in
``telemetry.detect_gpu_memory_gb``, so the same 48GiB card read as 48.0 or 44.7
depending on whether torch imported — a 7% swing in planning capacity.

These are estimates with stated assumptions, not measurements. Every constant
is a named, overridable argument for exactly that reason.
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

#: Multiplier over (weights + KV) covering activations and allocator
#: fragmentation. A guess, deliberately modest: activations for a decode-only
#: workload are small next to weights, and vLLM's own pool absorbs most
#: fragmentation. Raise it if loads OOM just above the estimate.
DEFAULT_FRAGMENTATION = 1.1

#: Fixed per-process CUDA context — driver, kernels, cuBLAS workspaces. Real
#: and invariant to model size, so it belongs outside the multiplier.
DEFAULT_CUDA_CONTEXT_GIB = 0.6


def _load_config(hf_model_id: str):
    """The checkpoint's HF config, or ``None`` if it can't be read.

    Never raises: an estimate is a planning hint, and failing to produce one
    must degrade to "unknown", never to a wrong number or a failed run.
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
    return 2  # bf16/fp16, what vLLM uses unless told otherwise


def estimate_weights_gib(
    hf_model_id: str,
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    tensor_parallel_size: int = 1,
) -> float | None:
    """Per-GPU weight footprint, in GiB.

    Parameter count is derived from the config's **shapes** rather than read
    from a ``num_parameters`` field, which most configs don't carry. The three
    terms that matter for a decoder-only transformer::

        embeddings  = vocab x hidden           (x2 when lm_head is untied)
        attention   = layers x hidden x (q_proj + k_proj + v_proj + o_proj)
        mlp         = layers x hidden x intermediate x n_mats

    ``n_mats`` is 3 for a gated MLP (gate/up/down — Llama, Mistral, Qwen) and 2
    otherwise; gated is detected from the config's activation function, since
    getting it wrong is a third of the MLP and the MLP dominates the count.

    MoE checkpoints are **not** handled: total parameters there are far larger
    than active ones, and which figure applies depends on expert placement.
    Such a model returns the dense estimate, which understates it — pin
    ``vram_gb`` explicitly for those.

    Args:
        hf_model_id: Checkpoint whose config to read.
        dtype: Weight dtype name; ``None`` assumes 2 bytes.
        quantization: Overrides ``dtype`` when set (``"awq"``/``"gptq"`` ->
            ~0.5 bytes/param, ``"int8"`` -> 1).
        tensor_parallel_size: Weights are sharded across TP ranks, so the
            per-GPU need is the total divided by this.

    Returns:
        GiB per GPU, or ``None`` when the config can't be read or lacks the
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

    # q and o are full-width; k and v are narrowed by GQA (kv_heads < heads),
    # which on a modern checkpoint is a several-fold saving worth modelling.
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

    The variable half of a local footprint. Weights are fixed once a checkpoint
    is chosen; KV scales with how much context you actually hold::

        2 (K and V) x layers x kv_heads x head_dim x seq_len x batch x bytes

    **A sequence's length is input + output**, not the completion budget.
    ``max_tokens`` only ever bounds the output, and nothing in this library
    bounds the prompt — so an obfuscated jailbreak (ASCII art, base64, stacked
    few-shot blocks) can be thousands of tokens of *input*. Pass a realistic
    ``seq_len`` when you know one; the default falls back to the engine's
    configured ``max_model_len``, which is the feasibility floor: the point
    below which even a single request cannot be served.

    ``batch=1`` therefore answers "can this run at all", and ``batch=N``
    answers "can it run the configured chunk size" — which is the question that
    predicts an OOM mid-run rather than at load.

    Args:
        hf_model_id: Checkpoint whose config to read.
        seq_len: Tokens held per sequence (input + output). Falls back to
            ``max_model_len``, then the model's ``max_position_embeddings``.
        max_model_len: The engine's configured context length, used as the
            fallback and as the cap — ``seq_len`` above it is clamped, since
            the engine would reject such a request anyway.
        dtype: KV dtype; ``None`` assumes 2 bytes.
        tensor_parallel_size: KV is sharded across TP ranks.
        batch: Concurrent sequences held at once.

    Returns:
        GiB per GPU, or ``None`` when the config can't be read or lacks the
        shape fields.
    """
    cfg = _load_config(hf_model_id)
    if cfg is None:
        return None

    layers = getattr(cfg, "num_hidden_layers", None)
    heads = getattr(cfg, "num_attention_heads", None)
    # GQA/MQA: only the KV heads are cached. Falling back to
    # num_attention_heads (i.e. MHA) is the conservative direction.
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
    """The number a planner should use: the parts, plus explicit headroom.

    ``weights + kv`` is a **lower bound no model can actually run in** —
    activations, allocator fragmentation and the CUDA context all sit on top.
    Previously that slack was implicit (vLLM's ``gpu_memory_utilization``
    reserving a card fraction, and hand-rounded ``vram_gb`` declarations), which
    meant a caller reading the number had to know it was reading a floor. Here
    it is a term you can see and change.

    Rounds **up** to the next 0.5 GiB: the inputs are estimates, so false
    precision would be misleading, and erring high is the safe direction.

    Args:
        weights_gib: From :func:`estimate_weights_gib`. ``None`` -> ``None``,
            since headroom over an unknown base is meaningless.
        kv_gib: From :func:`estimate_kv_gib`. ``None`` is treated as 0 — a
            weights-only figure is still a useful lower bound.
        fragmentation: Multiplier for activations + fragmentation.
        cuda_context_gib: Fixed per-process driver/kernel overhead, added
            outside the multiplier since it does not scale with the model.

    Returns:
        GiB per GPU, or ``None`` when the weights estimate is unavailable.
    """
    if weights_gib is None:
        return None
    total = (weights_gib + (kv_gib or 0.0)) * fragmentation + cuda_context_gib
    import math

    return round(math.ceil(total * 2) / 2, 2)
