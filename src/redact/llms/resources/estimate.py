"""Estimate what a local model needs on the GPU from its HF config.

Only the config is read (shapes, no weights), so a footprint is available
before anything is loaded. Measuring is no substitute for vLLM: it
preallocates ``gpu_memory_utilization x card`` at load, so memory measured
after a load describes that setting and not the model.

Where the config leaves a choice open, the larger figure is taken: an
estimate that is too high costs a GPU, one that is too low ends in an OOM.

All sizes are GiB (bytes / 1024**3).
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3

_DTYPE_BYTES = {
    "float64": 8, "double": 8,
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
}

#: Weight bit width per vLLM quantization method. Any other method is sized
#: unquantized, which errs high.
_QUANT_BITS = {"awq": 4, "gptq": 4, "fp8": 8, "fp8_per_tensor": 8}

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

#: Context length of a vLLM engine whose setup sets no ``max_model_len``.
DEFAULT_MAX_MODEL_LEN = 10_000

#: Fields of ``config.json`` that name the checkpoint's longest context, in
#: the order they are read.
_CONTEXT_FIELDS = ("max_position_embeddings", "max_sequence_length", "seq_length",
                   "n_positions", "model_max_length")

#: ``gpu_memory_utilization`` of a vLLM engine loaded without a plan, and the
#: share of a card the planner hands out. The rest is headroom: allocator
#: fragmentation and each process's CUDA context.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9

#: Default for ``config=``: the caller has not read the config yet.
_UNREAD = object()

_reported: set[tuple[str, str]] = set()


def _report_once(hf_model_id: str, reason: str) -> None:
    """Log at INFO why a checkpoint has no estimate, once per process."""
    if (hf_model_id, reason) not in _reported:
        _reported.add((hf_model_id, reason))
        logger.info("No VRAM estimate for %s: %s", hf_model_id, reason)


def _load_config(hf_model_id: str, revision: str | None = None):
    """The checkpoint's HF config, or ``None`` if it cannot be read. Never raises.

    ``hf_model_id`` may be a folder holding a ``config.json``, as a packaged
    one (``VLLMConfig.hf_config``) is. ``revision`` reads the config of a
    pinned snapshot (``vllm_kwargs["revision"]``) instead of the default
    branch's.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        _report_once(hf_model_id, "transformers is not installed")
        return None
    try:
        pinned = {"revision": revision} if revision else {}
        return AutoConfig.from_pretrained(hf_model_id, trust_remote_code=False,
                                          **pinned)
    except Exception as exc:  # noqa: BLE001 — a planning hint must not fail a load
        _report_once(hf_model_id,
                     f"its HF config could not be read ({type(exc).__name__}: {exc})")
        return None


def _read_config(hf_model_id: str, revision: str | None):
    """:func:`_load_config`, with ``revision`` passed only when one is set."""
    if revision:
        return _load_config(hf_model_id, revision=revision)
    return _load_config(hf_model_id)


def _field(cfg, name: str, default=None):
    """``getattr(cfg, name, default)`` that never raises.

    A transformers config whose attribute varies per layer (a heterogeneous
    config, as Gemma 4's under transformers 5.17) raises
    ``AmbiguousGlobalPerLayerAttributeError`` for the global attribute. That
    is a ``RuntimeError``, which ``getattr``'s default does not catch; here
    it reads as unset, and the per-layer values are read instead
    (:func:`_layer_attention`).
    """
    try:
        return getattr(cfg, name, default)
    except Exception:  # noqa: BLE001 — a planning hint must not fail a load
        return default


def _language_config(full):
    """The config holding the language model's shapes.

    Multimodal configs nest it as ``text_config``. Their vision tower and
    projector are not counted.
    """
    return _field(full, "text_config") or full


def _layer_attention(cfg, layers: int | None) -> list[tuple[int, int]] | None:
    """Each layer's ``(kv_heads, head_dim)`` when they differ by layer, read
    without touching an attribute that raises; ``None`` for a config with
    one attention shape for every layer, or when they cannot be read.

    Two forms are read: transformers' per-layer configs
    (``per_layer_config[i].num_key_value_heads`` and ``.head_dim``), and a
    config listing ``layer_types`` with the full-attention layers' own
    ``num_global_key_value_heads`` or ``global_head_dim`` beside the other
    layers' ``num_key_value_heads`` and ``head_dim``. A field of the two left
    out is the other layers': Gemma 4's ``config.json`` sets
    ``global_head_dim`` and ``attention_k_eq_v`` but no
    ``num_global_key_value_heads``, so its full-attention layers count every
    KV head, not the one transformers gives them under
    ``attention_k_eq_v``, which errs high (3.5 GiB of KV for the humanizer
    at 8,192 tokens, against 2.62 from ``per_layer_config``).
    """
    per_layer = _field(cfg, "per_layer_config")
    if per_layer:
        try:
            shapes = [(_field(layer, "num_key_value_heads"), _field(layer, "head_dim"))
                      for layer in per_layer]
        except TypeError:       # not a list of layer configs
            shapes = []
        if shapes and all(isinstance(kv, int) and isinstance(dim, int)
                          and kv > 0 and dim > 0 for kv, dim in shapes):
            return shapes
    types = _field(cfg, "layer_types")
    global_kv = _field(cfg, "num_global_key_value_heads")
    global_dim = _field(cfg, "global_head_dim")
    if not types or not (global_kv or global_dim):
        return None
    try:
        types = list(types)
    except TypeError:
        return None
    kv, dim = _field(cfg, "num_key_value_heads"), _field(cfg, "head_dim")
    if not (kv and dim) or (layers and len(types) != layers):
        return None
    return [((global_kv or kv), (global_dim or dim)) if kind == "full_attention"
            else (kv, dim) for kind in types]


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
        names += [_field(cfg, "torch_dtype", None), _field(cfg, "dtype", None)]
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
    qc = _field(full, "quantization_config", None)
    bits = qc.get("bits") if isinstance(qc, dict) else _field(qc, "bits", None)
    if isinstance(bits, bool) or not isinstance(bits, int):
        return usual
    return bits if 0 < bits <= _MAX_QUANT_BITS else usual


def _expert_count(cfg) -> int:
    """MLP experts per layer of an MoE config, shared ones included; else 0."""
    for name in ("num_local_experts", "num_experts", "n_routed_experts"):
        experts = _field(cfg, name)
        if experts:
            return experts + (_field(cfg, "n_shared_experts", None) or 0)
    return 0


def estimate_weights_gib(
    hf_model_id: str,
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    tensor_parallel_size: int | None = 1,
    float32_as_half: bool = False,
    config=_UNREAD,
    revision: str | None = None,
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
        revision: The snapshot whose config to read (a setup's pinned
            ``vllm_kwargs["revision"]``); ``None`` reads the default branch.

    Returns:
        GiB per GPU, or ``None`` when the config cannot be read or lacks the
        shape fields.
    """
    full = _read_config(hf_model_id, revision) if config is _UNREAD else config
    if full is None:
        return None
    cfg = _language_config(full)

    layers = _field(cfg, "num_hidden_layers")
    hidden = _field(cfg, "hidden_size", None)
    vocab = _field(cfg, "vocab_size", None)
    intermediate = _field(cfg, "intermediate_size", None)
    heads = _field(cfg, "num_attention_heads", None)
    if not all((layers, hidden, vocab, intermediate, heads)):
        _report_once(hf_model_id, "its config lacks the shape fields")
        return None

    # One attention shape per layer when it varies (Gemma 4), else the
    # config's; all heads at hidden / heads when neither can be read (errs
    # high).
    shapes = _layer_attention(cfg, layers) or [(
        _field(cfg, "num_key_value_heads") or heads,
        _field(cfg, "head_dim") or (hidden // heads))] * layers

    # q and o are full width; k and v are narrower under GQA (kv_heads < heads).
    attention = sum(
        hidden * heads * head_dim          # q_proj
        + 2 * hidden * kv_heads * head_dim  # k_proj + v_proj
        + heads * head_dim * hidden        # o_proj
        for kv_heads, head_dim in shapes)

    n_mats = 2 if _field(cfg, "model_type", None) in _UNGATED_MODEL_TYPES else 3
    experts = _expert_count(cfg)
    if experts:
        width = _field(cfg, "moe_intermediate_size", None) or intermediate
        # Each expert is a full MLP; the router adds hidden x experts.
        mlp_per_layer = experts * n_mats * hidden * width + hidden * experts
        # Qwen2-MoE: one always-on shared expert per layer, of its own width.
        shared = _field(cfg, "shared_expert_intermediate_size", None)
        if shared:
            mlp_per_layer += n_mats * hidden * shared
    else:
        mlp_per_layer = n_mats * hidden * intermediate

    linear = attention + layers * mlp_per_layer
    embed = vocab * hidden
    if not _field(cfg, "tie_word_embeddings", False):
        embed *= 2  # separate lm_head

    dtype_bytes = _resolve_dtype(dtype, full, float32_as_half)[1]
    bits = _quant_bits(quantization, full)
    linear_bytes = dtype_bytes
    if bits:
        linear_bytes = min(dtype_bytes, (bits + _QUANT_OVERHEAD_BITS) / 8)

    total = linear * linear_bytes + embed * dtype_bytes
    return round(total / _BYTES_PER_GIB / max(1, tensor_parallel_size or 1), 2)


def _config_json(hf_model_id: str) -> dict | None:
    """The checkpoint's ``config.json`` as plain data, from its directory or
    the hub cache: never downloaded, and no code of the checkpoint runs.
    ``None`` when it is in neither or is not a JSON object."""
    path = Path(hf_model_id) / "config.json"
    if not path.is_file():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(hf_model_id, "config.json")
        except Exception:  # noqa: BLE001 — not installed, or an id the hub rejects
            return None
        if not isinstance(cached, str):
            return None
        path = Path(cached)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _context_limit(hf_model_id: str, full) -> int | None:
    """The longest context the checkpoint takes, or ``None`` when unknown.

    The config's ``max_position_embeddings``, else the first of
    :data:`_CONTEXT_FIELDS` that ``config.json`` sets, the language model's
    (``text_config``) before the outer ones. The file is read for a config
    that needs remote code, or that names its context otherwise.
    """
    limit = _field(_language_config(full), "max_position_embeddings")
    if limit:
        return limit
    data = _config_json(hf_model_id) or {}
    for fields in (data.get("text_config"), data):
        for key in _CONTEXT_FIELDS if isinstance(fields, dict) else ():
            value = fields.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def effective_max_model_len(
    hf_model_id: str, max_model_len: int | None = None, config=_UNREAD,
    revision: str | None = None,
) -> int:
    """The context length a vLLM engine runs with.

    An explicit ``max_model_len`` wins. Otherwise it is
    :data:`DEFAULT_MAX_MODEL_LEN`, capped at the checkpoint's longest
    context when that is known (:func:`_context_limit`).
    """
    if max_model_len:
        return max_model_len
    full = _read_config(hf_model_id, revision) if config is _UNREAD else config
    limit = _context_limit(hf_model_id, full)
    return min(DEFAULT_MAX_MODEL_LEN, limit) if limit else DEFAULT_MAX_MODEL_LEN


def estimate_kv_gib(
    hf_model_id: str,
    *,
    max_model_len: int | None = None,
    dtype: str | None = None,
    tensor_parallel_size: int | None = 1,
    float32_as_half: bool = False,
    config=_UNREAD,
    revision: str | None = None,
) -> float | None:
    """Per-GPU KV-cache need, in GiB.

    Computed as::

        2 (K and V) x layers x kv_heads x head_dim x context length x bytes

    summed layer by layer when the attention shape varies by layer
    (:func:`_layer_attention`), each layer at the full context length.

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
        revision: As in :func:`estimate_weights_gib`.

    Returns:
        GiB per GPU, or ``None`` when the config cannot be read or lacks the
        shape fields.
    """
    full = _read_config(hf_model_id, revision) if config is _UNREAD else config
    if full is None:
        return None
    cfg = _language_config(full)

    layers = _field(cfg, "num_hidden_layers")
    shapes = _layer_attention(cfg, layers)
    if shapes is None:
        heads = _field(cfg, "num_attention_heads", None)
        # Under GQA/MQA only the KV heads are cached. Falling back to all
        # heads errs high.
        kv_heads = _field(cfg, "num_key_value_heads", None) or heads
        hidden = _field(cfg, "hidden_size", None)
        head_dim = _field(cfg, "head_dim", None) or (
            hidden // heads if hidden and heads else None
        )
        if not all((layers, kv_heads, head_dim)):
            return None
        shapes = [(kv_heads, head_dim)] * layers
    tokens = effective_max_model_len(hf_model_id, max_model_len, config=full)

    # Every layer at the full context, sliding-window ones too: vLLM 0.31
    # allocates so (Gemma 4: 0.33 MiB a token measured, 40 sliding layers at
    # 8 x 256 and 8 full ones at 1 x 512).
    kv_bytes = (
        2 * sum(kv_heads * head_dim for kv_heads, head_dim in shapes) * tokens
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
    revision: str | None = None,
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
    full = _read_config(hf_model_id, revision)
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
    heads = _field(_language_config(full), "num_attention_heads")
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
