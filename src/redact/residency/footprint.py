"""One local model's expected footprint and the engine it loads on. Sizes are GiB."""

from dataclasses import dataclass, field

from redact.llms.backends import resolve_setup
from redact.llms.model_config import get_model_config
from redact.llms.resources import estimate

# The two constants below are shared by placement.py and explain.py.

#: Share of a card the plan hands out once a vLLM engine is on it. The rest
#: is headroom: allocator fragmentation and each process's CUDA context.
_CARD_CEILING = 0.9

#: ``device_map`` values that let transformers split one model across cards.
_SPANNING_DEVICE_MAPS = frozenset({"auto", "balanced", "balanced_low_0", "sequential"})

_LOCAL_SETUPS = ("vllm", "introspect")


@dataclass
class ModelFootprint:
    """What one local model is expected to need."""

    model: str
    setup: str
    hf_model_id: str
    #: The padded need rounded up to 0.5 GiB, summed over the model's cards.
    gb: float | None
    min_gpus: int
    source: str  # "declared" | "estimated" | "unknown"
    #: Engine settings (max_model_len, device_map), the per-GPU weights and
    #: KV terms, the attention heads and the one-line description.
    parts: dict = field(default_factory=dict)
    #: The load this model uses. Models with the same value share one engine.
    engine: tuple = ()
    #: Cards the plan places the model on. Advisory: nothing applies it at
    #: load time.
    devices: tuple[int, ...] = ()
    #: GiB reserved per card: a vLLM engine's grant, or a transformers
    #: model's largest claim on one card. ``None`` when the card size is
    #: unknown or the model is on none of the cards.
    reserved_gb: float | None = None
    #: The ``gpu_memory_utilization`` planned for a vLLM engine. ``None`` for
    #: a transformers model, when the card size is unknown, or when the
    #: engine is left no memory.
    planned_utilization: float | None = None

    def __post_init__(self) -> None:
        if not self.engine:
            self.engine = (self.setup, self.hf_model_id)

    @property
    def known(self) -> bool:
        return self.gb is not None

    @property
    def per_gpu_gb(self) -> float | None:
        """The footprint on each of the model's cards."""
        return self.gb / self.min_gpus if self.known else None

    @property
    def need_gb(self) -> float | None:
        """Weights plus KV per card, without headroom."""
        if not self.known:
            return None
        return self.parts["weights_gib"] + (self.parts.get("kv_gib") or 0.0)

    @property
    def padded_gb(self) -> float | None:
        """The padded need per card (:func:`estimate.padded_gib`)."""
        return estimate.padded_gib(self.need_gb) if self.known else None

    def breakdown(self) -> str:
        """One-line derivation of the footprint, e.g.
        ``estimated: 43.9 weights + 5.00 KV, x1.1 +0.6 [bfloat16]``.

        The source says where the weights come from.
        """
        if self.known:
            return f"{self.source}: {self.parts['description']}"
        return self.source

    def as_dict(self) -> dict:
        """The footprint as plain data, for the trace and the run summary."""
        return {
            "model": self.model,
            "setup": self.setup,
            "hf_model_id": self.hf_model_id,
            "gib": self.gb,
            "source": self.source,
            "derivation": self.breakdown(),
            "min_gpus": self.min_gpus,
            "devices": list(self.devices),
            "reserved_gib_per_device": self.reserved_gb,
            "need_gib_per_device": self.need_gb,
            "padded_gib_per_device": self.padded_gb,
            "gpu_memory_utilization": self.planned_utilization,
        }


def _local_setup(model: str, backend_type: str | None = None):
    """``(config, setup)`` when the model resolves to a local setup, else ``None``."""
    config = get_model_config(model)
    setup = resolve_setup(config, backend_type)
    return (config, setup) if setup in _LOCAL_SETUPS else None


def footprint(model: str, backend_type: str | None = None) -> ModelFootprint | None:
    """What one model is expected to need, without loading anything.

    The weights are the setup's declared ``vram_gb`` if set, else an estimate
    from the checkpoint's HF config (:mod:`~redact.llms.resources.estimate`),
    else unknown: planning continues and the plan says so. For vLLM the KV
    cache is always estimated, on top of either.

    Args:
        backend_type: Optional setup override, e.g. ``"vllm"``.

    Returns:
        The footprint, or ``None`` when the resolved setup is not a local
        one.
    """
    found = _local_setup(model, backend_type)
    if found is None:
        return None
    config, setup = found
    local = getattr(config, setup)
    if setup == "vllm":
        kw = local.vllm_kwargs or {}
        span, max_len = local.min_gpus, kw.get("max_model_len")
        # vLLM's default dtype loads a float32 checkpoint as 16-bit, so that is
        # its real size. An explicit dtype=float32 is still sized at 32-bit.
        vllm_sizing = dict(dtype=kw.get("dtype"), max_model_len=max_len,
                           tensor_parallel_size=span, float32_as_half=True)
        parts = {"max_model_len": max_len}
    else:
        # A span of 1: the device_map decides the cards transformers uses.
        span, vllm_sizing = 1, None
        parts = {"max_model_len": None, "device_map": local.device_map}
    common = dict(model=model, setup=setup, hf_model_id=local.hf_model_id,
                  min_gpus=span, engine=_engine_id(local, setup))

    if local.vram_gb is not None:
        weights = local.vram_gb / span
        kv = (estimate.estimate_kv_gib(local.hf_model_id, **vllm_sizing)
              if vllm_sizing else None)
        terms = f"{weights:.1f} weights"
        if kv is not None:
            terms += f" + {kv:.2f} KV"
        text = (f"{terms}, x{estimate.DEFAULT_FRAGMENTATION} "
                f"+{estimate.DEFAULT_CUDA_CONTEXT_GIB}")
        if span > 1:
            text += f", per GPU of {span}"
        parts.update(weights_gib=weights, kv_gib=kv, description=text)
        return ModelFootprint(gb=estimate.planning_gib(weights, kv) * span,
                              source="declared", parts=parts, **common)

    if vllm_sizing:
        est = estimate.estimate(
            local.hf_model_id, quantization=local.quantization, **vllm_sizing)
    else:
        # No KV term: transformers does not preallocate one, and generation's
        # cache is small next to the weights.
        est = estimate.estimate(
            local.hf_model_id, dtype=local.torch_dtype, kv=False)
    if est is None:
        return ModelFootprint(gb=None, source="unknown", parts=parts, **common)
    parts.update(weights_gib=est.weights_gib, kv_gib=est.kv_gib,
                 attention_heads=est.attention_heads,
                 description=est.description)
    # gb is a total over the model's cards.
    return ModelFootprint(gb=est.total_gib, source="estimated", parts=parts,
                          **common)


def _engine_id(local, setup: str) -> tuple:
    """The load a local setup uses: the setup, then its backend's cache key.

    Two setups with the same id share one engine; anything else is a second
    load, even on the same checkpoint.
    """
    if setup == "introspect":
        return (setup, local.hf_model_id, local.device_map, local.torch_dtype,
                repr(sorted(local.load_kwargs.items())))
    return (setup, local.hf_model_id, local.quantization,
            repr(sorted(local.engine_kwargs.items())))
