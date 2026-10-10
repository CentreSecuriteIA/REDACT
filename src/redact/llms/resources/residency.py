"""Plan whether a run's local models fit on this machine.

Works out each local model's footprint, places the models on the cards, and
produces a plan with a readable explanation, before anything is loaded.

The plan describes what a run does:

- Nothing is unloaded between stages, so every local model of a run is
  resident at once. Models that cannot be resident together do not fit.
- No device is chosen at load time, so a single-GPU vLLM engine always loads
  on card 0 and a tensor-parallel one on cards 0..n-1.
- A vLLM engine is granted its padded need plus a share of its card's spare
  memory, and loads with that grant as its ``gpu_memory_utilization``.

Footprints are estimates, and all sizes are GiB. The plan is a warning ahead
of the run, not a guarantee against running out of memory.
"""

#TODO(driver script): schedule residency per stage instead of per run.
# Every local model of a run is resident at once, and a plan that needs more
# is reported as not fitting. To move into the driver:
#   - Stage-aware residency: keep a model resident only from the first to the
#     last stage that uses it, unload it after its last use, and preload the
#     next stage's models during the previous stage.
#   - Apply the planned card assignment (ModelFootprint.devices) when an
#     engine loads. Until then single-GPU vLLM engines all land on card 0.
#   - Confirm on a GPU that unload_local() returns a vLLM engine's memory.
#   - Recompute the plan at each stage transition. Until then a share lasts
#     until the next apply_plan().
#   - Replicas of one model across cards for throughput.
#   - Join or cancel the preload thread at exit.
#   - Paraphrase: plan its check model and the engines already resident.
#   - Free memory per card and CUDA_VISIBLE_DEVICES; mixed card sizes.
#   - Bill local cost on the union of engine lifetimes.
#   - A plan does not see engines already loaded, by this process or another.
#     The runner should own this; one process (container) per run is a candidate.

import logging
import math
import re
import threading
from dataclasses import dataclass, field

from .. import observe
from ..backends import resolve_setup
from ..model_config import get_model_config
from . import estimate, measure

logger = logging.getLogger(__name__)

#: Size used in the placement arithmetic for a transformers model whose
#: footprint is neither declared nor estimable. The plan reports it as unknown.
_UNKNOWN_GB = 0.0

#: Share of a card the plan hands out once a vLLM engine is on it. The rest
#: is headroom: allocator fragmentation and each process's CUDA context.
_CARD_CEILING = 0.9

#: Tensor-parallel sizes offered for a model too large for its cards.
_TENSOR_PARALLEL_SIZES = (2, 4, 8)

#: ``device_map`` values that let transformers split one model across cards.
_SPANNING_DEVICE_MAPS = frozenset({"auto", "balanced", "balanced_low_0", "sequential"})

_LOCAL_SETUPS = ("vllm", "introspect")

#: Slack for comparing GiB sums.
_EPS = 1e-6

_SEPARATE_RUNS = (
    "Run the stages that need different local models as separate "
    "run_pipeline invocations: process exit frees the card, and resume "
    "makes the rerun cheap."
)


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


class _Cards:
    """Memory per card while one group of models is placed."""

    def __init__(self, count: int, per_gpu_gb: float | None):
        self.per = per_gpu_gb
        #: GiB per card not held by transformers models, whose claims are
        #: fixed. ``None`` when the card size is unknown.
        self.free = [per_gpu_gb] * count if per_gpu_gb else None
        #: The vLLM engines on each card.
        self.shared: list[list[ModelFootprint]] = [[] for _ in self.free or ()]
        #: Cards a tensor-parallel engine holds whole.
        self.whole: set[int] = set()

    def _room(self, card: int, ceiling: bool = False) -> float:
        """GiB a new claim can take on a card."""
        if card in self.whole:
            # Negative, so a claim of 0 GiB (unknown size) cannot join either.
            return -math.inf
        room = self.free[card] - sum(fp.padded_gb or 0.0 for fp in self.shared[card])
        if ceiling or self.shared[card]:
            room -= (1 - _CARD_CEILING) * self.per
        return room

    def place(self, fp: ModelFootprint, force: bool = False) -> bool:
        """Claim the model's cards and record them on the footprint.

        Returns ``False``, claiming nothing, when the model does not fit
        beside what is already placed. ``force`` places it regardless, for a
        model that opens a group.
        """
        devices, reserve = self._claim(fp)
        vllm = fp.setup == "vllm"
        fits = self.free is None or all(
            d < len(self.free)
            # vLLM grants sharing a card stay under the ceiling together.
            and self._room(d, vllm) + _EPS >= reserve[d]
            # A tensor-parallel engine takes untouched cards only.
            and not (vllm and fp.min_gpus > 1
                     and (self.shared[d] or self.free[d] < self.per))
            for d in devices
        )
        if not (fits or force):
            return False
        fp.devices = devices
        if self.free is not None and devices:
            if not vllm:
                fp.reserved_gb = max(reserve.values())
            for d in devices:
                if d >= len(self.free):
                    continue
                if vllm:
                    self.shared[d].append(fp)
                else:
                    self.free[d] -= reserve[d]
                if vllm and fp.min_gpus > 1:
                    self.whole.add(d)
        return True

    def settle(self) -> None:
        """Grant each vLLM engine its padded need plus an equal share of
        the spare.

        Per card: ``pool = 0.9 x card - fixed claims`` and
        ``grant = padded need + (pool - sum of padded needs) / engines``,
        never below 0. A grant below its padded need makes the model oversized.
        """
        for card, engines in enumerate(self.shared):
            if not engines:
                continue
            pool = max(0.0, self.free[card] - (1 - _CARD_CEILING) * self.per)
            spare = pool - sum(fp.padded_gb or 0.0 for fp in engines)
            for fp in engines:
                padded = fp.padded_gb or 0.0
                fp.reserved_gb = max(0.0, padded + spare / len(engines))
                # Rounded down, to keep a card's fractions under the ceiling.
                fraction = math.floor(fp.reserved_gb / self.per * 1e4 + 1e-6) / 1e4
                if fp.reserved_gb + _EPS >= padded:
                    # Rounded up to cover the need, even where that takes
                    # the card's fractions a rounding step over the ceiling.
                    fraction = max(
                        fraction, math.ceil(padded / self.per * 1e4 - 1e-6) / 1e4)
                # vLLM rejects a fraction that is not positive.
                fp.planned_utilization = fraction if fraction > 0 else None

    def _claim(self, fp: ModelFootprint) -> tuple[tuple[int, ...], dict[int, float]]:
        """The cards a model would take and the GiB it claims on each."""
        if fp.setup == "vllm":
            # No device is chosen at load time, so vLLM takes the first cards.
            devices = tuple(range(fp.min_gpus))
            return devices, dict.fromkeys(devices, fp.padded_gb or 0.0)

        need = fp.gb or _UNKNOWN_GB
        device_map = str(fp.parts.get("device_map", "auto"))
        if device_map not in _SPANNING_DEVICE_MAPS:
            if device_map == "cpu":
                return (), {}
            index = re.search(r"(\d+)$", device_map)
            card = int(index.group(1)) if index else 0
            return (card,), {card: need}
        if self.free is None:
            return tuple(range(fp.min_gpus)), {}
        # A spanning device_map may split the model, so it is counted
        # against the cards' combined free memory.
        reserve: dict[int, float] = {}
        left = need
        for card in range(len(self.free)):
            take = min(max(self._room(card), 0.0), left)
            if take > _EPS:
                reserve[card] = take
                left -= take
        if left > _EPS or not reserve:
            # More than the cards hold: charge the excess to the last one.
            last = max(reserve, default=0)
            reserve[last] = reserve.get(last, 0.0) + left
        return tuple(reserve), reserve


@dataclass
class ResidencyPlan:
    """Whether a run's local models fit on a machine, and where."""

    #: Models that can be resident together. More than one group means the
    #: run does not fit: each later group cannot load beside the earlier ones.
    groups: list[list[ModelFootprint]] = field(default_factory=list)
    gpus_required: int = 0
    gpus_available: int = 0
    gpu_name: str | None = None
    per_gpu_gb: float | None = None
    #: Models whose grant, or whose cards, do not cover their need.
    oversized: list[ModelFootprint] = field(default_factory=list)
    #: True when the caller gave the card count or the card size.
    machine_given: bool = False

    @property
    def footprints(self) -> list[ModelFootprint]:
        return [fp for group in self.groups for fp in group]

    @property
    def co_resident(self) -> bool:
        """Whether every model can be resident at the same time."""
        return len(self.groups) <= 1

    @property
    def sequential(self) -> bool:
        """True when the models only fit one group at a time.

        A caller that unloads between groups itself (paraphrase) reads this.
        A run does not: for it, see :attr:`fits`.
        """
        return len(self.groups) > 1

    @property
    def fits(self) -> bool:
        """Whether a run can hold all its local models at once: no :attr:`problems`."""
        return not self.problems

    @property
    def problems(self) -> list[str]:
        """Why the plan does not fit, one entry per cause."""
        out = []
        if self.gpus_required > max(self.gpus_available, 0):
            if self.gpus_available <= 0:
                out.append("no GPU is available, so the local models cannot "
                           "load on this machine.")
            else:
                out.append(f"the plan needs {self.gpus_required} GPUs but "
                           f"{self.gpus_available} are available.")
        if not self.co_resident:
            out.append(self._co_residency_problem())
        out.extend("\n".join(self._oversized_advice(fp)) for fp in self.oversized)
        vllm = {fp.engine: fp for fp in self.footprints if fp.setup == "vllm"}
        for fp in vllm.values():
            heads = fp.parts.get("attention_heads")
            if fp.min_gpus > 1 and heads and heads % fp.min_gpus:
                out.append(
                    f"{fp.model}[vllm] has {heads} attention heads, which "
                    f"tensor_parallel_size={fp.min_gpus} does not divide, so "
                    f"vLLM cannot load it.")
        unsized = [fp.model for fp in vllm.values() if not fp.known]
        if unsized and len({fp.engine for fp in self.footprints}) > 1:
            out.append(
                f"{', '.join(unsized)} has no known size, so it cannot be "
                "planned beside other models: declare its vram_gb.")
        if not self.per_gpu_gb and self.gpus_available > 0 and len(vllm) > 1:
            out.append(
                "the card size is unknown, so no gpu_memory_utilization is "
                f"planned for the {len(vllm)} vLLM engines: the first takes "
                "vLLM's default share of the card and the next cannot load.")
        return out

    @property
    def warnings(self) -> list[str]:
        """What may still fail: an unchecked plan or model, or a KV cache not sized."""
        out = [
            f"WARNING: {fp.model} has no known size, so whether it fits was "
            "not checked."
            for fp in {fp.engine: fp for fp in self.footprints}.values()
            if not fp.known
        ]
        if not self.per_gpu_gb:
            if self.footprints and self.gpus_available > 0:
                out.append("WARNING: the card size is unknown, so placement "
                           "and memory grants were not checked.")
            return out
        seen = set()
        for fp in self.footprints:
            if (fp.setup != "vllm" or not fp.known or fp.engine in seen
                    or fp.parts.get("kv_gib") is not None):
                continue
            seen.add(fp.engine)
            out.append(
                f"WARNING: the KV cache of {fp.model} could not be estimated "
                f"(HF config unreadable or no context length), so its need "
                f"counts the weights only."
            )
        return out

    def as_dict(self) -> dict:
        """The plan as plain data, for the trace and the run summary."""
        group_of = {id(fp): i for i, g in enumerate(self.groups) for fp in g}
        return {
            "fits": self.fits,
            "co_resident": self.co_resident,
            "gpu": self.gpu_name,
            "gpus_available": self.gpus_available,
            "gpus_required": self.gpus_required,
            "per_gpu_gib": self.per_gpu_gb,
            "machine": "given" if self.machine_given else "detected",
            "groups": [[fp.model for fp in g] for g in self.groups],
            "models": [{**fp.as_dict(), "group": group_of[id(fp)]}
                       for fp in self.footprints],
            "oversized": [fp.model for fp in self.oversized],
            "problems": self.problems,
            "warnings": self.warnings,
        }

    def explain(self) -> str:
        """The plan as text: what loads on which cards, and what is wrong.

        Includes each model's derivation and grant, and the way out for a
        plan that does not fit.
        """
        if self.machine_given:
            gpu = f"planning for {self.gpus_available} GPU(s)"
        elif self.gpu_name:
            gpu = f"detected {self.gpus_available} x {self.gpu_name}"
        else:
            gpu = "no GPU detected"
        if self.per_gpu_gb:
            gpu += f", {self.per_gpu_gb:.1f}GiB each"
        lines = [f"plan needs {self.gpus_required} GPU(s); {gpu}"]

        for i, group in enumerate(self.groups, start=1):
            tail = "" if i == 1 else "  -> cannot be resident with the above"
            lines.append(f"  group {i}:{tail}")
            for fp in group:
                gb = f"~{fp.gb:.1f}GiB" if fp.known else "size unknown"
                lines.append(
                    f"    {fp.model}[{fp.setup}]  {gb}  ({fp.breakdown()})"
                    f"{self._placement(fp)}"
                )
                if fp.min_gpus > 1 and fp.known:
                    lines.append(
                        f"      -> {fp.min_gpus} GPUs "
                        f"(tensor_parallel_size={fp.min_gpus}), "
                        f"~{fp.per_gpu_gb:.1f}GiB each"
                    )

        lines.extend(f"  PROBLEM: {problem}" for problem in self.problems)
        lines.extend(f"  {warning}" for warning in self.warnings)

        on_card_0 = [fp for fp in self.footprints
                     if fp.setup == "vllm" and fp.min_gpus == 1]
        if on_card_0 and self.gpus_available > 1:
            lines.append(
                "  note: single-GPU vLLM engines all load on card 0, since "
                "nothing assigns a device at load time; the other card(s) "
                "do not add room for them"
            )
        seen = set()
        for fp in self.footprints:
            kv, length = fp.parts.get("kv_gib"), fp.parts.get("max_model_len")
            if not kv or kv <= fp.parts["weights_gib"] or fp.engine in seen:
                continue
            seen.add(fp.engine)
            at = (f"max_model_len={length}" if length else "the default context "
                  f"of up to {estimate.DEFAULT_MAX_MODEL_LEN} tokens")
            lines.append(
                f"  note: the KV cache of {fp.model} ({kv:.1f}GiB at {at}) is "
                f"larger than its weights ({fp.parts['weights_gib']:.1f}GiB); "
                f"lower max_model_len to shrink it"
            )
        unknown = [fp.model for fp in self.footprints if not fp.known]
        if unknown:
            lines.append(
                f"  note: no footprint for {', '.join(unknown)} — planned "
                f"optimistically; declare vram_gb to pin one"
            )
        return "\n".join(lines)

    def _placement(self, fp: ModelFootprint) -> str:
        """Where a model is planned and what it reserves there."""
        if not fp.devices or fp.reserved_gb is None:
            return ""
        cards = (f"card {fp.devices[0]}" if len(fp.devices) == 1
                 else f"cards {fp.devices[0]}-{fp.devices[-1]}")
        if fp.planned_utilization:
            each = " each" if len(fp.devices) > 1 else ""
            return (f"  -> {cards}, vLLM gets {fp.reserved_gb:.1f}GiB{each} "
                    f"(gpu_memory_utilization={fp.planned_utilization:.2f})")
        return f"  -> {cards}"

    def _co_residency_problem(self) -> str:
        """Why several groups do not fit a run, and what to do instead."""
        text = (
            "these groups cannot be resident together, and nothing is "
            "unloaded between stages, so a later group would load on top of "
            f"the earlier ones. {_SEPARATE_RUNS}"
        )
        engines = {fp.engine for fp in self.footprints
                   if fp.setup == "vllm" and fp.min_gpus == 1}
        if len(engines) > 1:
            text += (
                " vLLM engines share card 0 only when their needs with "
                f"overhead sum to at most {_CARD_CEILING} of it."
            )
        return text

    def _oversized_advice(self, fp: ModelFootprint) -> list[str]:
        """Advice for a model whose grant or cards do not cover its need.

        Lines after the first are indented for :meth:`explain`.
        """
        per = self.per_gpu_gb or 0
        if fp.setup != "vllm":
            pinned = str(fp.parts.get("device_map", "auto")) not in _SPANNING_DEVICE_MAPS
            cards = 1 if pinned else max(self.gpus_available, 1)
            return [
                f"{fp.model}[{fp.setup}] needs ~{fp.gb:.1f}GiB but "
                f"{'its card has' if pinned else 'the cards have'} "
                f"{per * cards:.1f}GiB.",
                "    alternatives: a 16-bit torch_dtype, a quantized "
                "checkpoint, or more GPU memory",
            ]
        if not fp.planned_utilization:
            return [f"{fp.model}[{fp.setup}] is left no memory on its card by "
                    f"the engines placed before it."]
        each = " per GPU" if fp.min_gpus > 1 else ""
        padded = fp.padded_gb or 0.0
        lines = [
            f"{fp.model}[{fp.setup}] needs {padded:.1f}GiB{each} with overhead "
            f"({fp.need_gb or 0.0:.1f} weights + KV, x{estimate.DEFAULT_FRAGMENTATION} "
            f"+{estimate.DEFAULT_CUDA_CONTEXT_GIB}) but "
            f"gpu_memory_utilization={fp.planned_utilization:.2f} grants "
            f"{fp.reserved_gb:.1f}GiB."
        ]
        kv = fp.parts.get("kv_gib")
        if kv and estimate.padded_gib(fp.need_gb - kv) <= fp.reserved_gb:
            lines.append(
                f"    the KV cache ({kv:.1f}GiB) is what does not fit: lower "
                f"max_model_len")
        if padded > _CARD_CEILING * per:
            lines.append(self._tensor_parallel_suggestion(fp, per))
        lines.append(
            "    alternatives: a lower max_model_len, a quantized checkpoint "
            "(~3x smaller at awq/gptq), or the .api setup"
        )
        return lines

    def _tensor_parallel_suggestion(self, fp: ModelFootprint, per: float) -> str:
        """Advice line: the smallest tensor-parallel size whose padded need
        per card fits under the ceiling, or that none does."""
        heads = fp.parts.get("attention_heads")
        for n in _TENSOR_PARALLEL_SIZES:
            # The size must divide the attention heads.
            if n <= fp.min_gpus or (heads and heads % n):
                continue
            share = estimate.padded_gib(fp.need_gb * fp.min_gpus / n)
            if share > _CARD_CEILING * per:
                continue
            more = "" if n <= self.gpus_available else "  (more GPUs than detected)"
            return (
                f"    suggested: min_gpus={n} (tensor_parallel_size={n})"
                f" -> {share:.1f}GiB per GPU with overhead{more}"
            )
        return (
            f"    no tensor-parallel size up to {_TENSOR_PARALLEL_SIZES[-1]} "
            f"fits a {per:.1f}GiB device"
        )


def plan_residency(
    models: list[str],
    backend_types: dict | None = None,
    *,
    gpus: int | None = None,
    gib_per_gpu: float | None = None,
) -> ResidencyPlan:
    """Place a run's local models on the machine's cards.

    Models are placed in the order given, each on the first cards its
    runtime would use. One that does not fit beside those already placed
    starts a new group, and more than one group means the run does not fit
    (:attr:`ResidencyPlan.fits`).

    Args:
        models: Model names. Unregistered and non-local ones are ignored.
        backend_types: Optional per-model setup override, as passed to
            ``ModelClient.create``.
        gpus: Plan for this many cards instead of the detected ones.
        gib_per_gpu: Plan for cards of this size instead of the detected
            one. Give both to plan for another machine, e.g. a rented pod.

    Returns:
        A :class:`ResidencyPlan`. A plan that does not fit is returned, not
        raised.
    """
    backend_types = backend_types or {}
    footprints = []
    for m in dict.fromkeys(models):  # de-dup, keep order
        try:
            fp = footprint(m, backend_types.get(m))
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping %s in residency plan (%s)", m, exc)
            continue
        if fp is not None:
            footprints.append(fp)

    given = gpus is not None or gib_per_gpu is not None
    gpu_name, gpus_available = (
        (None, gpus) if gpus is not None else measure.detect_gpus())
    plan = ResidencyPlan(
        gpus_available=gpus_available, gpu_name=gpu_name, machine_given=given,
        per_gpu_gb=gib_per_gpu,
    )
    # Nothing local: return before probing, so an API-only run never
    # imports torch.
    if not footprints:
        return plan

    planned = {fp.engine for fp in footprints}
    resident = sorted(
        f"{key[0]}[{setup}]" for setup, keys in loaded_engines().items()
        for key in keys if (setup, *key) not in planned)
    if resident and not given:
        logger.warning(
            "[residency] already loaded and not in this plan, so the plan "
            "hands their memory out again: %s", ", ".join(resident))

    if gib_per_gpu is None:
        # nvidia-smi first: asking torch starts CUDA in this process, before
        # vLLM launches its engine subprocess.
        plan.per_gpu_gb = measure.detect_gpu_memory_gib()
        if plan.per_gpu_gb is None:
            ft = measure.free_total_gib()
            plan.per_gpu_gb = ft[1] if ft else None
    per_gpu_gb = plan.per_gpu_gb

    groups: list[list[ModelFootprint]] = []
    current: list[ModelFootprint] = []
    # Per engine: the model that placed it, and the group it is in.
    placed: dict[tuple, tuple[ModelFootprint, list[ModelFootprint]]] = {}
    twins: list[tuple[ModelFootprint, ModelFootprint]] = []
    # CUDA without nvidia-smi reports a size and no count: that is one card.
    count = gpus_available if gpus is not None else max(gpus_available, 1)
    if per_gpu_gb and gpus is None:
        plan.gpus_available = count
    cards = _Cards(count, per_gpu_gb)
    for fp in footprints:
        # Two models on the same engine share one load, in the engine's group.
        if fp.engine in placed:
            twin, group = placed[fp.engine]
            twins.append((fp, twin))
            group.append(fp)
            continue
        if not cards.place(fp):
            if current:
                cards.settle()
                groups.append(current)
                current = []
                cards = _Cards(count, per_gpu_gb)
            cards.place(fp, force=True)
        current.append(fp)
        placed[fp.engine] = (fp, current)
    cards.settle()
    groups.append(current)
    for fp, twin in twins:
        fp.devices, fp.reserved_gb = twin.devices, twin.reserved_gb
        fp.planned_utilization = twin.planned_utilization

    plan.groups = groups
    plan.gpus_required = max(
        (max(fp.devices) + 1 for fp in footprints if fp.devices), default=0
    )
    if per_gpu_gb:
        plan.oversized = [
            fp for fp in footprints
            if _exceeds_its_cards(fp, per_gpu_gb, gpus_available)
        ]
    return plan


def _exceeds_its_cards(fp: ModelFootprint, per_gpu_gb: float, gpus: int) -> bool:
    """Whether a model needs more than its grant, or than its cards hold."""
    if fp.setup == "vllm":
        if fp.reserved_gb is None:
            return False
        return fp.reserved_gb <= 0 or (
            fp.known and fp.padded_gb > fp.reserved_gb + _EPS)
    # No devices: the model runs on the CPU and takes no card.
    if not fp.known or not fp.devices:
        return False
    spans = (fp.setup == "introspect"
             and str(fp.parts.get("device_map", "auto")) in _SPANNING_DEVICE_MAPS)
    return fp.gb > per_gpu_gb * (max(gpus, 1) if spans else 1)


def report(plan: ResidencyPlan, verbose: bool = True) -> None:
    """Log the plan and emit it as a telemetry event.

    A plan that does not fit is logged at ERROR, one with warnings at
    WARNING. Neither raises.
    """
    if verbose:
        if not plan.fits:
            log = logger.error
        elif plan.warnings:
            log = logger.warning
        else:
            log = logger.info
        log("[residency] %s", plan.explain())
    observe.record({"ev": "residency", **plan.as_dict()})


def apply_plan(plan: ResidencyPlan, replace: bool = True) -> None:
    """Hand the plan's ``gpu_memory_utilization`` values to the vLLM loader.

    Args:
        plan: The plan to load under.
        replace: ``False`` keeps values already set, for a caller that plans
            a subset inside a planned run.
    """
    from ..backends import vllm

    # Keyed like the engine cache: the engine id without the setup.
    vllm.set_planned_utilization(
        {fp.engine[1:]: fp.planned_utilization for fp in plan.footprints
         if fp.setup == "vllm" and fp.planned_utilization is not None},
        replace=replace)


# ---------------------------------------------------------------------------
# Preload
# ---------------------------------------------------------------------------


def preload(
    models: list[str],
    backend_types: dict | None = None,
    verbose: bool = True,
) -> threading.Thread | None:
    """Start loading local models on a background thread and return at once.

    Models load one after another, in the order given, so a long load can
    overlap with an earlier stage's API calls. A call that has to load an
    engine waits for the load in progress, because the backends lock their
    engine caches.

    A failed load does not stop the run. It is logged at ERROR and recorded
    as a ``preload_failed`` event, and the stage that needs the model fails
    when it tries to use it.

    Args:
        models: Model names to load. Unregistered and non-local ones are
            skipped.
        backend_types: Optional per-model setup override.

    Returns:
        The daemon thread, or ``None`` when there is nothing local to load.
    """
    from ..client import ModelClient

    backend_types = backend_types or {}
    local = []
    for m in dict.fromkeys(models):
        try:
            if _local_setup(m, backend_types.get(m)) is not None:
                local.append(m)
        except (KeyError, ValueError) as exc:
            logger.debug("Not preloading %s (%s)", m, exc)
    if not local:
        return None

    def _load_all() -> None:
        for name in local:
            try:
                if verbose:
                    logger.info("[preload] loading %s ...", name)
                ModelClient.create(name, backend_type=backend_types.get(name))
                if verbose:
                    logger.info("[preload] %s ready", name)
            except Exception as exc:  # noqa: BLE001 — must not kill the run
                logger.error(
                    "[preload] %s failed to load (%s: %s). The run continues; "
                    "the stage that needs it will fail at that point.",
                    name, type(exc).__name__, exc,
                )
                observe.record({
                    "ev": "preload_failed",
                    "model": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    thread = threading.Thread(target=_load_all, name="redact-preload", daemon=True)
    thread.start()
    return thread


def _engine_keys(models) -> dict[str, set]:
    """The backend cache keys of the models' local setups, per setup."""
    keys: dict[str, set] = {setup: set() for setup in _LOCAL_SETUPS}
    for name in models or ():
        try:
            config = get_model_config(name)
        except KeyError:
            continue
        # Every local setup: the model may be in use under a non-default one.
        for setup in _LOCAL_SETUPS:
            local = getattr(config, setup, None)
            if local is not None:
                # The backend's cache key is the engine id without the setup.
                keys[setup].add(_engine_id(local, setup)[1:])
    return keys


def loaded_engines(exclude: list[str] | None = None) -> dict[str, set]:
    """The cache keys of the engines loaded now, per setup.

    Args:
        exclude: Model names whose engines are left out.
    """
    from ..backends import introspection, vllm

    skip = _engine_keys(exclude)
    return {"vllm": set(vllm._engines) - skip["vllm"],
            "introspect": set(introspection._models) - skip["introspect"]}


def unload_local(keep: list[str] | None = None, engines: dict | None = None) -> None:
    """Release the cached local engines, except those the kept models use.

    A released engine is shut down and its memory freed, and its lifetime
    meter stops. Its planned grant stays, so a reload gets the same share. A
    client built on it raises until the checkpoint is loaded again, and a
    call in flight on it is not waited for. A vLLM version without a shutdown
    hook returns the card only when the engine is collected, or at process
    exit.

    Args:
        keep: Model names still in use. Their engines stay cached and timed,
            so a later client for the same model reuses the load.
        engines: Engines to keep as well, as :func:`loaded_engines` returns.
    """
    from ..backends.introspection import TransformersIntrospectionBackend
    from ..backends.vllm import VLLMBackend

    kept = _engine_keys(keep)
    for setup, keys in (engines or {}).items():
        kept[setup] |= keys
    VLLMBackend.clear_cache(keep=frozenset(kept["vllm"]))
    TransformersIntrospectionBackend.clear_cache(keep=frozenset(kept["introspect"]))
