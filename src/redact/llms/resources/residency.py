"""Plan which local models fit on this machine, and in what order to load them.

Works out each local model's footprint, packs the models onto the detected
GPUs, and produces a plan with a readable explanation, before anything is
loaded.

- A model with ``min_gpus > 1`` (its ``tensor_parallel_size``) takes whole
  devices and never shares them.
- For vLLM, the plan warns when a model needs more than its own
  ``gpu_memory_utilization`` share of a card. The combined shares of models
  on one card are not checked.

Footprints are estimates, and all sizes are GiB. The plan is a warning ahead
of the run, not a guarantee against running out of memory.
"""

import logging
import math
import threading
from dataclasses import dataclass, field

from .. import observe
from ..backends import resolve_setup
from ..model_config import get_model_config
from . import estimate, measure

logger = logging.getLogger(__name__)

#: Size used in the packing arithmetic for a model whose footprint is neither
#: declared nor estimable. The plan reports such a model as unknown.
_UNKNOWN_GB = 0.0


@dataclass
class ModelFootprint:
    """What one local model is expected to need."""

    model: str
    setup: str
    hf_model_id: str
    gb: float | None
    min_gpus: int
    source: str  # "declared" | "estimated" | "unknown"
    #: Engine settings (tensor parallel size, max_model_len,
    #: gpu_memory_utilization) and, for an estimate, its weights and KV terms.
    parts: dict = field(default_factory=dict)
    #: The load this model uses. Models with the same value share one engine.
    engine: tuple = ()

    def __post_init__(self) -> None:
        if not self.engine:
            self.engine = (self.setup, self.hf_model_id)

    @property
    def known(self) -> bool:
        return self.gb is not None

    def breakdown(self) -> str:
        """One-line derivation of the footprint.

        For an estimate, e.g. ``estimated: 43.9 weights + 2.1 KV, x1.1 +0.6``.
        """
        if self.source == "declared":
            return "declared"
        w, kv = self.parts.get("weights_gib"), self.parts.get("kv_gib")
        if w is None:
            return self.source
        terms = f"{w:.1f} weights"
        if kv:
            terms += f" + {kv:.1f} KV"
        return (
            f"estimated: {terms}, x{estimate.DEFAULT_FRAGMENTATION} "
            f"+{estimate.DEFAULT_CUDA_CONTEXT_GIB}"
        )


def footprint(
    model: str,
    backend_type: str | None = None,
    *,
    seq_len: int | None = None,
    batch: int = 1,
) -> ModelFootprint | None:
    """What one model is expected to need, without loading anything.

    Resolved in this order:

    1. The setup's declared ``vram_gb``, if set. It overrides the estimate.
    2. An estimate from the checkpoint's HF config
       (:mod:`~redact.llms.resources.estimate`): weights, KV and headroom.
    3. Unknown. Planning continues and the plan says so.

    Args:
        model: Registered model name.
        backend_type: Optional setup override, e.g. ``"vllm"``.
        seq_len: Tokens per sequence (input + output) to size the KV cache
            for. ``None`` uses the engine's configured ``max_model_len``.
        batch: Concurrent sequences to size the KV cache for.

    Returns:
        The footprint, or ``None`` when the resolved setup is not a local
        one.
    """
    config = get_model_config(model)
    setup = resolve_setup(config, backend_type)
    if setup not in ("vllm", "introspect"):
        return None
    local = getattr(config, setup)
    engine = _engine_id(local, setup)

    if local.vram_gb is not None:
        # A declared footprint still carries the engine settings, which the
        # grant check in explain() reads.
        return ModelFootprint(model, setup, local.hf_model_id, local.vram_gb,
                              local.min_gpus, "declared",
                              parts=_setup_parts(local, setup), engine=engine)

    parts = _estimate_parts(local, setup, seq_len=seq_len, batch=batch)
    gib = estimate.planning_gib(parts["weights_gib"], parts["kv_gib"])
    if gib is not None:
        # The estimate is per GPU; gb is a total, like a declared vram_gb.
        gib *= parts["tensor_parallel_size"]
        return ModelFootprint(model, setup, local.hf_model_id, gib,
                              local.min_gpus, "estimated", parts=parts,
                              engine=engine)
    return ModelFootprint(model, setup, local.hf_model_id, None,
                          local.min_gpus, "unknown", engine=engine)


def _engine_id(local, setup: str) -> tuple:
    """The load a local setup uses, matching its backend's cache key.

    Two setups with the same id share one engine; anything else is a second
    load, even on the same checkpoint.
    """
    if setup == "introspect":
        kwargs = local.extra_kwargs or {}
        return (setup, local.hf_model_id, local.device_map, local.torch_dtype,
                repr(sorted(kwargs.items())))
    return (setup, local.hf_model_id, local.quantization,
            repr(sorted(local.engine_kwargs.items())))


def _setup_parts(local, setup: str) -> dict:
    """Engine settings read from the setup, with no estimation involved."""
    if setup == "introspect":
        return {"tensor_parallel_size": 1, "max_model_len": None}
    kw = local.vllm_kwargs or {}
    return {
        "tensor_parallel_size": kw.get("tensor_parallel_size", local.min_gpus or 1),
        "max_model_len": kw.get("max_model_len"),
        "gpu_memory_utilization": kw.get("gpu_memory_utilization"),
    }


def _estimate_parts(local, setup: str, *, seq_len: int | None, batch: int) -> dict:
    """Engine settings plus weights and KV estimates, kept as separate terms.

    The terms are shown in :meth:`ResidencyPlan.explain`.
    """
    parts = _setup_parts(local, setup)
    tp = parts["tensor_parallel_size"]
    if setup == "introspect":
        parts["weights_gib"] = estimate.estimate_weights_gib(
            local.hf_model_id, dtype=local.torch_dtype)
        # KV is not modelled for transformers: it is transient and small next
        # to the weights.
        parts["kv_gib"] = None
        return parts

    kw = local.vllm_kwargs or {}
    parts["weights_gib"] = estimate.estimate_weights_gib(
        local.hf_model_id, dtype=kw.get("dtype"),
        quantization=local.quantization, tensor_parallel_size=tp)
    parts["kv_gib"] = estimate.estimate_kv_gib(
        local.hf_model_id, seq_len=seq_len, max_model_len=parts["max_model_len"],
        dtype=kw.get("dtype"), tensor_parallel_size=tp, batch=batch)
    return parts


@dataclass
class ResidencyPlan:
    """How to fit a run's local models onto this machine."""

    groups: list[list[ModelFootprint]] = field(default_factory=list)
    gpus_required: int = 0
    gpus_available: int = 0
    gpu_name: str | None = None
    per_gpu_gb: float | None = None
    #: Models larger than one device that declare ``min_gpus=1``. They cannot
    #: be placed without tensor parallelism, because packing never splits one
    #: model across cards.
    oversized: list[ModelFootprint] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        """Whether the run can be placed.

        False when it needs more devices than exist, or when any model is
        larger than one device while declaring ``min_gpus=1``.
        """
        if self.oversized:
            return False
        return self.gpus_required <= max(self.gpus_available, 0)

    @property
    def sequential(self) -> bool:
        """True when models must be unloaded between groups to make room."""
        return len(self.groups) > 1

    def explain(self) -> str:
        """The plan as text: what loads, in what order, on which devices.

        Includes each model's derivation, the per-device split of a
        tensor-parallel model, grant warnings, and a suggested configuration
        for any model too large for one device.
        """
        gpu = (
            f"{self.gpus_available} x {self.gpu_name}"
            if self.gpu_name else "no GPU detected"
        )
        if self.per_gpu_gb:
            gpu += f", {self.per_gpu_gb:.1f}GiB each"
        lines = [f"plan needs {self.gpus_required} GPU(s); detected {gpu}"]

        for i, group in enumerate(self.groups, start=1):
            tail = "" if i == 1 else "  -> sequential, unload between"
            lines.append(f"  group {i}:{tail}")
            for fp in group:
                gb = f"~{fp.gb:.1f}GiB" if fp.known else "size unknown"
                lines.append(f"    {fp.model}[{fp.setup}]  {gb}  ({fp.breakdown()})")
                if fp.min_gpus > 1 and fp.known:
                    lines.append(
                        f"      -> {fp.min_gpus} GPUs "
                        f"(tensor_parallel_size={fp.min_gpus}), "
                        f"~{fp.gb / fp.min_gpus:.1f}GiB each"
                    )
            lines.extend(self._grant_warnings(group))

        for fp in self.oversized:
            lines.extend(self._oversized_advice(fp))

        unknown = [fp.model for g in self.groups for fp in g if not fp.known]
        if unknown:
            lines.append(
                f"  note: no footprint for {', '.join(unknown)} — planned "
                f"optimistically; declare vram_gb to pin one"
            )
        return "\n".join(lines)

    def _grant_warnings(self, group: list[ModelFootprint]) -> list[str]:
        """Warn about each model that needs more than its own vLLM grant.

        ``gpu_memory_utilization`` caps what vLLM may reserve, so a model can
        fit the device and still run out of memory.
        """
        out = []
        if not self.per_gpu_gb:
            return out
        for fp in group:
            util = fp.parts.get("gpu_memory_utilization")
            if not (fp.known and util):
                continue
            granted = self.per_gpu_gb * util
            need = fp.gb / max(1, fp.min_gpus)
            if need > granted:
                out.append(
                    f"    WARNING: {fp.model} needs ~{need:.1f}GiB per GPU but "
                    f"gpu_memory_utilization={util} grants {granted:.1f}GiB. "
                    f"Raise it to >={min(0.98, need / self.per_gpu_gb + 0.02):.2f} "
                    f"or lower max_model_len."
                )
        return out

    def _oversized_advice(self, fp: ModelFootprint) -> list[str]:
        """Advice for a model too large for one device.

        Suggests the smallest tensor-parallel size among 2, 4 and 8 that
        fits, and what each device would then hold.
        """
        per = self.per_gpu_gb or 0
        lines = [
            f"  PROBLEM: {fp.model}[{fp.setup}] needs ~{fp.gb:.1f}GiB but a "
            f"device has {per:.1f}GiB — packing never splits one model across "
            f"cards."
        ]
        # Only powers of two are proposed, since the tensor-parallel size
        # must divide the attention heads.
        for n in (2, 4, 8):
            if per and fp.gb / n <= per:
                fits = "" if n <= self.gpus_available else "  (more GPUs than detected)"
                lines.append(
                    f"    suggested: min_gpus={n} (tensor_parallel_size={n})"
                    f" -> ~{fp.gb / n:.1f}GiB per GPU{fits}"
                )
                break
        else:
            if per:
                need = math.ceil(fp.gb / per)
                lines.append(
                    f"    would need >={need} GPUs of {per:.1f}GiB at "
                    f"tensor_parallel_size={need}"
                )
        lines.append(
            "    alternatives: a quantized checkpoint (~4x smaller at "
            "awq/gptq), a lower max_model_len, or the .api setup"
        )
        return lines


def plan_residency(
    models: list[str], backend_types: dict | None = None
) -> ResidencyPlan:
    """Group a run's local models into sets that can be loaded together.

    Greedy, in the order given. A model that does not fit beside
    the ones already placed starts a new group, which means the previous
    group is unloaded first.

    Args:
        models: Model names. Models whose setup is not local are ignored.
        backend_types: Optional per-model setup override, as passed to
            ``ModelClient.create``.

    Returns:
        A :class:`ResidencyPlan`. A plan that does not fit is returned, not
        raised.
    """
    from ...telemetry import detect_gpu_memory_gib, detect_gpus

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

    gpu_name, gpus_available = detect_gpus()
    # Per-device capacity from torch when CUDA is usable, else from nvidia-smi.
    ft = measure.free_total_gib()
    per_gpu_gb = ft[1] if ft else detect_gpu_memory_gib()

    plan = ResidencyPlan(
        gpus_available=gpus_available, gpu_name=gpu_name, per_gpu_gb=per_gpu_gb,
    )
    if not footprints:
        return plan

    # A group's budget is the whole machine: single-GPU models pack across
    # all cards, so two 20GiB models on two 24GiB cards form one group.
    capacity = (per_gpu_gb or 0.0) * max(gpus_available, 1)
    unknown_capacity = capacity <= 0
    groups: list[list[ModelFootprint]] = []
    current: list[ModelFootprint] = []
    seen_engines: set[tuple] = set()
    used_gb = 0.0
    for fp in footprints:
        # Two models on the same engine share one load, so the second one
        # costs nothing.
        engine = fp.engine
        shared = engine in seen_engines
        need_gb = 0.0 if shared else (fp.gb or _UNKNOWN_GB)

        # A single-GPU model fits if there is room, or if capacity is unknown
        # (then everything is grouped together). A multi-GPU model claims
        # whole devices, so it does not join a group that has models in it.
        # Known gap: a single-GPU model can still join a group that a
        # multi-GPU model started. The group may then need more devices than
        # exist, and the plan reports "does not fit" instead of splitting it.
        fits = shared or (
            fp.min_gpus == 1 and (unknown_capacity or used_gb + need_gb <= capacity)
        )
        if current and not fits:
            groups.append(current)
            current, used_gb, seen_engines = [], 0.0, set()
            need_gb = fp.gb or _UNKNOWN_GB
        current.append(fp)
        seen_engines.add(engine)
        used_gb += need_gb
    if current:
        groups.append(current)

    plan.groups = groups
    plan.gpus_required = max(
        (_group_gpus(group, per_gpu_gb) for group in groups), default=0
    )
    # A model larger than one card cannot be placed by packing. Only tensor
    # parallelism spans devices.
    if per_gpu_gb:
        plan.oversized = [
            fp for fp in footprints
            if fp.min_gpus == 1 and (fp.gb or 0.0) > per_gpu_gb
        ]
    return plan


def _group_gpus(group: list[ModelFootprint], per_gpu_gb: float | None) -> int:
    """Devices one group needs.

    Models sharing an engine are counted once. Single-GPU models share
    cards, so together they need ``ceil(total_gb / per_gpu_gb)`` devices.
    Multi-GPU models add their ``min_gpus`` each.
    """
    per_engine: dict[tuple, ModelFootprint] = {}
    for fp in group:
        per_engine.setdefault(fp.engine, fp)

    exclusive = sum(fp.min_gpus for fp in per_engine.values() if fp.min_gpus > 1)
    singles = [fp for fp in per_engine.values() if fp.min_gpus == 1]
    if not singles:
        return exclusive
    shared_gb = sum(fp.gb or 0.0 for fp in singles)
    cards = math.ceil(shared_gb / per_gpu_gb) if per_gpu_gb else 1
    return exclusive + max(1, cards)


def report(plan: ResidencyPlan, verbose: bool = True) -> None:
    """Log the plan and emit it as a telemetry event.

    A plan that does not fit is logged at ERROR and does not raise.
    """
    if verbose:
        log = logger.error if not plan.fits else logger.info
        log("[residency] %s", plan.explain())
    observe.record({
        "ev": "residency",
        "gpus_required": plan.gpus_required,
        "gpus_available": plan.gpus_available,
        "gpu": plan.gpu_name,
        "fits": plan.fits,
        "groups": [[fp.model for fp in g] for g in plan.groups],
    })


# ---------------------------------------------------------------------------
# Preload
# ---------------------------------------------------------------------------

#: The model roles each pipeline stage calls, for preloading only the models
#: a run will use.
STAGE_ROLES = {
    "constitution": ("constitution",),
    "inputs": ("gen", "check"),
    "outputs": ("gen", "check"),
    "paraphrase": ("paraphraser", "paraphrase_check"),
    "jailbreaks": ("gen", "translation"),
    "build": (),
}


def preload(
    models: list[str],
    backend_types: dict | None = None,
    verbose: bool = True,
) -> threading.Thread | None:
    """Start loading local models on a background thread and return at once.

    Models load one after another, so a long load can overlap with an earlier
    stage's API calls. A real call that arrives during a load waits for it,
    because the backends lock their engine caches.

    A failed load does not stop the run. It is logged at ERROR and recorded
    as a ``preload_failed`` event, and the stage that needs the model fails
    when it tries to use it.

    Args:
        models: Model names to load. Non-local ones are skipped.
        backend_types: Optional per-model setup override.
        verbose: Log progress.

    Returns:
        The daemon thread, or ``None`` when there is nothing local to load.
    """
    from ..client import ModelClient

    backend_types = backend_types or {}
    local = []
    for m in dict.fromkeys(models):
        try:
            if footprint(m, backend_types.get(m)) is not None:
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


def unload_local(keep: list[str] | None = None) -> None:
    """Clear the backend caches to free GPU memory.

    Memory is released when the last reference to an engine goes away, so an
    engine stays loaded while a caller still holds a client for it.

    Args:
        keep: Model names the caller still uses. They are only logged;
            nothing is protected by listing them.
    """
    from ..backends import clear_transport_caches

    if keep:
        logger.debug("[residency] unloading local models; %s still referenced", keep)
    clear_transport_caches()
