"""Which local models fit on this machine, and in what order to load them.

A run that needs more GPU than it has should say so **before** loading
anything, not die in a CUDA OOM four minutes into a 24B load. That is all this
module does: work out each local model's footprint, pack them onto the
available devices, and produce a plan plus a readable explanation.

Two things make the packing less trivial than "sum the gigabytes":

1. **Multi-GPU models claim whole devices.** A model with
   ``tensor_parallel_size > 1`` (declared as ``min_gpus``) does not fit "in the
   leftover GB on card 0" — it takes N cards outright. The obvious future case
   is a locally-served translator too large for a single card, which turns the
   plan into a mixed packing of whole-device and fractional-device claims.
2. **vLLM's ``gpu_memory_utilization`` is a fraction of the whole card**, and it
   preallocates. Two engines co-resident on one GPU therefore need their
   fractions to sum below 1 — a constraint that is expressible today and was
   simply never checked.

Everything here is an **estimate**. Real VRAM moves with ``max_model_len``,
quantization and KV cache; ``llama-3.2-3b-debug`` already hand-tunes
``gpu_memory_utilization`` and ``max_model_len`` for exactly that reason. The
plan is a readable pre-flight warning, not a guarantee — the OOM path still has
to be handled.
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

#: Assumed when a model declares no footprint and none was ever measured. Only
#: used to keep planning going; it is reported as unknown, not as fact.
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
    #: The terms behind an estimate (weights/KV/TP/max_model_len), so a report
    #: can show why the number is what it is. Empty for a declared footprint —
    #: a pinned number has no derivation to show.
    parts: dict = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.gb is not None

    def breakdown(self) -> str:
        """One-line derivation, e.g. ``43.9 weights + 2.1 KV, x1.1 +0.6``."""
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

    Resolution order — **declared beats estimated**, the inverse of the old
    measured-beats-declared rule. ``vram_gb`` is now the *override*: the place
    someone pins a number they know better than the estimate (an MoE
    checkpoint, an architecture the shape math doesn't model, a figure from a
    real OOM). Absent that, the estimate is computed from the checkpoint's HF
    config, so a newly registered model plans correctly with nothing
    hand-derived.

    1. The setup's declared ``vram_gb``, if set.
    2. :mod:`~redact.llms.resources.estimate` — weights + KV + explicit
       headroom, from config shapes alone.
    3. Unknown — planning continues optimistically and says so.

    Args:
        model: Registered model name.
        backend_type: Optional setup override, e.g. ``"vllm"``.
        seq_len: Tokens per sequence (input + output) to size the KV cache
            for. ``None`` uses the engine's configured ``max_model_len`` —
            the feasibility floor, i.e. one request at full context.
        batch: Concurrent sequences. ``1`` answers "can this run at all";
            the pipeline's real chunk size answers "can it run that batch".

    Returns:
        The footprint, or ``None`` when this model has no local setup at all
        (an API model occupies no GPU and is not the planner's business).
    """
    config = get_model_config(model)
    setup = resolve_setup(config, backend_type)
    if setup not in ("vllm", "introspect"):
        return None
    local = getattr(config, setup)

    if local.vram_gb is not None:
        # Still carry the setup facts: gpu_memory_utilization and TP describe
        # the *engine*, not the estimate, so the grant check below applies to a
        # pinned footprint exactly as much as to a computed one.
        return ModelFootprint(model, setup, local.hf_model_id, local.vram_gb,
                              local.min_gpus, "declared",
                              parts=_setup_parts(local, setup))

    parts = _estimate_parts(local, setup, seq_len=seq_len, batch=batch)
    gib = estimate.planning_gib(parts["weights_gib"], parts["kv_gib"])
    if gib is not None:
        return ModelFootprint(model, setup, local.hf_model_id, gib,
                              local.min_gpus, "estimated", parts=parts)
    return ModelFootprint(model, setup, local.hf_model_id, None,
                          local.min_gpus, "unknown")


def _setup_parts(local, setup: str) -> dict:
    """Engine facts read straight off the setup — no estimation involved.

    Split out from :func:`_estimate_parts` because these describe how the
    *engine* is configured, so they are just as true for a pinned ``vram_gb``
    as for a computed footprint. Keeping them together is what previously let
    the grant check silently skip every declared model.
    """
    if setup == "introspect":
        return {"tensor_parallel_size": 1, "max_model_len": None}
    kw = local.vllm_kwargs or {}
    return {
        "tensor_parallel_size": kw.get("tensor_parallel_size", local.min_gpus or 1),
        "max_model_len": kw.get("max_model_len"),
        "gpu_memory_utilization": kw.get("gpu_memory_utilization"),
    }


def _estimate_parts(local, setup: str, *, seq_len: int | None, batch: int) -> dict:
    """Weights and KV for one local setup, with the terms kept separate.

    Returned as parts rather than a total so :meth:`ResidencyPlan.explain` can
    show *why* a number is what it is — which is what lets someone decide
    whether to override it with ``vram_gb``.
    """
    parts = _setup_parts(local, setup)
    tp = parts["tensor_parallel_size"]
    if setup == "introspect":
        parts["weights_gib"] = estimate.estimate_weights_gib(
            local.hf_model_id, dtype=local.torch_dtype)
        # transformers holds no preallocated pool; its per-call KV is transient
        # and small next to the weights, so it is not modelled.
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
    #: Models too large for a single device that declare ``min_gpus=1``. These
    #: cannot be placed at all until tensor parallelism is configured — packing
    #: never splits one model across cards.
    oversized: list[ModelFootprint] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        """Can this run be placed at all?

        False when it needs more devices than exist, and also when any single
        model is larger than one device without declaring ``min_gpus`` — that
        one is unplaceable regardless of how many cards are available.
        """
        if self.oversized:
            return False
        return self.gpus_required <= max(self.gpus_available, 0)

    @property
    def sequential(self) -> bool:
        """True when models must be unloaded between groups to make room."""
        return len(self.groups) > 1

    def explain(self) -> str:
        """The pre-flight message — what loads, in what order, on what.

        Prints the whole allocation schema rather than only the grouping: the
        per-model derivation (so a number can be judged and overridden), the
        device split for a tensor-parallel model, and for anything that does
        not fit, a **concrete configuration that would** — not just advice to
        go find one.
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
        """Flag a model that fits the card but not its own vLLM grant.

        ``gpu_memory_utilization`` caps what vLLM may reserve, so a model can
        fit a device and still OOM because the *grant* is smaller than the
        need. That is the constraint which actually produces the failure, and
        nothing checked it before.
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
        """A model too large for one device — say which config *would* work.

        Packing never splits a model across cards; only tensor parallelism
        does. So the useful output is the smallest TP that fits and what each
        device would then hold, rather than "set min_gpus" and leave the
        arithmetic to the reader.
        """
        per = self.per_gpu_gb or 0
        lines = [
            f"  PROBLEM: {fp.model}[{fp.setup}] needs ~{fp.gb:.1f}GiB but a "
            f"device has {per:.1f}GiB — packing never splits one model across "
            f"cards."
        ]
        # TP must divide the attention heads, so only powers of two are worth
        # proposing without reading the config again.
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
    """Group a run's local models into what can be co-resident.

    Greedy first-fit: models are packed onto the detected devices in declared
    order; anything that cannot fit alongside what is already placed starts a
    new group, which means "unload the previous group first".

    Args:
        models: Model names. API-only models are ignored — they hold no GPU.
        backend_types: Optional per-model setup override, as passed to
            ``ModelClient.create``.

    Returns:
        A :class:`ResidencyPlan`. Never raises for a plan that does not fit:
        it reports, and the caller decides (``run_pipeline`` logs and continues).
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
    # torch knows the live figure, but needs torch installed AND CUDA usable.
    # nvidia-smi answers on any machine that has a card, which is most of them.
    ft = measure.free_total_gib()
    per_gpu_gb = ft[1] if ft else detect_gpu_memory_gib()

    plan = ResidencyPlan(
        gpus_available=gpus_available, gpu_name=gpu_name, per_gpu_gb=per_gpu_gb,
    )
    if not footprints:
        return plan

    # Greedy first-fit. Capacity is per-GPU GB when known; a multi-GPU model
    # claims whole devices and never shares them.
    # Budget for a group is the whole machine: single-GPU models pack across
    # however many cards there are, so two 20GB models on two 24GB cards are
    # one group (loaded together), not two.
    capacity = (per_gpu_gb or 0.0) * max(gpus_available, 1)
    unknown_capacity = capacity <= 0
    groups: list[list[ModelFootprint]] = []
    current: list[ModelFootprint] = []
    seen_engines: set[tuple[str, str]] = set()
    used_gb = 0.0
    for fp in footprints:
        # Two models on the SAME checkpoint share one engine (venice-uncensored's
        # .vllm setup and venice-paraphraser do exactly this), so the second one
        # is free — it is not a second load. Getting this wrong would split a
        # plan that actually fits.
        engine = (fp.setup, fp.hf_model_id)
        shared = engine in seen_engines
        need_gb = 0.0 if shared else (fp.gb or _UNKNOWN_GB)

        # With no way to read the card, do not *invent* a sequential plan —
        # that would claim knowledge we don't have. Group optimistically and
        # let explain()'s unknown-footprint note carry the uncertainty.
        # A single-GPU model competes for gigabytes; a multi-GPU one claims
        # whole devices and never shares, so it always starts its own group.
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
    # One model larger than one card cannot be split by packing it — only
    # tensor parallelism spans devices. Reporting "needs 7 GPUs" for a 50GB
    # model on 8GB cards would imply a placement that does not exist.
    if per_gpu_gb:
        plan.oversized = [
            fp for fp in footprints
            if fp.min_gpus == 1 and (fp.gb or 0.0) > per_gpu_gb
        ]
    return plan


def _group_gpus(group: list[ModelFootprint], per_gpu_gb: float | None) -> int:
    """Devices one group needs.

    Two corrections over "sum ``min_gpus``", both of which would otherwise
    over-count:

    - Models sharing a checkpoint share a load, so they share its GPUs. Counted
      once per engine, not once per registry row.
    - Several single-GPU models can sit on one card. What they need is
      ``ceil(total_gb / per_gpu_gb)``, not one card each.
    """
    per_engine: dict[tuple[str, str], ModelFootprint] = {}
    for fp in group:
        per_engine.setdefault((fp.setup, fp.hf_model_id), fp)

    exclusive = sum(fp.min_gpus for fp in per_engine.values() if fp.min_gpus > 1)
    singles = [fp for fp in per_engine.values() if fp.min_gpus == 1]
    if not singles:
        return exclusive
    shared_gb = sum(fp.gb or 0.0 for fp in singles)
    cards = math.ceil(shared_gb / per_gpu_gb) if per_gpu_gb else 1
    return exclusive + max(1, cards)


def report(plan: ResidencyPlan, verbose: bool = True) -> None:
    """Log the plan and emit it as a telemetry event.

    An over-capacity plan is logged at ERROR but is **not** fatal — an
    API-only stage that would have completed should still be allowed to.
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

#: Which model roles each pipeline stage actually calls. Used to preload only
#: what a run will really touch, rather than every model the recipe names.
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
    """Start loading local models in the background, and return immediately.

    A 24B vLLM load takes minutes. Running it while the constitution stage is
    still making Opus API calls is free wall-clock, and the backends' engine
    caches mean the first real use just finds it already there.

    Loads run **sequentially** on one background thread: concurrent loads
    compete for the same VRAM headroom, which is the opposite of helpful.
    Thread-safety of the caches themselves is handled by the double-checked
    locks in ``backends/vllm.py`` and ``backends/introspection.py`` — a real
    call arriving mid-preload blocks until the load finishes rather than
    starting a second one.

    **A failed preload never aborts the run.** It logs at ERROR and records a
    ``preload_failed`` event, then lets the run continue: an API-only stage
    like constitution is ledger-backed and its expensive output is saved as it
    goes, so finishing it is strictly better than killing it. The stage that
    genuinely needs the model fails on its own, at the point of use, with the
    cause already in the log.

    Args:
        models: Model names to warm. Non-local ones are skipped.
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
    """Free loaded local models so the next one has room.

    Drops the engine caches, and the rate-limit windows with them so a fresh
    run starts clean. Freeing is Python's business: the memory comes back when
    the *last* reference to an engine goes, not when this returns. Since
    clients are rebuilt per ``ModelClient.create()`` and nothing caches them,
    the cache is usually the last holder — but a caller still holding a client
    keeps its engine resident, correctly, because it is still in use.

    Args:
        keep: Model names whose clients are still in use by the caller. They
            are logged, not protected: a live local reference anywhere keeps
            its engine resident regardless of what this clears.
    """
    from ..backends import clear_transport_caches
    from ..client import clear_client_cache

    if keep:
        logger.debug("[residency] unloading local models; %s still referenced", keep)
    clear_client_cache()
    clear_transport_caches()
