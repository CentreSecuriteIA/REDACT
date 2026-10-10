"""The plan as text, and the report that logs it and emits the telemetry event."""

import logging
from typing import TYPE_CHECKING

from redact.llms import observe
from redact.llms.resources import estimate

from .footprint import _CARD_CEILING, _SPANNING_DEVICE_MAPS, ModelFootprint

if TYPE_CHECKING:
    from .plan import ResidencyPlan

logger = logging.getLogger(__name__)

#: Tensor-parallel sizes offered for a model too large for its cards.
_TENSOR_PARALLEL_SIZES = (2, 4, 8)

_SEPARATE_RUNS = (
    "Run the stages that need different local models as separate "
    "run_pipeline invocations: process exit frees the card, and resume "
    "makes the rerun cheap."
)


def explain(plan: "ResidencyPlan") -> str:
    """The plan as text: what loads on which cards, and what is wrong.

    Includes each model's derivation and grant, and the way out for a
    plan that does not fit.
    """
    if plan.machine_given:
        gpu = f"planning for {plan.gpus_available} GPU(s)"
    elif plan.gpu_name:
        gpu = f"detected {plan.gpus_available} x {plan.gpu_name}"
    else:
        gpu = "no GPU detected"
    if plan.per_gpu_gb:
        gpu += f", {plan.per_gpu_gb:.1f}GiB each"
    lines = [f"plan needs {plan.gpus_required} GPU(s); {gpu}"]

    for i, group in enumerate(plan.groups, start=1):
        tail = "" if i == 1 else "  -> cannot be resident with the above"
        lines.append(f"  group {i}:{tail}")
        for fp in group:
            gb = f"~{fp.gb:.1f}GiB" if fp.known else "size unknown"
            lines.append(
                f"    {fp.model}[{fp.setup}]  {gb}  ({fp.breakdown()})"
                f"{_placement(plan, fp)}"
            )
            if fp.min_gpus > 1 and fp.known:
                lines.append(
                    f"      -> {fp.min_gpus} GPUs "
                    f"(tensor_parallel_size={fp.min_gpus}), "
                    f"~{fp.per_gpu_gb:.1f}GiB each"
                )

    lines.extend(f"  PROBLEM: {problem}" for problem in plan.problems)
    lines.extend(f"  {warning}" for warning in plan.warnings)

    on_card_0 = [fp for fp in plan.footprints
                 if fp.setup == "vllm" and fp.min_gpus == 1]
    if on_card_0 and plan.gpus_available > 1:
        lines.append(
            "  note: single-GPU vLLM engines all load on card 0, since "
            "nothing assigns a device at load time; the other card(s) "
            "do not add room for them"
        )
    seen = set()
    for fp in plan.footprints:
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
    unknown = [fp.model for fp in plan.footprints if not fp.known]
    if unknown:
        lines.append(
            f"  note: no footprint for {', '.join(unknown)} — planned "
            f"optimistically; declare vram_gb to pin one"
        )
    return "\n".join(lines)

def _placement(plan: "ResidencyPlan", fp: ModelFootprint) -> str:
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

def co_residency_problem(plan: "ResidencyPlan") -> str:
    """Why several groups do not fit a run, and what to do instead."""
    text = (
        "these groups cannot be resident together, so a later group "
        f"would load on top of the earlier ones. {_SEPARATE_RUNS}"
    )
    engines = {fp.engine for fp in plan.footprints
               if fp.setup == "vllm" and fp.min_gpus == 1}
    if len(engines) > 1:
        text += (
            " vLLM engines share card 0 only when their needs with "
            f"overhead sum to at most {_CARD_CEILING} of it."
        )
    return text

def oversized_advice(plan: "ResidencyPlan", fp: ModelFootprint) -> list[str]:
    """Advice for a model whose grant or cards do not cover its need.

    Lines after the first are indented for :func:`explain`.
    """
    per = plan.per_gpu_gb or 0
    if fp.setup != "vllm":
        pinned = str(fp.parts.get("device_map", "auto")) not in _SPANNING_DEVICE_MAPS
        cards = 1 if pinned else max(plan.gpus_available, 1)
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
        lines.append(_tensor_parallel_suggestion(plan, fp, per))
    lines.append(
        "    alternatives: a lower max_model_len, a quantized checkpoint "
        "(~3x smaller at awq/gptq), or the .api setup"
    )
    return lines

def _tensor_parallel_suggestion(
    plan: "ResidencyPlan", fp: ModelFootprint, per: float
) -> str:
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
        more = "" if n <= plan.gpus_available else "  (more GPUs than detected)"
        return (
            f"    suggested: min_gpus={n} (tensor_parallel_size={n})"
            f" -> {share:.1f}GiB per GPU with overhead{more}"
        )
    return (
        f"    no tensor-parallel size up to {_TENSOR_PARALLEL_SIZES[-1]} "
        f"fits a {per:.1f}GiB device"
    )


def report(
    plan: "ResidencyPlan", verbose: bool = True, stages: list[str] | None = None,
) -> None:
    """Log the plan and emit it as a telemetry event.

    A plan that does not fit is logged at ERROR, one with warnings at
    WARNING. Neither raises. ``stages`` names the stages the plan is for.
    """
    label = f"stages {', '.join(stages)}: " if stages else ""
    if verbose:
        if not plan.fits:
            log = logger.error
        elif plan.warnings:
            log = logger.warning
        else:
            log = logger.info
        log("[residency] %s%s", label, plan.explain())
    event = {"ev": "residency", **plan.as_dict()}
    if stages:
        event["stages"] = list(stages)
    observe.record(event)
