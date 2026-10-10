"""The result of placing a run's local models: groups, grants, problems."""

from dataclasses import dataclass, field

from .explain import co_residency_problem, oversized_advice
from .explain import explain as _explain
from .footprint import ModelFootprint


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
            out.append(co_residency_problem(self))
        out.extend("\n".join(oversized_advice(self, fp)) for fp in self.oversized)
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
        """The plan as text (:func:`.explain.explain`)."""
        return _explain(self)
