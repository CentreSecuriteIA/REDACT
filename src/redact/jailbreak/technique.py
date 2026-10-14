"""What a technique is: a transform or a generator of model calls, its result,
and the tags the spec gives it.

A transform is ``(prompt) -> (text, info)``. A generator yields
:class:`~redact.llm_pipeline.LLMRequest` objects, is resumed with each reply,
and returns ``(text, info)``. ``info`` is ``""`` or ``key=value;...``; a
rejection's starts with :data:`DISCARDED`.
"""

import json
from collections.abc import Callable, Generator
from functools import lru_cache

from redact import paths
from redact.llm_pipeline import LLMRequest

_SPEC_PATH = paths.jailbreak_configs_dir() / "combination_spec.json"

DISCARDED = "DISCARDED"

TechniqueResult = tuple[str, str]
TechniqueGen = Generator[LLMRequest, str, TechniqueResult]


def is_discarded(info: str) -> bool:
    """Whether ``info`` is a rejected technique's."""
    return info.startswith(DISCARDED)


@lru_cache(maxsize=1)
def load_spec() -> dict:
    """The combination spec: families, techniques, layer order and rules."""
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def tag(funcs: list[Callable]) -> None:
    """Set each function's spec tags as attributes of the function.

    ``families``, ``complexity`` (0-3), ``requires_llm``, ``requires_benign``,
    ``encode_weight`` (``"light"``, ``"heavy"`` or None) and ``max_tokens``
    (the reply limit of its model calls, None for the model's default) come
    from the technique's entry; ``layer`` and ``within_layer_order`` from its
    first family. A function the spec does not name is skipped. Called once,
    when ``redact.jailbreak`` is imported.

    Raises:
        ValueError: A technique's ``max_tokens`` is not a whole number of at
            least 1. Checked here, so a bad spec fails at import instead of
            as an error row per sample.
    """
    spec = load_spec()
    families = spec["families"]
    techniques = spec["techniques"]

    family_layer = {name: f["layer"] for name, f in families.items()}
    family_wlo = {name: f.get("within_layer_order", 0) for name, f in families.items()}

    for fn in funcs:
        name = getattr(fn, "__name__", None)
        if name not in techniques:
            continue
        t = techniques[name]
        fn.families = t.get("families", [])
        fn.complexity = t.get("complexity", 1)
        fn.requires_llm = t.get("requires_llm", False)
        # Needs pre-generated benign Q&A data (FSH/DAP). Distinct from
        # requires_llm: the data is generated once up front, not per call — so
        # these aren't per-sample LLM techniques, but they still can't run in a
        # strictly no-LLM (pure_only) run.
        fn.requires_benign = t.get("requires_benign", False)
        fn.encode_weight = t.get("encode_weight", None)
        # Reply limit for the technique's LLM calls; None keeps the model's
        # default. chain._run_chain puts it on each request.
        limit = t.get("max_tokens")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError(
                f"Technique {name!r}: max_tokens must be a whole number of at "
                f"least 1, got {limit!r}.")
        fn.max_tokens = limit

        primary_family = fn.families[0] if fn.families else None
        fn.layer = family_layer.get(primary_family, "obfuscation")
        fn.within_layer_order = family_wlo.get(primary_family, 0)
