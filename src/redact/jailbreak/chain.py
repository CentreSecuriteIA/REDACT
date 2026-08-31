"""Running one sample through a (possibly combined) technique chain.

Split out of jailbreak/utils.py (see that module's docstring for the full
7-concern breakdown this split resolved). This one owns: normalizing a
technique's return value, chaining multiple techniques into one callable
(mixing pure transforms and LLM-dependent technique *generators*
transparently — see protocol.py), driving a single sample end-to-end
(apply_combination), and classifying the result (is_noop, rejection parsing).

Ported from reference utils.py combine_techniques (lines 24-49).
Retry wrappers (with_retries, with_feedback_retries) live in LLMs/wrappers.py.
"""

import inspect
from collections.abc import Callable

from .protocol import LLMRequest, TechniqueGen, run_sync
from .spec import load_spec


def _normalize_output(output) -> tuple[str, str]:
    """Coerce a technique return into ``(text, info)``.

    Cognitive/persona techniques may return a 3-tuple ``(text, info, scenario)``
    — the scenario is discarded in combined chains (callers needing it must run
    the technique standalone).
    """
    if isinstance(output, tuple) and len(output) == 3:  # noqa: PLR2004 — fixed 3-tuple protocol shape, not a tunable value
        text, info, _ = output
        return text, info
    return output


def _select_kwargs(technique: Callable, kwargs: dict) -> dict:
    """Filter ``kwargs`` to what ``technique`` accepts.

    Pure transforms have ``(prompt: str)`` with no ``**kwargs`` — passing
    unrecognised keys would raise TypeError. Techniques declaring ``**kwargs``
    receive everything.
    """
    sig_params = inspect.signature(technique).parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
    )
    if has_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig_params}


def _tag_yields(gen: TechniqueGen, root: str) -> TechniqueGen:
    """Wrap a technique generator's yields, tagging each with a sequential
    ``internals_id`` rooted at ``root`` — ``{root}/0``, ``{root}/1``, ...

    Numbered generically per yield within *this one* technique's execution,
    not distinguishing "a genuinely new step" (e.g. cognitive hacking's
    scenario -> construction) from "a retry of the same step" (e.g.
    translation's own translate -> check -> retry loop) — that distinction
    isn't visible at this level without each technique explicitly
    cooperating, which no technique currently does. Every LLM call still gets
    its own distinct, capturable path either way.

    Only tags a request when its *own* target model's backend actually
    supports internals capture (checked per request, not assumed from the
    caller) — so a chain that mixes an internals-capable ``gen_model`` with a
    non-capable ``translate_model`` (or vice versa) tags exactly the calls
    that can be captured, and never trips BatchCaller's guard on the other.
    """
    from redact.llms.model_config import (
        model_compute_config,  # lazy import — avoids import cycles
    )

    n = 0
    try:
        request = gen.send(None)
    except StopIteration as stop:
        return stop.value
    while True:
        # Read off the backend *class*, never a constructed one: this asks the
        # question for every model in the chain, including ones this sample
        # never actually calls, and building a local transport to answer it
        # would load model weights.
        if model_compute_config(request.model).supports_internals:
            request = LLMRequest(request.model, request.messages, internals_id=f"{root}/{n}")
            n += 1
        reply = yield request
        try:
            request = gen.send(reply)
        except StopIteration as stop:
            return stop.value


def _run_chain(techniques: list[Callable], text: str, internals_root: str | None = None, **kwargs) -> TechniqueGen:
    """Generator that chains techniques, threading text through.

    Pure transforms run inline (zero rounds); LLM-dependent technique
    generators are delegated to via ``yield from`` so their :class:`LLMRequest`
    yields propagate up to the batched engine (or to ``run_sync`` for the
    single-sample path) — tagged with ``internals_id`` (see :func:`_tag_yields`)
    when ``internals_root`` is given. Early-exits with a ``DISCARDED`` info
    string the moment any step rejects, returning the pre-failure text.
    Returns ``(text, info)``.
    """
    # gen-model techniques tag their requests with `gen_model`; accept the
    # legacy `model` kwarg as the source when gen_model isn't given explicitly.
    if not kwargs.get("gen_model") and kwargs.get("model"):
        kwargs = {**kwargs, "gen_model": kwargs["model"]}

    result = text
    info_parts: list[str] = []
    for technique in techniques:
        tech_kwargs = _select_kwargs(technique, kwargs)
        if inspect.isgeneratorfunction(technique):
            tech_gen = technique(result, **tech_kwargs)
            if internals_root:
                tech_gen = _tag_yields(tech_gen, f"{internals_root}/{technique.__name__}")
            output = yield from tech_gen
        else:
            output = technique(result, **tech_kwargs)
        new_text, info = _normalize_output(output)

        # Early exit on rejection — don't apply remaining techniques
        if info.startswith("DISCARDED"):
            return result, (
                f"DISCARDED; technique={technique.__name__}; "
                f"{info[len('DISCARDED; '):]}"
            )

        # Track no-ops
        if new_text == result:
            info_parts.append(f"noop={technique.__name__}")
        elif info:
            info_parts.append(info)
        result = new_text
    return result, ";".join(info_parts)


def combine_techniques(*techniques: Callable, sort_by_hierarchy: bool = True) -> Callable:
    """Chain multiple technique functions into one callable.

    Each technique receives the output of the previous one; info strings are
    joined with ';'. Mixes pure transforms and LLM-dependent technique
    *generators* transparently (see ``protocol.py``).

    If ``sort_by_hierarchy=True`` (default), techniques are reordered by
    (layer_order_index, within_layer_order) before chaining — so callers can
    pass techniques in any order and the correct semantic sequence is applied.

    The returned ``combined`` callable runs **synchronously**: calling
    ``combined(text, client=..., benign_data=...)``
    drives the chain to completion (issuing real LLM calls for generator steps
    via :func:`protocol.run_sync`) and returns ``(text, info)``. This preserves
    the single-sample / test contract. The batched engine does **not** call
    ``combined`` — it builds the chain generator directly via
    :func:`make_combination_gen` and interleaves many samples.

    ``combined.techniques`` exposes the ordered inner technique list (used by
    the engine and the manifest); ``combined.__name__`` is the '+'-joined names
    (or ``"identity"`` when empty).

    Returns:
        Combined callable: ``(str, **kwargs) -> (str, str)`` with
        ``.techniques`` and ``.__name__`` attributes.
    """
    spec = load_spec()
    layer_order = spec["layer_order"]

    def _sort_key(fn: Callable) -> tuple[int, int]:
        layer = getattr(fn, "layer", None)
        layer_idx = layer_order.index(layer) if layer in layer_order else len(layer_order)
        wlo = getattr(fn, "within_layer_order", 0)
        return (layer_idx, wlo)

    ordered = sorted(techniques, key=_sort_key) if sort_by_hierarchy else list(techniques)

    def combined(text: str, **kwargs) -> tuple[str, str]:
        gen = _run_chain(ordered, text, **kwargs)
        client = kwargs.get("client")

        def call(request: LLMRequest) -> str:
            # Lazy imports keep utils import-time light and avoid any
            # llms<->jailbreak import ordering surprises.
            from redact.llms.client import ModelClient
            from redact.llms.router import generate_sample
            c = client
            if c is None or c.model != request.model:
                # A chain can mix models (e.g. gen vs translate); resolve the
                # client for whichever model this request actually targets.
                c = ModelClient.create(request.model)
            return generate_sample(c, request.messages)

        return run_sync(gen, call)

    combined.__name__ = "+".join(t.__name__ for t in ordered) or "identity"
    combined.techniques = list(ordered)  # expose inner techniques for inspection
    return combined


def make_combination_gen(fn: Callable, text: str, **kwargs) -> TechniqueGen:
    """Build the chain *generator* for a combined technique, for the engine.

    Reads ``fn.techniques`` (set by :func:`combine_techniques`) and returns a
    generator that yields :class:`LLMRequest` per LLM step. The engine drives
    many of these concurrently, pooling yields by model. ``kwargs`` should
    carry ``gen_model`` and ``benign_data``.
    """
    techniques = getattr(fn, "techniques", None)
    if techniques is None:
        techniques = [fn]
    return _run_chain(techniques, text, **kwargs)


def _parse_rejection_info(info_str: str) -> tuple[bool, str]:
    """Parse technique info string to extract acceptance status and reason.

    Returns (accepted: bool, reasoning: str):
    - If info_str starts with "DISCARDED": accepted=False, reasoning=full info string
    - Otherwise: accepted=True, reasoning="" (empty)

    The "DISCARDED" prefix is a sentinel set by with_feedback_retries() when
    checks are exhausted, so checking it reliably identifies rejections.
    """
    if info_str.startswith("DISCARDED"):
        return False, info_str
    return True, ""


def apply_combination(
    fn: Callable,
    prompt: str,
    client=None,
    model: str | None = None,
    benign_data: dict | None = None,
    auto_benign: bool = True,
    benign_cache_path=None,
) -> tuple[str, str, bool, str]:
    """Apply a technique function (or combined function) to a prompt.

    Handles benign_data auto-generation for manipulation techniques:
    if the function (or any inner technique in a combined chain) requires
    benign_data and none is provided, it is loaded from cache or generated.

    Args:
        fn: A technique function or the result of combine_techniques().
        prompt: The input prompt string.
        client: ModelClient (required for LLM-dependent techniques).
        model: Model identifier used to tag gen-model requests; defaults to
            ``client.model`` when a client is given.
        benign_data: Pre-loaded benign data dict for FSH/DAP manipulation.
        auto_benign: If True, auto-load or auto-generate benign data when needed.
        benign_cache_path: Path to benign CSV cache. Uses default if None.

    Returns:
        (result_text, additional_info, accepted: bool, reasoning: str)
        - accepted: False if any step produced "DISCARDED; ..." output
        - reasoning: Full rejection message if accepted=False, empty string otherwise
    """
    # Check if benign_data is needed
    inner_fns = getattr(fn, "techniques", [fn])
    needs_benign = any("benign_data" in inspect.signature(f).parameters for f in inner_fns)

    if needs_benign and benign_data is None and auto_benign:
        from redact.jailbreak.manipulation.benign import get_or_generate_benign_data
        benign_data = get_or_generate_benign_data(
            client=client,
            cache_path=benign_cache_path,
        )

    output = fn(
        prompt,
        client=client,
        gen_model=model or (client.model if client is not None else None),
        benign_data=benign_data,
    )
    text, info = _normalize_output(output)

    # Parse rejection status from info string
    accepted, reasoning = _parse_rejection_info(info)
    return text, info, accepted, reasoning


def is_noop(original: str, result: str) -> bool:
    """Return True if a technique produced no change to the text.

    Used by pipeline code to discard samples where a transform (e.g.,
    sensitive_words with no detectable harmful words) had no effect.
    """
    return original == result
