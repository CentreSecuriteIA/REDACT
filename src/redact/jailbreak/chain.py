"""Run one input through its techniques: transforms inline, generators by
yielding their calls.
"""

import dataclasses
import inspect
import random
from collections.abc import Callable, Iterable
from pathlib import Path

from redact.dataset.dealing import stable_seed
from redact.llm_pipeline import LLMRequest, drive_sync
from redact.llms.client import ModelClient
from redact.llms.router import generate_sample

from .manipulation import get_or_generate_benign_data
from .technique import DISCARDED, TechniqueGen, TechniqueResult, is_discarded, load_spec


def _select_kwargs(technique: Callable, kwargs: dict) -> dict:
    """The part of ``kwargs`` that ``technique`` takes: all of it when it
    takes ``**kwargs``."""
    params = inspect.signature(technique).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _wrap(gen: TechniqueGen, limit: int | None, root: str | None) -> TechniqueGen:
    """``gen`` with each request capped at ``limit`` and, under ``root``,
    tagged for internals capture.

    A request that carries its own limit keeps it. A request whose model can
    capture internals gets the id ``{root}/{n}``, numbered in the order they
    are yielded. That is read off the request's client, or else the model's
    backend class, so no model is loaded to ask.
    """
    from redact.llms.model_config import model_compute_config  # avoids an import cycle

    n = 0
    try:
        request = gen.send(None)
    except StopIteration as stop:
        return stop.value
    while True:
        if limit is not None and request.max_tokens is None:
            request = dataclasses.replace(request, max_tokens=limit)
        if root is not None:
            config = (request.client.compute_config if request.client is not None
                      else model_compute_config(request.model))
            if config.supports_internals:
                request = dataclasses.replace(request, internals_id=f"{root}/{n}")
                n += 1
        reply = yield request
        try:
            request = gen.send(reply)
        except StopIteration as stop:
            return stop.value


def run_chain(
    techniques: list[Callable],
    text: str,
    *,
    internals_root: str | None = None,
    rng_seed: tuple | None = None,
    **kwargs,
) -> TechniqueGen:
    """Run ``text`` through ``techniques`` in order, yielding each model call.

    A transform runs inline. A generator's requests are yielded up, capped at
    the technique's ``max_tokens`` tag and, with ``internals_root``, tagged for
    capture under ``{internals_root}/{technique}``. Each technique gets the
    ``kwargs`` it takes (``gen_model``, ``translate_model``, ``check_model``,
    ``benign_data``, ``prompt_dir``, and ``source_prompt``: ``text``, the
    input the chain starts from, unless the caller gives one other than
    ``None``) and, with
    ``rng_seed``, an ``rng`` of its own:
    ``random.Random(stable_seed(*rng_seed, technique.__name__))``. The
    ``source_prompt`` is for a technique that reads the clean input rather
    than what the steps before it wrote: a selected ``fsh`` or ``dap``
    variant, applied last, picks its examples' subcategory on it and inserts
    the finished text among them.

    Returns ``(text, info)``: the steps' infos joined by ``;``, a step that
    left the text unchanged as ``noop=<name>``. The first rejection stops the
    chain and returns the text before it, with an info naming the technique;
    a step that empties the text (an empty model reply, a render with nothing
    drawn) is a rejection.
    """
    if kwargs.get("source_prompt") is None:
        kwargs["source_prompt"] = text
    result = text
    info_parts: list[str] = []
    for technique in techniques:
        if rng_seed is not None:
            kwargs["rng"] = random.Random(stable_seed(*rng_seed, technique.__name__))
        tech_kwargs = _select_kwargs(technique, kwargs)
        if inspect.isgeneratorfunction(technique):
            gen = technique(result, **tech_kwargs)
            limit = getattr(technique, "max_tokens", None)
            root = f"{internals_root}/{technique.__name__}" if internals_root else None
            if limit is not None or root is not None:
                gen = _wrap(gen, limit, root)
            new_text, info = yield from gen
        else:
            new_text, info = technique(result, **tech_kwargs)

        if is_discarded(info):
            return result, (
                f"{DISCARDED}; technique={technique.__name__}; "
                f"{info[len(f'{DISCARDED}; '):]}"
            )
        if new_text == result:
            info_parts.append(f"noop={technique.__name__}")
        elif not new_text.strip():
            return result, (
                f"{DISCARDED}; technique={technique.__name__}; reason=empty text")
        elif info:
            info_parts.append(info)
        result = new_text
    return result, ";".join(info_parts)


def in_layer_order(techniques: Iterable[Callable]) -> list[Callable]:
    """``techniques`` in the order they apply: by the spec's layer order
    (``rewrite``, ``hacking``, ``obfuscation``, ``requests``, then
    ``manipulation``, which inserts the finished question), then by rank
    within the layer (``within_layer_order``: in the obfuscation layer, the
    whole-text rewrites before the word-level steps)."""
    layer_order = load_spec()["layer_order"]

    def _sort_key(fn: Callable) -> tuple[int, int]:
        layer = getattr(fn, "layer", None)
        rank = layer_order.index(layer) if layer in layer_order else len(layer_order)
        return (rank, getattr(fn, "within_layer_order", 0))

    return sorted(techniques, key=_sort_key)


def _call_with(client: ModelClient | None) -> Callable[[LLMRequest], str]:
    """A blocking call for :func:`drive_sync`.

    A request goes to its own client, else to ``client`` when that serves the
    request's model, else to a client made for the model, with its reply
    limit and temperature.
    """
    def call(request: LLMRequest) -> str:
        c = request.client or client
        if c is None or c.model != request.model:
            c = ModelClient.create(request.model)
        return generate_sample(c, request.messages, max_tokens=request.max_tokens,
                               temperature=request.temperature)

    return call


def combine_techniques(
    *techniques: Callable, sort_by_hierarchy: bool = True,
) -> Callable:
    """``techniques`` as one technique, in layer order unless
    ``sort_by_hierarchy`` is False.

    ``combined(text, client=None, **kwargs)`` runs the chain to its end, one
    blocking model call at a time, and returns ``(text, info)``.
    ``combined.techniques`` is the ordered list and ``combined.__name__`` their
    names joined by ``+``, ``"identity"`` when there are none.
    """
    ordered = in_layer_order(techniques) if sort_by_hierarchy else list(techniques)

    def combined(
        text: str, client: ModelClient | None = None, **kwargs,
    ) -> TechniqueResult:
        return drive_sync(run_chain(ordered, text, **kwargs), _call_with(client))

    combined.__name__ = "+".join(t.__name__ for t in ordered) or "identity"
    combined.techniques = list(ordered)
    return combined


def apply_combination(
    fn: Callable,
    prompt: str,
    client: ModelClient | None = None,
    benign_data: dict | None = None,
    auto_benign: bool = True,
    benign_cache_path=None,
    prompt_dir=None,
) -> tuple[str, str, bool, str]:
    """Run one prompt through a technique or a combined one.

    Requests carry ``client.model`` as the generation model. When a technique
    is tagged ``requires_benign`` and no ``benign_data`` is given, it is loaded
    from ``benign_cache_path``, its missing pairs generated first with
    ``client`` when ``auto_benign``; without ``auto_benign`` only a cache that
    exists is loaded.
    ``prompt_dir`` is the prompt overlay of the techniques that take it and of
    the benign data.

    Returns:
        ``(text, info, accepted, reasoning)``: ``accepted`` is False when a
        step rejected, and ``reasoning`` is then the info, else ``""``.
    """
    techniques = getattr(fn, "techniques", [fn])
    cached = benign_cache_path is not None and Path(benign_cache_path).exists()
    if (benign_data is None and (auto_benign or cached)
            and any(getattr(t, "requires_benign", False) for t in techniques)):
        benign_data = get_or_generate_benign_data(
            client=client if auto_benign else None, cache_path=benign_cache_path,
            prompt_dir=prompt_dir)

    gen = run_chain(
        techniques, prompt,
        gen_model=client.model if client is not None else None,
        benign_data=benign_data,
        prompt_dir=prompt_dir,
    )
    text, info = drive_sync(gen, _call_with(client))
    accepted = not is_discarded(info)
    return text, info, accepted, "" if accepted else info
