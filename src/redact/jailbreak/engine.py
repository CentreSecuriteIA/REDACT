"""Batched, generator-driven jailbreak combination engine.

Advances a chunk of samples through their technique chains **round by round**:

1. Each sample's chain is a generator (``utils.make_combination_gen``). Pure
   transforms run inline as the generator advances; the generator pauses only
   to ``yield`` an :class:`LLMRequest` for an LLM step.
2. Each round, the engine collects every live sample's pending request, groups
   them by ``request.model``, and dispatches **one batch per model** through
   the process-wide router (``get_router().batch_generate``). The router picks
   the execution mode from backend capabilities — vLLM native batch, API
   thread-pool, or sequential — so the engine never branches on backend type.
3. Responses are fed back via ``gen.send(...)``; each sample advances to its
   next request or returns ``(text, info)``. Multi-round techniques
   (translation translate→check→retry, cognitive scenario→construction)
   re-enter the pool naturally on the next round.
4. A sample that returns ``DISCARDED`` info, or raises, exits the pool without
   blocking the others.

This is the pipeline-scale path. The single-sample path (``apply_combination``)
runs the same chains synchronously via ``protocol.run_sync``.
"""

from __future__ import annotations

from redact.dataset.io import _hash_text
from redact.llms import get_router
from redact.llms.conversation import drive_generators

from .utils import _parse_rejection_info, is_noop, make_combination_gen


def _rename_capture(sample: dict, gen_model: str, final_sample_id: str) -> None:
    """Relabel a sample's provisional internals capture to its final sample_id.

    No-op unless the sample actually carries a provisional root (i.e.
    capture_internals was on for this run) — safe to call unconditionally.
    """
    root = sample.get("_internals_root")
    if not root:
        return
    from redact.llms.api import get_backend
    get_backend(gen_model).rename_capture(root, f"{sample.get('id', '')}/jailbreak/{final_sample_id}")


def _finalize(sample: dict, value, gen_model: str) -> dict:
    """Build the output row for a completed sample from its ``(text, info)``."""
    text, info = value
    accepted, reasoning = _parse_rejection_info(info)
    fn = sample["combination"]
    techniques = list(getattr(fn, "techniques", []))
    sample_id = _hash_text(text)
    _rename_capture(sample, gen_model, sample_id)
    return {
        "input_id": sample.get("id", ""),
        "sample_id": sample_id,
        "jailbreak": text,
        "technique": getattr(fn, "__name__", "identity"),
        "technique_info": info,
        "complexity": sum(getattr(t, "complexity", 0) for t in techniques),
        "num_techniques": len(techniques),
        "is_noop": is_noop(sample["prompt"], text),
        "accepted": accepted,
        "reasoning": reasoning,
    }


def _finalize_error(sample: dict, exc: Exception, gen_model: str) -> dict:
    """Build the output row for a sample whose chain raised — original kept."""
    fn = sample["combination"]
    techniques = list(getattr(fn, "techniques", []))
    sample_id = _hash_text(sample["prompt"])
    _rename_capture(sample, gen_model, sample_id)
    return {
        "input_id": sample.get("id", ""),
        "sample_id": sample_id,
        "jailbreak": sample["prompt"],
        "technique": getattr(fn, "__name__", "identity"),
        "technique_info": f"ERROR: {exc}",
        "complexity": sum(getattr(t, "complexity", 0) for t in techniques),
        "num_techniques": len(techniques),
        "is_noop": True,
        "accepted": False,
        "reasoning": str(exc),
    }


def batch_apply_combinations(
    samples: list[dict],
    *,
    gen_model: str,
    translate_model: str | None = None,
    benign_data: dict | None = None,
    router=None,
    verbose: bool = False,
    capture_internals: bool = False,
) -> list[dict]:
    """Apply each sample's technique combination, batching LLM calls per round.

    Args:
        samples: ``[{"id": str, "prompt": str, "combination": fn}, ...]`` where
            ``fn`` is a combined callable from ``utils.combine_techniques`` /
            ``utils.build_combination`` (carries ``.techniques`` + ``.__name__``).
            An optional ``"iteration"`` key seeds the provisional internals
            path when ``capture_internals`` is on (defaults to 0).
        gen_model: Model name that generation-layer techniques tag their
            requests with (translation tags its own translation-role model and
            ignores this).
        benign_data: Pre-loaded benign Q&A dict for FSH/DAP techniques.
        router: Optional router override (defaults to the process-wide
            :func:`redact.llms.get_router`). Injectable for testing.
        verbose: When True, emit live per-round progress via the shared
            ``progress`` label on ``router.batch_generate`` (one tick stream
            per model per round). When False, dispatch with no progress kwarg
            (preserves the minimal router contract used by tests).
        capture_internals: When True, each sample's LLM-backed technique calls
            get tagged for internals capture (per-request, only for whichever
            target model's backend actually supports it — see
            ``utils._tag_yields``), under a provisional
            ``{id}/jailbreak/attempt_{iteration}`` root; ``_finalize`` relabels
            it to ``{id}/jailbreak/{sample_id}`` once the final jailbroken
            text (and thus its real sample_id) is known — a pure side effect,
            never a CSV column (see .claude/introspection_backend_plan.md).

    Returns:
        One result dict per input sample, in input order. Keys: ``input_id``,
        ``sample_id``, ``jailbreak``, ``technique``, ``technique_info``,
        ``complexity``, ``num_techniques``, ``is_noop``, ``accepted``,
        ``reasoning``.
    """
    router = router or get_router()
    n = len(samples)

    # Build one chain generator per sample; the generic round-driver
    # (llms.conversation.drive_generators) pools LLM calls by model per round.
    gens = {}
    for i, s in enumerate(samples):
        internals_root = None
        if capture_internals:
            internals_root = f"{s.get('id', '')}/jailbreak/attempt_{s.get('iteration', 0)}"
            s["_internals_root"] = internals_root  # read back in _finalize for the rename
        gens[i] = make_combination_gen(
            s["combination"], s["prompt"],
            gen_model=gen_model, translate_model=translate_model,
            benign_data=benign_data, internals_root=internals_root,
        )
    results = drive_generators(
        gens,
        router=router,
        finalize=lambda i, value: _finalize(samples[i], value, gen_model),
        on_error=lambda i, exc: _finalize_error(samples[i], exc, gen_model),
        verbose=verbose,
        progress="jailbreak" if verbose else None,
    )
    return [results[i] for i in range(n) if i in results]
