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

from collections import defaultdict

from redact.llms import get_router

from .protocol import LLMRequest
from .utils import make_combination_gen, _parse_rejection_info, is_noop


def _finalize(sample: dict, value) -> dict:
    """Build the output row for a completed sample from its ``(text, info)``."""
    text, info = value
    accepted, reasoning = _parse_rejection_info(info)
    fn = sample["combination"]
    techniques = list(getattr(fn, "techniques", []))
    return {
        "input_id": sample.get("id", ""),
        "jailbreak": text,
        "technique": getattr(fn, "__name__", "identity"),
        "technique_info": info,
        "complexity": sum(getattr(t, "complexity", 0) for t in techniques),
        "num_techniques": len(techniques),
        "is_noop": is_noop(sample["prompt"], text),
        "accepted": accepted,
        "reasoning": reasoning,
    }


def _finalize_error(sample: dict, exc: Exception) -> dict:
    """Build the output row for a sample whose chain raised — original kept."""
    fn = sample["combination"]
    techniques = list(getattr(fn, "techniques", []))
    return {
        "input_id": sample.get("id", ""),
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
) -> list[dict]:
    """Apply each sample's technique combination, batching LLM calls per round.

    Args:
        samples: ``[{"id": str, "prompt": str, "combination": fn}, ...]`` where
            ``fn`` is a combined callable from ``utils.combine_techniques`` /
            ``utils.build_combination`` (carries ``.techniques`` + ``.__name__``).
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

    Returns:
        One result dict per input sample, in input order. Keys: ``input_id``,
        ``jailbreak``, ``technique``, ``technique_info``, ``complexity``,
        ``num_techniques``, ``is_noop``, ``accepted``, ``reasoning``.
    """
    router = router or get_router()
    n = len(samples)
    results: list[dict | None] = [None] * n
    gens: dict[int, object] = {}
    pending: dict[int, LLMRequest] = {}

    def advance(i: int, response):
        """Resume sample ``i`` with ``response`` (None to prime); record/queue."""
        gen = gens[i]
        try:
            pending[i] = gen.send(response)
        except StopIteration as stop:
            results[i] = _finalize(samples[i], stop.value)
        except Exception as exc:  # noqa: BLE001 — isolate one sample's failure
            results[i] = _finalize_error(samples[i], exc)

    # Prime every sample: run inline pure steps up to the first LLM request
    # (or to completion for all-pure chains).
    for i, s in enumerate(samples):
        gens[i] = make_combination_gen(
            s["combination"], s["prompt"],
            gen_model=gen_model, translate_model=translate_model,
            benign_data=benign_data,
        )
        advance(i, None)

    # Drive remaining samples round by round: one batch call per model per round.
    round_idx = 0
    while pending:
        round_idx += 1
        groups: dict[str, list[int]] = defaultdict(list)
        for i, req in pending.items():
            groups[req.model].append(i)

        round_requests = dict(pending)
        pending.clear()

        for model, idxs in groups.items():
            messages_list = [round_requests[i].messages for i in idxs]
            # Pass a progress label only when verbose, so non-verbose runs keep
            # the minimal router.batch_generate(model, messages) contract.
            kwargs = (
                {"progress": f"round {round_idx} ({model})"} if verbose else {}
            )
            try:
                responses = router.batch_generate(model, messages_list, **kwargs)
            except Exception as exc:  # noqa: BLE001 — whole batch failed
                for i in idxs:
                    results[i] = _finalize_error(samples[i], exc)
                continue
            for i, resp in zip(idxs, responses):
                advance(i, resp)

    return [r for r in results if r is not None]
