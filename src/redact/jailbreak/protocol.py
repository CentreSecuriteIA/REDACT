"""Request protocol and generator contract for the batched jailbreak engine.

The batched engine ([engine.py]) advances every sample through its technique
chain layer-by-layer: pure transforms run inline, LLM steps are pooled per
model and dispatched in one batch call. To make multi-round LLM techniques
(translation's translate→check→retry loop, cognitive's scenario→construction)
participate in that round-by-round scheme uniformly, each LLM technique is a
**generator**: it ``yield``s an :class:`LLMRequest` and resumes (via
``.send(response)``) with the model's reply.

Technique generator contract::

    def to_x(prompt, *, gen_model=None, ...) -> TechniqueGen:
        response = yield LLMRequest(model, messages)   # hand request to engine
        ...                                            # may yield again (N rounds)
        return text, info                              # info "" ok / "DISCARDED; ..."

Pure transforms stay plain ``(str, **kwargs) -> (str, str)`` functions; the
combined chain in ``utils.combine_techniques`` runs them inline (zero rounds).

:func:`run_sync` drives a technique generator to completion with a blocking
call function. It is the single-sample / test path used by
``utils.apply_combination`` — the engine never calls it (it interleaves many
generators instead), but it guarantees the generator form behaves identically
when run one sample at a time.
"""

from __future__ import annotations

from typing import Callable, Generator

# LLMRequest is a model-layer type — defined in redact.llms.conversation and
# re-exported here so existing jailbreak imports (`from .protocol import LLMRequest`)
# keep working while `llms/` and `multi_turn/` share the same class.
from redact.llms.conversation import LLMRequest  # noqa: F401


# A technique generator yields LLMRequests, is resumed with the response
# string, and returns the (text, info) result. info is "" on success or
# starts with "DISCARDED; " on rejection.
TechniqueResult = tuple[str, str]
TechniqueGen = Generator[LLMRequest, str, TechniqueResult]


def run_sync(gen: TechniqueGen, call: Callable[[LLMRequest], str]):
    """Drive a technique generator to completion with a blocking call fn.

    Args:
        gen: A primed-or-unprimed technique generator object.
        call: ``callable(LLMRequest) -> str`` that performs the LLM call
            (e.g. routing each request to its backend by ``request.model``).

    Returns:
        The generator's return value — ``(text, info)`` for technique
        generators, or whatever a combined-chain generator returns.
    """
    try:
        request = next(gen)
        while True:
            response = call(request)
            request = gen.send(response)
    except StopIteration as stop:
        return stop.value
