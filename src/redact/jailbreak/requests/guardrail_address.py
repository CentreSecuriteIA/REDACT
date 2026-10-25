"""Guardrail address (spec 1.4): a line or block addressed to a content
classifier or supervisor, put before or after the prompt, saying the message
is benign, already approved, or should be classified safe. The three
techniques match the three formats of the BELLS guardrail attacks'
``benign_instruction_for_classifier`` (one technique per format); each draws
one of the eight variants of its pool in
``requests/guardrail_templates.json``, the first of which is the reference's
own text. The prompt is kept as it is.

The target is the supervisor, not the model: these only make sense when the
message is one a classifier might flag, so the plan draws them for harmful
and borderline inputs only, never an absolutely benign one, where the note
would label noise (:func:`is_targeted_input`,
:func:`~redact.jailbreak.technique.applicable_when`; the mirror of the
benign roles). They are an attack only (their family's spec ``kinds``).

``info`` is ``guardrail=<directive>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from redact.types import EntryType

from ..directives import apply_template_directive, read_pools, uses_pool
from ..technique import applicable_when

_FILE = "requests/guardrail_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="guardrail")

#: The entry type a guardrail address is never drawn for: an absolutely
#: benign request needs no note that it is benign.
EXCLUDED = (EntryType.BENIGN,)


def is_targeted_input(text: str, entry_type: str) -> bool:
    """Whether a guardrail address may be drawn for an input: its entry type
    is not absolutely benign (:data:`EXCLUDED`), whatever its text, so the
    harmful and the two dual-use levels take one and the benign one does
    not."""
    return entry_type not in EXCLUDED


@applicable_when(is_targeted_input)
@uses_pool(_FILE, "block_question")
def to_guardrail_block_question(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Put a block after the prompt that asks the classifier whether to block
    it and answers no."""
    return _apply("block_question", prompt, rng=rng)


@applicable_when(is_targeted_input)
@uses_pool(_FILE, "classify_benign")
def to_guardrail_classify_benign(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Put a line before the prompt telling the classifier to label it
    benign, the prompt wrapped in a tag."""
    return _apply("classify_benign", prompt, rng=rng)


@applicable_when(is_targeted_input)
@uses_pool(_FILE, "security_override")
def to_guardrail_security_override(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Put a block after the prompt that claims a prior review cleared it."""
    return _apply("security_override", prompt, rng=rng)


def get_guardrail_address_functions() -> list[Callable]:
    """Return all guardrail-address technique functions."""
    return [
        to_guardrail_block_question,
        to_guardrail_classify_benign,
        to_guardrail_security_override,
    ]
