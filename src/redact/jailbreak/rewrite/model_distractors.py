"""Model-distractor techniques: a model writes a few sentences in the voice of
the person asking (``jailbreak/model_distractor``, its ``{position}``), and
they are joined to the prompt, before it or after it. The prompt itself is not
rewritten, and no check model reads the reply.

A reply with no text is asked again, up to :data:`MAX_REPLIES` replies, then
the prompt is discarded (:data:`EMPTY_DISTRACTOR`). ``info`` is
``position=<position>``.
"""

from collections.abc import Callable

from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import extracted
from redact.llms.prompting import build_messages, load_prompt

#: The most replies asked for one prompt: a reply with no text is asked again.
MAX_REPLIES = 3

#: The info of a prompt whose replies never held a sentence.
EMPTY_DISTRACTOR = f"{DISCARDED}; reason=empty distractor"


def _sentences_of(reply: str) -> list[str]:
    """``[reply]`` stripped, or ``[]`` for a reply with no text, which
    :func:`~redact.llm_pipeline.extracted` asks for again."""
    return [reply.strip()] if reply.strip() else []


def _distractor_gen(
    prompt: str, position: str, *, gen_model: str | None = None, prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Ask the model for sentences to place ``position`` (``before`` or
    ``after``) the prompt, and join them to it with a space."""
    messages = build_messages(
        load_prompt("jailbreak", "model_distractor", prompt_dir),
        position=position, prompt=prompt)
    sentences, _attempts = yield from extracted(
        gen_model, lambda: messages, _sentences_of, max_attempts=MAX_REPLIES)
    if not sentences:
        return prompt, EMPTY_DISTRACTOR
    parts = [sentences[0], prompt] if position == "before" else [prompt, sentences[0]]
    return " ".join(parts), f"position={position}"


def to_model_distractor_prefix(prompt: str, **kwargs) -> TechniqueGen:
    """Put sentences a model wrote before the prompt."""
    return (yield from _distractor_gen(prompt, "before", **kwargs))


def to_model_distractor_suffix(prompt: str, **kwargs) -> TechniqueGen:
    """Put sentences a model wrote after the prompt."""
    return (yield from _distractor_gen(prompt, "after", **kwargs))


def get_model_distractor_functions() -> list[Callable]:
    """Return the model-distractor techniques."""
    return [to_model_distractor_prefix, to_model_distractor_suffix]
