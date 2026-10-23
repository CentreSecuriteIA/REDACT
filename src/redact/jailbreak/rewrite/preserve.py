"""What the persuasion and tense-rewrite techniques share: a model rewrites the
whole prompt, and a check model reads the rewrite against the prompt for the
same request (``jailbreak/preserve_check``).

``info`` is the technique's own label (``category=<key>``, ``tense=<tense>``);
a rewrite the check rejects every time is ``DISCARDED; reason=preserve check
failed; <label>; feedback=<verdict>``, and an accepted rewrite with no text, or
the prompt itself, is ``DISCARDED; reason=empty rewrite; <label>`` or
``unchanged rewrite``.
"""

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate

#: The info of an accepted rewrite that has no text, or is the prompt unchanged;
#: the label follows.
EMPTY_REWRITE = f"{DISCARDED}; reason=empty rewrite"
UNCHANGED_REWRITE = f"{DISCARDED}; reason=unchanged rewrite"


def preserved_rewrite_gen(
    prompt: str,
    messages: list[dict],
    label: str,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
) -> TechniqueGen:
    """Have ``gen_model`` answer ``messages`` with a rewrite of ``prompt``, and
    ``check_model`` (``gen_model`` when None) check the rewrite against it
    (``jailbreak/preserve_check``, at :data:`CHECK_TEMPERATURE`).

    A rejected rewrite is made again with the verdict as feedback, up to
    ``check_attempts`` rewrites, so one application makes 2 to
    ``2 * check_attempts`` calls. Returns ``(rewrite, label)``, or the last
    rewrite with a ``DISCARDED`` info when every one is rejected. A rewrite
    the check accepts but that has no text, or is the prompt, is discarded
    too (:data:`EMPTY_REWRITE`, :data:`UNCHANGED_REWRITE`), the prompt
    returned: a check model may accept either.
    """
    check = PromptTemplate.load("jailbreak", "preserve_check", prompt_dir=prompt_dir)
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(messages, feedback),
        check_model or gen_model,
        lambda original, rewritten: check(original=original, rewritten=rewritten),
        original=prompt, max_attempts=check_attempts,
        temperatures=(None, CHECK_TEMPERATURE),
    )
    rewrite = text.strip()
    if not accepted:
        return rewrite, (f"{DISCARDED}; reason=preserve check failed; "
                         f"{label}; feedback={verdict}")
    if not rewrite:
        return prompt, f"{EMPTY_REWRITE}; {label}"
    if rewrite == prompt.strip():
        return prompt, f"{UNCHANGED_REWRITE}; {label}"
    return rewrite, label
