"""What the persuasion and tense-rewrite techniques share: a model rewrites the
whole prompt, and a check model reads the rewrite against the prompt for the
same request (``jailbreak/preserve_check``).

``info`` is the technique's own label (``category=<key>``, ``tense=<tense>``);
a rewrite the check rejects every time is ``DISCARDED; reason=preserve check
failed; <label>; feedback=<verdict>``.
"""

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate


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
    rewrite with a ``DISCARDED`` info when every one is rejected.
    """
    check = PromptTemplate.load("jailbreak", "preserve_check", prompt_dir=prompt_dir)
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(messages, feedback),
        check_model or gen_model,
        lambda original, rewritten: check(original=original, rewritten=rewritten),
        original=prompt, max_attempts=check_attempts,
        temperatures=(None, CHECK_TEMPERATURE),
    )
    if not accepted:
        return text.strip(), (f"{DISCARDED}; reason=preserve check failed; "
                              f"{label}; feedback={verdict}")
    return text.strip(), label
