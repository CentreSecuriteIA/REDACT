"""Prompt loading and message assembly — the one place LLM-ready chat messages
get built from a prompt JSON file (see CLAUDE.md's "Prompt JSON Schema").

``load_prompt()`` reads the file; ``PromptTemplate`` (and its one-shot wrapper
``build_messages()``) renders it into ``[system, *few_shot, user]`` messages,
in two stages so a fixed system prompt is built once and reused across many
per-item calls (e.g. a checker validating many samples).

**Prompts resolve per file, not per tree.** ``prompt_dir`` is an overlay: each
prompt comes from there if present and from the packaged copy otherwise. So
customizing one checker means copying one file, not forking all 34 and then
silently missing every later improvement to the rest.
:func:`report_prompt_sources` prints the resulting picture once at startup,
since with two sources "which prompt ran?" is no longer obvious.
"""

import json
import logging
import string
from pathlib import Path

from ... import paths

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_DIR = paths.prompts_dir()

#: Above this many overrides, the startup report gives a count instead of one
#: line each — the summary exists to replace per-load chatter, not to become
#: its own.
_MAX_LISTED_OVERRIDES = 10


def _prompt_file(root: Path, pipeline: str, category: str) -> Path | None:
    """The JSON file for one prompt under ``root``, or ``None`` if absent.

    A directory holding exactly one ``.json`` uses it; otherwise
    ``template.json``. Returns ``None`` rather than raising, since a miss is
    the normal case when probing a user's override directory.
    """
    base = root / pipeline / category
    if not base.is_dir():
        return None
    json_files = list(base.glob("*.json"))
    target = json_files[0] if len(json_files) == 1 else base / "template.json"
    return target if target.exists() else None


def resolve_prompt(
    pipeline: str,
    category: str,
    prompt_dir: str | Path | None = None,
) -> tuple[Path, bool]:
    """Which file a prompt actually resolves to, and whether it is an override.

    Split out from :func:`load_prompt` so the startup report can answer "which
    prompt will run?" without reading every file — with two sources, that
    question has to be answerable or a stale override in someone's directory
    is invisible.

    Returns:
        ``(path, is_override)``.

    Raises:
        FileNotFoundError: If neither the override directory nor the packaged
            tree has this prompt.
    """
    if prompt_dir is not None:
        override = _prompt_file(Path(prompt_dir), pipeline, category)
        if override is not None:
            return override, True

    packaged = _prompt_file(_DEFAULT_PROMPT_DIR, pipeline, category)
    if packaged is not None:
        return packaged, False

    searched = [str(_DEFAULT_PROMPT_DIR / pipeline / category)]
    if prompt_dir is not None:
        searched.insert(0, str(Path(prompt_dir) / pipeline / category))
    raise FileNotFoundError(
        f"No prompt found for {pipeline}/{category}. Looked in: "
        + ", ".join(searched)
        + ". Expected a single .json file or template.json."
    )


def load_prompt(
    pipeline: str,
    category: str,
    prompt_dir: str | Path | None = None,
) -> dict:
    """Load one prompt, preferring a user's override over the packaged copy.

    ``prompt_dir`` is an **overlay, not a replacement**: each prompt is taken
    from there if present and from the library otherwise. That is what lets
    someone customize one checker by copying one file, instead of forking the
    whole tree and then silently missing every later improvement to the other
    33 — the failure mode a whole-directory override has and this does not.

    Args:
        pipeline: Pipeline name (e.g. "input", "output", "jailbreak",
            "constitution").
        category: Category within it (e.g. "quality_check",
            "generation/standalone").
        prompt_dir: Optional directory searched first. ``None`` uses the
            packaged prompts only.

    Returns:
        Parsed JSON dict.

    Raises:
        FileNotFoundError: If neither source has this prompt.

    Example:
        >>> load_prompt("input", "quality_check")  # doctest: +SKIP
    """
    target, _ = resolve_prompt(pipeline, category, prompt_dir)
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def report_prompt_sources(
    prompt_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Log once, at startup, which prompts come from where.

    Two sources means "which prompt ran?" stops being obvious, and a stale
    override is the kind of thing you debug in the library for an hour before
    thinking to look in your own directory. One line at startup beats a log
    line per load.

    Also flags directories in ``prompt_dir`` matching **no** packaged prompt.
    A typo'd ``input/quality_chek/`` overrides nothing and appears nowhere, so
    the edit silently has no effect — that is the common mistake, and it is
    invisible without this.

    Args:
        prompt_dir: The override directory, or ``None``.

    Returns:
        ``{"overridden": [...], "unused": [...], "packaged": [...]}`` — all
        three are name lists, so a caller can treat the result uniformly and
        call ``len()`` on whichever it cares about. (An earlier version
        returned a count for ``packaged`` on the grounds that nobody reads 34
        names back; that saved nothing and forced the whole return type down
        to ``dict[str, object]``.)
    """
    packaged = sorted(
        f"{p.parent.relative_to(_DEFAULT_PROMPT_DIR)}".replace("\\", "/")
        for p in _DEFAULT_PROMPT_DIR.rglob("*.json")
    )
    if prompt_dir is None:
        logger.debug("prompts: %d from library, no override directory", len(packaged))
        return {"overridden": [], "unused": [], "packaged": packaged}

    root = Path(prompt_dir)
    user = sorted(
        f"{p.parent.relative_to(root)}".replace("\\", "/") for p in root.rglob("*.json")
    ) if root.is_dir() else []
    overridden = [name for name in user if name in packaged]
    unused = [name for name in user if name not in packaged]

    logger.info(
        "prompts: %d from library, %d overridden from %s",
        len(packaged) - len(overridden), len(overridden), root,
    )
    # Named individually only while the list is short enough to read. Someone
    # overriding the whole tree would otherwise turn a one-block summary into
    # 34 lines, which is the thing this was meant to replace.
    if overridden:
        if len(overridden) <= _MAX_LISTED_OVERRIDES:
            for name in overridden:
                logger.info("  override: %s", name)
        else:
            logger.info("  (%d overrides; run with --debug to list them)",
                        len(overridden))
            logger.debug("  overrides: %s", ", ".join(overridden))
    # Always named, however many: an unused override is a mistake, and the
    # name is the whole content of the message.
    for name in unused:
        logger.warning(
            "  unused override: %s — no packaged prompt by that name, so this "
            "file is never loaded (typo?)", name,
        )
    return {"overridden": overridden, "unused": unused, "packaged": packaged}


def _render(text: str, kwargs: dict[str, str], what: str) -> str:
    """Render one field, or explain why it could not be rendered.

    Fast path first — a bare ``format_map``, since almost every call succeeds
    and analysis would be wasted on it. The template is parsed only once
    something raises, by :func:`_explain`, which is where the real diagnosis
    happens.

    Rendering used to be guarded by ``if text and kwargs``, so with *no*
    kwargs a prompt containing ``{Category}`` went to the model with the
    braces still in it, silently, while *partial* kwargs raised a bare
    ``KeyError`` — two outcomes for one mistake, one of them invisible.

    Args:
        text: Raw ``system_prompt`` or ``template`` from the config.
        kwargs: Values to substitute.
        what: Which field this is, for the error message.

    Raises:
        ValueError: On any rendering failure, with the field named and the
            cause identified.
    """
    if not text:
        return text
    try:
        return text.format_map(kwargs)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(_explain(text, kwargs, what, exc)) from None


def _explain(text: str, kwargs: dict[str, str], what: str, exc: Exception) -> str:
    """Work out why rendering failed. Only ever runs on the error path.

    ``format_map`` stops at the first problem, so *which* one it names depends
    on where in the string each sits — a malformed brace after a missing key
    is invisible, and vice versa. One parse here settles it.

    Order matters: an unparseable template is checked first, because no amount
    of supplied values would fix it. Reporting a missing key there would be a
    wrong diagnosis — supply it and the next call dies on the braces anyway.
    """
    try:
        fields = [n for _, n, _, _ in string.Formatter().parse(text) if n]
    except ValueError as parse_exc:
        return (
            f"Prompt {what} is not a valid template: {parse_exc}. Check the "
            f"braces in the prompt JSON — a literal brace must be doubled "
            f"({{{{ }}}}), and every placeholder must be closed."
        )
    # "{a.b}" / "{a[0]}" look up the root name; only that can be missing.
    roots = (n.split(".")[0].split("[")[0] for n in fields)
    missing = list(dict.fromkeys(r for r in roots if r and r not in kwargs))
    if not missing:
        # Parsed cleanly and every field was supplied, so the failure is
        # something else — a bad format spec, or a value whose __format__
        # raised. Pass it through rather than claiming a cause.
        return f"Prompt {what} could not be rendered: {type(exc).__name__}: {exc}"
    return (
        f"Prompt {what} is missing {', '.join(f'{{{n}}}' for n in missing)}. "
        f"Pass {', '.join(f'{n}=...' for n in missing)}, or remove "
        f"{'them' if len(missing) > 1 else 'it'} from the prompt JSON."
    )


class PromptTemplate:
    """One loaded prompt config, turned into chat messages.

    Produces ``[system, *few_shot, user]``. What makes this a class rather
    than a function is that its parts have **different lifetimes**:

    **1. System prompt** — rendered once, at construction, from
    ``build_kwargs``, then reused unchanged for every call. That split is the
    whole point: a checker validating 200 samples shares one identical system
    prompt, so re-rendering it per sample is pure waste. Every checker in
    ``content_moderation/checker.py`` used to hand-roll this.

    **2. Template** — kept raw and rendered fresh on **each** call from
    ``call_kwargs``, because it holds what varies per item (the sample being
    checked, the entry being expanded). Becomes the user message.

    **3. Few-shot examples** — already message dicts; nothing is rendered.

    **4. Format instruction** *(optional)* — when ``format_style`` is set, the
    matching multi-sample instruction (``llms/extraction.py``'s
    ``EXTRACTION_STYLES``) is appended to the system prompt at construction.
    It **requires** ``num_samples``, since stating how many samples to return
    is its entire job; omitting ``format_style`` is how you say "just one".
    ``self.extraction_style`` mirrors the choice so whoever parses the reply
    reads the style off the same object instead of tracking it separately —
    which is what keeps instruction and parser from drifting apart.

    Both stages take the same kind of kwargs and both render through
    ``str.format_map``, so a placeholder works in either field; which one you
    pass it to decides only *when* it is substituted.

    :func:`build_messages` is the one-shot wrapper — construct and call with
    a single set of kwargs — and covers every caller that doesn't reuse the
    object.
    """

    def __init__(
        self,
        prompt_config: dict,
        few_shot: bool = True,
        format_style: str | None = None,
        **build_kwargs: str,
    ):
        """Fix everything that does not vary per call.

        Args:
            prompt_config: Parsed prompt JSON (see CLAUDE.md's "Prompt JSON
                Schema").
            few_shot: Include the config's ``few_shot_examples``.
            format_style: One of ``llms.extraction.EXTRACTION_STYLES``, or
                ``None`` for no format instruction.
            **build_kwargs: Values for the ``system_prompt``'s placeholders,
                plus ``num_samples`` when ``format_style`` is set.

        Raises:
            ValueError: If the system prompt references a placeholder that was
                not supplied, or ``format_style`` is set without
                ``num_samples``.
        """
        # --- 1. System prompt: rendered once, fixed for this object's life.
        system_prompt = _render(
            prompt_config.get("system_prompt", ""), build_kwargs, "system_prompt"
        )

        # --- 2. Format instruction: appended to the system prompt, so it is
        # fixed alongside it rather than re-derived on every call.
        if format_style:
            # Deferred import: extraction.py imports load_prompt from this
            # module, so this stays a function-local import to avoid a
            # module-level cycle rather than restructuring either module
            # around a dependency that's only needed for this one optional
            # feature.
            from .extraction import get_format_instruction

            # Required, not defaulted. The instruction's whole job is to say
            # how many samples to return; silently falling back to 1 would ask
            # for one while the template asks for fifteen, and the mismatch
            # only shows up as a short extraction much later.
            if "num_samples" not in build_kwargs:
                raise ValueError(
                    f"format_style={format_style!r} needs num_samples, since the "
                    f"format instruction states how many samples to return. "
                    f"Pass num_samples=..., or omit format_style for a prompt "
                    f"that asks for one."
                )
            system_prompt += get_format_instruction(
                format_style, num_samples=int(build_kwargs["num_samples"])
            )

        self.system_prompt = system_prompt
        self.extraction_style = format_style

        # --- 3. Few-shot examples: already message dicts, nothing to render.
        self.few_shot_examples = (
            list(prompt_config.get("few_shot_examples", [])) if few_shot else []
        )

        # --- 4. Template: deliberately NOT rendered here. It holds the
        # per-item values, so it stays raw until __call__.
        self._template = prompt_config.get("template", "")

    def __call__(self, **call_kwargs: str) -> list[dict]:
        """Render the template and assemble the messages for one item.

        Args:
            **call_kwargs: Values for the ``template``'s placeholders — the
                per-item ones. Anything fixed across calls belongs in
                ``build_kwargs`` instead.

        Returns:
            ``[system, *few_shot, user]``, with empty sections omitted.

        Raises:
            ValueError: If the template references a placeholder that was not
                supplied.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.few_shot_examples)

        user_content = _render(self._template, call_kwargs, "template")
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages


def build_messages(
    prompt_config: dict,
    few_shot: bool = True,
    format_style: str | None = None,
    **kwargs: str,
) -> list[dict]:
    """Build a chat message list from a prompt config dict — the one-shot
    case of :class:`PromptTemplate` (construct and call together, same
    kwargs used for both ``system_prompt`` and ``template``).

    Args:
        prompt_config: Parsed prompt JSON (see CLAUDE.md's "Prompt JSON
            Schema" section for the file format).
        few_shot: Whether to include few-shot examples from the config.
        format_style: Optional multi-sample output format to append to
            ``system_prompt`` — one of ``llms.extraction.EXTRACTION_STYLES``
            (e.g. ``"numbered"``). See :class:`PromptTemplate` for why this
            is preferable to manually calling ``get_format_instruction()``
            and concatenating it onto the config beforehand.
        **kwargs: Variable template parameters to inject into both
                  system_prompt and template. Pass whatever the specific
                  prompt needs (e.g. Category="violence", SeedPrompts="...",
                  num_samples="15", style="journalistic").

    Returns:
        List of message dicts ready for LLM generate().

    Example:
        >>> config = load_prompt("input", "quality_check")  # doctest: +SKIP
        >>> msgs = build_messages(config, Category="violence", entry_type="harmful",
        ...                       subcategory="", sample="...")  # doctest: +SKIP
    """
    return PromptTemplate(
        prompt_config, few_shot=few_shot, format_style=format_style, **kwargs
    )(**kwargs)
