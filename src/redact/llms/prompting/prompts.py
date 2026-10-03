"""Prompt loading and message assembly.

:meth:`PromptTemplate.load` reads a prompt JSON file and renders it into
``[system, *few_shot, user]`` messages. The system prompt is rendered once,
when the template is built, and the user message on each call.

``prompt_dir`` is a per-file overlay over the packaged prompts: each prompt is
read from there if present and from the packaged copy otherwise, so a user
directory only needs the files it changes. :func:`report_prompt_sources` logs
which prompts are overridden.
"""

import json
import logging
import string
from pathlib import Path
from typing import Any

from ... import paths

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_DIR = paths.prompts_dir()

#: Above this many overrides, the report logs a count instead of one line
#: each.
_MAX_LISTED_OVERRIDES = 10


def _prompt_file(root: Path, pipeline: str, category: str) -> Path | None:
    """The JSON file for one prompt under ``root``, or ``None`` if absent.

    A directory with exactly one ``.json`` file uses it; otherwise
    ``template.json``.
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
    """Find the file a prompt resolves to, and whether it is an override.

    Args:
        pipeline: Pipeline name, e.g. ``"input"``.
        category: Category within it, e.g. ``"quality_check"``.
        prompt_dir: Optional override directory, searched first.

    Returns:
        ``(path, is_override)``.

    Raises:
        FileNotFoundError: Neither the override directory nor the packaged
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
    """Load one prompt as a dict, preferring an override over the packaged copy.

    Prefer :meth:`PromptTemplate.load` in new code.

    Args:
        pipeline: Pipeline name (e.g. "input", "output", "jailbreak",
            "constitution").
        category: Category within it (e.g. "quality_check",
            "generation/standalone").
        prompt_dir: Optional directory searched first. ``None`` uses the
            packaged prompts only.

    Raises:
        FileNotFoundError: Neither source has this prompt.

    Example:
        >>> load_prompt("input", "quality_check")  # doctest: +SKIP
    """
    target, _ = resolve_prompt(pipeline, category, prompt_dir)
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def report_prompt_sources(
    prompt_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Log which prompts come from the library and which from ``prompt_dir``.

    Also warns about directories in ``prompt_dir`` that match no packaged
    prompt, such as a misspelled ``input/quality_chek/``. Those files are
    never loaded.

    Args:
        prompt_dir: The override directory, or ``None``.

    Returns:
        ``{"overridden": [...], "unused": [...], "packaged": [...]}``, each a
        list of prompt names.
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
    # Overrides are listed one per line only while the list is short.
    if overridden:
        if len(overridden) <= _MAX_LISTED_OVERRIDES:
            for name in overridden:
                logger.info("  override: %s", name)
        else:
            logger.info("  (%d overrides; run with --debug to list them)",
                        len(overridden))
            logger.debug("  overrides: %s", ", ".join(overridden))
    # Unused overrides are always listed, since each one is a mistake.
    for name in unused:
        logger.warning(
            "  unused override: %s — no packaged prompt by that name, so this "
            "file is never loaded (typo?)", name,
        )
    return {"overridden": overridden, "unused": unused, "packaged": packaged}


def _render(text: str, kwargs: dict[str, str], what: str) -> str:
    """Render one field with ``str.format_map``.

    Raises:
        ValueError: Rendering failed. The message names the field (``what``)
            and the cause.
    """
    if not text:
        return text
    try:
        return text.format_map(kwargs)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(_explain(text, kwargs, what, exc)) from None


def _explain(text: str, kwargs: dict[str, str], what: str, exc: Exception) -> str:
    """Build the error message for a failed render.

    A malformed template is reported before a missing placeholder, because
    supplying values would not fix it.
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
        # The template parsed and every field was supplied, so report the
        # original error (e.g. a bad format spec).
        return f"Prompt {what} could not be rendered: {type(exc).__name__}: {exc}"
    return (
        f"Prompt {what} is missing {', '.join(f'{{{n}}}' for n in missing)}. "
        f"Pass {', '.join(f'{n}=...' for n in missing)}, or remove "
        f"{'them' if len(missing) > 1 else 'it'} from the prompt JSON."
    )


class PromptTemplate:
    """One prompt config, rendered into ``[system, *few_shot, user]`` messages.

    Rendering happens in two stages:

    - The system prompt is rendered once, at construction, from the
      construction kwargs only. With ``format_style`` set, the matching
      format instruction (see ``llms/prompting/extraction.py``) is appended
      to it, which requires ``num_samples``.
    - The template is rendered on each call and becomes the user message. It
      sees the construction kwargs too; a call kwarg of the same name wins.

    Few-shot examples are message dicts and are passed through unrendered.
    ``extraction_style`` holds the ``format_style``, so the reply can be
    parsed with the matching extractor.

    Build one per stage and call it once per item. :func:`build_messages` is
    the one-shot form, which passes the same kwargs to both stages.
    """

    def __init__(
        self,
        prompt_config: dict,
        few_shot: bool = True,
        format_style: str | None = None,
        *,
        name: str | None = None,
        source: Path | None = None,
        is_override: bool = False,
        prompt_dir: str | Path | None = None,
        **build_kwargs: Any,
    ):
        """Render the system prompt and store the rest for later calls.

        Prefer :meth:`load`, which fills ``name``, ``source`` and
        ``is_override`` so a render error can name the file.

        Args:
            prompt_config: Parsed prompt JSON.
            few_shot: Include the config's ``few_shot_examples``.
            format_style: One of ``EXTRACTION_STYLES`` (in
                ``llms/prompting/extraction.py``), or ``None`` for no format
                instruction.
            name: ``"pipeline/category"``, for error messages.
            source: File the config was read from, for error messages.
            is_override: Whether ``source`` is a user override.
            prompt_dir: Override directory the format instruction is read
                from, when ``format_style`` is set.
            **build_kwargs: Values for the ``system_prompt`` placeholders,
                plus ``num_samples`` when ``format_style`` is set.

        Raises:
            ValueError: The system prompt has a placeholder that was not
                supplied, or ``format_style`` is set without ``num_samples``.
        """
        self.name = name
        self.source = source
        self.is_override = is_override

        system_prompt = _render(
            prompt_config.get("system_prompt", ""), build_kwargs,
            self._where("system_prompt"),
        )

        if format_style:
            # Imported here because extraction.py imports this module.
            from .extraction import get_format_instruction

            # num_samples is required. A default of 1 could contradict the
            # number of samples the template asks for.
            if "num_samples" not in build_kwargs:
                raise ValueError(
                    f"format_style={format_style!r} needs num_samples, since the "
                    f"format instruction states how many samples to return. "
                    f"Pass num_samples=..., or omit format_style for a prompt "
                    f"that asks for one."
                )
            system_prompt += get_format_instruction(
                format_style, num_samples=int(build_kwargs["num_samples"]),
                prompt_dir=prompt_dir,
            )

        self.system_prompt = system_prompt
        self.extraction_style = format_style

        self.few_shot_examples = (
            list(prompt_config.get("few_shot_examples", [])) if few_shot else []
        )

        # The template stays raw here and is rendered in __call__.
        self._template = prompt_config.get("template", "")
        self._build_kwargs = build_kwargs

    def _where(self, field: str) -> str:
        """Name a field for an error message, with the prompt's name and file."""
        if self.name is None:
            return field
        origin = "override" if self.is_override else "packaged"
        return f"{field} of {self.name} ({origin}: {self.source})"

    @classmethod
    def load(
        cls,
        pipeline: str,
        category: str,
        *,
        prompt_dir: str | Path | None = None,
        few_shot: bool = True,
        format_style: str | None = None,
        **build_kwargs: Any,
    ) -> "PromptTemplate":
        """Find a prompt file, read it and build a template from it.

        Args:
            pipeline: Pipeline name, e.g. ``"input"`` or ``"jailbreak"``.
            category: Category within it, e.g. ``"quality_check"``.
            prompt_dir: Optional override directory, searched first.
            few_shot: Include the config's ``few_shot_examples``.
            format_style: One of ``EXTRACTION_STYLES``, or ``None``.
            **build_kwargs: Values for the ``system_prompt`` placeholders,
                plus ``num_samples`` when ``format_style`` is set.

        Returns:
            A callable template: ``template(**per_item_kwargs) -> messages``.

        Raises:
            FileNotFoundError: Neither source has this prompt.
            ValueError: The system prompt cannot be rendered with these
                values.
        """
        path, is_override = resolve_prompt(pipeline, category, prompt_dir)
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
        return cls(
            config,
            few_shot=few_shot,
            format_style=format_style,
            name=f"{pipeline}/{category}",
            source=path,
            is_override=is_override,
            prompt_dir=prompt_dir,
            **build_kwargs,
        )

    def __call__(self, **call_kwargs: str) -> list[dict]:
        """Render the template and assemble the messages for one item.

        Args:
            **call_kwargs: Values for the ``template`` placeholders.

        Returns:
            ``[system, *few_shot, user]``, with empty sections omitted.

        Raises:
            ValueError: The template has a placeholder that was not supplied.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.few_shot_examples)

        call_kwargs = {**self._build_kwargs, **call_kwargs}
        user_content = _render(self._template, call_kwargs, self._where("template"))
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages


def build_messages(
    prompt_config: dict,
    few_shot: bool = True,
    format_style: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Build a chat message list from a prompt config in one step.

    Constructs a :class:`PromptTemplate` and calls it with the same kwargs,
    so they apply to both ``system_prompt`` and ``template``. Prefer
    :meth:`PromptTemplate.load` in new code.

    Args:
        prompt_config: Parsed prompt JSON.
        few_shot: Include the config's ``few_shot_examples``.
        format_style: One of ``EXTRACTION_STYLES`` (e.g. ``"numbered"``).
            Appends the matching format instruction to the system prompt and
            requires ``num_samples`` in ``kwargs``.
        **kwargs: Placeholder values, whatever the prompt needs (e.g.
            ``Category="violence"``, ``num_samples="15"``).

    Returns:
        List of message dicts ready for ``generate()``.

    Example:
        >>> config = load_prompt("input", "quality_check")  # doctest: +SKIP
        >>> msgs = build_messages(config, Category="violence", entry_type="harmful",
        ...                       subcategory="", sample="...")  # doctest: +SKIP
    """
    return PromptTemplate(
        prompt_config, few_shot=few_shot, format_style=format_style, **kwargs
    )(**kwargs)
