"""Prompt loading and message assembly.

:meth:`PromptTemplate.load` reads a prompt JSON file and renders it into
``[system, *few_shot, user]`` messages. The system prompt is rendered once,
when the template is built, and the user message on each call.

``prompt_dir`` is a per-file overlay over the packaged prompts: each prompt is
read from there if present and from the packaged copy otherwise, so a user
directory only needs the files it changes. :func:`report_prompt_sources` logs
which prompts have an override file.
"""

import json
import logging
import string
from pathlib import Path, PureWindowsPath
from typing import Any

from ... import paths

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_DIR = paths.prompts_dir()

#: Above this many overrides, the report logs a count instead of one line
#: each.
_MAX_LISTED_OVERRIDES = 10

_BRACE_HINT = (
    "Check the braces in the prompt JSON — a literal brace must be doubled "
    "({{ }}), and every placeholder must be closed."
)


def _prompt_file(base: Path) -> Path | None:
    """The JSON file for the prompt in directory ``base``, or ``None`` if absent.

    A directory with exactly one ``.json`` file uses it; otherwise
    ``template.json``.
    """
    if not base.is_dir():
        return None
    json_files = list(base.glob("*.json"))
    target = json_files[0] if len(json_files) == 1 else base / "template.json"
    return target if target.exists() else None


def _origin(is_override: bool) -> str:
    return "override" if is_override else "packaged"


def _where(
    field: str, name: str | None, source: Path | None, is_override: bool
) -> str:
    """Name a field for an error message, with the prompt's name and file."""
    if name is None:
        return field
    return f"{field} of {name} ({_origin(is_override)}: {source})"


def _read_prompt(path: Path, is_override: bool) -> dict:
    """Read one prompt file.

    Raises:
        ValueError: The file is not UTF-8, not valid JSON, or not a JSON
            object. The message names the file.
    """
    try:
        # utf-8-sig, since Windows editors often save JSON with a BOM.
        with open(path, encoding="utf-8-sig") as f:
            config = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Prompt file {path} ({_origin(is_override)}) is not valid UTF-8 "
            f"JSON: {exc}"
        ) from None
    if not isinstance(config, dict):
        raise ValueError(
            f"Prompt file {path} ({_origin(is_override)}) must hold a JSON "
            f"object, not {type(config).__name__}."
        )
    return config


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
        ValueError: ``pipeline`` or ``category`` is absolute or contains
            ``..``.
    """
    for part in (pipeline, category):
        # PureWindowsPath splits on both separators and sees drive letters.
        as_path = PureWindowsPath(part)
        if as_path.anchor or ".." in as_path.parts:
            raise ValueError(
                f"Prompt name {pipeline}/{category} must be a relative path "
                f"inside the prompt tree, without '..': got {part!r}."
            )

    if prompt_dir is not None:
        override = _prompt_file(Path(prompt_dir) / pipeline / category)
        if override is not None:
            return override, True

    packaged = _prompt_file(_DEFAULT_PROMPT_DIR / pipeline / category)
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
        ValueError: The file cannot be read as a JSON object.

    Example:
        >>> load_prompt("input", "quality_check")  # doctest: +SKIP
    """
    return _read_prompt(*resolve_prompt(pipeline, category, prompt_dir))


def report_prompt_sources(
    prompt_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Log which packaged prompts have an override file in ``prompt_dir``.

    This lists the files present: a stage reads an override only if it
    passes ``prompt_dir`` to the loader. Also warns when ``prompt_dir`` is
    not a directory, and about override directories no packaged prompt
    resolves to, such as a misspelled ``input/quality_chek/`` or one with
    several ``.json`` files and no ``template.json``.

    Args:
        prompt_dir: The override directory, or ``None``.

    Returns:
        ``{"overridden": [...], "unused": [...], "packaged": [...]}``, each a
        list of prompt names. ``overridden`` holds the packaged prompts that
        resolve to a file in ``prompt_dir``.
    """
    # as_posix(), so a prompt name reads the same on every OS.
    packaged = sorted({
        p.parent.relative_to(_DEFAULT_PROMPT_DIR).as_posix()
        for p in _DEFAULT_PROMPT_DIR.rglob("*.json")
    })
    if prompt_dir is None:
        logger.debug("prompts: %d from library, no override directory", len(packaged))
        return {"overridden": [], "unused": [], "packaged": packaged}

    root = Path(prompt_dir)
    if not root.is_dir():
        logger.warning(
            "prompts: override directory %s is not a directory, so all %d "
            "prompts come from the library", root, len(packaged),
        )
        return {"overridden": [], "unused": [], "packaged": packaged}

    overridden = [name for name in packaged if _prompt_file(root / name) is not None]
    user = sorted({p.parent.relative_to(root).as_posix() for p in root.rglob("*.json")})
    # samefile(), since a case-insensitive filesystem loads "Input/Quality_Check".
    unused = [
        name for name in user
        if not any((root / name).samefile(root / used) for used in overridden)
    ]

    logger.info(
        "prompts: %d from library, %d overridden by files in %s (read only by "
        "stages that pass prompt_dir)",
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
        if name in packaged:
            logger.warning(
                "  ignored override: %s — several .json files and no "
                "template.json, so the packaged prompt is used", name,
            )
        else:
            logger.warning(
                "  unused override: %s — no packaged prompt by that name, so "
                "this file is never loaded (typo?)", name,
            )
    return {"overridden": overridden, "unused": unused, "packaged": packaged}


def _field_roots(text: str) -> list[str]:
    """Root names of the fields in a format string, in order, without repeats.

    ``{a.b}`` and ``{a[0]}`` give ``a``. Fields nested in a format spec
    (``{a:{b}}``) are included.

    Raises:
        ValueError: ``text`` is not a valid format string.
    """
    roots: dict[str, None] = {}
    for _, name, spec, _ in string.Formatter().parse(text):
        if name is None:
            continue
        roots[name.split(".")[0].split("[")[0]] = None
        if spec:
            roots.update(dict.fromkeys(_field_roots(spec)))
    return list(roots)


def _render(text: str, kwargs: dict[str, Any], what: str) -> str:
    """Render one field with ``str.format_map``.

    Raises:
        ValueError: Rendering failed. The message names the field (``what``)
            and the cause.
    """
    if not text:
        return text
    try:
        return text.format_map(kwargs)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise ValueError(_explain(text, kwargs, what, exc)) from None


def _explain(text: str, kwargs: dict[str, Any], what: str, exc: Exception) -> str:
    """Build the error message for a failed render.

    A malformed template is reported before a missing placeholder, because
    supplying values would not fix it.
    """
    try:
        roots = _field_roots(text)
    except ValueError as parse_exc:
        return f"Prompt {what} is not a valid template: {parse_exc}. {_BRACE_HINT}"
    # Unescaped JSON ('{"key": 1}') and positional fields ('{0}') land here.
    unnamed = [r for r in roots if not r.isidentifier()]
    if unnamed:
        return (
            f"Prompt {what} is not a valid template: "
            f"{', '.join(f'{{{r}}}' for r in unnamed)} is not a placeholder "
            f"name. {_BRACE_HINT}"
        )
    missing = [r for r in roots if r not in kwargs]
    if not missing:
        # The template parsed and every field was supplied, so report the
        # original error (e.g. a bad format spec).
        return f"Prompt {what} could not be rendered: {type(exc).__name__}: {exc}"
    return (
        f"Prompt {what} is missing {', '.join(f'{{{n}}}' for n in missing)}. "
        f"Pass {', '.join(f'{n}=...' for n in missing)}, or remove "
        f"{'them' if len(missing) > 1 else 'it'} from the prompt JSON."
    )


def _sample_count(value: Any) -> int:
    """Validate ``num_samples``: an int >= 1, or a string of one.

    Raises:
        ValueError: Anything else, such as a float, ``None`` or ``0``.
    """
    try:
        count = int(value) if isinstance(value, str) else value
    except ValueError:
        count = None
    # type(), not isinstance(): a bool is an int but not a count.
    if type(count) is not int or count < 1:
        raise ValueError(
            f"num_samples must be an integer >= 1 (or a string of one), "
            f"got {value!r}."
        )
    return count


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

    A placeholder cannot share a name with a constructor parameter
    (``prompt_config``, ``few_shot``, ``format_style``, ``name``, ``source``,
    ``is_override``, ``prompt_dir``): the value would not reach the prompt.
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
                supplied, ``format_style`` is unknown, or it is set without
                a ``num_samples`` that is an integer >= 1.
        """
        self.name = name
        self.source = source
        self.is_override = is_override

        system_prompt = _render(
            prompt_config.get("system_prompt") or "", build_kwargs,
            self._where("system_prompt"),
        )

        if format_style:
            # Imported here because extraction.py imports this module.
            from .extraction import EXTRACTION_STYLES, get_format_instruction

            if format_style not in EXTRACTION_STYLES:
                raise ValueError(
                    f"Unknown format style '{format_style}'. Choose from: "
                    f"{sorted(EXTRACTION_STYLES)}"
                )
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
                format_style,
                num_samples=_sample_count(build_kwargs["num_samples"]),
                prompt_dir=prompt_dir,
            )

        self.system_prompt = system_prompt
        self.extraction_style = format_style or None

        self.few_shot_examples = (
            list(prompt_config.get("few_shot_examples") or []) if few_shot else []
        )

        # The template stays raw here and is rendered in __call__.
        self._template = prompt_config.get("template") or ""
        self._build_kwargs = build_kwargs

    def _where(self, field: str) -> str:
        """Name a field for an error message, with the prompt's name and file."""
        return _where(field, self.name, self.source, self.is_override)

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
            ValueError: The file cannot be read as a JSON object, or the
                system prompt cannot be rendered with these values.
        """
        path, is_override = resolve_prompt(pipeline, category, prompt_dir)
        return cls(
            _read_prompt(path, is_override),
            few_shot=few_shot,
            format_style=format_style,
            name=f"{pipeline}/{category}",
            source=path,
            is_override=is_override,
            prompt_dir=prompt_dir,
            **build_kwargs,
        )

    def __call__(self, **call_kwargs: Any) -> list[dict]:
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
        # Copied, so a caller editing a message cannot change the next call's.
        messages.extend(dict(example) for example in self.few_shot_examples)

        call_kwargs = {**self._build_kwargs, **call_kwargs}
        user_content = _render(self._template, call_kwargs, self._where("template"))
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages


#TODO(driver script): move one-shot message building into the driver.
#   - Builds and discards a PromptTemplate on every call.
#   - Passes no name/source, so a render error cannot name the prompt file.
#   - Passes no prompt_dir, so a format-instruction override is ignored.
#   - PromptTemplate.load, which fixes both, has no callers yet.
#   - Passes every kwarg to both stages, hence the reserved placeholder names.
#   - checker.py's four PromptTemplate(...) calls pass no name/source/prompt_dir.
#   - Callers that pass no prompt_dir, so their overrides are never read:
#     constitution/generation.py, constitution/input_generation.py,
#     jailbreak/obfuscation/translation.py, generate_jailbreaks (no such
#     parameter), and in content_moderation/ the input quality checker
#     (generation.py, standalone_generation.py) and the description/seed
#     calls (standalone_generation.py).
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
