"""Prompt loading and message assembly — the one place LLM-ready chat
messages get built from a prompt JSON file (see CLAUDE.md's "Prompt JSON
Schema" section for the file format). ``load_prompt()`` reads the file;
``PromptTemplate`` (and its one-shot wrapper ``build_messages()``) renders it
into ``[system, *few_shot, user]`` messages, in two stages so a fixed
system prompt can be built once and reused across many per-item calls
(e.g. a checker validating many samples) without re-rendering it each time.
"""

import json
from pathlib import Path

from .. import paths

_DEFAULT_PROMPT_DIR = paths.prompts_dir()


def load_prompt(
    pipeline: str,
    category: str,
    prompt_dir: str | Path | None = None,
) -> dict:
    """Load a prompt JSON file by pipeline and category.

    Searches for a JSON file in ``prompt_dir/pipeline/category/``.
    If the directory contains exactly one ``.json`` file, loads it.
    Otherwise looks for ``template.json`` as the default.

    Args:
        pipeline: Pipeline name (e.g. "input", "output", "jailbreak",
            "constitution/generation").
        category: Category name within that pipeline (e.g. "quality_check",
            "generation/standalone").
        prompt_dir: Root prompt directory path.

    Returns:
        Parsed JSON dict.

    Example:
        >>> load_prompt("input", "quality_check")  # doctest: +SKIP
    """
    if prompt_dir is None:
        prompt_dir = _DEFAULT_PROMPT_DIR
    base = Path(prompt_dir) / pipeline / category
    if not base.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {base}")

    json_files = list(base.glob("*.json"))
    target = json_files[0] if len(json_files) == 1 else base / "template.json"

    if not target.exists():
        raise FileNotFoundError(
            f"No prompt JSON found in {base}. "
            f"Expected a single .json file or template.json."
        )

    with open(target, encoding="utf-8") as f:
        return json.load(f)


def render_template(template: str, **kwargs: str) -> str:
    """Render one template string via ``str.format_map``.

    A low-level primitive — ``PromptTemplate``/``build_messages()`` are the
    real entry points for turning a loaded prompt config into chat messages;
    this is exposed mainly for a caller that already has a bare template
    string outside that flow.

    Raises:
        KeyError: If the template contains a placeholder not provided in kwargs.
    """
    return template.format_map(kwargs)


class PromptTemplate:
    """A loaded prompt config, rendered in two stages.

    Stage 1 (construction): ``system_prompt`` and few-shot examples are
    rendered once from build-time kwargs and fixed for the object's
    lifetime. Stage 2 (each call): ``template`` is rendered fresh from
    call-time kwargs and assembled into ``[system, *few_shot, user]``.

    This split is the point: a caller validating many samples with the same
    checker system prompt (e.g. ``check_sample``/``batch_check_samples`` in
    ``router.py``) builds one ``PromptTemplate`` and calls it once per
    sample, instead of re-rendering an identical system prompt every time —
    the pattern every checker in ``content_moderation/checker.py`` used to
    hand-roll independently. ``build_messages()`` below is the single-shot
    special case (construct once, call once with the same kwargs) that
    covers every other real caller in the codebase.

    ``format_style`` (optional) appends the matching multi-sample output
    format instruction — see ``llms/extraction.py``'s
    ``get_format_instruction()`` / ``EXTRACTION_STYLES`` — to
    ``system_prompt`` at construction time, reading ``num_samples`` out of
    ``build_kwargs`` (defaulting to 1 if absent). Replaces manually calling
    ``get_format_instruction()`` and concatenating it onto the config's
    ``system_prompt`` before construction, which is easy to forget to keep
    in sync with the extraction style actually used downstream. The
    resulting ``self.extraction_style`` mirrors it, so a caller that stores
    its ``PromptTemplate`` can read the style back off the same object
    later (at extraction time) instead of tracking it separately.
    """

    def __init__(
        self,
        prompt_config: dict,
        few_shot: bool = True,
        format_style: str | None = None,
        **build_kwargs: str,
    ):
        system_prompt = prompt_config.get("system_prompt", "")
        if system_prompt and build_kwargs:
            system_prompt = system_prompt.format_map(build_kwargs)
        if format_style:
            # Deferred import: extraction.py imports load_prompt from this
            # module, so this stays a function-local import to avoid a
            # module-level cycle rather than restructuring either module
            # around a dependency that's only needed for this one optional
            # feature.
            from .extraction import get_format_instruction

            num_samples = build_kwargs.get("num_samples", 1)
            system_prompt += get_format_instruction(format_style, num_samples=int(num_samples))
        self.system_prompt = system_prompt
        self.extraction_style = format_style
        self.few_shot_examples = (
            list(prompt_config.get("few_shot_examples", [])) if few_shot else []
        )
        self._template = prompt_config.get("template", "")

    def __call__(self, user_input: str = "", **call_kwargs: str) -> list[dict]:
        """Render ``template`` with ``call_kwargs`` and assemble the message list."""
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.few_shot_examples)

        if self._template and call_kwargs:
            user_content = render_template(self._template, **call_kwargs)
        else:
            user_content = self._template

        if user_input:
            user_content = f"{user_content}\n{user_input}" if user_content else user_input

        if user_content:
            messages.append({"role": "user", "content": user_content})

        return messages


def build_messages(
    prompt_config: dict,
    user_input: str = "",
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
        user_input: Optional additional user text appended after the
                    rendered template.
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
    return PromptTemplate(prompt_config, few_shot=few_shot, format_style=format_style, **kwargs)(
        user_input=user_input, **kwargs
    )


def _placeholder_prompt(config: dict) -> dict:
    """Redact one prompt config's real text, keeping its structure real.

    ``category``/``pipeline``/``seed_fields``/``metadata.version`` are
    preserved as-is (structural facts calling code and this schema itself
    depend on); ``system_prompt``/``template``/``instruction`` (whichever
    the file has — the last is the ``format_instructions/`` shape) become a
    documented placeholder that still mentions every ``seed_fields`` entry
    as a real ``{placeholder}`` token, so a scaffolded file round-trips
    through ``build_messages()``/``PromptTemplate`` without a ``KeyError``
    even before anyone fills it in. ``few_shot_examples`` is always cleared
    — never carry real example content into a placeholder.
    """
    stub = dict(config)
    seed_fields = config.get("seed_fields", [])
    fields_note = (
        " ".join(f"{{{f}}}" for f in seed_fields) if seed_fields else "(no seed_fields declared)"
    )
    for key in ("system_prompt", "template", "instruction"):
        if key in stub:
            stub[key] = (
                f"TODO: replace with a real {key}. Available placeholder fields: {fields_note}"
            )
    stub["few_shot_examples"] = []
    stub["metadata"] = {
        "version": "0.0",
        "notes": "PLACEHOLDER — replace before use. See CLAUDE.md's Public Release Notes.",
    }
    return stub


def scaffold_prompt_tree(target_dir: str | Path, source_dir: str | Path | None = None) -> int:
    """Write a placeholder ``prompts/``-shaped tree, mirroring the real one.

    Implements CLAUDE.md's Public Release Notes pre-release step ("Redact or
    replace all files in prompts/ with documented placeholders"): walks
    every ``.json`` file under ``source_dir`` (default: the package's own
    ``prompts/``) and writes a same-shaped, same-path stub under
    ``target_dir`` via :func:`_placeholder_prompt` — external users get a
    starting directory structure that matches every real ``load_prompt()``
    call site without needing this (private) repo's actual prompt text.

    This redacts prompt *text*; it does not audit for harmful content
    leaking through directory/category names themselves (those describe harm
    categories by design and are left as-is) — review the output before
    actually publishing, this is a starting point, not a guarantee.

    Args:
        target_dir: Root directory to write the placeholder tree into.
        source_dir: Root directory to scaffold from. Defaults to this
            package's own ``prompts/``.

    Returns:
        Number of files written.
    """
    source = Path(source_dir) if source_dir is not None else _DEFAULT_PROMPT_DIR
    target = Path(target_dir)
    written = 0
    for src_path in source.rglob("*.json"):
        with open(src_path, encoding="utf-8") as f:
            config = json.load(f)
        dst_path = target / src_path.relative_to(source)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(_placeholder_prompt(config), f, indent=4)
            f.write("\n")
        written += 1
    return written
