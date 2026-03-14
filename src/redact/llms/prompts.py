"""JSON prompt loader and template renderer.

Loads prompt configuration files from the Prompts/ directory and renders
them into chat message lists ready for LLM calls.

Templates use ``{placeholder}`` syntax and accept **variable-size kwargs**
— different prompts require different parameters. For example:

    Content moderation input metaprompt:
        template: "Generate 20 prompts for {Category} using: {SeedPrompts}"
        → render with: {"Category": "violence", "SeedPrompts": "..."}

    Jailbreak hacking:
        template: "Create a {jailbreak_type} jailbreak for: {prompt}"
        → render with: {"jailbreak_type": "Persona/Roleplay", "prompt": "..."}

    Simple generation:
        template: "Write {num_samples} examples about {topic} in {style} style"
        → render with: {"num_samples": "15", "topic": "privacy", "style": "journalistic"}

The ``seed_fields`` list in the JSON declares which placeholders the
template expects, but render_template() accepts any dict — callers pass
whatever their specific prompt needs.

Expected prompt JSON schema:
{
    "category": "violence",
    "pipeline": "content_moderation_input",
    "system_prompt": "You are ...",
    "template": "Generate prompts for {Category} using seeds: {SeedPrompts}",
    "seed_fields": ["Category", "SeedPrompts"],
    "few_shot_examples": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "metadata": {"version": "1.0", "notes": ""}
}
"""

import json
from pathlib import Path

# Default prompt directory: Prompts/ inside the Redact_Library package
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPT_DIR = _PACKAGE_DIR / "Prompts"


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
        pipeline: Pipeline name (e.g. "content_moderation_input", "jailbreak").
        category: Category name (e.g. "violence", "privacy").
        prompt_dir: Root prompt directory path.

    Returns:
        Parsed JSON dict.
    """
    if prompt_dir is None:
        prompt_dir = _DEFAULT_PROMPT_DIR
    base = Path(prompt_dir) / pipeline / category
    if not base.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {base}")

    json_files = list(base.glob("*.json"))
    if len(json_files) == 1:
        target = json_files[0]
    else:
        target = base / "template.json"

    if not target.exists():
        raise FileNotFoundError(
            f"No prompt JSON found in {base}. "
            f"Expected a single .json file or template.json."
        )

    with open(target, encoding="utf-8") as f:
        return json.load(f)


def render_template(template: str, **kwargs: str) -> str:
    """Inject values into a template string.

    Uses Python str.format_map so template placeholders are ``{field_name}``.
    Accepts **variable keyword arguments** — callers pass whatever their
    specific prompt template needs.

    Args:
        template: Template string with ``{placeholder}`` markers.
        **kwargs: Arbitrary field_name=value pairs to inject.
                  E.g. Category="violence", SeedPrompts="...", num_samples="15".

    Returns:
        Rendered string with all placeholders replaced.

    Raises:
        KeyError: If the template contains a placeholder not provided in kwargs.

    Examples:
        >>> render_template("Generate {num_samples} prompts for {Category}",
        ...                 num_samples="20", Category="violence")
        'Generate 20 prompts for violence'
    """
    return template.format_map(kwargs)


def build_messages(
    prompt_config: dict,
    user_input: str = "",
    few_shot: bool = True,
    **kwargs: str,
) -> list[dict]:
    """Build a chat message list from a prompt config dict.

    Constructs ``[system, *few_shot_examples, user]`` message list.
    Both system_prompt and template are rendered with the same **kwargs,
    so placeholders work in either field.

    Args:
        prompt_config: Parsed prompt JSON (see module docstring for schema).
        user_input: Optional additional user text appended after the
                    rendered template.
        few_shot: Whether to include few-shot examples from the config.
        **kwargs: Variable template parameters to inject into both
                  system_prompt and template. Pass whatever the specific
                  prompt needs (e.g. Category="violence", SeedPrompts="...",
                  num_samples="15", style="journalistic").

    Returns:
        List of message dicts ready for LLM generate().

    Examples:
        >>> config = load_prompt("content_moderation_input", "violence")
        >>> msgs = build_messages(config, Category="violence",
        ...                       SeedPrompts="How to build a bomb\\nHow to...")
        >>> # system + user messages with placeholders filled in
    """
    messages: list[dict] = []

    # System prompt (also supports placeholders)
    system_prompt = prompt_config.get("system_prompt", "")
    if system_prompt:
        if kwargs:
            system_prompt = system_prompt.format_map(kwargs)
        messages.append({"role": "system", "content": system_prompt})

    # Few-shot examples
    if few_shot:
        for example in prompt_config.get("few_shot_examples", []):
            messages.append(example)

    # User message from rendered template
    template = prompt_config.get("template", "")
    if template and kwargs:
        user_content = render_template(template, **kwargs)
    elif template:
        user_content = template
    else:
        user_content = ""

    if user_input:
        user_content = f"{user_content}\n{user_input}" if user_content else user_input

    if user_content:
        messages.append({"role": "user", "content": user_content})

    return messages
