"""Write a prompt tree a user can edit — empty skeletons or the real thing.

Split from ``prompts.py`` because this is the only code here that **writes**
rather than reads: setup and release tooling, one caller
(``scripts/scaffold_prompts.py``), not part of any generation path.

Both modes exist because they answer different questions. ``mode="copy"``
gives the real prompts to edit in place — what someone tuning an existing
pipeline wants. ``mode="empty"`` gives skeletons that mirror each file's real
key set — what someone authoring a new prompt wants, and what a release build
would use if the prompts were ever withheld again.

Neither is required to customize a prompt: ``load_prompt`` falls back to the
packaged copy per file, so a user's directory only needs the files they
actually changed. These just give them a starting point.
"""

import json
import shutil
from pathlib import Path

from ... import paths

_DEFAULT_PROMPT_DIR = paths.prompts_dir()

def _placeholder_prompt(config: dict) -> dict:
    """Redact one prompt config's real text, keeping its structure real.

    What survives:

    - ``seed_fields`` — the prompt's contract with its callers, and the only
      field here that does real work (it drives the placeholder text below).
    - ``system_prompt``/``template``/``instruction``, whichever the file has
      (the last is the ``format_instructions/`` shape) — replaced by a TODO
      that still names every ``seed_fields`` entry as a real ``{placeholder}``
      token, so a scaffolded file round-trips through
      ``build_messages()``/``PromptTemplate`` before anyone fills it in.

    What does not:

    - ``few_shot_examples`` is emptied — never carry real example content into
      a placeholder.
    - ``metadata`` is **replaced**, not preserved: a redacted file must not
      claim the real prompt's version, so it is reset to ``0.0`` with a note.

    **The stub's key set matches its source file's exactly**, so the scaffold
    teaches the shape rather than one generic template. A chat prompt keeps
    ``system_prompt``/``template``/``few_shot_examples``; a
    ``format_instructions/`` snippet keeps ``instruction`` and gains none of
    them. Every file under one directory therefore looks like its siblings,
    and nobody has to work out which fields their pipeline actually uses —
    which a single all-fields example could not show, since no real prompt
    has both ``template`` and ``instruction``.
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
    # Only empty what was there. Assigning unconditionally gave the three
    # format_instructions/ files a few_shot_examples they never had, so the
    # scaffold stopped mirroring the shape it is meant to document.
    if "few_shot_examples" in stub:
        stub["few_shot_examples"] = []
    stub["metadata"] = {
        "version": "0.0",
        "notes": "PLACEHOLDER — replace before use. See CLAUDE.md's Public Release Notes.",
    }
    return stub


def scaffold_prompt_tree(
    target_dir: str | Path,
    source_dir: str | Path | None = None,
    mode: str = "copy",
) -> int:
    """Write an editable ``prompts/``-shaped tree, mirroring the real one.

    A starting point, never a requirement: ``load_prompt`` falls back to the
    packaged copy per file, so a user's directory only needs the prompts they
    actually changed. Delete what you don't intend to override.

    Args:
        target_dir: Root directory to write into.
        source_dir: Root to scaffold from. Defaults to the package's own
            ``prompts/``.
        mode: ``"copy"`` (default) writes the real prompts, to edit in place —
            what someone tuning an existing pipeline wants, and readable as a
            worked example. ``"empty"`` writes skeletons with the text
            replaced by TODOs that still name every ``seed_fields`` entry as a
            real ``{placeholder}``, so the file round-trips through
            ``build_messages()`` before anyone fills it in — what someone
            authoring a new prompt wants, and what a release build would use
            if the prompts were ever withheld again.

    Returns:
        Number of files written.

    Raises:
        ValueError: On an unknown ``mode``.

    Note:
        ``"empty"`` redacts prompt *text* only. Directory and category names
        describe harm categories by design and are left as-is, so review the
        output before publishing — a starting point, not a guarantee.
    """
    if mode not in ("copy", "empty"):
        raise ValueError(f"mode must be 'copy' or 'empty', got {mode!r}")

    source = Path(source_dir) if source_dir is not None else _DEFAULT_PROMPT_DIR
    target = Path(target_dir)
    written = 0
    for src_path in source.rglob("*.json"):
        dst_path = target / src_path.relative_to(source)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(src_path, dst_path)
        else:
            with open(src_path, encoding="utf-8") as f:
                config = json.load(f)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(_placeholder_prompt(config), f, indent=4)
                f.write("\n")
        written += 1
    return written
