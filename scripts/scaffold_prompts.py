"""Write an editable copy of the prompts/ tree.

    python scripts/scaffold_prompts.py ./my_prompts            # real prompts
    python scripts/scaffold_prompts.py ./my_prompts --empty    # skeletons

Then point a run at it (``prompt_dir="./my_prompts"``). The directory is an
**overlay**: each prompt is read from there if present and from the library
otherwise, so delete every file you don't intend to change — keeping only
your edits means the rest stay current when the library's prompts improve.

``--empty`` replaces system_prompt/template/instruction with TODOs that keep
each field's own {placeholders}, so a scaffolded file renders with the same
values as the original before anyone fills it in. It redacts prompt *text*
only: directory and category names describe harm categories by design and are
left as-is, so review the output before publishing anything built from it.
"""

import argparse
import sys

from redact.llms.prompting import scaffold_prompt_tree


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an editable copy of the prompts/ tree."
    )
    parser.add_argument("target_dir", help="Directory to write the tree into.")
    parser.add_argument(
        "--source-dir", default=None,
        help="Directory to scaffold from (default: this package's own prompts/).",
    )
    parser.add_argument(
        "--empty", action="store_true",
        help="Write TODO skeletons instead of the real prompts.",
    )
    args = parser.parse_args(argv)

    mode = "empty" if args.empty else "copy"
    written = scaffold_prompt_tree(
        args.target_dir, source_dir=args.source_dir, mode=mode
    )
    kind = "placeholder" if args.empty else "prompt"
    print(f"Wrote {written} {kind} file(s) under {args.target_dir}")
    print("Overlay: delete any file you don't intend to override.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
