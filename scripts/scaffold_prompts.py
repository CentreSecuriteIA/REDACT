"""Generate a placeholder prompts/ tree for the public release.

    python scripts/scaffold_prompts.py ./redacted_prompts

Implements CLAUDE.md's Public Release Notes pre-release step: mirrors the
real prompts/ directory structure, but with system_prompt/template/
instruction text replaced by a documented placeholder — external users get
a starting structure matching every real load_prompt() call site, without
this repo's actual prompt content. Review the output before publishing; this
is a starting point, not a guarantee nothing sensitive leaks through
directory/category names themselves.
"""

import argparse
import sys

from redact.llms.prompts import scaffold_prompt_tree


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a placeholder prompts/ tree.")
    parser.add_argument("target_dir", help="Directory to write the placeholder tree into.")
    parser.add_argument(
        "--source-dir", default=None,
        help="Directory to scaffold from (default: this package's own prompts/).",
    )
    args = parser.parse_args(argv)

    written = scaffold_prompt_tree(args.target_dir, source_dir=args.source_dir)
    print(f"Wrote {written} placeholder prompt file(s) under {args.target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
