"""Write a setup folder: two recipes, their params files and the configs they overlay.

    python scripts/scaffold_setup.py ./my_setup              # refuses to overwrite
    python scripts/scaffold_setup.py ./my_setup --overwrite

Then run a recipe from it (``python scripts/run.py ./my_setup/eval_recipe.json``;
the folder's README gives the command to type from inside it). Each recipe's
``"config_dir": "."`` makes the folder an **overlay**: its ``dataset_types/``,
``taxonomy/``, ``jailbreak/augmentations.json``, ``generation_axes.json`` and
``dimensions.json`` files are read in place of the library's, so delete every
file you don't change and the rest stay current when the library's improve.
Each recipe's ``data_dir`` is ``runs/eval`` or ``runs/training`` in the
folder, as an absolute path. Prompts are a separate tier
(``scripts/scaffold_prompts.py`` and a recipe's ``prompt_dir``).
"""

import argparse
import sys

from redact import scaffold_setup


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a setup folder of recipes and the configs they overlay."
    )
    parser.add_argument("target_dir", help="Directory to write the folder into.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace files the directory already holds.",
    )
    args = parser.parse_args(argv)
    try:
        written = scaffold_setup(args.target_dir, overwrite=args.overwrite)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Wrote {len(written)} file(s) under {args.target_dir}")
    print("Overlay: delete any config file you don't intend to change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
