"""Config-driven pipeline runner.

Execute a whole REDACT run from a recipe JSON (eval or training)::

    python scripts/run.py path/to/recipe.json
    python scripts/run.py path/to/recipe.json --params path/to/params.json
    python scripts/run.py path/to/recipe.json --data-dir ./runs/my_run

The recipe's ``params_file`` is used unless ``--params`` overrides it, and
``--data-dir`` overrides the recipe's ``data_dir``. Each stage writes a
``{stage}.run.json`` manifest under ``{data_dir}/Datasets/``.
"""

import argparse
import json
import sys

from redact import run_pipeline, load_recipe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a REDACT pipeline from a recipe JSON.")
    parser.add_argument("recipe", help="Path to the recipe JSON.")
    parser.add_argument("--params", help="Override the recipe's params_file.", default=None)
    parser.add_argument("--data-dir", help="Override the recipe's data_dir.", default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-stage progress.")
    args = parser.parse_args(argv)

    recipe = load_recipe(args.recipe)
    if args.data_dir:
        recipe["data_dir"] = args.data_dir

    summary = run_pipeline(recipe, params=args.params, verbose=not args.quiet)
    print("\n=== Run summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
