"""Config-driven pipeline runner.

Execute a whole REDACT run from a recipe JSON (eval or training)::

    python scripts/run.py path/to/recipe.json
    python scripts/run.py path/to/recipe.json --params path/to/params.json
    python scripts/run.py path/to/recipe.json --data-dir ./runs/my_run

The recipe's ``params_file`` is used unless ``--params`` overrides it (a
relative path is read from the current directory, else from the recipe's
folder), and ``--data-dir`` overrides the recipe's ``data_dir``. Each stage
writes a ``{stage}.run.json`` run record under ``{data_dir}/Datasets/``.

The first Ctrl-C (or ``kill <pid>``) releases the models and exits; a
second one ends vLLM's engine processes and exits at once.
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import sys

from redact import run_pipeline
from redact.llms.backends.vllm import terminate_engine_processes


def _say(line: str) -> None:
    """Write a line to stderr from a signal handler, where ``print`` and
    logging may raise."""
    with contextlib.suppress(OSError):
        os.write(2, (line + "\n").encode())


def _exit_through_release() -> None:
    """Make Ctrl-C and SIGTERM exit through the run's ``finally``, which
    releases the models, and a second one of either exit at once.

    SIGTERM's default action would leave the engine process holding the
    card. A release that hangs, such as a model load that never returns,
    keeps the process up until the second signal.
    """
    received = []

    def handle(signum, frame):
        received.append(signum)
        if len(received) > 1:
            _say("Interrupted again: ending the engine processes and exiting at "
                 "once. Check with nvidia-smi that the card is free.")
            # Quiet: logging here may re-enter a write this signal interrupted.
            left = terminate_engine_processes(all=True, quiet=True)
            if left:
                _say(f"Engine processes still alive: end them with "
                     f"kill -9 {' '.join(map(str, left))}")
            os._exit(128 + signum)
        _say("Interrupted: releasing the models. Interrupt again to exit at once.")
        if signum == signal.SIGINT:
            signal.default_int_handler(signum, frame)   # raises KeyboardInterrupt
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a REDACT pipeline from a recipe JSON.")
    parser.add_argument("recipe", help="Path to the recipe JSON.")
    parser.add_argument(
        "--params", default=None,
        help="Override the recipe's params_file; a relative path is read from "
             "the current directory, else from the recipe's folder.")
    parser.add_argument("--data-dir", help="Override the recipe's data_dir.", default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-stage progress.")
    parser.add_argument("--debug", action="store_true", help="Show per-item DEBUG detail.")
    args = parser.parse_args(argv)

    # redact's pipelines log progress via the standard `logging` module rather than
    # print() — as the entry point, we're responsible for giving it somewhere to go.
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    _exit_through_release()

    summary = run_pipeline(args.recipe, params=args.params, verbose=not args.quiet,
                           data_dir=args.data_dir)
    print("\n=== Run summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
