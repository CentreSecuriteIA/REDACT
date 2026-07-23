"""Summarize the stage manifests of a completed (or in-progress) run.

    python scripts/inspect_run.py ./runs/my_run

Prints each ``{stage}.run.json`` under ``{data_dir}/Datasets/`` with its counts,
models, and timestamp — a quick way to see what a run produced and how the next
stage should continue.
"""

import argparse
import sys

from redact.runconfig import STAGES, read_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a run's stage manifests.")
    parser.add_argument("data_dir", help="The run's data_dir.")
    args = parser.parse_args(argv)

    found = False
    for stage in STAGES:
        m = read_manifest(stage, args.data_dir)
        if m is None:
            continue
        found = True
        counts = m.get("counts", {})
        ts = m.get("timestamp", "?")
        models = {k: v for k, v in (m.get("models") or {}).items() if v}
        print(f"[{stage}] {ts}  counts={counts}  models={models or 'role-defaults'}")

    if not found:
        print(f"No stage manifests found under {args.data_dir}/Datasets/.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
