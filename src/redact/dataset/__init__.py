"""Rows on disk: identity, the two sidecar files, resume, and the per-category tree.

A stage gives each row a content id, plans its units in a manifest, and saves
finished units in batches: their rows first, their acks in the ledger after.
A resumed run skips the units the ledger records.

Usage::

    from redact.dataset import (
        CommitBuffer, Ledger, Manifest, append_rows, read_rows, resume_state, row_ids,
    )

    ids = row_ids(inputs, "sample")
    ledger = Ledger.sidecar(out, key_fields=("input_id",), casters={"input_id": str})
    manifest = Manifest.sidecar(out)
    # Takes the stage's lock. A fresh run clears the output and both sidecars;
    # a resumed one cuts off what a save that did not finish left in the output.
    done = resume_state(ledger, resume=resume, artifact=out, manifest=manifest)
    manifest.write({"input_id": i, "status": "planned"} for i in ids)
    with CommitBuffer(append=lambda rows: append_rows(out, rows), ledger=ledger) as buf:
        for i in ids:
            if i not in done:
                buf.add(rows_of(i), {"input_id": i})
    return read_rows(out)

A stage that resumes and finds nothing to save opens no buffer and calls
``release_lock(ledger)`` itself.
"""

# The per-category tree of input samples
from .categories import (
    append_samples,
    discover_categories,
    get_existing_samples,
    input_files,
    merge_all,
    merge_inputs,
    sample_row,
    take_per_group,
)

# Rows before acks
from .save import CommitBuffer, commit

# Starting part of a stage over
from .drop import drop_units

# Row identity
from .identity import content_id, row_id, row_ids

# Published datasets
from .loading import load_from_config, load_hf_dataset

# What a run skips
from .resume import release_lock, resume_state

# Rows in one CSV
from .rows import (
    Part,
    accepted_mask,
    append_rows,
    check_columns,
    count_rows,
    has_rows,
    iter_rows,
    keys_in,
    read_rows,
    write_concat,
)

# The ledger and the manifest
from .sidecar import JsonlSidecar, Ledger, Manifest

# Taxonomy and seed files
from .taxonomy import (
    create_taxonomy,
    get_seed_prompts,
    iter_categories,
    load_seeds,
    load_taxonomy,
)

__all__ = [
    "CommitBuffer",
    "JsonlSidecar",
    "Ledger",
    "Manifest",
    "Part",
    "accepted_mask",
    "append_rows",
    "append_samples",
    "check_columns",
    "commit",
    "content_id",
    "count_rows",
    "create_taxonomy",
    "discover_categories",
    "drop_units",
    "get_existing_samples",
    "get_seed_prompts",
    "has_rows",
    "input_files",
    "iter_categories",
    "iter_rows",
    "keys_in",
    "load_from_config",
    "load_hf_dataset",
    "load_seeds",
    "load_taxonomy",
    "merge_all",
    "merge_inputs",
    "read_rows",
    "release_lock",
    "resume_state",
    "row_id",
    "row_ids",
    "sample_row",
    "take_per_group",
    "write_concat",
]
