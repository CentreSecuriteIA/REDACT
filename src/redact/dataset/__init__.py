"""Dataset handling utilities — CSV I/O, merging, deduplication, splitting, loading, taxonomy."""

from .categories import (
    SAMPLE_COLUMNS,
    append_samples,
    discover_categories,
    get_existing_samples,
    merge_all,
    merge_category_csvs,
    merge_inputs,
    read_category_csv,
    sample_row,
    take_per_group,
    write_category_csv,
)
from .identity import content_id, row_id, row_ids
from .loading import load_from_config, load_hf_dataset
from .resume import CommitBuffer, commit, drop_units, release_lock, resume_state
from .rows import append_rows, check_columns, count_rows, iter_rows, keys_in, read_rows
from .sidecar import JsonlSidecar, Ledger, Manifest
from .taxonomy import get_seed_prompts, iter_categories, load_seeds, load_taxonomy
