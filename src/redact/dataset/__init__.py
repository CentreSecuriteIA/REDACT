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
from .dedup import exact_dedup, normalized_dedup
from .identity import content_id, row_id, row_ids
from .loading import filter_dataset, load_from_config, load_hf_dataset
from .merge import (
    CONSTITUTION_INPUT_COLUMN_MAP,
    CONSTITUTION_INPUT_KEEP_COLUMNS,
    CONTENT_MOD_COLUMN_MAP,
    CONTENT_MOD_KEEP_COLUMNS,
    JAILBREAK_COLUMN_MAP,
    JAILBREAK_KEEP_COLUMNS,
    TYPE_COLUMN_MAP,
    merge_constitution_input_csvs,
    merge_content_mod_csvs,
    merge_csvs_from_dirs,
    merge_technique_csvs,
    normalize_csv,
    normalize_technique_csv,
)
from .resume import CommitBuffer, commit, drop_units, release_lock, resume_state
from .rows import append_rows, check_columns, count_rows, iter_rows, keys_in, read_rows
from .sidecar import JsonlSidecar, Ledger, Manifest
from .split import (
    balanced_counts,
    deterministic_balanced_assign,
    normalize_origin,
    split_by_column,
    split_by_functions,
)
from .taxonomy import (
    filter_by_group,
    get_categories,
    get_category_descriptions,
    get_group_categories,
    get_seed_prompts,
    get_subcategories,
    iter_categories,
    load_seeds,
    load_taxonomy,
    normalize_categories,
)
