"""Dataset handling utilities — CSV I/O, merging, deduplication, splitting, loading, taxonomy."""

from .io import (
    SAMPLE_COLUMNS,
    read_category_csv,
    write_category_csv,
    append_samples,
    get_existing_samples,
)
from .sidecar import JsonlSidecar
from .ledger import Ledger
from .manifest import Manifest
from .merge import (
    merge_category_csvs,
    discover_categories,
    merge_all,
    normalize_csv,
    merge_csvs_from_dirs,
    normalize_technique_csv,
    merge_technique_csvs,
    merge_content_mod_csvs,
    merge_constitution_input_csvs,
    JAILBREAK_KEEP_COLUMNS,
    JAILBREAK_COLUMN_MAP,
    CONTENT_MOD_KEEP_COLUMNS,
    CONTENT_MOD_COLUMN_MAP,
    CONSTITUTION_INPUT_KEEP_COLUMNS,
    CONSTITUTION_INPUT_COLUMN_MAP,
    TYPE_COLUMN_MAP,
)
from .dedup import exact_dedup, normalized_dedup
from .split import (
    normalize_origin,
    balanced_counts,
    deterministic_balanced_assign,
    split_by_column,
    split_by_functions,
    take_per_group,
)
from .loading import load_hf_dataset, load_from_config, filter_dataset
from .taxonomy import (
    load_taxonomy,
    get_categories,
    get_category_descriptions,
    get_subcategories,
    normalize_categories,
    filter_by_group,
    get_group_categories,
    iter_categories,
    load_seeds,
    get_seed_prompts,
)
