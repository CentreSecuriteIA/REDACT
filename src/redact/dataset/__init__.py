"""Dataset handling utilities — CSV I/O, merging, deduplication, splitting, loading, taxonomy."""

from .dedup import exact_dedup, normalized_dedup
from .io import (
    SAMPLE_COLUMNS,
    append_samples,
    get_existing_samples,
    read_category_csv,
    write_category_csv,
)
from .ledger import Ledger
from .loading import filter_dataset, load_from_config, load_hf_dataset
from .manifest import Manifest
from .merge import (
    CONSTITUTION_INPUT_COLUMN_MAP,
    CONSTITUTION_INPUT_KEEP_COLUMNS,
    CONTENT_MOD_COLUMN_MAP,
    CONTENT_MOD_KEEP_COLUMNS,
    JAILBREAK_COLUMN_MAP,
    JAILBREAK_KEEP_COLUMNS,
    TYPE_COLUMN_MAP,
    discover_categories,
    merge_all,
    merge_category_csvs,
    merge_constitution_input_csvs,
    merge_content_mod_csvs,
    merge_csvs_from_dirs,
    merge_technique_csvs,
    normalize_csv,
    normalize_technique_csv,
)
from .sidecar import JsonlSidecar
from .split import (
    balanced_counts,
    deterministic_balanced_assign,
    normalize_origin,
    split_by_column,
    split_by_functions,
    take_per_group,
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
