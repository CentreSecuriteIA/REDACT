"""High-level pipeline functions for end-to-end dataset generation.

Thin wrappers over the existing building blocks (LLMs, content_moderation,
jailbreak, dataset). Each function saves intermediate results to
Datasets/ and returns a merged DataFrame.

Usage::

    from redact import generate_inputs, generate_jailbreaks, build_dataset

    inputs = generate_inputs(samples_per_category=15, num_categories=3)
    jailbreaks = generate_jailbreaks(inputs=inputs)
    dataset = build_dataset()
"""

import json
import logging
import warnings
from math import ceil
from pathlib import Path

import pandas as pd

from redact import paths
from redact.content_moderation import (
    InputPipeline,
    run_output_generation,
)
from redact.content_moderation.paraphrase import (
    paraphrase_pool,
    run_paraphrase_target,
)
from redact.dataset import (
    Ledger,
    iter_categories,
    load_seeds,
    load_taxonomy,
    merge_all,
    take_per_group,
)
from redact.dataset.merge import (
    discover_categories,
    merge_content_mod_csvs,
)
from redact.jailbreak import (
    batch_apply_combinations,
    build_combination,
    build_function_registry,
    completed_from_output,
    compute_sample_id,
    default_manifest_path,
    get_all_hacking_functions,
    get_all_manipulation_functions,
    get_all_obfuscation_functions,
    get_all_request_functions,
    load_plan,
    load_spec,
    plan_run,
)
from redact.jailbreak.manipulation import get_or_generate_benign_data
from redact.llms import (
    RateLimiter,
    get_backend,
    get_router,
    load_prompt,
)
from redact.llms.base import LLMBackend
from redact.llms.model_config import default_model_for_role

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent  # src/redact/
_DEFAULT_TAXONOMY_DIR = paths.taxonomy_dir()


# Thin delegators to the single-source path module (redact/paths.py). Kept as
# local names so existing call sites read unchanged; all resolution lives in
# one place now.
def _default_dataset_dir() -> Path:
    return paths.datasets()


def _default_jailbreak_path() -> Path:
    return paths.jailbreaks_csv()


def _default_benign_path() -> Path:
    return paths.benign_csv()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_backend(
    backend: LLMBackend | None = None,
    model: str = "venice-uncensored",
) -> tuple[LLMBackend, RateLimiter]:
    """Resolve backend — auto-select from model name if None.

    Returns the process-wide rate limiter (owned by :func:`get_router`) so
    every pipeline shares one RPM budget per model. Previously each call
    constructed a fresh ``RateLimiter()``, which silently allowed each
    pipeline to consume the full budget independently.
    """
    if backend is None:
        backend = get_backend(model)
    return backend, get_router().rate_limiter


def _ensure_benign_data(
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter,
    benign_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Load benign data, generating if it doesn't exist."""
    return get_or_generate_benign_data(
        backend=backend,
        model=model,
        rate_limiter=rate_limiter,
        cache_path=benign_path or _default_benign_path(),
        verbose=verbose,
    )



# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def create_taxonomy(
    name: str,
    categories: dict[str, str | dict],
    description: str = "",
    aliases: dict[str, str] | None = None,
    groups: dict[str, list[str]] | None = None,
    taxonomy_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Create and save a taxonomy JSON file.

    A small authoring helper: writes a taxonomy JSON (same shape as the bundled
    ones in ``configs/taxonomy/``) so it can be loaded by name via
    ``generate_inputs(taxonomy=name, taxonomy_dir=...)``.

    Args:
        name: Taxonomy name (used as filename).
        categories: Category definitions. Values can be description strings
            or full dicts with ``description``, ``subcategories``, etc.
        description: Top-level taxonomy description.
        aliases: Mapping of alternate names to canonical names.
        groups: Named groups of categories.
        taxonomy_dir: Directory to save the JSON file. Defaults to the bundled
            ``configs/taxonomy/`` (same dir ``load_taxonomy`` reads by default).
        verbose: Print the saved path.

    Returns:
        The taxonomy dict (usable directly in ``generate_inputs(taxonomy=...)``)
    """
    normalized = {}
    for cat_name, cat_info in categories.items():
        if isinstance(cat_info, str):
            normalized[cat_name] = {"description": cat_info}
        else:
            normalized[cat_name] = cat_info

    taxonomy = {
        "name": name,
        "description": description,
        "categories": normalized,
        "aliases": aliases or {},
        "groups": groups or {},
    }

    save_dir = Path(taxonomy_dir or _DEFAULT_TAXONOMY_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    if verbose:
        logger.info("Taxonomy '%s' saved to %s", name, path)
    return taxonomy


# ---------------------------------------------------------------------------
# Constitution generation
# ---------------------------------------------------------------------------


def generate_constitution(
    taxonomy: dict | str = "content_moderation_categories",
    entry_types: list[str] | None = None,
    num_categories: int = 10,
    model: str | None = None,
    num_taxonomy_categories: int | None = None,
    include_standalone_benign: bool = False,
    standalone_benign_categories: int = 10,
    data_dir: str | Path | None = None,
    taxonomy_dir: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate a constitution (category hierarchy) for classifier training.

    For each taxonomy category, generates harmful, benign, and dual-use
    constitution entries using the specified model. Each entry defines a
    subcategory with sample descriptions that can later drive input sample
    generation.

    Args:
        taxonomy: Taxonomy name or pre-loaded dict.
        entry_types: Which entry types to generate. Choices:
            ``"harmful"``, ``"benign"``, ``"dual_use_benign"``,
            ``"dual_use_harmful"``. Default: all four.
        num_categories: Number of constitution categories per entry type per
            taxonomy category. Range 5-15 recommended.
        model: Generation model. ``None`` (default) resolves to the
            ``constitution_gen`` role — Claude Opus. The backend is auto-selected
            from the model name.
        num_taxonomy_categories: Limit to first N taxonomy categories
            (None = all).
        include_standalone_benign: If True, also generate category-free
            benign entries in a single LLM call (no taxonomy influence).
            Saved to general_benign.csv.
        standalone_benign_categories: Number of benign constitution categories
            to generate in the standalone benign call.
        data_dir: Working root; the constitution CSVs land in
            ``{data_dir}/Data_cache/constitution/``. Defaults to the project root
            (``get_output_dir()``).
        taxonomy_dir: Directory to load the taxonomy from (when ``taxonomy`` is a
            name). Defaults to the bundled ``configs/taxonomy/``.
        resume: When True (default), skip ``(source_category, entry_type)``
            units already recorded in the sidecar ``constitution.state.jsonl``
            ledger beside the CSVs and append to the existing CSVs (a crash
            loses at most one unit's work). When False, clear the CSVs and the
            ledger first and regenerate from scratch.
        verbose: Print progress.

    Returns:
        DataFrame of all constitution entries, backed by the on-disk
        ``merged.csv`` (the full dataset, including rows from prior resumed
        runs — not just this run's delta).
    """
    from redact.constitution import ConstitutionPipeline, EntryType

    # Resolve taxonomy
    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(taxonomy, config_dir=taxonomy_dir)

    # Resolve entry types
    resolved_types: list[EntryType] | None = None
    if entry_types is not None:
        resolved_types = [EntryType(t) for t in entry_types]

    # Resolve model (role default) + backend from the model name
    model = model or default_model_for_role("constitution_gen")
    backend = get_backend(model)

    rate_limiter = get_router().rate_limiter

    const_dir = paths.constitution_dir(data_dir)
    pipeline = ConstitutionPipeline(
        backend=backend,
        model=model,
        rate_limiter=rate_limiter,
        output_dir=const_dir,
    )

    result = pipeline.run(
        taxonomy=taxonomy,
        entry_types=resolved_types,
        num_categories=num_categories,
        num_taxonomy_categories=num_taxonomy_categories,
        include_standalone_benign=include_standalone_benign,
        standalone_benign_categories=standalone_benign_categories,
        save=True,
        resume=resume,
        verbose=verbose,
    )

    return result.to_dataframe()


# ---------------------------------------------------------------------------
# Content moderation input generation
# ---------------------------------------------------------------------------


def generate_inputs(
    taxonomy: dict | str = "content_moderation_categories",
    samples_per_category: int = 15,
    use_metaprompt: bool = True,
    seeds_name: str = "content_moderation_seeds",
    samples_per_request: int = 5,
    num_seeds: int = 8,
    model: str | None = None,
    check_model: str | None = None,
    num_categories: int | None = None,
    data_dir: str | Path | None = None,
    taxonomy_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    constitution_df: pd.DataFrame | None = None,
    style: str = "long",
    samples_per_entry: int = 3,
    entry_types: list[str] | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Generate content moderation input samples (standalone or constitution-seeded).

    Two modes share one implementation:

    - **Standalone** (``constitution_df is None``, default): meta-prompt
      seeds drive per-category multi-turn generation with the
      entry-type-aware quality checker (entry_type defaults to ``harmful``).
      **Deprecated** (emits ``DeprecationWarning``) in favour of the
      constitution-seeded mode, which gives better/more adjustable coverage;
      still fully functional as the cheap/simple "quick eval" path.
    - **Constitution-seeded** (``constitution_df`` provided): each
      constitution entry becomes one generation request via
      ``InputPipeline.run_from_constitution()``. ``style``,
      ``samples_per_entry``, ``entry_types``, ``batch_size`` configure this
      mode. ``taxonomy`` / ``samples_per_category`` / ``num_seeds`` are
      ignored.

    Args:
        taxonomy: Taxonomy name or pre-loaded dict. (Standalone mode only.)
        samples_per_category: Target accepted samples per category. (Standalone.)
        use_metaprompt: LLM-generate descriptions+seeds vs. hand-written. (Standalone.)
        seeds_name: Seeds JSON name when ``use_metaprompt=False``. (Standalone.)
        samples_per_request: Samples per LLM call per turn. (Standalone.)
        num_seeds: Number of seed prompts to generate. (Standalone.)
        model: Generation model. ``None`` (default) resolves to the
            ``uncensored_gen`` role. Backend is auto-selected from the name.
        check_model: Checker model; defaults to ``model``.
        num_categories: First N categories from taxonomy. (Standalone.)
        data_dir: Working root. Standalone inputs land under
            ``{data_dir}/Datasets/``; constitution-seeded inputs auto-route to
            ``{data_dir}/Datasets/constitution_inputs/`` so they never clash.
            Defaults to the project root (``get_output_dir()``).
        taxonomy_dir: Directory to load the taxonomy from (name mode).
            Defaults to the bundled ``configs/taxonomy/``.
        prompt_dir: Root prompt directory for the generation prompt. Defaults to
            the bundled ``prompts/``. (Checker prompts still use the default;
            deeper threading is a follow-up.)
        resume: When True (default), keep and extend existing per-category CSVs
            (standalone dedups against them; constitution mode skips entries
            already present). When False, clear the relevant CSVs first and
            regenerate from scratch.
        verbose: Print progress.
        constitution_df: DataFrame of constitution entries. Triggers
            constitution-seeded mode when non-None.
        style: Template style for constitution mode (``"long"`` / ``"short"`` /
            any directory under ``prompts/input/generation/from_constitution/``).
        samples_per_entry: Prompts to generate per constitution entry.
        entry_types: Filter constitution entries to these types
            (e.g. ``["harmful", "benign"]``). None = all.
        batch_size: Entries per LLM engine pass (constitution mode).

    Returns:
        Merged DataFrame of accepted samples.
    """
    # Generation model defaults to the uncensored_gen role (both modes).
    model = model or default_model_for_role("uncensored_gen")

    # ------------------------------------------------------------------
    # Constitution-seeded mode (delegates to InputPipeline.run_from_constitution)
    # ------------------------------------------------------------------
    if constitution_df is not None:
        backend, rate_limiter = _get_backend(None, model)
        check_model = check_model or model

        # Constitution-seeded inputs auto-route to a dedicated subfolder so they
        # never collide with standalone content-moderation inputs.
        ds_dir = paths.constitution_inputs_dir(data_dir)
        pipeline = InputPipeline(
            gen_backend=backend,
            gen_model=model,
            check_backend=backend,
            check_model=check_model,
            rate_limiter=rate_limiter,
            extraction_style="numbered",
            dataset_dir=ds_dir,
        )

        if entry_types is not None:
            constitution_df = constitution_df[
                constitution_df["entry_type"].isin(entry_types)
            ].reset_index(drop=True)

        prompt_config = load_prompt(
            "input", f"generation/from_constitution/{style}", prompt_dir=prompt_dir
        )

        if verbose:
            logger.info("Generate Inputs (constitution-seeded)")
            logger.info("Entries: %d | Style: %s | Samples/entry: %d | Batch: %d",
                        len(constitution_df), style, samples_per_entry, batch_size)

        pipeline.run_from_constitution(
            constitution_df=constitution_df,
            prompt_config=prompt_config,
            samples_per_entry=samples_per_entry,
            use_checker=True,
            save=True,
            verbose=verbose,
            batch_size=batch_size,
            fresh=not resume,
            style=style,
        )

        result = merge_all(ds_dir, accepted_only=True)
        if style and not result.empty and "template_style" in result.columns:
            result = result[result["template_style"] == style].reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Standalone (meta-prompt) mode
    # ------------------------------------------------------------------
    warnings.warn(
        "Standalone meta-prompt input generation (generate_inputs without "
        "constitution_df) is deprecated in favour of constitution-seeded "
        "generation (generate_inputs(constitution_df=...)); it remains functional.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Resolve taxonomy
    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(taxonomy, config_dir=taxonomy_dir)

    categories = list(iter_categories(taxonomy))
    if num_categories is not None:
        categories = categories[:num_categories]

    # Backend (auto-resolved from the model name)
    backend, rate_limiter = _get_backend(None, model)
    check_model = check_model or model

    # Pipeline
    ds_dir = paths.datasets(data_dir)
    pipeline = InputPipeline(
        gen_backend=backend,
        gen_model=model,
        check_backend=backend,
        check_model=check_model,
        rate_limiter=rate_limiter,
        extraction_style="numbered",
        dataset_dir=ds_dir,
    )

    prompt_config = load_prompt("input", "generation/standalone", prompt_dir=prompt_dir)

    # Seeds (simple mode)
    seeds_db = None
    if not use_metaprompt:
        seeds_db = load_seeds(seeds_name)

    # Clear existing data when not resuming
    if not resume:
        actual_dir = Path(ds_dir) if ds_dir else _default_dataset_dir()
        for cat_name, _ in categories:
            csv_path = paths.category_csv(cat_name, base=actual_dir)
            if csv_path.exists():
                csv_path.unlink()
                if verbose:
                    logger.info("Cleared existing %s", csv_path)

    if verbose:
        mode = "automated (LLM descriptions + seeds)" if use_metaprompt else "simple (taxonomy + hand-written seeds)"
        logger.info("Generate Content Moderation Inputs")
        logger.info("Mode: %s", mode)
        logger.info("Categories: %d | Target: %d samples each (%d/request)",
                    len(categories), samples_per_category, samples_per_request)

    pipeline.run_standalone(
        categories=categories,
        prompt_config=prompt_config,
        samples_per_category=samples_per_category,
        samples_per_request=samples_per_request,
        use_metaprompt=use_metaprompt,
        num_seeds=num_seeds,
        seeds_db=seeds_db,
        verbose=verbose,
    )

    return merge_all(ds_dir, accepted_only=True)


def generate_outputs(
    inputs: pd.DataFrame | None = None,
    model: str | None = None,
    check_outputs: bool = True,
    check_model: str | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
    max_per_category: int | None = None,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate model responses for input samples, batched and quality-checked.

    Pipeline per ``batch_size`` chunk:
      1. Build all messages upfront.
      2. ``BatchCaller.from_model(...).batch_generate(messages_list, model)``
         — single vLLM engine pass (or thread-pool / sequential per backend
         capability). One rate-limit slot per batch.
      3. ``batch_check_samples`` over (input, output) pairs with the
         entry-type-aware output checker. Refusals on harmful inputs are
         rejected; refusals on benign inputs are evaluated normally.
      4. Incremental append to the output CSV per batch — crash-resilient.

    Resume: each input gets a stable content-hash id (its ``id`` column when
    present, else ``_hash_text(prompt)``). Completed ids are recorded in a
    sidecar ``*.state.jsonl`` ledger beside the output CSV. On a re-run
    (``resume=True``) inputs already in the ledger are skipped, so a crash loses
    at most one chunk. ``resume=False`` clears both the CSV and the ledger first.

    Args:
        inputs: Input samples DataFrame. If None, loads accepted samples
            from ``{data_dir}/Datasets/`` via :func:`merge_all`.
        model: Generation model. ``None`` (default) resolves to the
            ``uncensored_gen`` role. Backend is auto-selected from the name.
        check_outputs: Run the entry-type-aware output quality checker.
            Set False to skip checking (accept everything).
        check_model: Checker model; defaults to ``model``.
        batch_size: Inputs per engine pass.
        max_samples: Global cap on inputs processed (``head(N)`` over the whole
            frame); None = all. Applied *after* ``max_per_category``.
        max_per_category: Deterministic cap of N inputs per
            ``(category, entry_type)`` group (falls back to per-``category``,
            then to a plain head, when those columns are absent). Unlike
            ``max_samples`` this keeps every category and severity level
            represented instead of skewing to the first categories in the
            (category-ordered) merged frame. None = no per-group cap.
        data_dir: Working root. Inputs are read from ``{data_dir}/Datasets/``
            (when ``inputs`` is None) and the responses CSV defaults to
            ``{data_dir}/Datasets/output_responses.csv``. Defaults to the project
            root (``get_output_dir()``).
        output_path: Explicit override for the output CSV path. Takes precedence
            over ``data_dir``.
        prompt_dir: Root prompt directory for the generation **and** output-check
            prompts. ``None`` (default) falls back to the bundled ``prompts/``.
        resume: When True (default), skip inputs already recorded in the sidecar
            state ledger. When False, clear the output CSV and ledger first.
        verbose: Print per-batch progress.

    Returns:
        DataFrame of all rows in the output CSV (the full dataset, including
        rows from prior resumed runs) with the unified output schema:
        ``input_id``, ``sample_id``, ``input_prompt``, ``category``,
        ``subcategory``, ``entry_type``, ``output_response``, ``accepted``,
        ``rejection_reason``, ``model``, ``source``. ``sample_id`` is this
        row's own content-hash identity (``_hash_text(output_response)``),
        distinct from ``input_id`` (the origin sample it responds to).
    """
    ds_dir = paths.datasets(data_dir)

    if inputs is None:
        inputs = merge_all(ds_dir, accepted_only=True)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    # Per-group cap first (keeps every category/severity represented), then the
    # global head as a final safety cap.
    if max_per_category is not None:
        inputs = take_per_group(inputs, max_per_category)
        if verbose:
            logger.info("Per-category cap: %d -> %d samples", max_per_category, len(inputs))
    if max_samples is not None:
        inputs = inputs.head(max_samples)

    model = model or default_model_for_role("uncensored_gen")
    backend, rate_limiter = _get_backend(None, model)
    check_model = check_model or model
    check_backend, _ = _get_backend(None, check_model) if check_outputs else (None, None)

    out_path = (
        Path(output_path) if output_path
        else paths.output_responses_csv(data_dir)
    )

    return run_output_generation(
        inputs=inputs, model=model, backend=backend, rate_limiter=rate_limiter,
        check_outputs=check_outputs, check_model=check_model, check_backend=check_backend,
        batch_size=batch_size, out_path=out_path, prompt_dir=prompt_dir,
        resume=resume, verbose=verbose,
    )


def generate_jailbreaks(
    inputs: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    benign_path: str | Path | None = None,
    max_complexity: int = 6,
    max_obfuscations: int = 2,
    seed: int = 42,
    pure_only: bool = False,
    entry_types: list[str] | None = None,
    include_hacking: bool = True,
    include_manipulation: bool = True,
    include_obfuscation: bool = True,
    include_requests: bool = True,
    include_translation: bool = True,
    sampling_probs: dict | None = None,
    model: str | None = None,
    translation_model: str | None = None,
    auto_generate_benign: bool = True,
    batch_size: int = 256,
    iterations: int = 1,
    settings_per_iteration: list[dict] | None = None,
    resume: bool = True,
    manifest_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Augment input prompts with jailbreak technique combinations (batched).

    Two explicit phases:

    1. **Plan** — :func:`redact.jailbreak.plan_run` lays out the full run before
       any generation: every input sample × ``iterations`` gets a
       deterministically assigned combination (seeded on the prompt-content id,
       deduped per sample) written to a JSONL manifest beside the output CSV.
       Skipped when ``resume=True`` and a manifest already exists.
    2. **Execute** — the manifest is streamed in ``batch_size`` chunks; each
       chunk is run through the batched engine
       (:func:`redact.jailbreak.batch_apply_combinations`), which pools LLM calls
       per model per round via the router (vLLM native batch / API multi-worker).
       Output rows are appended to the CSV per chunk, so a crash loses at most
       one chunk and a re-run resumes from the output (its
       ``(input_id, iteration)`` pairs are the source of truth).

    Args:
        inputs: Input prompts DataFrame. If None, loads accepted samples from
            ``Datasets/`` via :func:`merge_all`.
        data_dir: Working root. Inputs are read from ``{data_dir}/Datasets/``
            (when ``inputs`` is None), the jailbreaks CSV defaults to
            ``{data_dir}/Datasets/jailbreaks.csv``, and the benign cache to
            ``{data_dir}/Data_cache/benign/``. Defaults to the project root
            (``get_output_dir()``).
        output_path: Explicit override for the jailbreaks CSV. Defaults to the
            ``data_dir``-derived path. The manifest defaults to
            ``<output>.manifest.jsonl``.
        benign_path: Explicit override for the FSH/DAP benign-cache CSV. Defaults
            to the ``data_dir``-derived path.
        max_complexity / max_obfuscations: Combination-sampler limits.
        seed: Global run seed (combined with each sample's id + iteration).
        pure_only: Exclude LLM-dependent techniques (no router calls).
        entry_types: If set, filter inputs to these entry_type values.
        include_hacking / include_manipulation / include_obfuscation /
            include_requests: Layer toggles for the pool.
        include_translation: When False, drop the translation family from the pool
            so it is never sampled. Translation is the most expensive technique
            family (each is a translate→check→retry loop = 2–8 LLM calls on the
            translation-role model) and adds little to a first-pass dataset.
            Defaults to True.
        sampling_probs: Override per-layer inclusion probabilities (see
            combination_spec.json ``sampling_probs``).
        model: Generation model for LLM-dependent techniques. ``None`` (default)
            resolves to the ``uncensored_gen`` role. Backend auto-resolved from
            the name; the engine routes via the process-wide router.
        translation_model: Model for the translation family. ``None`` (default)
            uses the ``translation`` role (DeepSeek). Translation always routes to
            its own model independently of ``model``.
        auto_generate_benign: Pre-load/generate benign data for FSH/DAP.
        batch_size: Manifest units per engine batch.
        iterations: Combinations to assign per sample (1 = single-round). Ignored when
            ``settings_per_iteration`` is given (its length wins).
        settings_per_iteration: Per-round sampler kwargs for a multi-round
            increasing-complexity run — one dict per round, each merged into the
            combination sampler for that iteration. When provided, ``iterations`` is
            set to ``len(settings_per_iteration)`` and every sample is augmented once
            per round (e.g. 4 rounds → 4 augmentations per sample). The technique
            **pool** is still built once from the top-level ``include_*`` flags (the
            full superset); per-round ``include_*`` / ``sampling_probs`` /
            ``exact_techniques`` keys are *sampling* restrictions, not pool membership.
            See :func:`redact.jailbreak.default_escalation_schedule` for the built-in
            default. ``None`` (default) → single flat-settings run.
        resume: Reuse an existing manifest and skip already-written output rows.
        manifest_path: Override manifest location.
        verbose: Print progress.

    Returns:
        DataFrame of all jailbreak rows from the output CSV (one per planned
        unit), with the input columns plus ``sample_id``, ``jailbreak``,
        ``technique``, ``technique_info``, ``complexity``, ``num_techniques``,
        ``is_noop``, ``accepted``, ``reasoning``, ``iteration``,
        ``combination_spec_version``. ``sample_id`` is this row's own
        content-hash identity (``_hash_text(jailbreak_text)``), distinct from
        ``input_id`` (the origin sample it was derived from).
    """
    out = Path(output_path) if output_path else paths.jailbreaks_csv(data_dir)
    man_path = Path(manifest_path) if manifest_path else default_manifest_path(out)

    # ------------------------------------------------------------------
    # Load + normalize inputs (ensure a prompt column and a content-hash id)
    # ------------------------------------------------------------------
    if inputs is None:
        inputs = merge_all(paths.datasets(data_dir), accepted_only=True)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    inputs = inputs.copy()
    if "sample" in inputs.columns and "prompt" not in inputs.columns:
        inputs = inputs.rename(columns={"sample": "prompt"})
    if "prompt" not in inputs.columns:
        raise ValueError("inputs must contain a 'prompt' (or 'sample') column.")
    if entry_types:
        inputs = inputs[inputs["entry_type"].isin(entry_types)].reset_index(drop=True)

    if "sample_id" not in inputs.columns:
        inputs["sample_id"] = inputs["prompt"].map(lambda p: compute_sample_id(str(p)))
    else:
        missing = inputs["sample_id"].isna() | (inputs["sample_id"].astype(str).isin(["", "nan"]))
        if missing.any():
            inputs.loc[missing, "sample_id"] = inputs.loc[missing, "prompt"].map(
                lambda p: compute_sample_id(str(p))
            )
    inputs["sample_id"] = inputs["sample_id"].astype(str)

    # Generation model defaults to the uncensored_gen role; translation defaults
    # to the translation role (DeepSeek). Backend auto-resolved from the name.
    model = model or default_model_for_role("uncensored_gen")
    backend, rate_limiter = _get_backend(None, model)
    # Internals capture (opt-in, non-interfering — no CSV column). Gated on the
    # gen_model's backend; batch_apply_combinations/_tag_yields separately check
    # each individual request's own target model, so a chain that also routes to
    # translate_model only captures the calls whose model actually supports it.
    capture_jailbreak = getattr(backend, "supports_internals", False)

    # ------------------------------------------------------------------
    # Build technique pool + assignment settings
    # ------------------------------------------------------------------
    # The registry maps every known technique name → callable for reconstructing
    # a planned combination at execution time. It is built from the FULL technique
    # set, independent of the include_* / pure_only / include_translation filters:
    # those filters govern what new plans may *sample*, not what an existing
    # manifest can *reconstruct*. Building it from the filtered pool would make a
    # stale manifest (e.g. one planned with translation, now run with
    # include_translation=False) raise KeyError in build_combination.
    full_pool = (
        get_all_obfuscation_functions()
        + get_all_hacking_functions()
        + get_all_manipulation_functions()
        + get_all_request_functions()
    )
    registry = build_function_registry(full_pool)

    # The sampling pool is the filtered subset — what a fresh plan may draw from.
    pool = []
    if include_obfuscation:
        pool += get_all_obfuscation_functions()
    if include_hacking:
        pool += get_all_hacking_functions()
    if include_manipulation:
        pool += get_all_manipulation_functions()
    if include_requests:
        pool += get_all_request_functions()
    if pure_only:
        # Exclude both per-call LLM techniques and benign-data-dependent ones
        # (FSH/DAP): the latter need a one-time LLM benign-data generation, so
        # they aren't safe in a strictly no-LLM run.
        pool = [
            f for f in pool
            if not getattr(f, "requires_llm", False)
            and not getattr(f, "requires_benign", False)
        ]
    if not include_translation:
        # Translation is the most expensive family per technique (each is a
        # translate→check→retry loop = 2–8 LLM calls) and adds little to a first-pass
        # dataset. Drop it from the pool so it is never sampled.
        pool = [f for f in pool if "translation" not in getattr(f, "families", [])]

    sample_kwargs = {
        "max_complexity": max_complexity,
        "max_obfuscations": max_obfuscations,
        "include_hacking": include_hacking,
        "include_manipulation": include_manipulation,
        "include_obfuscation": include_obfuscation,
        "include_requests": include_requests,
    }
    if sampling_probs is not None:
        sample_kwargs["sampling_probs"] = sampling_probs

    # Multi-round escalation: the per-iteration settings list drives the round count.
    if settings_per_iteration is not None:
        if not settings_per_iteration:
            raise ValueError("settings_per_iteration must be a non-empty list of dicts.")
        iterations = len(settings_per_iteration)

    spec_version = load_spec().get("version", "")

    if verbose:
        logger.info("Generate Jailbreaks (plan + batched execute)")
        logger.info("Pool: %d techniques | Samples: %d | Iterations: %d | Model: %s | Chunk: %d",
                    len(pool), len(inputs), iterations, model, batch_size)

    # ------------------------------------------------------------------
    # Phase 1 — Plan (lay out every unit before generation)
    # ------------------------------------------------------------------
    if resume and man_path.exists():
        if verbose:
            logger.info("Reusing existing manifest: %s", man_path)
    else:
        plan_run(
            inputs, pool, manifest_path=man_path, seed=seed,
            iterations=iterations, settings_per_iteration=settings_per_iteration,
            sample_kwargs=sample_kwargs, verbose=verbose,
        )

    plan_rows = load_plan(man_path)

    # Sidecar completion ledger (same crash-safe pattern as every other stage),
    # keyed on the planned unit ``(input_id, iteration)``.
    jb_ledger = Ledger.sidecar(
        out, key_fields=("input_id", "iteration"),
        casters={"input_id": str, "iteration": int},
    )

    # Fresh (non-resume) run overwrites prior output + ledger; resume keeps + skips.
    if not resume:
        if out.exists():
            out.unlink()
        jb_ledger.reset()
    # Resume source of truth is the ledger, unioned with the output CSV so a run
    # created before the ledger existed still resumes (back-compat).
    completed = (jb_ledger.completed() | completed_from_output(out)) if resume else set()

    # ------------------------------------------------------------------
    # Phase 2 — Execute (stream manifest in chunks through the engine)
    # ------------------------------------------------------------------
    has_manipulation = any(
        "fsh" in getattr(f, "families", []) or "dap" in getattr(f, "families", [])
        for f in pool
    )
    benign_data = None
    if auto_generate_benign and has_manipulation:
        benign_data = _ensure_benign_data(
            backend, model, rate_limiter,
            benign_path=benign_path or paths.benign_csv(data_dir),
            verbose=verbose,
        )

    prompt_map = dict(zip(inputs["sample_id"], inputs["prompt"]))
    # meta_map's records carry the CM input's own "sample_id" (this row's origin
    # identity) — _finalize()'s result overwrites it with the jailbreak's own
    # sample_id via row.update(res) below, since res is applied after this seed.
    meta_map = {r["sample_id"]: r for r in inputs.to_dict("records")}

    pending = [
        r for r in plan_rows
        if str(r["sample_id"]) in prompt_map
        and (str(r["sample_id"]), int(r["iteration"])) not in completed
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    router = get_router()
    total = len(pending)
    written = 0
    n_chunks = ceil(total / batch_size) if (total and batch_size) else 0

    if verbose and total:
        logger.info("Executing %d units in %d chunk(s)...", total, n_chunks)

    for start in range(0, total, batch_size):
        chunk = pending[start : start + batch_size]
        if verbose:
            logger.info("chunk %d/%d: %d units", start // batch_size + 1, n_chunks, len(chunk))
        samples = [
            {
                "id": str(r["sample_id"]),
                "prompt": prompt_map[str(r["sample_id"])],
                "combination": build_combination(r["combination"], registry),
                "iteration": int(r["iteration"]),
            }
            for r in chunk
        ]
        results = batch_apply_combinations(
            samples, gen_model=model, translate_model=translation_model,
            benign_data=benign_data, router=router, verbose=verbose,
            capture_internals=capture_jailbreak,
        )

        rows = []
        for plan_row, res in zip(chunk, results):
            row = dict(meta_map.get(res["input_id"], {}))
            row.update(res)
            row["iteration"] = plan_row["iteration"]
            row["combination_spec_version"] = spec_version
            rows.append(row)

        chunk_df = pd.DataFrame(rows)
        if out.exists():
            chunk_df.to_csv(out, mode="a", header=False, index=False)
        else:
            chunk_df.to_csv(out, index=False)
        # Ack only after the CSV append succeeds (crash-safe): a crash mid-chunk
        # leaves these units un-acked so they re-run next time.
        jb_ledger.record([
            {"input_id": str(x["input_id"]), "iteration": int(x["iteration"])}
            for x in rows
        ])
        written += len(rows)

        if verbose:
            n_acc = sum(1 for x in rows if x.get("accepted"))
            logger.info("[%d/%d] %d accepted -> appended to %s",
                        start + len(rows), total, n_acc, out.name)

    jailbreaks = pd.read_csv(out) if out.exists() else pd.DataFrame()

    if verbose:
        logger.info("Jailbreak Summary")
        logger.info("Planned units: %d | Newly written: %d", len(plan_rows), written)
        logger.info("Total rows in %s: %d", out.name, len(jailbreaks))
        if not jailbreaks.empty and "accepted" in jailbreaks.columns:
            n_accepted = int(jailbreaks["accepted"].sum())
            n_noop = int(jailbreaks["is_noop"].sum()) if "is_noop" in jailbreaks.columns else 0
            logger.info("Accepted: %d | Rejected: %d", n_accepted, len(jailbreaks) - n_accepted)
            logger.info("No-ops: %d", n_noop)
        logger.info("Manifest: %s", man_path)
        logger.info("Saved to: %s", out)

    return jailbreaks


# ---------------------------------------------------------------------------
# Paraphrase / fingerprint removal
#
# The plan/ledger/execute orchestration (model-pool resolution +
# run_paraphrase_target's plan->execute loop) lives in
# content_moderation/paraphrase.py — generate_paraphrases() below just
# validates args and dispatches per target.
# ---------------------------------------------------------------------------


def generate_paraphrases(
    inputs: pd.DataFrame | None = None,
    outputs: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    paraphraser: str | None = None,
    check_model: str | None = None,
    paraphrases_per_sample: int = 1,
    target: str = "both",
    check: bool = True,
    resume: bool = True,
    batch_size: int = 256,
    seed: int = 42,
    prompt_dir: str | Path | None = None,
    inputs_path: str | Path | None = None,
    outputs_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Paraphrase base inputs and/or accepted base outputs into additive artifacts.

    Each base sample is paraphrased ``paraphrases_per_sample`` times (K repeated 1:1
    calls, one per unit), with paraphraser models assigned round-robin from the
    ``paraphraser`` pool. Each paraphrase is validated by a **separate**
    meaning-preservation checker (check→drop, no retry) and deduped so no two stored
    samples are identical. Rows land in ``paraphrases_inputs.csv`` /
    ``paraphrases_outputs.csv`` under ``data_dir``, each keyed by
    ``(input_id, iteration)`` and carrying ``paraphrase_model`` provenance. Per
    artifact: a ``*.manifest.jsonl`` holds the full plan (every unit -> its model),
    and a sidecar ``*.state.jsonl`` **ledger** records every attempted unit + outcome
    — resume reads the ledger and re-runs only unrecorded units (so drops are never
    silently retried), mirroring the constitution / output-response ledgers.

    Args:
        inputs: Base inputs DataFrame; if None loads accepted base inputs from
            ``data_dir``. (target ``"inputs"``/``"both"``.)
        outputs: Base outputs DataFrame; if None loads ``output_responses.csv`` and
            keeps accepted rows. (target ``"outputs"``/``"both"``.)
        data_dir: Working root.
        paraphraser: ``None`` → ``paraphraser`` role default; a model name; or
            ``"distribution"`` → round-robin over all ``paraphraser``-role models.
        check_model: Validator model. ``None`` → ``uncensored_gen`` role (kept
            separate from the paraphraser; warns if they coincide).
        paraphrases_per_sample: K units per base sample (warns if K>1 with 1 model).
        target: ``"inputs"`` | ``"outputs"`` | ``"both"``.
        check: Run the meaning-preservation checker (drop failures if it rejects).
        resume: Skip already-produced ``(input_id, iteration)`` units.
        batch_size: Units per paraphrase/check batch.
        seed: Round-robin assignment seed.
        prompt_dir: Root prompt directory for the paraphrase **and** paraphrase-check
            prompts. ``None`` (default) falls back to the bundled ``prompts/``.
        inputs_path / outputs_path: Artifact path overrides.
        verbose: Print progress.

    Returns:
        ``{"inputs": df, "outputs": df}`` for whichever targets ran.
    """
    pool = paraphrase_pool(paraphraser)
    if paraphrases_per_sample > 1 and len(pool) == 1:
        warnings.warn(
            f"paraphrases_per_sample={paraphrases_per_sample} with a single paraphraser "
            f"({pool[0]}): the K paraphrases rely on sampling variety and identical ones are "
            f"deduped, so the realized count may be < K.",
            stacklevel=2,
        )
    if check:
        check_model = check_model or default_model_for_role("uncensored_gen")
        if check_model in pool:
            warnings.warn(
                f"paraphrase check_model ({check_model}) is also a paraphraser - validation is "
                f"not independent (expected under the placeholder paraphraser setup).",
                stacklevel=2,
            )

    if verbose:
        logger.info("Generate Paraphrases (target=%s)", target)
        logger.info("Paraphraser(s): %s | K: %d | Checker: %s",
                    pool, paraphrases_per_sample, check_model if check else "disabled")

    targets = ["inputs", "outputs"] if target == "both" else [target]
    if not set(targets) <= {"inputs", "outputs"}:
        raise ValueError("target must be 'inputs', 'outputs', or 'both'.")

    results: dict[str, pd.DataFrame] = {}
    for tgt in targets:
        source = inputs if tgt == "inputs" else outputs
        override = inputs_path if tgt == "inputs" else outputs_path
        results[tgt] = run_paraphrase_target(
            tgt, source, data_dir, pool, check, check_model, paraphrases_per_sample,
            resume, batch_size, seed, prompt_dir, override, verbose,
        )
    return results


def build_dataset(
    data_dir: str | Path | None = None,
    jailbreak_path: str | Path | None = None,
    output_path: str | Path | None = None,
    inputs_path: str | Path | None = None,
    responses_path: str | Path | None = None,
    paraphrase_inputs_path: str | Path | None = None,
    paraphrase_outputs_path: str | Path | None = None,
    mode: str = "training",
    include_inputs: bool = True,
    include_jailbreaks: bool = True,
    include_outputs: bool = True,
    include_paraphrases: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge all generated data into a single dataset.

    Combines content moderation inputs, jailbreak variants, and output
    responses from their saved CSVs on disk.

    Inputs and output responses each have two on-disk shapes depending on the
    upstream pipeline: a single merged handoff CSV (the constitution and
    content-moderation notebooks write ``constitution_inputs_merged.csv`` /
    ``cm_inputs_merged.csv`` and ``constitution_output_responses.csv`` /
    ``output_responses.csv``), or per-category ``{category}/samples.csv``
    folders (standalone generation). Pass ``inputs_path`` / ``responses_path``
    to pick the exact files; otherwise they are auto-detected, preferring an
    explicit merged CSV, then the per-category scan.

    Args:
        data_dir: Working root. Inputs are discovered under
            ``{data_dir}/Datasets/``, the jailbreaks CSV defaults to
            ``{data_dir}/Datasets/jailbreaks.csv``, and the merged CSV to
            ``{data_dir}/Datasets/complete_dataset.csv``. Defaults to the project
            root (``get_output_dir()``).
        jailbreak_path: Path to the jailbreaks CSV. Defaults to
            ``Datasets/jailbreaks.csv``.
        output_path: Where to save merged CSV. Defaults to
            ``Datasets/complete_dataset.csv``.
        inputs_path: Explicit merged-inputs CSV (e.g.
            ``Datasets/constitution_inputs_merged.csv``). If omitted, a merged
            CSV is auto-detected, falling back to the per-category scan.
        responses_path: Explicit output-responses CSV (e.g.
            ``Datasets/constitution_output_responses.csv``). If omitted, a
            responses CSV is auto-detected.
        paraphrase_inputs_path / paraphrase_outputs_path: Explicit paraphrase
            artifact CSVs. Default to ``Datasets/paraphrases_{inputs,outputs}.csv``.
        mode: ``"training"`` merges paraphrases into ``complete_dataset.csv``;
            ``"eval"`` keeps them out of the complete dataset and writes them to a
            separate ``Datasets/paraphrased.csv``. (Maps from the recipe
            ``dataset_type``.)
        include_inputs: Include content moderation input samples.
        include_jailbreaks: Include jailbreak samples.
        include_outputs: Include output response samples.
        include_paraphrases: Include paraphrase artifacts (per ``mode``).
        verbose: Print progress.

    Returns:
        Complete merged DataFrame (the base+jailbreak+output set, plus paraphrases
        when ``mode="training"``).
    """
    if mode not in ("training", "eval"):
        raise ValueError("mode must be 'training' or 'eval'.")
    ds_dir = paths.datasets(data_dir)
    jb_path = Path(jailbreak_path) if jailbreak_path else paths.jailbreaks_csv(data_dir)
    out_path = Path(output_path) if output_path else (ds_dir / paths.COMPLETE_DATASET_FILENAME)

    parts = []

    if verbose:
        logger.info("Build Complete Dataset")

    # Content moderation inputs.
    # Prefer an explicit/auto-detected merged handoff CSV (constitution or
    # content-moderation), then fall back to the per-category samples.csv scan.
    if include_inputs:
        in_path = Path(inputs_path) if inputs_path else None
        if in_path is None:
            for candidate in ("constitution_inputs_merged.csv", "cm_inputs_merged.csv"):
                if (ds_dir / candidate).exists():
                    in_path = ds_dir / candidate
                    break

        if in_path is not None and in_path.exists():
            cm_df = pd.read_csv(in_path)
            cm_df["dataset_type"] = "content_moderation_input"
            parts.append(cm_df)
            if verbose:
                logger.info("Inputs: %d samples from %s", len(cm_df), in_path.name)
        else:
            categories = discover_categories(ds_dir)
            if categories:
                cm_df = merge_content_mod_csvs(ds_dir, accepted_only=True)
                cm_df["dataset_type"] = "content_moderation_input"
                parts.append(cm_df)
                if verbose:
                    logger.info("Inputs: %d samples from %d categories", len(cm_df), len(categories))
            elif verbose:
                logger.info("Inputs: none found")

    # Jailbreaks
    if include_jailbreaks and jb_path.exists():
        jb_df = pd.read_csv(jb_path)
        if not jb_df.empty:
            jb_df["dataset_type"] = "jailbreak"
            parts.append(jb_df)
            if verbose:
                logger.info("Jailbreaks: %d samples", len(jb_df))
        elif verbose:
            logger.info("Jailbreaks: none found")
    elif verbose and include_jailbreaks:
        logger.info("Jailbreaks: file not found (%s)", jb_path)

    # Output responses (explicit/auto-detected: constitution or content-moderation).
    if include_outputs:
        resp_path = Path(responses_path) if responses_path else None
        if resp_path is None:
            for candidate in ("constitution_output_responses.csv", "output_responses.csv"):
                if (ds_dir / candidate).exists():
                    resp_path = ds_dir / candidate
                    break

        if resp_path is not None and resp_path.exists():
            out_df = pd.read_csv(resp_path)
            out_df["dataset_type"] = "content_moderation_output"
            parts.append(out_df)
            if verbose:
                logger.info("Outputs: %d samples from %s", len(out_df), resp_path.name)
        elif verbose:
            logger.info("Outputs: none found")

    # Paraphrases — additive. In ``training`` mode they merge into the complete
    # dataset; in ``eval`` mode they're kept separate in ``paraphrased.csv`` and
    # excluded from the complete set (base always stays put either way).
    if include_paraphrases:
        pi_path = Path(paraphrase_inputs_path) if paraphrase_inputs_path else paths.paraphrases_inputs_csv(data_dir)
        po_path = Path(paraphrase_outputs_path) if paraphrase_outputs_path else paths.paraphrases_outputs_csv(data_dir)
        para_parts = []
        for p in (pi_path, po_path):
            if p.exists():
                pdf = pd.read_csv(p)
                if not pdf.empty:
                    if "accepted" in pdf.columns:
                        pdf = pdf[pdf["accepted"] == True]  # noqa: E712
                    pdf["dataset_type"] = "content_moderation_paraphrase"
                    para_parts.append(pdf)
        if para_parts:
            para_df = pd.concat(para_parts, ignore_index=True)
            if mode == "eval":
                para_out = ds_dir / paths.PARAPHRASED_FILENAME
                para_out.parent.mkdir(parents=True, exist_ok=True)
                para_df.to_csv(para_out, index=False)
                if verbose:
                    logger.info("Paraphrases: %d -> separate %s (eval mode)",
                                len(para_df), para_out.name)
            else:  # training — merge in
                parts.append(para_df)
                if verbose:
                    logger.info("Paraphrases: %d merged (training mode)", len(para_df))
        elif verbose:
            logger.info("Paraphrases: none found")

    if not parts:
        if verbose:
            logger.info("No data found. Run generate_inputs/jailbreaks/outputs first.")
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    if verbose:
        logger.info("Total: %d samples", len(merged))
        if "dataset_type" in merged.columns:
            for dtype, count in merged["dataset_type"].value_counts().items():
                logger.info("%s: %d", dtype, count)
        logger.info("Saved to: %s", out_path)

    return merged
