"""High-level pipeline functions for end-to-end dataset generation.

Thin wrappers over the existing building blocks (LLMs, content_moderation,
jailbreak, dataset). Each function saves intermediate results to
Datasets/ and returns a merged DataFrame.

Usage — constitution-seeded (recommended; see ``generate_constitution``)::

    from redact import (
        generate_constitution, generate_inputs, generate_jailbreaks, build_dataset,
    )

    constitution = generate_constitution(num_taxonomy_categories=3)
    inputs = generate_inputs(constitution_df=constitution, samples_per_entry=3)
    jailbreaks = generate_jailbreaks(inputs=inputs)
    dataset = build_dataset()

Usage — standalone meta-prompt inputs (DEPRECATED path, still functional;
skips constitution generation, so it's the cheaper "quick eval" option)::

    inputs = generate_inputs(samples_per_category=15, num_categories=3)
"""

#TODO(driver script): the generate_* functions should not be called on their
# own; only the pipeline entry point should call them. Fix them to that later.

#TODO(dataset): what the stages do not do yet for a large run (around 1M rows):
#   - A stage holds its inputs in memory, and unless called with
#     read_back=False reads the full output CSV back to return it.
#   - The paraphrase stage reads its whole output CSV on every run, for the
#     texts already in it.
#   - Nothing locks the constitution's files or the standalone inputs against
#     a second process.

import logging
import warnings
from collections import Counter
from contextlib import closing
from pathlib import Path

import pandas as pd

from redact import paths
from redact.content_moderation import (
    InputPipeline,
    run_output_generation,
)
from redact.content_moderation.inputs import _pool_size
from redact.content_moderation.paraphrase import (
    paraphrase_pool,
    run_paraphrase_target,
)
from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    Part,
    append_rows,
    discover_categories,
    has_rows,
    iter_categories,
    load_seeds,
    load_taxonomy,
    merge_all,
    merge_inputs,
    read_rows,
    resume_state,
    row_ids,
    take_per_group,
    write_concat,
)
from redact.jailbreak import (
    build_combination,
    build_function_registry,
    get_all_hacking_functions,
    get_all_manipulation_functions,
    get_all_obfuscation_functions,
    get_all_request_functions,
    load_spec,
    plan_run,
)
from redact.jailbreak.engine import _finalize, _finalize_error, _sample_gen
from redact.jailbreak.manipulation import get_or_generate_benign_data
from redact.llm_pipeline import drive_units
from redact.llms import ModelClient, load_prompt
from redact.llms.model_config import default_model_for_role

logger = logging.getLogger(__name__)


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
    read_back: bool = True,
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
        verbose: Log progress.
        read_back: False returns an empty DataFrame and reads no entry back.

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

    # Resolve model (role default) + client from the model name
    model = model or default_model_for_role("constitution_gen")
    client = ModelClient.create(model)

    const_dir = paths.constitution_dir(data_dir)
    pipeline = ConstitutionPipeline(
        client=client,
        output_dir=const_dir,
    )

    return pipeline.run(
        taxonomy=taxonomy,
        entry_types=resolved_types,
        num_categories=num_categories,
        num_taxonomy_categories=num_taxonomy_categories,
        include_standalone_benign=include_standalone_benign,
        standalone_benign_categories=standalone_benign_categories,
        resume=resume,
        verbose=verbose,
        read_back=read_back,
    )


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
    max_attempts: int = 3,
    check_max_tokens: int | None = None,
    read_back: bool = True,
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
        verbose: Log progress.
        constitution_df: DataFrame of constitution entries. Triggers
            constitution-seeded mode when non-None.
        style: Template style for constitution mode (``"long"`` / ``"short"`` /
            any directory under ``prompts/input/generation/from_constitution/``).
        samples_per_entry: Prompts to generate per constitution entry.
        entry_types: Filter constitution entries to these types
            (e.g. ``["harmful", "benign"]``). None = all.
        batch_size: Entries per LLM engine pass (constitution mode).
        max_attempts: Generations per entry before a reply with the wrong
            number of samples is given up on (constitution mode).
        check_max_tokens: Reply limit for the check calls (constitution
            mode). ``None`` keeps the checker model's default.
        read_back: False returns an empty DataFrame and reads no sample back.

    Returns:
        Merged DataFrame of accepted samples.
    """
    # Generation model defaults to the uncensored_gen role (both modes).
    model = model or default_model_for_role("uncensored_gen")

    # ------------------------------------------------------------------
    # Constitution-seeded mode (delegates to InputPipeline.run_from_constitution)
    # ------------------------------------------------------------------
    if constitution_df is not None:
        client = ModelClient.create(model)
        check_model = check_model or model
        check_client = ModelClient.create(check_model)

        # Constitution-seeded inputs auto-route to a dedicated subfolder so they
        # never collide with standalone content-moderation inputs.
        ds_dir = paths.constitution_inputs_dir(data_dir)
        pipeline = InputPipeline(
            gen=client,
            check=check_client,
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
            verbose=verbose,
            batch_size=batch_size,
            fresh=not resume,
            style=style,
            max_attempts=max_attempts,
            check_max_tokens=check_max_tokens,
        )

        # Write the root-level handoff CSV build_dataset() actually looks for
        # (paths.constitution_inputs_merged_csv) —
        # without this, build_dataset() silently drops every constitution-
        # seeded input row, since its own category-folder discovery scans
        # Datasets/ directly and never finds the one-level-deeper
        # Datasets/constitution_inputs/{category}/samples.csv layout.
        # Unfiltered by style so build_dataset() sees every style generated
        # under this data_dir, not just whichever style this call used.
        # Copied from the category files in chunks; left as it is when no
        # sample is accepted.
        files = (paths.category_csv(c, root=ds_dir)
                 for c in discover_categories(ds_dir))
        write_concat(
            paths.constitution_inputs_merged_csv(data_dir),
            [Part(f, accepted_only=True) for f in files if has_rows(f)],
            keep_empty=False,
        )
        if not read_back:
            return pd.DataFrame()
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
    client = ModelClient.create(model)
    check_model = check_model or model

    # Pipeline
    ds_dir = paths.datasets(data_dir)
    pipeline = InputPipeline(
        gen=client,
        check=ModelClient.create(check_model),
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
        for cat_name, _ in categories:
            csv_path = paths.category_csv(cat_name, root=ds_dir)
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

    return merge_all(ds_dir, accepted_only=True) if read_back else pd.DataFrame()


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
    max_attempts: int = 1,
    check_max_tokens: int | None = None,
    read_back: bool = True,
) -> pd.DataFrame:
    """Generate model responses for input samples, batched and quality-checked.

    Pipeline, in calls of up to ``batch_size`` rows:
      1. Generation calls.
      2. Check calls, each row with its own entry-type-aware output
         checker. Refusals on harmful inputs are rejected; refusals on benign
         inputs are evaluated normally. With ``max_attempts > 1`` rejected
         rows are regenerated with the checker's verdict as feedback.
      3. Incremental append to the output CSV as rows finish — crash-resilient.

    Resume: each input gets a stable content-hash id (its ``id`` column when
    present, else ``content_id(prompt)``). Completed ids are recorded in a
    sidecar ``*.state.jsonl`` ledger beside the output CSV. On a re-run
    (``resume=True``) inputs already in the ledger are skipped, so a crash loses
    only the rows still in progress. ``resume=False`` clears both the CSV and
    the ledger first.

    Args:
        inputs: Input samples DataFrame. If None, the accepted inputs of the
            run under ``data_dir`` (:func:`redact.dataset.merge_inputs`).
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
        data_dir: Working root. Inputs are read from under
            ``{data_dir}/Datasets/`` (when ``inputs`` is None) and the responses
            CSV defaults to ``{data_dir}/Datasets/output_responses.csv``.
            Defaults to the project root (``get_output_dir()``).
        output_path: Explicit override for the output CSV path. Takes precedence
            over ``data_dir``.
        prompt_dir: Root prompt directory for the generation **and** output-check
            prompts. ``None`` (default) falls back to the bundled ``prompts/``.
        resume: When True (default), skip inputs already recorded in the sidecar
            state ledger. When False, clear the output CSV and ledger first.
        verbose: Print per-batch progress.
        max_attempts: Generations per input; more than 1 retries a rejected
            response with the checker's verdict as feedback.
        check_max_tokens: Reply limit for the check calls; a verdict needs
            few tokens. ``None`` keeps the checker model's default.
        read_back: False returns an empty DataFrame and does not read the
            output CSV.

    Returns:
        DataFrame of all rows in the output CSV (the full dataset, including
        rows from prior resumed runs) with the unified output schema:
        ``input_id``, ``sample_id``, ``input_prompt``, ``category``,
        ``subcategory``, ``entry_type``, ``output_response``, ``accepted``,
        ``rejection_reason``, ``model``, ``source``. ``sample_id`` is this
        row's own content-hash identity (``content_id(output_response)``),
        distinct from ``input_id`` (the origin sample it responds to).
    """
    if inputs is None:
        inputs = merge_inputs(data_dir)
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
    client = ModelClient.create(model)
    check_model = check_model or model
    check_client = ModelClient.create(check_model) if check_outputs else None

    out_path = (
        Path(output_path) if output_path
        else paths.output_responses_csv(data_dir)
    )

    return run_output_generation(
        inputs=inputs, client=client,
        check_outputs=check_outputs, check=check_client,
        batch_size=batch_size, out_path=out_path, prompt_dir=prompt_dir,
        resume=resume, verbose=verbose, max_attempts=max_attempts,
        check_max_tokens=check_max_tokens, read_back=read_back,
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
    read_back: bool = True,
) -> pd.DataFrame:
    """Augment input prompts with jailbreak technique combinations (batched).

    Two explicit phases:

    1. **Plan** — :func:`redact.jailbreak.plan_run` lays out the full run before
       any generation: every input sample × ``iterations`` gets a
       deterministically assigned combination (seeded on the prompt-content id,
       deduped per sample) written to a JSONL manifest beside the output CSV.
       Skipped when ``resume=True`` and a manifest already exists.
    2. **Execute** — the pending units of the manifest run through
       :func:`redact.llm_pipeline.drive_units`, ``2 * batch_size`` at a time
       or as many as the generation model's async lane holds if that is more,
       which pools LLM calls per model, up to ``batch_size`` per call, via each
       model's client (vLLM native batch / API multi-worker). Output rows are
       appended to the CSV as units finish, every 32 units, so a crash loses
       only the live units and a re-run resumes from the output (its
       ``(input_id, iteration)`` pairs are the source of truth).

    Args:
        inputs: Input prompts DataFrame. If None, the accepted inputs of the
            run under ``data_dir`` (:func:`redact.dataset.merge_inputs`).
        data_dir: Working root. Inputs are read from under ``{data_dir}/Datasets/``
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
        pure_only: Exclude LLM-dependent techniques (no LLM calls).
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
            the name; the engine resolves a client per model.
        translation_model: Model for the translation family. ``None`` (default)
            uses the ``translation`` role (DeepSeek). Translation always routes to
            its own model independently of ``model``.
        auto_generate_benign: Pre-load/generate benign data for FSH/DAP.
        batch_size: Requests per model call; twice as many units are live.
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
        verbose: Log progress.
        read_back: False returns an empty DataFrame and does not read the
            output CSV.

    Returns:
        DataFrame of all jailbreak rows from the output CSV (one per planned
        unit), with the input columns plus ``sample_id``, ``jailbreak``,
        ``technique``, ``technique_info``, ``complexity``, ``num_techniques``,
        ``is_noop``, ``accepted``, ``reasoning``, ``iteration``,
        ``combination_spec_version``. ``sample_id`` is this row's own
        content-hash identity (``content_id(jailbreak_text)``), distinct from
        ``input_id`` (the origin sample it was derived from).
    """
    out = Path(output_path) if output_path else paths.jailbreaks_csv(data_dir)
    man_path = Path(manifest_path) if manifest_path else Manifest.sidecar(out).path

    # ------------------------------------------------------------------
    # Load + normalize inputs (ensure a prompt column and a content-hash id)
    # ------------------------------------------------------------------
    if inputs is None:
        inputs = merge_inputs(data_dir)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    inputs = inputs.copy()
    if "sample" in inputs.columns and "prompt" not in inputs.columns:
        inputs = inputs.rename(columns={"sample": "prompt"})
    if "prompt" not in inputs.columns:
        raise ValueError("inputs must contain a 'prompt' (or 'sample') column.")
    if entry_types:
        inputs = inputs[inputs["entry_type"].isin(entry_types)].reset_index(drop=True)

    inputs["sample_id"] = row_ids(inputs, "prompt")

    # Generation model defaults to the uncensored_gen role; translation defaults
    # to the translation role (DeepSeek). Backend auto-resolved from the name.
    model = model or default_model_for_role("uncensored_gen")
    client = ModelClient.create(model)
    # Internals capture (opt-in, non-interfering — no CSV column). Gated on the
    # gen model's transport; _sample_gen/_tag_yields separately check
    # each individual request's own target model, so a chain that also routes to
    # translate_model only captures the calls whose model actually supports it.
    capture_jailbreak = client.compute_config.supports_internals

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
        logger.info("Pool: %d techniques | Samples: %d | Iterations: %d | Model: %s | Batch: %d",
                    len(pool), len(inputs), iterations, model, batch_size)

    # Sidecar completion ledger (same crash-safe pattern as every other stage),
    # keyed on the planned unit ``(input_id, iteration)``.
    jb_ledger = Ledger.sidecar(
        out, key_fields=("input_id", "iteration"),
        casters={"input_id": str, "iteration": int},
    )

    # Before the plan is written: the resume takes the stage's lock. A fresh
    # (non-resume) run clears prior output + ledger + plan, so that a run
    # stopped while it plans does not leave the old plan to be reused; a
    # resume keeps + skips what the ledger records, and cuts off what an
    # unfinished save left.
    manifest = Manifest(man_path)
    completed = resume_state(
        jb_ledger,
        resume=resume,
        artifact=out,
        manifest=manifest,
    )

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

    # ------------------------------------------------------------------
    # Phase 2 — Execute (stream the manifest through the refill pool)
    # ------------------------------------------------------------------
    has_manipulation = any(
        "fsh" in getattr(f, "families", []) or "dap" in getattr(f, "families", [])
        for f in pool
    )
    benign_data = None
    if auto_generate_benign and has_manipulation:
        benign_data = get_or_generate_benign_data(
            client=client,
            cache_path=benign_path or paths.benign_csv(data_dir),
            verbose=verbose,
        )

    # Where each input is in the frame, by its id; of two inputs with one id
    # the later one. A unit's prompt and its row's input columns are read
    # there when needed, so no copy of every input is held.
    position = dict(zip(inputs["sample_id"], range(len(inputs))))
    prompts = inputs["prompt"]

    def _pending(plan_row: dict) -> bool:
        """Whether a planned unit is of these inputs and not finished."""
        sid = str(plan_row["sample_id"])
        return sid in position and (sid, int(plan_row["iteration"])) not in completed

    out.parent.mkdir(parents=True, exist_ok=True)
    planned = total = written = 0
    if verbose:     # one pass over the plan, for the numbers in the log lines
        for plan_row in manifest:
            planned += 1
            total += _pending(plan_row)

    if verbose and total:
        logger.info("Executing %d units...", total)

    def _append(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out, rows)
        written += len(rows)
        if verbose:
            n_acc = sum(1 for x in rows if x.get("accepted"))
            logger.info("[%d/%d] %d accepted -> appended to %s",
                        written, total, n_acc, out.name)
        return sizes

    # Plan row and engine sample of each live unit, by its place among the
    # pending units, so two plan rows with one (input_id, iteration) stay two
    # units.
    live: dict[int, tuple[dict, dict]] = {}

    def _units():
        # The plan is read one row at a time, as the driver takes units.
        # Closed with this generator: Windows does not replace an open file.
        with closing(iter(manifest)) as plan:
            for i, r in enumerate(filter(_pending, plan)):
                sid = str(r["sample_id"])
                sample = {
                    "id": sid,
                    "prompt": prompts.iat[position[sid]],
                    "combination": build_combination(r["combination"], registry),
                    "iteration": int(r["iteration"]),
                }
                live[i] = (r, sample)
                yield i, _sample_gen(
                    sample, gen_model=model, translate_model=translation_model,
                    benign_data=benign_data, capture_internals=capture_jailbreak,
                )

    def _add(plan_row: dict, res: dict) -> None:
        # The input's columns seed the row, its own "sample_id" among them,
        # which the jailbreak's replaces in the update below.
        row = inputs.iloc[position[res["input_id"]]].to_dict()
        row.update(res)
        row["iteration"] = plan_row["iteration"]
        row["combination_spec_version"] = spec_version
        # Records are explicit so the on-disk key types stay str/int whatever
        # the rows happen to carry.
        buf.add([row], {"input_id": str(row["input_id"]),
                        "iteration": int(row["iteration"])})

    def _on_done(i, value):
        plan_row, sample = live.pop(i)
        _add(plan_row, _finalize(sample, value, model))

    def _on_error(i, exc):
        plan_row, sample = live.pop(i)
        _add(plan_row, _finalize_error(sample, exc, model))

    # A unit whose chain raises gets an error row and is acked. The buffer
    # appends, then acks (see dataset/save.py), every 32 units and on any
    # exit.
    with (
        closing(_units()) as units,
        CommitBuffer(append=_append, ledger=jb_ledger) as buf,
    ):
        drive_units(
            units,
            on_done=_on_done,
            on_error=_on_error,
            pool_size=_pool_size(batch_size, client),
            cap=batch_size,
            progress="jailbreak" if verbose else None,
        )

    jailbreaks = read_rows(out) if read_back else pd.DataFrame()

    if verbose:
        logger.info("Jailbreak Summary")
        logger.info("Planned units: %d | Newly written: %d", planned, written)
        if read_back:
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
    read_back: bool = True,
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
        inputs: Base inputs DataFrame; if None the accepted inputs of the run
            under ``data_dir`` (:func:`redact.dataset.merge_inputs`). (target
            ``"inputs"``/``"both"``.)
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
        verbose: Log progress.
        read_back: False returns empty DataFrames and does not read the
            artifacts back.

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
            read_back=read_back,
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
    read_back: bool = True,
) -> pd.DataFrame:
    """Merge all generated data into a single dataset.

    Combines content moderation inputs, jailbreak variants, and output
    responses from their saved CSVs on disk. Each CSV is copied in chunks
    into a temporary file that then replaces the merged one; a column that
    only some of them have is empty in the rows of the others.

    The inputs are read from ``constitution_inputs_merged.csv``, which the
    seeded inputs stage writes, or else from the run's per-category
    ``{category}/samples.csv`` files (:func:`redact.dataset.merge_inputs`);
    the outputs from ``output_responses.csv``. ``inputs_path`` /
    ``responses_path`` pick other files.

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
        inputs_path: Merged-inputs CSV. Defaults to
            ``Datasets/constitution_inputs_merged.csv``, falling back to the
            run's category files when it does not exist.
        responses_path: Output-responses CSV. Defaults to
            ``Datasets/output_responses.csv``.
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
        verbose: Log progress.
        read_back: False returns an empty DataFrame and does not read the
            merged CSV.

    Returns:
        The rows of the merged CSV (the base+jailbreak+output set, plus paraphrases
        when ``mode="training"``).
    """
    if mode not in ("training", "eval"):
        raise ValueError("mode must be 'training' or 'eval'.")
    ds_dir = paths.datasets(data_dir)
    jb_path = Path(jailbreak_path) if jailbreak_path else paths.jailbreaks_csv(data_dir)
    out_path = Path(output_path) if output_path else (ds_dir / paths.COMPLETE_DATASET_FILENAME)

    parts: list[Part] = []     # in the order their rows are written

    if verbose:
        logger.info("Build Complete Dataset")

    if include_inputs:
        in_path = (Path(inputs_path) if inputs_path
                   else paths.constitution_inputs_merged_csv(data_dir))
        if in_path.exists():
            parts.append(Part(in_path, "content_moderation_input",
                              label=f"Inputs from {in_path.name}"))
        else:
            # The one part read whole. Without a row in any category file it
            # has no column.
            cm_df = merge_inputs(data_dir)
            if len(cm_df.columns):
                parts.append(Part(cm_df, "content_moderation_input",
                                  label="Inputs from the category files"))
            elif verbose:
                logger.info("Inputs: none found")

    # Jailbreaks. A file without a row adds no column to the merged CSV.
    if include_jailbreaks and jb_path.exists():
        if has_rows(jb_path):
            parts.append(Part(jb_path, "jailbreak", label="Jailbreaks"))
        elif verbose:
            logger.info("Jailbreaks: none found")
    elif verbose and include_jailbreaks:
        logger.info("Jailbreaks: file not found (%s)", jb_path)

    if include_outputs:
        resp_path = (Path(responses_path) if responses_path
                     else paths.output_responses_csv(data_dir))
        if resp_path.exists():
            parts.append(Part(resp_path, "content_moderation_output",
                              label=f"Outputs from {resp_path.name}"))
        elif verbose:
            logger.info("Outputs: none found")

    # Paraphrases — additive. In ``training`` mode they merge into the complete
    # dataset; in ``eval`` mode they're kept separate in ``paraphrased.csv`` and
    # excluded from the complete set (base always stays put either way).
    if include_paraphrases:
        pi_path = Path(paraphrase_inputs_path) if paraphrase_inputs_path else paths.paraphrases_inputs_csv(data_dir)
        po_path = Path(paraphrase_outputs_path) if paraphrase_outputs_path else paths.paraphrases_outputs_csv(data_dir)
        para_parts = [
            Part(p, "content_moderation_paraphrase", accepted_only=True,
                 label=f"Paraphrases from {p.name}")
            for p in (pi_path, po_path) if has_rows(p)
        ]
        if para_parts and mode == "eval":
            para_out = ds_dir / paths.PARAPHRASED_FILENAME
            separate = sum(write_concat(para_out, para_parts))
            if verbose:
                logger.info("Paraphrases: %d -> separate %s (eval mode)",
                            separate, para_out.name)
        elif para_parts:  # training — merge in
            parts += para_parts
        elif verbose:
            logger.info("Paraphrases: none found")

    if not parts:
        if verbose:
            logger.info("No data found. Run generate_inputs/jailbreaks/outputs first.")
        return pd.DataFrame()

    written = write_concat(out_path, parts)

    if verbose:
        by_type: Counter = Counter()
        for part, rows in zip(parts, written):
            logger.info("%s: %d samples", part.label, rows)
            by_type[part.dataset_type] += rows
        logger.info("Total: %d samples", sum(written))
        for dtype, count in by_type.most_common():
            logger.info("%s: %d", dtype, count)
        logger.info("Saved to: %s", out_path)

    if not read_back:
        return pd.DataFrame()
    # low_memory=False: a column that only some parts have would otherwise be
    # typed piece by piece, with a warning about its mixed types.
    return pd.read_csv(out_path, low_memory=False)
