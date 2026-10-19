"""The stages: resolve what each needs, call its core, return its rows.

Each front (``generate_*``, ``build_dataset``) resolves its inputs, its models
by role and its files under ``data_dir``, then calls the core in its
subpackage. :data:`STAGES` describes the fronts to the driver: the roles each
calls, the files it writes and the kwargs a recipe gives it.
"""

# Known limits for a large run (around 1M rows):
#   - A stage holds its inputs in memory: about 1.1 GB per million input
#     rows. Called with read_back=False, as a recipe run calls every stage,
#     it reads none of its output back.
#   - Nothing locks the constitution's files against a second process; the
#     stage is tens of units.
#TODO(dataset): the paraphrase stage reads its whole output CSV on every run,
#   for the texts already in it. This goes when the stage moves onto
#   drive_units with its ids on the ledger.

import filecmp
import inspect
import json
import logging
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd

from redact import dataset, paths
from redact.config_overlay import taxonomy_folder
from redact.constitution import ConstitutionPipeline, read_constitution
from redact.content_moderation import InputPipeline, run_output_generation
from redact.content_moderation import axes as generation_axes
from redact.content_moderation.modes import resolve_modes
from redact.content_moderation.paraphrase import (
    paraphrase_pool,
    run_paraphrase_target,
)
from redact.dataset import (
    Exclusion,
    Manifest,
    Part,
    accepted_mask,
    clear_stage,
    discover_categories,
    exclusion_files,
    has_rows,
    input_files,
    iter_rows,
    load_taxonomy,
    merge_all,
    merge_inputs,
    take_per_group,
    write_concat,
)
from redact.dataset.planfile import coverage_path, plan_path
from redact.jailbreak import augmentation as augmentation_kinds
from redact.jailbreak import pool as jailbreak_pool
from redact.jailbreak.obfuscation import langid
from redact.jailbreak.obfuscation import translation as jailbreak_translation
from redact.jailbreak.pipeline import run_augmentation
from redact.llms import ModelClient, load_prompt
from redact.llms.model_config import default_model_for_role
from redact.types import EntryType

logger = logging.getLogger(__name__)


def generate_constitution(
    taxonomy: dict | str = "harmful",
    entry_types: list[str] | None = None,
    num_categories: int = 10,
    model: str | None = None,
    num_taxonomy_categories: int | None = None,
    include_general_benign: bool = False,
    general_benign_categories: int = 10,
    benign_taxonomy: dict | str = "benign",
    data_dir: str | Path | None = None,
    taxonomy_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    read_back: bool = True,
    config_dir: str | Path | None = None,
    include_standalone_benign: bool | None = None,
    standalone_benign_categories: int | None = None,
) -> pd.DataFrame:
    """Generate a constitution (category hierarchy) for classifier training.

    For each taxonomy category, generates harmful, benign, and dual-use
    constitution entries using the specified model. Each entry defines a
    subcategory with sample descriptions that can later drive input sample
    generation. The general-benign part adds benign entries for each
    category of a benign taxonomy.

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
        num_taxonomy_categories: Limit each taxonomy, ``taxonomy`` and
            ``benign_taxonomy``, to its first N categories (None = all).
        include_general_benign: Also generate the general-benign part: one
            call per category of ``benign_taxonomy``, with its description
            and subcategories, through the ``general_benign`` prompt. Its
            rows are ``benign`` entries whose ``source_category`` is the
            benign category and ``source_taxonomy`` is ``benign``, saved to
            general_benign.csv.
        general_benign_categories: Constitution categories asked for per
            general-benign call.
        benign_taxonomy: The benign taxonomy's name under ``taxonomy_dir``,
            or the loaded taxonomy; read only with ``include_general_benign``.
        data_dir: Working root; the constitution CSVs land in
            ``{data_dir}/Data_cache/constitution/``. Defaults to the project root
            (``get_output_dir()``).
        taxonomy_dir: Directory to load the taxonomy from (when ``taxonomy`` is a
            name). Defaults to the bundled ``configs/taxonomy/``.
        prompt_dir: Root prompt directory for the constitution prompts.
            ``None`` (default) falls back to the bundled ``prompts/``.
        resume: When True (default), skip ``(source_category, entry_type)``
            units already recorded in the sidecar ``constitution.state.jsonl``
            ledger beside the CSVs and append to the existing CSVs (a crash
            loses at most one unit's work). When False, clear the CSVs and the
            ledger first and regenerate from scratch.
        verbose: Log progress.
        read_back: False returns an empty DataFrame and reads no entry back.
        config_dir: A config folder whose ``taxonomy/`` files are read in
            place of the packaged ones, when ``taxonomy_dir`` is not given
            (:mod:`redact.config_overlay`). ``None``: the packaged only.
        include_standalone_benign: The former name of
            ``include_general_benign``, read for this release with a
            ``DeprecationWarning``; given, it wins.
        standalone_benign_categories: The former name of
            ``general_benign_categories``, read as the one above.

    Returns:
        DataFrame of all constitution entries, backed by the on-disk
        ``merged.csv`` (the full dataset, including rows from prior resumed
        runs — not just this run's delta).
    """
    if (include_standalone_benign, standalone_benign_categories) != (None, None):
        warnings.warn(
            "generate_constitution: include_standalone_benign and "
            "standalone_benign_categories are now include_general_benign and "
            "general_benign_categories; the former names are read for this "
            "release.", DeprecationWarning, stacklevel=2)
        if include_standalone_benign is not None:
            include_general_benign = include_standalone_benign
        if standalone_benign_categories is not None:
            general_benign_categories = standalone_benign_categories
    # Resolve taxonomy
    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(
            taxonomy, config_dir=taxonomy_folder(taxonomy, taxonomy_dir, config_dir))
    if include_general_benign and isinstance(benign_taxonomy, str):
        benign_taxonomy = load_taxonomy(benign_taxonomy, config_dir=taxonomy_folder(
            benign_taxonomy, taxonomy_dir, config_dir))

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
        prompt_dir=prompt_dir,
    )

    return pipeline.run(
        taxonomy=taxonomy,
        entry_types=resolved_types,
        num_categories=num_categories,
        num_taxonomy_categories=num_taxonomy_categories,
        include_general_benign=include_general_benign,
        general_benign_categories=general_benign_categories,
        benign_taxonomy=benign_taxonomy if include_general_benign else None,
        resume=resume,
        verbose=verbose,
        read_back=read_back,
    )


def generate_inputs(
    constitution_df: pd.DataFrame | None = None,
    model: str | None = None,
    check_model: str | None = None,
    data_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    style: str = "long",
    samples_per_entry: int = 3,
    entry_types: list[str] | None = None,
    batch_size: int = 32,
    max_attempts: int = 3,
    check_max_tokens: int | None = None,
    axes: dict | bool | None = None,
    axes_weighting: str = "table",
    seed: int = 42,
    config_dir: str | Path | None = None,
    read_back: bool = True,
) -> pd.DataFrame:
    """Generate content moderation input samples seeded from a constitution.

    Each constitution entry becomes ``samples_per_entry`` units of one sample
    each via ``InputPipeline.run_from_constitution()``, each dealt a spec of
    the generation axes; the samples are checked with the entry-type-aware
    quality checker.

    Args:
        constitution_df: DataFrame of constitution entries. ``None`` reads
            the constitution saved under ``data_dir``.
        model: Generation model. ``None`` (default) resolves to the
            ``uncensored_gen`` role. Backend is auto-selected from the name.
        check_model: Checker model; defaults to ``model``.
        data_dir: Working root. The inputs land under
            ``{data_dir}/Datasets/constitution_inputs/``. Defaults to the
            project root (``get_output_dir()``).
        prompt_dir: Root prompt directory for the generation, format and
            check prompts. ``None`` (default) falls back to the bundled
            ``prompts/``.
        resume: When True (default), skip entries already processed. When
            False, start this style over.
        verbose: Log progress.
        style: Template style (``"long"`` / ``"short"`` / any directory under
            ``prompts/input/generation/from_constitution/``).
        samples_per_entry: Samples per constitution entry, one call each.
        entry_types: Filter constitution entries to these types
            (e.g. ``["harmful", "benign"]``). None = all.
        batch_size: Entries per LLM engine pass.
        max_attempts: Generations per unit before a reply without its
            ``[END]`` marker or with another number of samples is given up on.
        check_max_tokens: Reply limit for the check calls. ``None`` keeps the
            checker model's default.
        axes: The generation axes each unit is dealt a value of, the
            ``inputs`` section of ``configs/generation_axes.json``: ``None``
            for those, ``False`` for none, or ``{axis: {value: weight}}``
            over them, an axis as ``None`` to leave it out. With axes the
            length axis sets each sample's length; ``style`` picks the
            prompt only.
        axes_weighting: ``"table"`` deals the axes' weights as they are,
            ``"log"`` spreads them more evenly over each axis's values
            (:func:`redact.content_moderation.axes.resolve`).
        seed: The seed of the deal.
        config_dir: A config folder whose ``generation_axes.json`` and
            ``dimensions.json`` are read in place of the packaged ones when
            it has them. ``None``: the packaged ones.
        read_back: False returns an empty DataFrame and reads no sample back.

    Returns:
        Merged DataFrame of accepted samples of ``style``.

    Raises:
        ValueError: ``constitution_df`` is None and no constitution is saved
            under ``data_dir``, ``axes`` names an unknown axis or value or
            ``axes_weighting`` another weighting, or the folder or its plan
            file stops the stage
            (``InputPipeline.run_from_constitution``).
    """
    axes = generation_axes.resolve(
        "inputs", axes, weighting=axes_weighting, config_dir=config_dir)
    if constitution_df is None:
        constitution_df = _saved_constitution(data_dir)
    model = model or default_model_for_role("uncensored_gen")
    client = ModelClient.create(model)
    check_model = check_model or model
    check_client = ModelClient.create(check_model)

    ds_dir = paths.constitution_inputs_dir(data_dir)
    pipeline = InputPipeline(
        gen=client,
        check=check_client,
        extraction_style="numbered",
        dataset_dir=ds_dir,
        prompt_dir=prompt_dir,
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

    result = pipeline.run_from_constitution(
        constitution_df=constitution_df,
        prompt_config=prompt_config,
        samples_per_entry=samples_per_entry,
        use_checker=True,
        verbose=verbose,
        batch_size=batch_size,
        resume=resume,
        style=style,
        max_attempts=max_attempts,
        check_max_tokens=check_max_tokens,
        axes=axes,
        seed=seed,
    )
    if result.total_entries_processed:
        _warn_on_low_acceptance(
            result.total_prompts_accepted, result.total_prompts_generated)

    # The merged file build_dataset() reads: the accepted rows of every
    # style under this data_dir, copied from the category files in chunks;
    # removed when no sample is accepted, so the build takes no row the files
    # no longer hold.
    files = (paths.category_csv(c, root=ds_dir)
             for c in discover_categories(ds_dir))
    merged = paths.constitution_inputs_merged_csv(data_dir)
    written = write_concat(
        merged,
        [Part(f, accepted_only=True) for f in files if has_rows(f)],
        keep_empty=False,
    )
    if not any(written):
        merged.unlink(missing_ok=True)
    if not read_back:
        return pd.DataFrame()
    result = merge_all(ds_dir, accepted_only=True)
    if style and not result.empty and "template_style" in result.columns:
        result = result[result["template_style"] == style].reset_index(drop=True)
    return result


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
    axes: dict | bool | None = None,
    axes_weighting: str = "table",
    seed: int = 42,
    modes: dict | None = None,
    config_dir: str | Path | None = None,
    read_back: bool = True,
) -> pd.DataFrame:
    """Generate model responses for input samples, batched and quality-checked.

    Each row is dealt an answer mode, ``harmful`` or ``helpful``, from those
    its input's entry type allows (``modes``), then a spec of the response
    axes. Pipeline, in calls of up to ``batch_size`` rows:
      1. Generation calls, each with its mode's prompt
         (``output/generation/<mode>``), asking for a response of the spec
         dealt to its row.
      2. Check calls, each with its mode's checker
         (``output/quality_check/<mode>``), shown the category and not the
         entry type: a refusal is rejected in both modes. With
         ``max_attempts > 1`` rejected rows are regenerated with the
         checker's verdict as feedback.
      3. Incremental append to the output CSV as rows finish — crash-resilient.

    Resume: each input gets a stable content-hash id (its ``id`` column when
    present, else ``content_id(prompt)``). Completed ids are recorded in a
    sidecar ``*.state.jsonl`` ledger beside the output CSV. On a re-run
    (``resume=True``) inputs already in the ledger are skipped, so a crash loses
    only the rows still in progress, and the modes and specs are dealt as
    before: ``output_responses.plan.json`` stops a resume with another
    ``seed``, other ``modes`` or other axes' weights. ``resume=False`` clears
    both the CSV and the ledger first.

    Args:
        inputs: Input samples DataFrame. If None, the accepted inputs of the
            run under ``data_dir`` (:func:`redact.dataset.merge_inputs`).
            With no row, the stage warns, writes nothing and returns an
            empty frame.
        model: Generation model. ``None`` (default) resolves to the
            ``uncensored_gen`` role. Backend is auto-selected from the name.
        check_outputs: Run each row's mode's output quality checker.
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
        axes: The response axes each row is dealt a value of, the
            ``outputs`` section of ``configs/generation_axes.json``: ``None``
            for those, ``False`` for none, or ``{axis: {value: weight}}``
            over them, an axis as ``None`` to leave it out.
        axes_weighting: ``"table"`` deals the axes' weights as they are,
            ``"log"`` spreads them more evenly over each axis's values
            (:func:`redact.content_moderation.axes.resolve`).
        seed: The seed of the deal.
        modes: The answer modes each entry type allows, laid over the
            default per entry type: ``harmful`` → ``["harmful"]``, both
            dual-use types → ``["harmful", "helpful"]``, ``benign`` →
            ``["helpful"]`` (:func:`redact.content_moderation.modes.resolve_modes`).
        config_dir: A config folder whose ``generation_axes.json`` and
            ``dimensions.json`` are read in place of the packaged ones when
            it has them. ``None``: the packaged ones.
        read_back: False returns an empty DataFrame and does not read the
            output CSV.

    Returns:
        DataFrame of all rows in the output CSV (the full dataset, including
        rows from prior resumed runs) with the unified output schema:
        ``input_id``, ``input_prompt``, ``category``, ``subcategory``,
        ``entry_type``, ``model``, ``source``, ``sample_id``,
        ``output_response``, ``accepted``, ``rejection_reason``,
        ``response_mode``, ``output_label``, ``response_label``, then with
        axes a ``response_<axis>`` column per axis. ``entry_type`` is the
        input's; ``response_label`` is ``<entry_type>_<output_label>``, the
        output label ``harmful`` for the harmful mode and ``benign`` for the
        helpful one. ``sample_id`` is this row's own content-hash identity
        (``content_id(output_response)``), distinct from ``input_id`` (the
        origin sample it responds to).

    Raises:
        ValueError: ``inputs`` is None and the run has no input sample at all,
            ``axes`` names an unknown axis or value or ``axes_weighting``
            another weighting, ``modes`` is not a mapping of entry types to
            modes, an input's entry type is one ``modes`` does not map, or
            the plan file or the file's columns stop a resume
            (:func:`run_output_generation`).
    """
    axes = generation_axes.resolve(
        "outputs", axes, weighting=axes_weighting, config_dir=config_dir)
    modes = resolve_modes(modes)
    out_path = (
        Path(output_path) if output_path
        else paths.output_responses_csv(data_dir)
    )
    inputs = _given_or_run_inputs(inputs, data_dir)
    if inputs.empty:
        manifest = Manifest.sidecar(out_path).path
        return _skipped("outputs", "input sample", clear=[] if resume else [{
            "artifact": out_path,
            "also": (coverage_path(manifest), plan_path(manifest))}])

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

    return run_output_generation(
        inputs=inputs, client=client,
        check_outputs=check_outputs, check=check_client,
        batch_size=batch_size, out_path=out_path, prompt_dir=prompt_dir,
        resume=resume, verbose=verbose, max_attempts=max_attempts,
        check_max_tokens=check_max_tokens, read_back=read_back,
        axes=axes, seed=seed, modes=modes,
    )


def generate_jailbreaks(
    inputs: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    benign_path: str | Path | None = None,
    benign_taxonomy: dict | str = "benign",
    taxonomy_dir: str | Path | None = None,
    max_complexity: int | None = None,
    max_obfuscations: int | None = None,
    seed: int = 42,
    pure_only: bool | None = None,
    entry_types: list[str] | None = None,
    include_hacking: bool | None = None,
    include_manipulation: bool | None = None,
    include_obfuscation: bool | None = None,
    include_requests: bool | None = None,
    include_translation: bool | None = None,
    sampling_probs: dict | None = None,
    model: str | None = None,
    translation_model: str | None = None,
    auto_generate_benign: bool = True,
    batch_size: int = 256,
    iterations: int | None = None,
    settings_per_iteration: list[dict] | None = None,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    manifest_path: str | Path | None = None,
    verbose: bool = True,
    read_back: bool = True,
    augmentation: dict | None = None,
    variations_path: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Augment input prompts with jailbreak technique combinations and natural
    variations (batched).

    Resolves the inputs, the generation client and the two kinds of the
    augmentation section, then runs
    :func:`redact.jailbreak.pipeline.run_augmentation` once per kind that is
    on: the adversarial kind to the jailbreaks CSV, then the natural kind to
    the variations CSV. Each run plans whole into its own manifest, ledger and
    coverage table and streams its units through the batched driver.

    Args:
        inputs: Input prompts DataFrame. If None, the accepted inputs of the
            run under ``data_dir`` (:func:`redact.dataset.merge_inputs`).
            With no row, the stage warns, writes nothing and returns an
            empty frame.
        data_dir: Working root. Inputs are read from under
            ``{data_dir}/Datasets/`` (when ``inputs`` is None), the jailbreaks
            CSV defaults to ``{data_dir}/Datasets/jailbreaks.csv`` and the
            benign cache to ``{data_dir}/Data_cache/benign/``.
        output_path: Override for the jailbreaks CSV. The manifest defaults to
            ``<output>.manifest.jsonl``.
        variations_path: Override for the variations CSV, by default
            ``variations.csv`` beside ``output_path`` when it is given, else
            ``{data_dir}/Datasets/variations.csv``.
        benign_path: Override for the FSH/DAP benign-cache CSV.
        benign_taxonomy: The benign taxonomy, whose subcategories the FSH/DAP
            benign Q&A pairs are generated for: a taxonomy file's name, or a
            loaded taxonomy. Defaults to the packaged ``benign``.
        taxonomy_dir: Directory to load ``benign_taxonomy`` from (when it is
            a name). Defaults to the bundled ``configs/taxonomy/``.
        seed: Global run seed (combined with each sample's id + iteration).
        model: Generation model. ``None`` resolves to the ``uncensored_gen``
            role.
        translation_model: Model for the translation family. ``None`` uses the
            ``translation`` role, independently of ``model``.
        auto_generate_benign: Pre-load or generate benign data for FSH/DAP.
        batch_size: Requests per model call; twice as many units are live.
        prompt_dir: Prompt overlay forwarded to the LLM-dependent techniques.
        resume: Reuse each kind's existing manifest and skip its finished
            units; False starts both kinds that are on over.
        manifest_path: Override the jailbreaks manifest's location.
        verbose: Log progress.
        read_back: False returns an empty DataFrame and does not read the CSV.
        augmentation: The augmentation section, ``{"adversarial": ...,
            "natural": ...}``, each kind overlaying the packaged default
            (:mod:`redact.jailbreak.augmentation`); None is the default. A
            kind given as None is off.
        config_dir: A config folder whose ``jailbreak/augmentations.json``
            and ``taxonomy/`` files are read in place of the packaged ones
            (``taxonomy_dir``, given, wins for the benign taxonomy;
            :mod:`redact.config_overlay`). ``None``: the packaged only.

    Deprecated aliases:
        ``max_complexity``, ``max_obfuscations``, ``pure_only``,
        ``entry_types``, ``include_hacking``, ``include_manipulation``,
        ``include_obfuscation``, ``include_requests``,
        ``include_translation``, ``sampling_probs``, ``iterations`` and
        ``settings_per_iteration`` are read into
        ``augmentation["adversarial"]``, applied over it with a
        ``DeprecationWarning``: ``include_<layer>`` denies or admits a
        layer, ``include_translation`` is ``translation``,
        ``sampling_probs`` is ``layer_rates`` and
        ``settings_per_iteration`` is ``rounds``.

    Returns:
        The rows of the jailbreaks CSV (input columns plus ``sample_id``,
        ``jailbreak``, ``technique``, ``technique_info``, ``complexity``,
        ``num_techniques``, ``is_noop``, ``accepted``, ``reasoning``,
        ``iteration``, ``combination_spec_version``), or an empty frame when
        the adversarial kind is off. ``sample_id`` is this row's own
        content-hash identity, distinct from ``input_id``. The variations
        CSV (see ``variations_path``) has the same columns, its rows keeping
        their input's ``entry_type``; a variation that leaves its input
        unchanged is not accepted.

    Raises:
        ValueError: ``inputs`` is None and the run has no input sample at all.
    """
    flat = {name: value for name, value in (
        ("include_hacking", include_hacking),
        ("include_manipulation", include_manipulation),
        ("include_obfuscation", include_obfuscation),
        ("include_requests", include_requests),
        ("include_translation", include_translation),
        ("pure_only", pure_only), ("max_complexity", max_complexity),
        ("max_obfuscations", max_obfuscations), ("sampling_probs", sampling_probs),
        ("iterations", iterations),
        ("settings_per_iteration", settings_per_iteration),
        ("entry_types", entry_types),
    ) if value is not None}
    section = _augmentation_section(augmentation, flat, config_dir)
    adversarial = augmentation_kinds.resolve(section, "adversarial",
                                             config_dir=config_dir)
    natural = augmentation_kinds.resolve(section, "natural", config_dir=config_dir)
    if adversarial is None and natural is None:
        return pd.DataFrame()
    # Each kind that is on, with its CSV and its manifest.
    kinds = [(name, kind, out, man) for name, kind, out, man in (
        ("adversarial", adversarial,
         Path(output_path) if output_path else paths.jailbreaks_csv(data_dir),
         Path(manifest_path) if manifest_path else None),
        ("natural", natural, _variations_csv(data_dir, output_path, variations_path),
         None),
    ) if kind is not None]

    inputs = _given_or_run_inputs(inputs, data_dir)
    if inputs.empty:
        return _skipped("jailbreaks", "input sample", clear=[] if resume else [
            _kind_files(out, man) for _name, _kind, out, man in kinds])

    inputs = inputs.copy()
    if "sample" in inputs.columns and "prompt" not in inputs.columns:
        inputs = inputs.rename(columns={"sample": "prompt"})
    if "prompt" not in inputs.columns:
        raise ValueError("inputs must contain a 'prompt' (or 'sample') column.")

    if any(jailbreak_pool.uses_translation(jailbreak_pool.technique_pool(kind))
           and not _of_entry_types(inputs, kind["entry_types"]).empty
           for _name, kind, _out, _man in kinds):
        _check_machine_translation(translation_model, prompt_dir)
    client = ModelClient.create(model or default_model_for_role("uncensored_gen"))
    benign = Path(benign_path) if benign_path else paths.benign_csv(data_dir)

    def run(name: str, kind: dict, out_path: Path, manifest_path: Path | None,
            **kwargs) -> pd.DataFrame:
        of_kind = _of_entry_types(inputs, kind["entry_types"])
        if of_kind.empty:
            logger.warning(
                "%s kind skipped: entry_types %s match none of the inputs' (%s); "
                "it writes nothing.", name, kind["entry_types"],
                sorted(set(inputs["entry_type"].astype(str))))
            if not resume:
                clear_stage(**_kind_files(out_path, manifest_path))
            return pd.DataFrame()
        return run_augmentation(
            of_kind,
            jailbreak_pool.technique_pool(kind),
            client=client, translation_model=translation_model, out_path=out_path,
            benign_path=benign, benign_taxonomy=benign_taxonomy,
            taxonomy_dir=taxonomy_folder(benign_taxonomy, taxonomy_dir, config_dir),
            prompt_dir=prompt_dir, seed=seed,
            rounds=augmentation_kinds.rounds_of(kind),
            family_weights=kind["family_weights"] or None,
            auto_generate_benign=auto_generate_benign, batch_size=batch_size,
            resume=resume, verbose=verbose, manifest_path=manifest_path, **kwargs,
        )

    rows = pd.DataFrame()
    for name, kind, out, man in kinds:
        if name == "adversarial":
            rows = run(name, kind, out, man,
                       layer_rates=kind["layer_rates"] or None,
                       read_back=read_back, label="jailbreak")
        else:
            run(name, kind, out, man,
                read_back=False, label="variation", accept_noop=False)
    return rows


def _check_machine_translation(translation_model: str | None, prompt_dir) -> None:
    """Before any call, for a machine-translation translator: its prompt
    renders in the model's format, and a missing language identifier is
    warned about once."""
    translator = translation_model or jailbreak_translation._resolve_translate_model()
    if not jailbreak_translation.is_machine_translation(translator):
        return
    jailbreak_translation.check_mt_prompt(prompt_dir)
    if not langid.available():
        logger.warning(
            "jailbreaks: %s is a machine-translation model and GlotLID is not "
            "available, so the language of its translations is not checked; "
            "install redact[langid] and fetch its model with "
            "scripts/download_models.py.", translator)


def _kind_files(out_path: Path, manifest_path: Path | None) -> dict:
    """:func:`redact.dataset.clear_stage`'s kwargs for the files of one kind
    of the jailbreak stage: its CSV, manifest, coverage table and plan."""
    manifest = manifest_path or Manifest.sidecar(out_path).path
    return {"artifact": out_path, "manifest_path": manifest,
            "also": (coverage_path(manifest), plan_path(manifest))}


def _variations_csv(data_dir, output_path, variations_path) -> Path:
    """``variations_path``; else ``variations.csv`` beside ``output_path``, or
    the run's when no ``output_path`` is given."""
    if variations_path:
        return Path(variations_path)
    if output_path:
        return Path(output_path).with_name(paths.VARIATIONS_FILENAME)
    return paths.variations_csv(data_dir)


def _augmentation_section(
    augmentation: dict | None, flat: dict, config_dir: str | Path | None = None,
) -> dict:
    """``augmentation`` with the deprecated flat keywords applied over its
    adversarial kind.

    Raises:
        ValueError: A flat keyword is given while the adversarial kind is off.
    """
    section = dict(augmentation or {})
    if not flat:
        return section
    warnings.warn(
        f"generate_jailbreaks: {sorted(flat)} are deprecated; write them in "
        "augmentation['adversarial'] (include_<layer> as a layers deny, "
        "include_translation as translation, sampling_probs as layer_rates, "
        "settings_per_iteration as rounds).",
        DeprecationWarning, stacklevel=3)
    kind = augmentation_kinds.resolve(section, "adversarial", config_dir=config_dir)
    if kind is None:
        raise ValueError(
            f"augmentation['adversarial'] is null (off), and {sorted(flat)} set "
            "it; leave them out, or switch the kind on.")
    section["adversarial"] = augmentation_kinds.apply_flat_keywords(kind, flat)
    return section


def _of_entry_types(inputs: pd.DataFrame, entry_types: list | None) -> pd.DataFrame:
    """The rows of ``inputs`` of ``entry_types``; all of them for None."""
    if not entry_types:
        return inputs
    return inputs[inputs["entry_type"].isin(entry_types)].reset_index(drop=True)


def _has_inputs(data_dir: str | Path | None) -> bool:
    """Whether the run under ``data_dir`` has input samples, accepted or not."""
    return bool(discover_categories(paths.constitution_inputs_dir(data_dir)))


def _given_or_run_inputs(
    inputs: pd.DataFrame | None, data_dir: str | Path | None
) -> pd.DataFrame:
    """``inputs``, or for None the accepted inputs of the run under ``data_dir``.

    Raises:
        ValueError: ``inputs`` is None and the run has no input sample at all.
    """
    if inputs is not None:
        return inputs
    if not _has_inputs(data_dir):
        raise ValueError("No input samples found. Run generate_inputs() first.")
    return merge_inputs(data_dir)


def _skipped(stage: str, what: str, *, clear=()) -> pd.DataFrame:
    """Warn that ``stage`` has no accepted ``what`` and runs on nothing.

    ``clear`` holds :func:`redact.dataset.clear_stage`'s kwargs for each
    output a fresh run starts over, since the core that would clear it does
    not run.
    """
    logger.warning(
        "%s skipped: no %s was accepted, so it has nothing to work on and "
        "writes nothing. The rejection_reason column of the rejected rows says why.",
        stage, what)
    for files in clear:
        clear_stage(**files)
    return pd.DataFrame()


# The inputs stage warns when it accepts fewer than this share of its samples.
_LOW_ACCEPTANCE = 0.5


def _warn_on_low_acceptance(accepted: int, generated: int) -> None:
    """Warn when the inputs stage accepted under half the samples it generated."""
    if accepted and accepted >= _LOW_ACCEPTANCE * generated:
        return
    logger.warning(
        "inputs: only %d of %d generated samples were accepted (under half). "
        "The later stages work on accepted samples only; the rejection_reason "
        "column of the category files says why each was rejected.",
        accepted, generated)


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
            keeps accepted rows. (target ``"outputs"``/``"both"``.) A target
            with rows but none accepted, or given an empty frame, is skipped
            with a warning; one with no rows at all is skipped silently.
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
        inputs_path: Path override for the ``inputs`` target's artifact.
        outputs_path: Path override for the ``outputs`` target's artifact.
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
    check_client = ModelClient.create(check_model) if check else None

    results: dict[str, pd.DataFrame] = {}
    for tgt in targets:
        if tgt == "inputs":
            # Rows of any verdict; a target without any is skipped silently.
            has_rows = inputs is not None or _has_inputs(data_dir)
            source = inputs if inputs is not None else merge_inputs(data_dir)
            out_path = (Path(inputs_path) if inputs_path
                        else paths.paraphrases_inputs_csv(data_dir))
        else:
            source = (outputs if outputs is not None
                      else dataset.read_rows(paths.output_responses_csv(data_dir)))
            has_rows = outputs is not None or not source.empty
            if "accepted" in source.columns:
                source = source[accepted_mask(source["accepted"])]
            out_path = (Path(outputs_path) if outputs_path
                        else paths.paraphrases_outputs_csv(data_dir))
        if has_rows and source.empty:
            what = "input sample" if tgt == "inputs" else "output response"
            results[tgt] = _skipped(f"paraphrase ({tgt})", what, clear=[] if resume
                                    else [{"artifact": out_path}])
            continue
        results[tgt] = run_paraphrase_target(
            tgt, source, out_path,
            pool=pool, check=check_client, k=paraphrases_per_sample,
            resume=resume, batch_size=batch_size, seed=seed, prompt_dir=prompt_dir,
            verbose=verbose, read_back=read_back,
        )
    return results


def build_dataset(
    data_dir: str | Path | None = None,
    jailbreak_path: str | Path | None = None,
    output_path: str | Path | None = None,
    inputs_path: str | Path | None = None,
    variations_path: str | Path | None = None,
    responses_path: str | Path | None = None,
    paraphrase_inputs_path: str | Path | None = None,
    paraphrase_outputs_path: str | Path | None = None,
    mode: str = "training",
    include_inputs: bool = True,
    include_jailbreaks: bool = True,
    include_variations: bool = True,
    include_outputs: bool = True,
    include_paraphrases: bool = True,
    exclude_from: Iterable[str | Path] | str | Path | None = None,
    verbose: bool = True,
    read_back: bool = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Merge the accepted rows of the stage files into the dataset of a type.

    ``mode`` is the dataset type. A ``"training"`` build copies the inputs,
    jailbreaks, natural variations, outputs and paraphrases, one part after
    another, into ``complete_dataset.csv``; a column that only some parts
    have is empty in the rows of the others. An ``"eval"`` build writes
    three files of one flat schema instead, each with a ``text`` and a
    ``label`` column: ``eval_inputs.csv`` (the inputs, then their
    paraphrases, marked ``paraphrased``), ``eval_outputs.csv`` (the
    responses, then their paraphrases) and ``eval_jailbreaks.csv`` (the
    attacks that changed their input, ``is_jailbreak`` true, then the bare
    benign inputs, false); the variations are in none of them. With
    ``exclude_from``, an eval build leaves out the rows another run holds by
    id or text, and flags those from a constitution entry it has too
    (:class:`redact.dataset.Exclusion`); it counts both in
    ``Datasets/eval_excluded.json``, which the run record's counts take.

    Each file is copied in chunks into a temporary file that then replaces
    it. Only the rows whose ``accepted`` is true are copied; a CSV without
    that column is copied whole. A row whose ``sample_id`` an earlier part
    of its file wrote is left out, and so is a benign or dual-use benign
    variation made with a technique the natural kind does not draw (a plan
    made before it left the kind), with one WARNING that counts them.

    The inputs are read from ``constitution_inputs_merged.csv``, which the
    inputs stage writes, or else from the category files of
    ``constitution_inputs/`` (:func:`redact.dataset.merge_inputs`); the
    outputs from ``output_responses.csv``. ``inputs_path`` /
    ``responses_path`` pick other files.

    Args:
        data_dir: Working root. Every file is read and written under
            ``{data_dir}/Datasets/`` unless its ``*_path`` says otherwise.
            Defaults to the project root (``get_output_dir()``).
        jailbreak_path: Path to the jailbreaks CSV. Defaults to
            ``Datasets/jailbreaks.csv``.
        variations_path: Path to the variations CSV. Defaults to
            ``Datasets/variations.csv``.
        output_path: Where to save the complete dataset of a training build.
            Defaults to ``Datasets/complete_dataset.csv``.
        inputs_path: Merged-inputs CSV. Defaults to
            ``Datasets/constitution_inputs_merged.csv``, falling back to the
            run's category files when it does not exist.
        responses_path: Output-responses CSV. Defaults to
            ``Datasets/output_responses.csv``.
        paraphrase_inputs_path: Explicit paraphrase-inputs CSV. Defaults to
            ``Datasets/paraphrases_inputs.csv``.
        paraphrase_outputs_path: Explicit paraphrase-outputs CSV. Defaults to
            ``Datasets/paraphrases_outputs.csv``.
        mode: The dataset type, ``"training"`` or ``"eval"`` (the recipe's
            ``dataset_type``).
        include_inputs: Include content moderation input samples; in eval
            mode, the benign negatives of the jailbreak file too.
        include_jailbreaks: Include jailbreak samples.
        include_variations: Include natural variations (training only).
        include_outputs: Include output response samples.
        include_paraphrases: Include paraphrases.
        exclude_from: Eval only: the runs whose rows the eval files leave
            out, each a run's folder (its ``Datasets/*.csv`` and its
            constitution) or a CSV, such as a training run's
            ``complete_dataset.csv``.
        verbose: Log progress.
        read_back: False returns empty frames and reads no file back.

    Returns:
        A training build: the rows of the complete dataset. An eval build:
        the rows of each eval file, by ``"inputs"``, ``"outputs"`` and
        ``"jailbreaks"``.

    Raises:
        ValueError: An unknown ``mode``, an ``output_path`` in eval mode, or
            an ``exclude_from`` in training mode.
        FileNotFoundError: An ``exclude_from`` path that does not exist, or a
            folder with no file to read. Nothing is written.
    """
    if mode not in ("training", "eval"):
        raise ValueError("mode must be 'training' or 'eval'.")
    if mode == "eval" and output_path:
        raise ValueError("output_path moves complete_dataset.csv, which an eval "
                         "build does not write; leave it out.")
    held_by = _excluded_paths(exclude_from)
    if held_by and mode != "eval":
        raise ValueError("exclude_from leaves out of an eval build what another "
                         "run holds; a training build takes none.")
    held_files = _held_files(held_by)
    if mode == "eval":
        # Its counts are those of this build or of none.
        paths.eval_excluded_json(data_dir).unlink(missing_ok=True)
    if verbose:
        logger.info("Build %s", "Complete Dataset" if mode == "training"
                    else "Eval Datasets")
    parts = _stage_parts(
        data_dir, inputs_path=inputs_path, jailbreak_path=jailbreak_path,
        variations_path=variations_path, responses_path=responses_path,
        paraphrase_inputs_path=paraphrase_inputs_path,
        paraphrase_outputs_path=paraphrase_outputs_path,
        include={"inputs": include_inputs, "jailbreaks": include_jailbreaks,
                 "variations": include_variations and mode == "training",
                 "outputs": include_outputs, "paraphrases": include_paraphrases},
        verbose=verbose)
    if not parts:
        if verbose:
            logger.info("No data found. Run generate_inputs/jailbreaks/outputs first.")
        return pd.DataFrame() if mode == "training" else {}
    if mode == "eval":
        exclusion = None
        if held_files:
            _warn_on_shared_benign(data_dir, held_by)
            exclusion = Exclusion(held_files)
        built = _build_eval(data_dir, parts, exclusion, verbose=verbose,
                            read_back=read_back)
        if exclusion is not None:
            _save_excluded(data_dir, held_files, exclusion.counts, verbose=verbose)
        return built

    out_path = (Path(output_path) if output_path
                else paths.complete_dataset_csv(data_dir))
    # A row whose id an earlier part wrote is left out: an adversarial no-op
    # is its input, and a variation can be a jailbreak's text.
    written = write_concat(out_path, list(parts.values()), unique="sample_id")
    guard = parts["variations"].where if "variations" in parts else None
    if guard is not None and guard.left_out:
        logger.warning(
            "Variations: left out %d benign or dual-use benign rows made with a "
            "technique the natural kind does not draw (%s); run the natural kind "
            "again with resume=False to replace them.",
            guard.left_out, ", ".join(sorted(guard.techniques)))

    if verbose:
        by_type: Counter = Counter()
        for part, rows in zip(parts.values(), written):
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


def _stage_parts(
    data_dir: str | Path | None, *, inputs_path, jailbreak_path, variations_path,
    responses_path, paraphrase_inputs_path, paraphrase_outputs_path,
    include: dict[str, bool], verbose: bool,
) -> dict[str, Part]:
    """The stage files a build copies, by part, in the order a training build
    writes them.

    A part that ``include`` leaves out, or that has nothing to copy, is
    missing, and with ``verbose`` the log says so.
    """
    def at(given: str | Path | None, default: Callable) -> Path:
        return Path(given) if given else default(data_dir)

    parts: dict[str, Part] = {}
    if include["inputs"]:
        in_path = at(inputs_path, paths.constitution_inputs_merged_csv)
        if in_path.exists():
            parts["inputs"] = Part(in_path, "content_moderation_input",
                                   accepted_only=True,
                                   label=f"Inputs from {in_path.name}")
        else:
            # The one part read whole. Without a row in any category file it
            # has no column.
            cm_df = merge_inputs(data_dir)
            if len(cm_df.columns):
                parts["inputs"] = Part(cm_df, "content_moderation_input",
                                       label="Inputs from the category files")
            elif verbose:
                logger.info("Inputs: none found")

    # A file without a row adds no column to the merged CSV.
    for name, path, dataset_type, where in (
            ("jailbreaks", at(jailbreak_path, paths.jailbreaks_csv), "jailbreak",
             None),
            ("variations", at(variations_path, paths.variations_csv),
             "natural_variation", _NaturalVariations())):
        label = name.capitalize()
        if not include[name]:
            continue
        if not path.exists():
            if verbose:
                logger.info("%s: file not found (%s)", label, path)
        elif has_rows(path):
            parts[name] = Part(path, dataset_type, accepted_only=True, label=label,
                               where=where)
        elif verbose:
            logger.info("%s: none found", label)

    if include["outputs"]:
        resp_path = at(responses_path, paths.output_responses_csv)
        if resp_path.exists():
            parts["outputs"] = Part(resp_path, "content_moderation_output",
                                    accepted_only=True,
                                    label=f"Outputs from {resp_path.name}")
        elif verbose:
            logger.info("Outputs: none found")

    if include["paraphrases"]:
        for name, path in (
                ("paraphrases_inputs",
                 at(paraphrase_inputs_path, paths.paraphrases_inputs_csv)),
                ("paraphrases_outputs",
                 at(paraphrase_outputs_path, paths.paraphrases_outputs_csv))):
            if has_rows(path):
                parts[name] = Part(path, "content_moderation_paraphrase",
                                   accepted_only=True,
                                   label=f"Paraphrases from {path.name}")
        if verbose and not {"paraphrases_inputs", "paraphrases_outputs"} & set(parts):
            logger.info("Paraphrases: none found")
    return parts


#: The entry types whose variations a build keeps only when the natural kind
#: draws every technique that made them.
_BENIGN_ENTRY_TYPES = (EntryType.BENIGN.value, EntryType.DUAL_USE_BENIGN.value)


class _NaturalVariations:
    """The ``where`` of the variations part: it leaves out a benign or
    dual-use benign row whose ``technique`` holds one the natural kind does
    not draw (its spec ``kinds`` lack ``natural``), as a plan made before the
    technique left the natural kind holds, counting them in ``left_out`` and
    their techniques in ``techniques``. A harmful-labelled row stays.
    """

    def __init__(self) -> None:
        self.natural = {fn.__name__ for fn in jailbreak_pool.all_techniques()
                        if "natural" in getattr(fn, "kinds", ())}
        self.left_out = 0
        self.techniques: set[str] = set()

    def __call__(self, chunk: pd.DataFrame) -> pd.Series:
        if not {"technique", "entry_type"} <= set(chunk.columns):
            return pd.Series(True, index=chunk.index)
        unnatural = chunk["technique"].fillna("").astype(str).map(
            lambda chain: [t for t in chain.split("+") if t and t not in self.natural])
        out = chunk["entry_type"].isin(_BENIGN_ENTRY_TYPES) & unnatural.map(bool)
        self.left_out += int(out.sum())
        self.techniques.update(t for chain in unnatural[out] for t in chain)
        return ~out


#: The files of an eval build, by the key its front returns their rows under.
_EVAL_FILES = {"inputs": paths.eval_inputs_csv, "outputs": paths.eval_outputs_csv,
               "jailbreaks": paths.eval_jailbreaks_csv}

#: The columns a derived eval row takes from its input when its own stage
#: file lacks them.
_FROM_INPUT = ("subcategory", "source_taxonomy")


def _build_eval(
    data_dir: str | Path | None, parts: dict[str, Part],
    exclusion: Exclusion | None, *, verbose: bool, read_back: bool,
) -> dict[str, pd.DataFrame]:
    """Write the three eval files from the stage ``parts``, less the rows
    ``exclusion`` leaves out; their rows, by file."""
    files = _eval_parts(parts)
    if exclusion is not None:
        if "inputs" in parts:
            exclusion.read_inputs(parts["inputs"].source)
        files = {name: [_excluding(part, exclusion) for part in file_parts]
                 for name, file_parts in files.items()}
    built = {}
    for name, file_parts in files.items():
        path = _EVAL_FILES[name](data_dir)
        written = write_concat(path, file_parts, unique="sample_id")
        if verbose:
            for part, rows in zip(file_parts, written):
                if part.label:
                    logger.info("%s: %s: %d samples", path.name, part.label, rows)
            logger.info("Saved %d samples to: %s", sum(written), path)
        built[name] = (pd.read_csv(path, dtype={"sample_id": str, "input_id": str},
                                   low_memory=False)
                       if read_back else pd.DataFrame())
    return built


def _eval_parts(parts: dict[str, Part]) -> dict[str, list[Part]]:
    """The parts of each eval file, in the order they are written, each
    mapped onto the file's columns.

    A stage part the build did not find is one without rows, so each file
    has its columns whatever the run made.
    """
    by_input = _by_input(parts.get("inputs"), _FROM_INPUT)

    def part(name: str, **changes) -> Part:
        found = parts.get(name, Part(pd.DataFrame(), ""))
        return found._replace(**changes)

    origin = {"entry_type": "entry_type", "category": "category",
              "subcategory": "subcategory", "source_taxonomy": "source_taxonomy"}
    derived = {**origin, **{column: _of_input(column, by_input[column])
                            for column in _FROM_INPUT}}
    inputs = {"sample_id": "sample_id", "text": "sample", "label": "entry_type",
              "paraphrased": False, "input_id": None, "paraphrase_model": None,
              **origin}
    input_paraphrases = {**inputs, "paraphrased": True, "input_id": "input_id",
                         "paraphrase_model": "paraphrase_model", **derived}
    outputs = {"sample_id": "sample_id", "text": "output_response",
               "label": _response_label("Outputs"), "output_label": "output_label",
               "paraphrased": False, "input_id": "input_id", "model": "model",
               "paraphrase_model": None, **derived}
    output_paraphrases = {**outputs, "text": "sample",
                          "label": _response_label("Output paraphrases"),
                          "paraphrased": True, "model": None,
                          "paraphrase_model": "paraphrase_model"}
    attacks = {"sample_id": "sample_id", "text": "jailbreak", "label": "entry_type",
               "is_jailbreak": True, "technique": "technique",
               "complexity": "complexity", "num_techniques": "num_techniques",
               "iteration": "iteration", "input_id": "input_id", **derived}
    negatives = {**attacks, "text": "sample", "is_jailbreak": False,
                 "technique": None, "complexity": None, "num_techniques": None,
                 "iteration": None, "input_id": None, **origin}
    return {
        "inputs": [part("inputs", columns=inputs),
                   part("paraphrases_inputs", columns=input_paraphrases)],
        "outputs": [part("outputs", columns=outputs),
                    part("paraphrases_outputs", columns=output_paraphrases)],
        "jailbreaks": [part("jailbreaks", columns=attacks, where=_changed),
                       part("inputs", columns=negatives, where=_benign,
                            label="Benign inputs" if "inputs" in parts else "")],
    }


def _excluding(part: Part, exclusion: Exclusion) -> Part:
    """``part``, also leaving out the rows ``exclusion`` holds, of those its
    own ``where`` keeps."""
    origin = "input_id" if part.columns["input_id"] == "input_id" else "sample_id"
    held = exclusion.keeps(part.columns["text"], origin)
    if part.where is None:
        return part._replace(where=held)
    first = part.where

    def where(chunk: pd.DataFrame) -> pd.Series:
        kept = first(chunk).copy()
        kept[kept] = held(chunk[kept])
        return kept

    return part._replace(where=where)


def _excluded_paths(
    exclude_from: Iterable[str | Path] | str | Path | None,
) -> list[Path]:
    """The paths ``exclude_from`` names: one, or each of several."""
    if exclude_from is None:
        return []
    if isinstance(exclude_from, (str, Path)):
        return [Path(exclude_from)]
    return [Path(path) for path in exclude_from]


def _held_files(held_by: list[Path]) -> list[Path]:
    """The CSVs the paths of ``held_by`` name (:func:`dataset.exclusion_files`).

    Raises:
        FileNotFoundError: A path that does not exist, or a folder with no
            file to read.
    """
    files = []
    for path in held_by:
        found = exclusion_files([path]) if path.exists() else []
        if not found:
            raise FileNotFoundError(
                f"exclude_from names {path}, which is neither a CSV nor a run's "
                "folder with a Datasets/*.csv or a constitution.")
        files += found
    return files


def _warn_on_shared_benign(data_dir: str | Path | None, held_by: list[Path]) -> None:
    """Warn when a run of ``held_by`` has the same benign pairs as this one.

    The fsh and dap rows embed the pairs as they are.
    """
    own = paths.benign_csv(data_dir)
    for folder in (path for path in held_by if path.is_dir()):
        theirs = paths.benign_csv(folder)
        if own.exists() and theirs.exists() and filecmp.cmp(own, theirs, shallow=False):
            logger.warning(
                "%s holds the same %s as this run: the fsh and dap rows of both "
                "embed the same benign pairs. Generate them in each run's own "
                "data_dir.", folder, paths.BENIGN_FILENAME)


def _save_excluded(data_dir: str | Path | None, files: list[Path],
                   counts: dict[str, int], *, verbose: bool) -> None:
    """Save the counts of the rows an eval build left out, and of those it
    kept from a shared constitution entry, and log them."""
    if verbose:
        logger.info("Left out %d rows that %d other files hold: %d by id, %d by "
                    "normalised text", counts["exact"] + counts["normalised"],
                    len(files), counts["exact"], counts["normalised"])
    if counts["descriptions"]:
        logger.warning(
            "%d rows come from a constitution entry that another run's "
            "constitution has too; they are kept.", counts["descriptions"])
    path = paths.eval_excluded_json(data_dir)
    path.write_text(json.dumps({"sources": [str(f) for f in files],
                                "excluded": counts}, indent=2), encoding="utf-8")


def _changed(chunk: pd.DataFrame) -> pd.Series:
    """The attack rows whose text is not their input's (``is_noop`` false)."""
    if "is_noop" not in chunk.columns:
        return pd.Series(True, index=chunk.index)
    return ~accepted_mask(chunk["is_noop"])


def _benign(chunk: pd.DataFrame) -> pd.Series:
    """The rows of entry type ``benign``."""
    if "entry_type" not in chunk.columns:
        return pd.Series(False, index=chunk.index)
    return chunk["entry_type"] == EntryType.BENIGN.value


def _response_label(part: str) -> Callable[[pd.DataFrame], pd.Series]:
    """A response's eval ``label``: its ``response_label`` cell, else, for a
    row written before the answer modes, its ``entry_type``. The first chunk
    of the part that falls back logs a WARNING naming ``part``."""
    warned = False

    def cells(chunk: pd.DataFrame) -> pd.Series:
        nonlocal warned
        entry_type = (chunk["entry_type"] if "entry_type" in chunk.columns
                      else pd.Series("", index=chunk.index))
        if "response_label" in chunk.columns:
            label = chunk["response_label"]
            missing = label.isna() | (label.astype(str) == "")
        else:
            label = pd.Series("", index=chunk.index)
            missing = pd.Series(True, index=chunk.index)
        if not missing.any():
            return label
        if not warned:
            warned = True
            logger.warning(
                "%s: rows without a response_label, written before the answer "
                "modes, take their entry_type as their label in %s.", part,
                paths.EVAL_OUTPUTS_FILENAME)
        return label.where(~missing, entry_type)
    return cells


def _of_input(column: str, by_input: dict) -> Callable[[pd.DataFrame], pd.Series]:
    """A derived row's ``column``: its own cell, else that of the input its
    ``input_id`` names."""
    def cells(chunk: pd.DataFrame) -> pd.Series:
        if column in chunk.columns:
            return chunk[column]
        if "input_id" not in chunk.columns:
            return pd.Series("", index=chunk.index)
        return chunk["input_id"].map(by_input)
    return cells


def _by_input(inputs: Part | None, columns: tuple[str, ...]) -> dict[str, dict]:
    """Each of ``columns`` of the inputs, by the inputs' ``sample_id``.

    A file is read in chunks; a column the inputs lack maps no id.
    """
    by_input: dict[str, dict] = {column: {} for column in columns}
    if inputs is None:
        return by_input
    if isinstance(inputs.source, pd.DataFrame):
        header, rows = inputs.source.columns, inputs.source.to_dict("records")
    else:
        header = pd.read_csv(inputs.source, nrows=0).columns
        rows = iter_rows(inputs.source,
                         [c for c in ("sample_id", *columns) if c in header])
    if "sample_id" not in header:
        return by_input
    for row in rows:
        for column in columns:
            if column in row:
                by_input[column][str(row["sample_id"])] = row[column]
    return by_input


@dataclass(frozen=True)
class RunContext:
    """What one run of a recipe hands its stages."""

    data_dir: str | Path | None
    dataset_type: str
    models: dict                # recipe role -> model name, None for its default
    prompt_dir: str | Path | None
    taxonomy: dict | str
    taxonomy_dir: str | Path | None
    params: dict                # the params file: stage name -> the front's kwargs
                                # (over the dataset type's, see Stage.params)
    augmentations: dict         # the section the jailbreak stage runs, resolved
    resume: bool
    verbose: bool
    benign_taxonomy: dict | str = "benign"     # the categories of fsh/dap's pairs
    config_dir: str | Path | None = None       # a folder overlaid on configs/


@dataclass(frozen=True)
class Stage:
    """A stage a recipe can name: its front, the roles it calls, the files it
    reads and writes.

    ``roles`` pairs each recipe role the stage calls with the registry role
    of its default model. ``recipe_kwargs`` is what the run passes the front
    from the recipe; a value an earlier stage of the run saves is a
    :class:`Saved`. ``files`` is the CSVs that hold the stage's rows and
    the values a row holds to be counted, for
    :func:`redact.driver.file_counts`. ``sources`` is the files the stage
    reads its inputs from; those that exist are its run record's
    ``sources``. ``folded`` names the params keys the
    run applies to a kwarg it passes itself (the jailbreak stage's flat
    keywords, to its ``augmentation``), so the front does not get them again.
    ``extra_counts`` reads what the stage counts beyond its files' rows,
    after it ran (the rows an eval build left out).
    """

    name: str
    front: str
    roles: tuple[tuple[str, str], ...]
    files: Callable[[RunContext], tuple[list[Path], dict]]
    recipe_kwargs: Callable[[RunContext], dict]
    sources: Callable[[RunContext], list[Path]]
    folded: tuple[str, ...] = ()
    extra_counts: Callable[[RunContext], dict] | None = None

    @property
    def run(self) -> Callable[..., object]:
        """The front, as :mod:`redact` exports it when the stage runs."""
        import redact

        return getattr(redact, self.front)

    def params(self, ctx: RunContext) -> dict:
        """The stage's params: the dataset type's, overlaid key by key by the
        params file's.

        Raises:
            ValueError: A key that is not a parameter of the front.
        """
        given = _params(ctx, self.name)
        taken = inspect.signature(self.run).parameters
        if any(p.kind is p.VAR_KEYWORD for p in taken.values()):
            return given
        unknown = [key for key in given if key not in taken]
        if unknown:
            raise ValueError(
                f"params[{self.name!r}] names {unknown}, which {self.front} does "
                "not take; write only its parameters.")
        return given

    def check(self, ctx: RunContext) -> dict:
        """The params the front gets, checked without reading any stage
        file, so a run checks every stage before the first runs.

        Raises:
            ValueError: A params key that is not a parameter of the front, or
                one the run passes it itself.
        """
        given = self.params(ctx)
        clash = sorted(set(given) & {*self.recipe_kwargs(ctx), "read_back"})
        if clash:
            raise ValueError(
                f"params[{self.name!r}] sets {clash}, which the run passes "
                f"{self.front} itself; leave {'it' if len(clash) == 1 else 'them'} "
                "out of the params.")
        return {key: value for key, value in given.items() if key not in self.folded}

    def kwargs(self, ctx: RunContext) -> dict:
        """The front's kwargs: the recipe's, then the params.

        Raises:
            ValueError: As :meth:`check`, or a saved value that is missing.
        """
        given = self.check(ctx)
        recipe = {key: value.read() if isinstance(value, Saved) else value
                  for key, value in self.recipe_kwargs(ctx).items()}
        return {**recipe, **given}


@dataclass(frozen=True)
class Saved:
    """A kwarg the run reads when its stage starts, from what an earlier
    stage saved."""

    read: Callable[[], object]


def _params(ctx: RunContext, stage: str) -> dict:
    """The params of ``stage``: the dataset type file's, then the params file's."""
    from redact.recipe import stage_params

    return stage_params(stage, ctx.dataset_type, ctx.params, ctx.config_dir)


def _one_file(stage: str, default: Callable) -> Callable:
    """``files`` of a stage writing one CSV, which its ``output_path`` moves."""
    def files(ctx: RunContext) -> tuple[list[Path], dict]:
        return [_params(ctx, stage).get("output_path") or default(ctx.data_dir)], {}
    return files


def _constitution_kwargs(ctx: RunContext) -> dict:
    return dict(
        taxonomy=ctx.taxonomy, taxonomy_dir=ctx.taxonomy_dir,
        benign_taxonomy=ctx.benign_taxonomy, config_dir=ctx.config_dir,
        model=ctx.models["constitution"], data_dir=ctx.data_dir,
        prompt_dir=ctx.prompt_dir, resume=ctx.resume, verbose=ctx.verbose,
    )


def includes_general_benign(params: dict) -> bool:
    """Whether the constitution ``params`` switch the general-benign part on,
    by its name or its former name."""
    former = params.get("include_standalone_benign")
    return bool(params.get("include_general_benign", False)
                if former is None else former)


def _constitution_files(ctx: RunContext) -> tuple[list[Path], dict]:
    return [paths.constitution_merged_csv(ctx.data_dir)], {}


def _json_file(folder: Path, name: str) -> Path:
    """The file a config ``name`` under ``folder`` is read from, with or
    without its ``.json``."""
    path = folder / name
    return path if path.suffix == ".json" else path.with_name(path.name + ".json")


def _taxonomy_files(ctx: RunContext) -> list[Path]:
    """The files of the run's taxonomy and, with the general-benign part on,
    its benign taxonomy; none for one given as an object."""
    names = [ctx.taxonomy]
    if includes_general_benign(_params(ctx, "constitution")):
        names.append(ctx.benign_taxonomy)
    return [_json_file(taxonomy_folder(name, ctx.taxonomy_dir, ctx.config_dir)
                       or paths.taxonomy_dir(), name)
            for name in names if isinstance(name, str)]


def _inputs_kwargs(ctx: RunContext) -> dict:
    return dict(
        data_dir=ctx.data_dir, prompt_dir=ctx.prompt_dir, model=ctx.models["gen"],
        check_model=ctx.models["check"], resume=ctx.resume, verbose=ctx.verbose,
        constitution_df=Saved(partial(_saved_constitution, ctx.data_dir)),
        config_dir=ctx.config_dir,
    )


def _saved_constitution(data_dir: str | Path | None) -> pd.DataFrame:
    """The constitution saved under ``data_dir``, which seeds the run's inputs.

    Raises:
        ValueError: No constitution is saved there.
    """
    merged = paths.constitution_merged_csv(data_dir)
    if not merged.exists():
        raise ValueError(
            f"No constitution at {merged}. Its entries seed the inputs, so run "
            "the 'constitution' stage (generate_constitution) before 'inputs', "
            "in this recipe or in an earlier run on the same data_dir.")
    return read_constitution(paths.constitution_dir(data_dir))


def _inputs_sources(ctx: RunContext) -> list[Path]:
    """The saved constitution, whose entries seed the inputs."""
    return [paths.constitution_merged_csv(ctx.data_dir)]


def _run_inputs(ctx: RunContext) -> list[Path]:
    return input_files(ctx.data_dir)


def _inputs_files(ctx: RunContext) -> tuple[list[Path], dict]:
    tree = paths.constitution_inputs_dir(ctx.data_dir)
    files = [paths.category_csv(c, root=tree) for c in discover_categories(tree)]
    keep = {"accepted": True}
    # Inputs of another style may share the tree.
    style = _params(ctx, "inputs").get("style", "long")
    if style:
        keep["template_style"] = style
    return files, keep


def _outputs_kwargs(ctx: RunContext) -> dict:
    return dict(
        data_dir=ctx.data_dir, model=ctx.models["gen"],
        check_model=ctx.models["check"], prompt_dir=ctx.prompt_dir,
        resume=ctx.resume, verbose=ctx.verbose, config_dir=ctx.config_dir,
    )


def _paraphrase_kwargs(ctx: RunContext) -> dict:
    return dict(
        data_dir=ctx.data_dir, paraphraser=ctx.models["paraphraser"],
        check_model=ctx.models["paraphrase_check"], prompt_dir=ctx.prompt_dir,
        resume=ctx.resume, verbose=ctx.verbose,
    )


def _paraphrase_sources(ctx: RunContext) -> list[Path]:
    target = _params(ctx, "paraphrase").get("target", "both")
    inputs = input_files(ctx.data_dir) if target in ("inputs", "both") else []
    outputs = ([paths.output_responses_csv(ctx.data_dir)]
               if target in ("outputs", "both") else [])
    return [*inputs, *outputs]


def _paraphrase_files(ctx: RunContext) -> tuple[list[Path], dict]:
    given = _params(ctx, "paraphrase")
    target = given.get("target", "both")
    default = {"inputs": paths.paraphrases_inputs_csv,
               "outputs": paths.paraphrases_outputs_csv}
    return [given.get(f"{name}_path") or default[name](ctx.data_dir)
            for name in default if target in (name, "both")], {}


def _jailbreaks_kwargs(ctx: RunContext) -> dict:
    return dict(
        data_dir=ctx.data_dir, model=ctx.models["gen"],
        translation_model=ctx.models["translation"], prompt_dir=ctx.prompt_dir,
        benign_taxonomy=ctx.benign_taxonomy, taxonomy_dir=ctx.taxonomy_dir,
        resume=ctx.resume, verbose=ctx.verbose, augmentation=ctx.augmentations,
        config_dir=ctx.config_dir,
    )


def _jailbreaks_files(ctx: RunContext) -> tuple[list[Path], dict]:
    """The file of each kind that is on: a kind left out is on, one given as
    None is off."""
    given = _params(ctx, "jailbreaks")
    output, variations = given.get("output_path"), given.get("variations_path")
    files = {
        "adversarial": Path(output) if output else paths.jailbreaks_csv(ctx.data_dir),
        "natural": _variations_csv(ctx.data_dir, output, variations),
    }
    return [path for kind, path in files.items()
            if ctx.augmentations.get(kind, {}) is not None], {}


def _build_kwargs(ctx: RunContext) -> dict:
    return dict(data_dir=ctx.data_dir, mode=ctx.dataset_type, verbose=ctx.verbose)


def _build_excluded(ctx: RunContext) -> dict:
    """The rows an eval build left out because another run holds them."""
    path = paths.eval_excluded_json(ctx.data_dir)
    if ctx.dataset_type != "eval" or not path.exists():
        return {}
    return {"excluded": json.loads(path.read_text(encoding="utf-8"))["excluded"]}


def _build_files(ctx: RunContext) -> tuple[list[Path], dict]:
    """The three eval files of an eval build, else the complete dataset."""
    if ctx.dataset_type == "eval":
        return [path(ctx.data_dir) for path in _EVAL_FILES.values()], {}
    return _one_file("build", paths.complete_dataset_csv)(ctx)


def _build_sources(ctx: RunContext) -> list[Path]:
    """The part files the build copies, in its order: each where its
    ``*_path`` key puts it, and none of a part that ``include_*`` leaves out
    or that the dataset type's build does not copy."""
    given = _params(ctx, "build")

    def at(key: str, default: Callable) -> Path:
        return Path(given[key]) if given.get(key) else default(ctx.data_dir)

    merged = at("inputs_path", paths.constitution_inputs_merged_csv)
    training = ctx.dataset_type == "training"
    parts = {
        "inputs": [merged] if merged.exists() else input_files(ctx.data_dir),
        "jailbreaks": [at("jailbreak_path", paths.jailbreaks_csv)],
        "variations": [at("variations_path", paths.variations_csv)] if training else [],
        "outputs": [at("responses_path", paths.output_responses_csv)],
        "paraphrases": [at("paraphrase_inputs_path", paths.paraphrases_inputs_csv),
                        at("paraphrase_outputs_path", paths.paraphrases_outputs_csv)],
    }
    held = exclusion_files(_excluded_paths(given.get("exclude_from")))
    return [*(path for part, files in parts.items()
              if given.get(f"include_{part}", True) for path in files), *held]


_GEN = ("gen", "uncensored_gen")
_CHECK = ("check", "uncensored_gen")

#: The stages a recipe can name, in the order a run takes them.
STAGES: dict[str, Stage] = {stage.name: stage for stage in (
    Stage("constitution", "generate_constitution",
          roles=(("constitution", "constitution_gen"),),
          files=_constitution_files, recipe_kwargs=_constitution_kwargs,
          sources=_taxonomy_files),
    Stage("inputs", "generate_inputs", roles=(_GEN, _CHECK),
          files=_inputs_files, recipe_kwargs=_inputs_kwargs,
          sources=_inputs_sources),
    Stage("outputs", "generate_outputs", roles=(_GEN, _CHECK),
          files=_one_file("outputs", paths.output_responses_csv),
          recipe_kwargs=_outputs_kwargs, sources=_run_inputs),
    Stage("paraphrase", "generate_paraphrases",
          roles=(("paraphraser", "paraphraser"),
                 ("paraphrase_check", "uncensored_gen")),
          files=_paraphrase_files, recipe_kwargs=_paraphrase_kwargs,
          sources=_paraphrase_sources),
    Stage("jailbreaks", "generate_jailbreaks",
          roles=(_GEN, ("translation", "translation")),
          files=_jailbreaks_files,
          recipe_kwargs=_jailbreaks_kwargs, sources=_run_inputs,
          folded=augmentation_kinds.FLAT_KEYWORDS),
    Stage("build", "build_dataset", roles=(),
          files=_build_files, recipe_kwargs=_build_kwargs, sources=_build_sources,
          extra_counts=_build_excluded),
)}
