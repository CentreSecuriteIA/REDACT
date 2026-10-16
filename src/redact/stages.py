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
#     stage is tens of units. The deprecated standalone inputs have no lock
#     either.
#TODO(dataset): the paraphrase stage reads its whole output CSV on every run,
#   for the texts already in it. This goes when the stage moves onto
#   drive_units with its ids on the ledger.

import inspect
import logging
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd

from redact import dataset, paths
from redact.constitution import ConstitutionPipeline, read_constitution
from redact.content_moderation import (
    InputPipeline,
    run_output_generation,
    run_standalone_inputs,
)
from redact.content_moderation.paraphrase import (
    paraphrase_pool,
    run_paraphrase_target,
)
from redact.dataset import (
    Part,
    discover_categories,
    has_rows,
    iter_categories,
    load_taxonomy,
    merge_all,
    merge_inputs,
    row_ids,
    take_per_group,
    write_concat,
)
from redact.jailbreak import augmentation as augmentation_kinds
from redact.jailbreak import pool as jailbreak_pool
from redact.jailbreak.pipeline import run_augmentation
from redact.llms import ModelClient, load_prompt
from redact.llms.model_config import default_model_for_role
from redact.types import EntryType

logger = logging.getLogger(__name__)


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
        )
        if result.total_entries_processed:
            _warn_on_low_acceptance(
                result.total_prompts_accepted, result.total_prompts_generated)

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

    warnings.warn(
        "Standalone meta-prompt input generation (generate_inputs without "
        "constitution_df) is deprecated in favour of constitution-seeded "
        "generation (generate_inputs(constitution_df=...)); it remains functional.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(taxonomy, config_dir=taxonomy_dir)
    categories = list(iter_categories(taxonomy))
    if num_categories is not None:
        categories = categories[:num_categories]

    ds_dir = paths.datasets(data_dir)
    results = run_standalone_inputs(
        categories,
        ModelClient.create(model),
        ModelClient.create(check_model or model),
        ds_dir,
        prompt_dir=prompt_dir,
        samples_per_category=samples_per_category,
        samples_per_request=samples_per_request,
        use_metaprompt=use_metaprompt,
        seeds_name=seeds_name,
        num_seeds=num_seeds,
        resume=resume,
        verbose=verbose,
    )
    if results:
        _warn_on_low_acceptance(sum(r.total_accepted for r in results),
                                sum(r.total_extracted for r in results))
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
            With no row, the stage warns, writes nothing and returns an
            empty frame.
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

    Raises:
        ValueError: ``inputs`` is None and the run has no input sample at all.
    """
    inputs = _given_or_run_inputs(inputs, data_dir)
    if inputs.empty:
        return _skipped("outputs", "input sample")

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
        max_complexity, max_obfuscations, pure_only, entry_types,
            include_hacking, include_manipulation, include_obfuscation,
            include_requests, include_translation, sampling_probs,
            iterations, settings_per_iteration: Deprecated aliases into
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
    section = _augmentation_section(augmentation, flat)
    adversarial = augmentation_kinds.resolve(section, "adversarial")
    natural = augmentation_kinds.resolve(section, "natural")
    if adversarial is None and natural is None:
        return pd.DataFrame()

    inputs = _given_or_run_inputs(inputs, data_dir)
    if inputs.empty:
        return _skipped("jailbreaks", "input sample")

    inputs = inputs.copy()
    if "sample" in inputs.columns and "prompt" not in inputs.columns:
        inputs = inputs.rename(columns={"sample": "prompt"})
    if "prompt" not in inputs.columns:
        raise ValueError("inputs must contain a 'prompt' (or 'sample') column.")
    inputs["sample_id"] = row_ids(inputs, "prompt")

    client = ModelClient.create(model or default_model_for_role("uncensored_gen"))
    benign = Path(benign_path) if benign_path else paths.benign_csv(data_dir)

    def run(kind: dict, out_path: Path, **kwargs) -> pd.DataFrame:
        return run_augmentation(
            _of_entry_types(inputs, kind["entry_types"]),
            jailbreak_pool.technique_pool(kind),
            client=client, translation_model=translation_model, out_path=out_path,
            benign_path=benign, prompt_dir=prompt_dir, seed=seed,
            rounds=augmentation_kinds.rounds_of(kind),
            family_weights=kind["family_weights"] or None,
            auto_generate_benign=auto_generate_benign, batch_size=batch_size,
            resume=resume, verbose=verbose, **kwargs,
        )

    rows = pd.DataFrame()
    if adversarial is not None:
        rows = run(
            adversarial,
            Path(output_path) if output_path else paths.jailbreaks_csv(data_dir),
            manifest_path=Path(manifest_path) if manifest_path else None,
            layer_rates=adversarial["layer_rates"] or None,
            read_back=read_back, label="jailbreak")
    if natural is not None:
        run(natural, _variations_csv(data_dir, output_path, variations_path),
            read_back=False, label="variation", accept_noop=False)
    return rows


def _variations_csv(data_dir, output_path, variations_path) -> Path:
    """``variations_path``; else ``variations.csv`` beside ``output_path``, or
    the run's when no ``output_path`` is given."""
    if variations_path:
        return Path(variations_path)
    if output_path:
        return Path(output_path).with_name(paths.VARIATIONS_FILENAME)
    return paths.variations_csv(data_dir)


def _augmentation_section(augmentation: dict | None, flat: dict) -> dict:
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
    kind = augmentation_kinds.resolve(section, "adversarial")
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
    return bool(discover_categories(paths.constitution_inputs_dir(data_dir))
                or discover_categories(paths.datasets(data_dir)))


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


def _skipped(stage: str, what: str) -> pd.DataFrame:
    """Warn that ``stage`` has no accepted ``what`` and runs on nothing."""
    logger.warning(
        "%s skipped: no %s was accepted, so it has nothing to work on and "
        "writes nothing. The rejection_reason column of the rejected rows says why.",
        stage, what)
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
                source = source[source["accepted"] == True]  # noqa: E712
            out_path = (Path(outputs_path) if outputs_path
                        else paths.paraphrases_outputs_csv(data_dir))
        if has_rows and source.empty:
            what = "input sample" if tgt == "inputs" else "output response"
            results[tgt] = _skipped(f"paraphrase ({tgt})", what)
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
    verbose: bool = True,
    read_back: bool = True,
) -> pd.DataFrame:
    """Merge all generated data into a single dataset.

    Combines content moderation inputs, jailbreak variants, natural
    variations and output responses from their saved CSVs on disk. Each CSV
    is copied in chunks into a temporary file that then replaces the merged
    one; a column that only some of them have is empty in the rows of the
    others.

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
        variations_path: Path to the variations CSV. Defaults to
            ``Datasets/variations.csv``.
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
        include_variations: Include the accepted natural variations.
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

    # The natural variations the jailbreak stage writes beside its jailbreaks.
    var_path = (Path(variations_path) if variations_path
                else paths.variations_csv(data_dir))
    if include_variations and has_rows(var_path):
        parts.append(Part(var_path, "natural_variation", accepted_only=True,
                          label="Variations"))

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
    augmentations: dict         # the section the jailbreak stage runs, resolved
    resume: bool
    verbose: bool
    stages: tuple[str, ...] = ()    # the stages the recipe names


@dataclass(frozen=True)
class Stage:
    """A stage a recipe can name: its front, the roles it calls, the files it writes.

    ``roles`` pairs each recipe role the stage calls with the registry role
    of its default model. ``recipe_kwargs`` is what the run passes the front
    from the recipe; a value an earlier stage of the run saves is a
    :class:`Saved`. ``files`` is the CSVs that hold the stage's rows and
    the values a row holds to be counted, for
    :func:`redact.driver.file_counts`. ``folded`` names the params keys the
    run applies to a kwarg it passes itself (the jailbreak stage's flat
    keywords, to its ``augmentation``), so the front does not get them again.
    """

    name: str
    front: str
    roles: tuple[tuple[str, str], ...]
    files: Callable[[RunContext], tuple[list[Path], dict]]
    recipe_kwargs: Callable[[RunContext], dict]
    folded: tuple[str, ...] = ()

    @property
    def run(self) -> Callable[..., object]:
        """The front, as :mod:`redact` exports it when the stage runs."""
        import redact

        return getattr(redact, self.front)

    def params(self, ctx: RunContext) -> dict:
        """The params file's kwargs for the stage.

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
        """The params file's kwargs the front gets, checked without reading
        any file, so a run checks every stage before the first runs.

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
        """The front's kwargs: the recipe's, then the params file's.

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
    """The params file's entry for ``stage``."""
    return dict(ctx.params.get(stage) or {})


def _one_file(stage: str, default: Callable) -> Callable:
    """``files`` of a stage writing one CSV, which its ``output_path`` moves."""
    def files(ctx: RunContext) -> tuple[list[Path], dict]:
        return [_params(ctx, stage).get("output_path") or default(ctx.data_dir)], {}
    return files


def _constitution_kwargs(ctx: RunContext) -> dict:
    return dict(
        taxonomy=ctx.taxonomy, taxonomy_dir=ctx.taxonomy_dir,
        model=ctx.models["constitution"], data_dir=ctx.data_dir,
        resume=ctx.resume, verbose=ctx.verbose,
    )


def _constitution_files(ctx: RunContext) -> tuple[list[Path], dict]:
    return [paths.constitution_merged_csv(ctx.data_dir)], {}


def _inputs_kwargs(ctx: RunContext) -> dict:
    kwargs = dict(
        data_dir=ctx.data_dir, taxonomy=ctx.taxonomy, taxonomy_dir=ctx.taxonomy_dir,
        prompt_dir=ctx.prompt_dir, model=ctx.models["gen"],
        check_model=ctx.models["check"], resume=ctx.resume, verbose=ctx.verbose,
    )
    if _seeded(ctx):
        kwargs["constitution_df"] = Saved(partial(_saved_constitution, ctx.data_dir))
    return kwargs


def _seeded(ctx: RunContext) -> bool:
    """Whether the run's inputs are seeded from a constitution: a training
    run's always are, an eval run's when it has one (the stage is in its
    recipe, or a constitution is saved under ``data_dir``). The others are
    standalone."""
    return (ctx.dataset_type == "training" or "constitution" in ctx.stages
            or paths.constitution_merged_csv(ctx.data_dir).exists())


def _saved_constitution(data_dir: str | Path | None) -> pd.DataFrame:
    """The constitution saved under ``data_dir``, which seeds the run's inputs.

    Raises:
        ValueError: No constitution is saved there.
    """
    merged = paths.constitution_merged_csv(data_dir)
    if not merged.exists():
        raise ValueError(
            f"No constitution at {merged}. Its entries seed the inputs, so run "
            "the 'constitution' stage before 'inputs', in this recipe or in an "
            "earlier run on the same data_dir.")
    return read_constitution(paths.constitution_dir(data_dir))


def _inputs_files(ctx: RunContext) -> tuple[list[Path], dict]:
    seeded = _seeded(ctx)
    tree = (paths.constitution_inputs_dir if seeded else paths.datasets)(ctx.data_dir)
    files = [paths.category_csv(c, root=tree) for c in discover_categories(tree)]
    keep = {"accepted": True}
    # Seeded inputs of another style may share the tree.
    style = _params(ctx, "inputs").get("style", "long")
    if seeded and style:
        keep["template_style"] = style
    return files, keep


def _outputs_kwargs(ctx: RunContext) -> dict:
    return dict(
        data_dir=ctx.data_dir, model=ctx.models["gen"],
        check_model=ctx.models["check"], prompt_dir=ctx.prompt_dir,
        resume=ctx.resume, verbose=ctx.verbose,
    )


def _paraphrase_kwargs(ctx: RunContext) -> dict:
    return dict(
        data_dir=ctx.data_dir, paraphraser=ctx.models["paraphraser"],
        check_model=ctx.models["paraphrase_check"], prompt_dir=ctx.prompt_dir,
        resume=ctx.resume, verbose=ctx.verbose,
    )


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
        resume=ctx.resume, verbose=ctx.verbose, augmentation=ctx.augmentations,
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


_GEN = ("gen", "uncensored_gen")
_CHECK = ("check", "uncensored_gen")

#: The stages a recipe can name, in the order a run takes them.
STAGES: dict[str, Stage] = {stage.name: stage for stage in (
    Stage("constitution", "generate_constitution",
          roles=(("constitution", "constitution_gen"),),
          files=_constitution_files, recipe_kwargs=_constitution_kwargs),
    Stage("inputs", "generate_inputs", roles=(_GEN, _CHECK),
          files=_inputs_files, recipe_kwargs=_inputs_kwargs),
    Stage("outputs", "generate_outputs", roles=(_GEN, _CHECK),
          files=_one_file("outputs", paths.output_responses_csv),
          recipe_kwargs=_outputs_kwargs),
    Stage("paraphrase", "generate_paraphrases",
          roles=(("paraphraser", "paraphraser"),
                 ("paraphrase_check", "uncensored_gen")),
          files=_paraphrase_files, recipe_kwargs=_paraphrase_kwargs),
    Stage("jailbreaks", "generate_jailbreaks",
          roles=(_GEN, ("translation", "translation")),
          files=_jailbreaks_files,
          recipe_kwargs=_jailbreaks_kwargs,
          folded=augmentation_kinds.FLAT_KEYWORDS),
    Stage("build", "build_dataset", roles=(),
          files=_one_file("build", paths.complete_dataset_csv),
          recipe_kwargs=_build_kwargs),
)}
