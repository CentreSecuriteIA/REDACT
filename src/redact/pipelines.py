"""High-level pipeline functions for end-to-end dataset generation.

Thin wrappers over the existing building blocks (LLMs, Content_Moderation,
Jailbreak, Dataset_Functions). Each function saves intermediate results to
Datasets/ and returns a merged DataFrame.

Usage::

    from redact import generate_inputs, generate_jailbreaks, build_dataset

    inputs = generate_inputs(samples_per_category=15, num_categories=3)
    jailbreaks = generate_jailbreaks(inputs=inputs)
    dataset = build_dataset()
"""

import json
from math import ceil
from pathlib import Path
from typing import Callable

import pandas as pd

from redact import Config, get_output_dir
from redact.llms import (
    get_backend,
    RateLimiter,
    load_prompt,
    build_messages,
    generate_sample,
    get_router,
    BatchCaller,
)
from redact.llms.base import LLMBackend
from redact.content_moderation import (
    InputPipeline,
    CategoryResult,
    generate_category_description,
    generate_seeds,
)
from redact.content_moderation.checker import (
    build_quality_checker,
    build_output_quality_checker,
)
from redact.dataset.io import get_existing_samples
from redact.dataset import (
    load_taxonomy,
    iter_categories,
    load_seeds,
    get_seed_prompts,
    merge_all,
)
from redact.dataset.merge import (
    merge_content_mod_csvs,
    discover_categories,
)
from redact.jailbreak import (
    get_all_obfuscation_functions,
    get_all_hacking_functions,
    get_all_manipulation_functions,
    get_all_request_functions,
    load_spec,
    build_combination,
    build_function_registry,
    batch_apply_combinations,
    plan_run,
    load_plan,
    completed_from_output,
    compute_sample_id,
    default_manifest_path,
)
from redact.jailbreak.manipulation import (
    BENIGN_CATEGORIES,
    load_benign_data,
    process_category,
    get_or_generate_benign_data,
)


_PACKAGE_DIR = Path(__file__).resolve().parent  # src/redact/
_DEFAULT_TAXONOMY_DIR = _PACKAGE_DIR / "configs" / "taxonomy"


def _default_dataset_dir() -> Path:
    return get_output_dir() / "Datasets"


def _default_jailbreak_path() -> Path:
    return _default_dataset_dir() / "jailbreaks.csv"


def _default_benign_path() -> Path:
    return get_output_dir() / "Data_cache" / "benign" / "benign_samples.csv"


def _default_scenario_dir() -> Path:
    return get_output_dir() / "Data_cache" / "scenarios"


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
    config_dir: str | Path | None = None,
) -> dict:
    """Create and save a taxonomy JSON file.

    Args:
        name: Taxonomy name (used as filename).
        categories: Category definitions. Values can be description strings
            or full dicts with ``description``, ``subcategories``, etc.
        description: Top-level taxonomy description.
        aliases: Mapping of alternate names to canonical names.
        groups: Named groups of categories.
        config_dir: Directory to save the JSON file. Defaults to
            ``Dataset_Configs/taxonomy/``.

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

    save_dir = Path(config_dir or _DEFAULT_TAXONOMY_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"Taxonomy '{name}' saved to {path}")
    return taxonomy


# ---------------------------------------------------------------------------
# Constitution generation
# ---------------------------------------------------------------------------


def generate_constitution(
    taxonomy: dict | str = "content_moderation_categories",
    entry_types: list[str] | None = None,
    num_categories: int = 10,
    model: str = "claude-opus-4-6",
    backend: LLMBackend | None = None,
    num_taxonomy_categories: int | None = None,
    include_standalone_benign: bool = False,
    standalone_benign_categories: int = 10,
    output_dir: str | Path | None = None,
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
        model: Model for generation (default: Claude Opus).
        backend: LLM backend. If None, auto-selects from model name.
        num_taxonomy_categories: Limit to first N taxonomy categories
            (None = all).
        include_standalone_benign: If True, also generate category-free
            benign entries in a single LLM call (no taxonomy influence).
            Saved to general_benign.csv.
        standalone_benign_categories: Number of benign constitution categories
            to generate in the standalone benign call.
        output_dir: Where to save CSVs. Defaults to
            ``Data_cache/constitution/``.
        verbose: Print progress.

    Returns:
        DataFrame of all constitution entries.
    """
    from redact.constitution import ConstitutionPipeline, EntryType

    # Resolve taxonomy
    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(taxonomy)

    # Resolve entry types
    resolved_types: list[EntryType] | None = None
    if entry_types is not None:
        resolved_types = [EntryType(t) for t in entry_types]

    # Resolve backend
    if backend is None:
        backend = get_backend(model)

    rate_limiter = get_router().rate_limiter

    pipeline = ConstitutionPipeline(
        backend=backend,
        model=model,
        rate_limiter=rate_limiter,
        output_dir=output_dir,
    )

    result = pipeline.run(
        taxonomy=taxonomy,
        entry_types=resolved_types,
        num_categories=num_categories,
        num_taxonomy_categories=num_taxonomy_categories,
        include_standalone_benign=include_standalone_benign,
        standalone_benign_categories=standalone_benign_categories,
        save=True,
        verbose=verbose,
    )

    return result.to_dataframe()


# ---------------------------------------------------------------------------
# Constitution-to-input generation
# ---------------------------------------------------------------------------


def generate_inputs_from_constitution(
    style: str = "long",
    samples_per_entry: int = 3,
    entry_types: list[str] | None = None,
    source_categories: list[str] | None = None,
    model: str = "venice-uncensored",
    check_model: str | None = None,
    backend: LLMBackend | None = None,
    use_checker: bool = True,
    constitution_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    batch_size: int = 32,
) -> pd.DataFrame:
    """[Deprecated] Generate input prompts from constitution entries.

    Loads constitution CSVs from disk, then delegates to the unified
    ``InputPipeline.run_from_constitution()`` via ``ConstitutionInputPipeline``.

    **Prefer** the composable API:
    ``df = generate_constitution(...); generate_inputs(constitution_df=df, ...)``.
    This wrapper is kept for backward compatibility with existing notebooks
    and runner scripts.

    Args:
        style: Template style ("long", "short", or custom). Controls
            prompt length/detail. See prompts/constitution/input_generation/.
        samples_per_entry: Number of prompts to generate per constitution entry.
        entry_types: Filter to specific entry types (e.g. ["harmful", "benign",
            "dual_use_harmful", "dual_use_benign", "general_benign"]).
        source_categories: Filter to specific taxonomy categories.
        model: Model for generation.
        check_model: Model for quality checking. Defaults to same as model.
        backend: LLM backend. If None, auto-selects from model name.
        use_checker: Whether to quality-check generated prompts.
        constitution_dir: Where to read constitution CSVs. Defaults to
            Data_cache/constitution/.
        output_dir: Where to save generated prompts. Defaults to
            Datasets/constitution_inputs/.
        verbose: Print progress.

    Returns:
        DataFrame of all generated prompts with constitution metadata.
    """
    import warnings
    from redact.constitution.input_generation import ConstitutionInputPipeline

    warnings.warn(
        "generate_inputs_from_constitution() is deprecated; prefer "
        "generate_inputs(constitution_df=generate_constitution(...)).",
        DeprecationWarning,
        stacklevel=2,
    )

    backend, rate_limiter = _get_backend(backend, model)
    check_model = check_model or model

    pipeline = ConstitutionInputPipeline(
        gen_backend=backend,
        gen_model=model,
        check_backend=backend,
        check_model=check_model,
        rate_limiter=rate_limiter,
        constitution_dir=constitution_dir,
        output_dir=output_dir,
    )

    pipeline.run(
        style=style,
        samples_per_entry=samples_per_entry,
        entry_types=entry_types,
        source_categories=source_categories,
        use_checker=use_checker,
        save=True,
        verbose=verbose,
        batch_size=batch_size,
    )

    # Return merged DataFrame from saved CSVs
    actual_dir = pipeline.output_dir
    return merge_content_mod_csvs(actual_dir, accepted_only=True)


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
    model: str = "venice-uncensored",
    check_model: str | None = None,
    backend: LLMBackend | None = None,
    base_url: str = "https://api.venice.ai/api/v1",
    num_categories: int | None = None,
    dataset_dir: str | Path | None = None,
    fresh: bool = False,
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
      This is the existing content-moderation path.
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
        model: Generation model.
        check_model: Checker model; defaults to ``model``.
        backend: LLM backend; auto-resolved if None.
        base_url: API base URL (legacy).
        num_categories: First N categories from taxonomy. (Standalone.)
        dataset_dir: Where to save per-category CSVs.
        fresh: Clear existing category CSVs first. (Standalone.)
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
    # ------------------------------------------------------------------
    # Constitution-seeded mode (delegates to InputPipeline.run_from_constitution)
    # ------------------------------------------------------------------
    if constitution_df is not None:
        backend, rate_limiter = _get_backend(backend, model)
        check_model = check_model or model

        ds_dir = Path(dataset_dir) if dataset_dir else None
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
            "input", f"generation/from_constitution/{style}"
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Generate Inputs (constitution-seeded)")
            print(f"{'='*60}")
            print(
                f"Entries: {len(constitution_df)} | Style: {style} | "
                f"Samples/entry: {samples_per_entry} | Batch: {batch_size}"
            )

        pipeline.run_from_constitution(
            constitution_df=constitution_df,
            prompt_config=prompt_config,
            samples_per_entry=samples_per_entry,
            use_checker=True,
            save=True,
            verbose=verbose,
            batch_size=batch_size,
        )

        return merge_all(ds_dir, accepted_only=True)

    # ------------------------------------------------------------------
    # Standalone (meta-prompt) mode
    # ------------------------------------------------------------------
    # Resolve taxonomy
    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(taxonomy)

    categories = list(iter_categories(taxonomy))
    if num_categories is not None:
        categories = categories[:num_categories]

    # Backend
    backend, rate_limiter = _get_backend(backend, model)
    check_model = check_model or model

    # Pipeline
    ds_dir = Path(dataset_dir) if dataset_dir else None
    pipeline = InputPipeline(
        gen_backend=backend,
        gen_model=model,
        check_backend=backend,
        check_model=check_model,
        rate_limiter=rate_limiter,
        extraction_style="numbered",
        dataset_dir=ds_dir,
    )

    prompt_config = load_prompt("input", "generation/standalone")
    # Max turns as safety cap: 3x what a perfect run would need
    max_turns = ceil(samples_per_category / samples_per_request) * 3

    # Seeds (simple mode)
    seeds_db = None
    if not use_metaprompt:
        seeds_db = load_seeds(seeds_name)

    # Clear existing data for a fresh run
    if fresh:
        actual_dir = Path(ds_dir) if ds_dir else _default_dataset_dir()
        for cat_name, _ in categories:
            csv_path = actual_dir / cat_name / "samples.csv"
            if csv_path.exists():
                csv_path.unlink()
                if verbose:
                    print(f"  Cleared existing {csv_path}")

    if verbose:
        mode = "automated (LLM descriptions + seeds)" if use_metaprompt else "simple (taxonomy + hand-written seeds)"
        print(f"\n{'='*60}")
        print(f"Generate Content Moderation Inputs")
        print(f"{'='*60}")
        print(f"Mode: {mode}")
        print(f"Categories: {len(categories)} | Target: {samples_per_category} samples each ({samples_per_request}/request)")

    all_results = []
    for i, (category_name, category_info) in enumerate(categories):
        if verbose:
            print(f"\n[{i+1}/{len(categories)}] {category_name}")

        if use_metaprompt:
            if verbose:
                print(f"  Generating description...")
            description = generate_category_description(
                backend, model, category_name, rate_limiter
            )
            if verbose:
                print(f"  Description: {len(description)} chars")
                print(f"  Generating seeds...")
            seed_text = generate_seeds(
                backend, model, category_name, description,
                num_seeds=num_seeds, rate_limiter=rate_limiter,
            )
            if verbose:
                print(f"  Seeds: {seed_text.count(chr(10)) + 1} generated")
        else:
            description = category_info.get("description", category_name)
            seed_text = get_seed_prompts(seeds_db, category_name)

        if not seed_text:
            if verbose:
                print(f"  No seeds available, skipping.")
            continue

        seed_kwargs = {
            "Category": category_name,
            "category_description": description,
            "SeedPrompts": seed_text,
        }

        # Run turns until target is reached or max_turns exceeded
        checker = build_quality_checker(category_name)
        result = CategoryResult(category=category_name)
        feedback = ""
        prohibited = get_existing_samples(category_name, ds_dir)

        for turn_idx in range(max_turns):
            if result.total_accepted >= samples_per_category:
                break

            remaining = samples_per_category - result.total_accepted
            if verbose:
                print(f"  Turn {turn_idx + 1}: {result.total_accepted}/{samples_per_category} accepted, requesting {samples_per_request}...")

            turn_result = pipeline.run_turn(
                prompt_config=prompt_config,
                build_check_messages=checker,
                turn_index=turn_idx,
                samples_per_request=samples_per_request,
                feedback=feedback,
                category=category_name,
                save=True,
                prohibited=prohibited,
                **seed_kwargs,
            )
            result.turns.append(turn_result)

            # Update prohibited set
            for sr in turn_result.samples:
                prohibited.add(sr.text)

            # Collect feedback for next turn
            rejections = [
                r.reasoning for r in turn_result.samples
                if not r.accepted and r.reasoning
            ]
            feedback = "\n".join(f"- {r[:200]}" for r in rejections[:3]) if rejections else ""

            if verbose:
                print(f"    -> {turn_result.accepted_count} accepted, {turn_result.rejected_count} rejected ({turn_result.acceptance_rate:.0%})")

            # Stop early if no samples were extracted at all (model issue)
            if turn_result.extracted_count == 0:
                if verbose:
                    print(f"    No samples extracted, stopping category.")
                break

        all_results.append(result)

        if verbose:
            print(f"  -> {result.total_accepted} accepted / {result.total_extracted} extracted ({result.overall_acceptance_rate:.0%})")

    # Summary
    if verbose and all_results:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for result in all_results:
            print(f"  {result.category:30s} {result.total_accepted:3d} / {result.total_extracted:3d} ({result.overall_acceptance_rate:.0%})")
        total_accepted = sum(r.total_accepted for r in all_results)
        total_extracted = sum(r.total_extracted for r in all_results)
        print(f"\n  Total: {total_accepted} accepted / {total_extracted} extracted")

    return merge_all(ds_dir, accepted_only=True)


def generate_outputs(
    inputs: pd.DataFrame | None = None,
    model: str = "venice-uncensored",
    check_outputs: bool = True,
    check_model: str | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
    backend: LLMBackend | None = None,
    base_url: str = "https://api.venice.ai/api/v1",
    dataset_dir: str | Path | None = None,
    output_path: str | Path | None = None,
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

    Args:
        inputs: Input samples DataFrame. If None, loads accepted samples
            from ``dataset_dir`` via :func:`merge_all`.
        model: Generation model identifier.
        check_outputs: Run the entry-type-aware output quality checker.
            Set False to skip checking (accept everything).
        check_model: Checker model; defaults to ``model``.
        batch_size: Inputs per engine pass.
        max_samples: Cap on inputs processed; None = all.
        backend: LLM backend; auto-resolved if None.
        base_url: API base URL (legacy, kept for callers).
        dataset_dir: Where to find input CSVs (when ``inputs`` is None).
        output_path: Where to save the output CSV. Defaults to
            ``Datasets/output_responses.csv``.
        verbose: Print per-batch progress.

    Returns:
        DataFrame with the unified output schema: ``input_id``,
        ``input_prompt``, ``category``, ``subcategory``, ``entry_type``,
        ``output_response``, ``accepted``, ``rejection_reason``, ``model``,
        ``source``.
    """
    ds_dir = Path(dataset_dir) if dataset_dir else None

    if inputs is None:
        inputs = merge_all(ds_dir, accepted_only=True)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    if max_samples is not None:
        inputs = inputs.head(max_samples)

    backend, rate_limiter = _get_backend(backend, model)
    check_model = check_model or model
    if check_outputs:
        check_backend, _ = _get_backend(None, check_model)
    else:
        check_backend = None
    prompt_config = load_prompt("output", "generation")
    gen_caller = BatchCaller.from_model(backend, model, rate_limiter=rate_limiter)

    text_col = "sample" if "sample" in inputs.columns else "prompt"
    inputs = inputs.reset_index(drop=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Generate Output Responses")
        print(f"{'='*60}")
        print(
            f"Samples: {len(inputs)} | Model: {model} | "
            f"Checker: {'enabled (' + check_model + ')' if check_outputs else 'disabled'} | "
            f"Batch: {batch_size}"
        )

    out_path = (
        Path(output_path) if output_path
        else (_default_dataset_dir() / "output_responses.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-(category, entry_type) checker cache so we build each prompt once.
    checker_cache: dict[tuple[str, str], "Callable[[str], list[dict]]"] = {}

    def _checker_for(category: str, entry_type: str):
        key = (category, entry_type)
        if key not in checker_cache:
            checker_cache[key] = build_output_quality_checker(
                category=category, entry_type=entry_type
            )
        return checker_cache[key]

    all_rows: list[dict] = []

    for batch_start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[batch_start : batch_start + batch_size]

        # 1. Build all messages for this chunk
        messages_list = []
        chunk_rows = []
        for _, row in chunk.iterrows():
            input_text = row[text_col]
            category = str(row.get("category", "unknown"))
            entry_type = str(row.get("entry_type", "harmful"))
            subcategory = str(row.get("subcategory", ""))

            messages_list.append(
                build_messages(
                    prompt_config,
                    input_prompt=input_text,
                    Category=category,
                )
            )
            chunk_rows.append({
                "input_id": str(row.get("id", "")),
                "input_prompt": input_text,
                "category": category,
                "subcategory": subcategory,
                "entry_type": entry_type,
                "model": model,
                "source": str(row.get("source", "")),
            })

        # 2. Single batched generation
        responses = gen_caller.batch_generate(messages_list, model)

        # 3. Batched output checking (per-row checker, flat batch)
        if check_outputs and check_backend is not None:
            check_msgs_list: list[list[dict]] = []
            for r, resp in zip(chunk_rows, responses):
                payload = f"INPUT:\n{r['input_prompt']}\n\nOUTPUT:\n{resp}"
                check_msgs_list.append(
                    _checker_for(r["category"], r["entry_type"])(payload)
                )
            check_responses = check_backend.batch_generate(
                check_msgs_list, check_model
            )
            check_results = []
            for cr in check_responses:
                accepted = any(
                    cr.strip().lower().startswith(p)
                    for p in ("yes", "ok", "accept", "pass")
                )
                check_results.append((accepted, "" if accepted else cr))
        else:
            check_results = [(True, "")] * len(chunk_rows)

        # 4. Assemble + incremental append
        for r, resp, (accepted, reasoning) in zip(chunk_rows, responses, check_results):
            r["output_response"] = resp
            r["accepted"] = accepted
            r["rejection_reason"] = reasoning
        all_rows.extend(chunk_rows)

        new_df = pd.DataFrame(chunk_rows)
        if out_path.exists():
            new_df.to_csv(out_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(out_path, index=False)

        if verbose:
            accepted_count = sum(1 for r in chunk_rows if r["accepted"])
            print(
                f"  [{batch_start + len(chunk_rows)}/{len(inputs)}] "
                f"{accepted_count}/{len(chunk_rows)} accepted "
                f"-> appended to {out_path.name}"
            )

    output_df = pd.DataFrame(all_rows)
    if verbose:
        accepted_total = int(output_df["accepted"].sum()) if not output_df.empty else 0
        print(
            f"\n  Saved {len(output_df)} responses "
            f"({accepted_total} accepted) to {out_path}"
        )

    return output_df


def generate_jailbreaks(
    inputs: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    max_complexity: int = 6,
    max_obfuscations: int = 2,
    seed: int = 42,
    pure_only: bool = False,
    entry_types: list[str] | None = None,
    include_hacking: bool = True,
    include_manipulation: bool = True,
    include_obfuscation: bool = True,
    include_requests: bool = True,
    sampling_probs: dict | None = None,
    model: str = "venice-uncensored",
    backend: LLMBackend | None = None,
    base_url: str = "https://api.venice.ai/api/v1",
    auto_generate_benign: bool = True,
    chunk_size: int = 256,
    iterations: int = 1,
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
    2. **Execute** — the manifest is streamed in ``chunk_size`` chunks; each
       chunk is run through the batched engine
       (:func:`redact.jailbreak.batch_apply_combinations`), which pools LLM calls
       per model per round via the router (vLLM native batch / API multi-worker).
       Output rows are appended to the CSV per chunk, so a crash loses at most
       one chunk and a re-run resumes from the output (its
       ``(input_id, iteration)`` pairs are the source of truth).

    Args:
        inputs: Input prompts DataFrame. If None, loads accepted samples from
            ``Datasets/`` via :func:`merge_all`.
        output_path: Output CSV path. Defaults to ``Datasets/jailbreaks.csv``.
            The manifest defaults to ``<output>.manifest.jsonl``.
        max_complexity / max_obfuscations: Combination-sampler limits.
        seed: Global run seed (combined with each sample's id + iteration).
        pure_only: Exclude LLM-dependent techniques (no router calls).
        entry_types: If set, filter inputs to these entry_type values.
        include_hacking / include_manipulation / include_obfuscation /
            include_requests: Layer toggles for the pool.
        sampling_probs: Override per-layer inclusion probabilities (see
            combination_spec.json ``sampling_probs``).
        model: Generation model for LLM-dependent techniques.
        backend: LLM backend (used only for benign pre-load); engine routes via
            the process-wide router.
        auto_generate_benign: Pre-load/generate benign data for FSH/DAP.
        chunk_size: Manifest units per engine batch.
        iterations: Combinations to assign per sample (1 = single-round).
        resume: Reuse an existing manifest and skip already-written output rows.
        manifest_path: Override manifest location.
        verbose: Print progress.

    Returns:
        DataFrame of all jailbreak rows from the output CSV (one per planned
        unit), with the input columns plus ``jailbreak``, ``technique``,
        ``technique_info``, ``complexity``, ``num_techniques``, ``is_noop``,
        ``accepted``, ``reasoning``, ``iteration``, ``combination_spec_version``.
    """
    out = Path(output_path) if output_path else _default_jailbreak_path()
    man_path = Path(manifest_path) if manifest_path else default_manifest_path(out)

    # ------------------------------------------------------------------
    # Load + normalize inputs (ensure a prompt column and a content-hash id)
    # ------------------------------------------------------------------
    if inputs is None:
        inputs = merge_all(accepted_only=True)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    inputs = inputs.copy()
    if "sample" in inputs.columns and "prompt" not in inputs.columns:
        inputs = inputs.rename(columns={"sample": "prompt"})
    if "prompt" not in inputs.columns:
        raise ValueError("inputs must contain a 'prompt' (or 'sample') column.")
    if entry_types:
        inputs = inputs[inputs["entry_type"].isin(entry_types)].reset_index(drop=True)

    if "id" not in inputs.columns:
        inputs["id"] = inputs["prompt"].map(lambda p: compute_sample_id(str(p)))
    else:
        missing = inputs["id"].isna() | (inputs["id"].astype(str).isin(["", "nan"]))
        if missing.any():
            inputs.loc[missing, "id"] = inputs.loc[missing, "prompt"].map(
                lambda p: compute_sample_id(str(p))
            )
    inputs["id"] = inputs["id"].astype(str)

    backend, rate_limiter = _get_backend(backend, model)

    # ------------------------------------------------------------------
    # Build technique pool + assignment settings
    # ------------------------------------------------------------------
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
        pool = [f for f in pool if not getattr(f, "requires_llm", False)]
    registry = build_function_registry(pool)

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
    spec_version = load_spec().get("version", "")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Generate Jailbreaks (plan + batched execute)")
        print(f"{'='*60}")
        print(
            f"Pool: {len(pool)} techniques | Samples: {len(inputs)} | "
            f"Iterations: {iterations} | Model: {model} | Chunk: {chunk_size}"
        )

    # ------------------------------------------------------------------
    # Phase 1 — Plan (lay out every unit before generation)
    # ------------------------------------------------------------------
    if resume and man_path.exists():
        if verbose:
            print(f"  Reusing existing manifest: {man_path}")
    else:
        plan_run(
            inputs, pool, manifest_path=man_path, seed=seed,
            iterations=iterations, sample_kwargs=sample_kwargs, verbose=verbose,
        )

    plan_rows = load_plan(man_path)

    # Fresh (non-resume) run overwrites prior output; resume keeps + skips.
    if not resume and out.exists():
        out.unlink()
    completed = completed_from_output(out) if resume else set()

    # ------------------------------------------------------------------
    # Phase 2 — Execute (stream manifest in chunks through the engine)
    # ------------------------------------------------------------------
    has_manipulation = any(
        "fsh" in getattr(f, "families", []) or "dap" in getattr(f, "families", [])
        for f in pool
    )
    benign_data = None
    if auto_generate_benign and has_manipulation:
        benign_data = _ensure_benign_data(backend, model, rate_limiter, verbose=verbose)

    prompt_map = dict(zip(inputs["id"], inputs["prompt"]))
    meta_map = {r["id"]: r for r in inputs.to_dict("records")}

    pending = [
        r for r in plan_rows
        if str(r["sample_id"]) in prompt_map
        and (str(r["sample_id"]), int(r["iteration"])) not in completed
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    router = get_router()
    total = len(pending)
    written = 0

    for start in range(0, total, chunk_size):
        chunk = pending[start : start + chunk_size]
        samples = [
            {
                "id": str(r["sample_id"]),
                "prompt": prompt_map[str(r["sample_id"])],
                "combination": build_combination(r["combination"], registry),
            }
            for r in chunk
        ]
        results = batch_apply_combinations(
            samples, gen_model=model, benign_data=benign_data, router=router,
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
        written += len(rows)

        if verbose:
            n_acc = sum(1 for x in rows if x.get("accepted"))
            print(f"  [{start + len(rows)}/{total}] {n_acc} accepted -> appended to {out.name}")

    jailbreaks = pd.read_csv(out) if out.exists() else pd.DataFrame()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Jailbreak Summary")
        print(f"{'='*60}")
        print(f"  Planned units: {len(plan_rows)} | Newly written: {written}")
        print(f"  Total rows in {out.name}: {len(jailbreaks)}")
        if not jailbreaks.empty and "accepted" in jailbreaks.columns:
            n_accepted = int(jailbreaks["accepted"].sum())
            n_noop = int(jailbreaks["is_noop"].sum()) if "is_noop" in jailbreaks.columns else 0
            print(f"  Accepted: {n_accepted} | Rejected: {len(jailbreaks) - n_accepted}")
            print(f"  No-ops: {n_noop}")
        print(f"  Manifest: {man_path}")
        print(f"  Saved to: {out}")

    return jailbreaks


def build_dataset(
    dataset_dir: str | Path | None = None,
    jailbreak_path: str | Path | None = None,
    output_path: str | Path | None = None,
    include_inputs: bool = True,
    include_jailbreaks: bool = True,
    include_outputs: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge all generated data into a single dataset.

    Combines content moderation inputs, jailbreak variants, and output
    responses from their saved CSVs on disk.

    Args:
        dataset_dir: Root dataset directory (for inputs). Defaults to
            ``redact/Datasets/``.
        jailbreak_path: Path to the jailbreaks CSV. Defaults to
            ``Datasets/jailbreaks.csv``.
        output_path: Where to save merged CSV. Defaults to
            ``Datasets/complete_dataset.csv``.
        include_inputs: Include content moderation input samples.
        include_jailbreaks: Include jailbreak samples.
        include_outputs: Include output response samples.
        verbose: Print progress.

    Returns:
        Complete merged DataFrame.
    """
    ds_dir = Path(dataset_dir) if dataset_dir else _default_dataset_dir()
    jb_path = Path(jailbreak_path) if jailbreak_path else _default_jailbreak_path()
    out_path = Path(output_path) if output_path else (ds_dir / "complete_dataset.csv")

    parts = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Build Complete Dataset")
        print(f"{'='*60}")

    # Content moderation inputs
    if include_inputs:
        categories = discover_categories(ds_dir)
        if categories:
            cm_df = merge_content_mod_csvs(ds_dir, accepted_only=True)
            cm_df["dataset_type"] = "content_moderation_input"
            parts.append(cm_df)
            if verbose:
                print(f"  Inputs: {len(cm_df)} samples from {len(categories)} categories")
        elif verbose:
            print(f"  Inputs: none found")

    # Jailbreaks
    if include_jailbreaks and jb_path.exists():
        jb_df = pd.read_csv(jb_path)
        if not jb_df.empty:
            jb_df["dataset_type"] = "jailbreak"
            parts.append(jb_df)
            if verbose:
                print(f"  Jailbreaks: {len(jb_df)} samples")
        elif verbose:
            print(f"  Jailbreaks: none found")
    elif verbose and include_jailbreaks:
        print(f"  Jailbreaks: file not found ({jb_path})")

    # Output responses
    output_csv = ds_dir / "output_responses.csv"
    if include_outputs and output_csv.exists():
        out_df = pd.read_csv(output_csv)
        out_df["dataset_type"] = "content_moderation_output"
        parts.append(out_df)
        if verbose:
            print(f"  Outputs: {len(out_df)} samples")
    elif verbose and include_outputs:
        print(f"  Outputs: none found")

    if not parts:
        if verbose:
            print(f"\n  No data found. Run generate_inputs/jailbreaks/outputs first.")
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    if verbose:
        print(f"\n  Total: {len(merged)} samples")
        if "dataset_type" in merged.columns:
            for dtype, count in merged["dataset_type"].value_counts().items():
                print(f"    {dtype}: {count}")
        print(f"  Saved to: {out_path}")

    return merged
