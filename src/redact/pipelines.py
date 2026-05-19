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
import random as _random
from math import ceil
from pathlib import Path

import pandas as pd

from redact import Config, get_output_dir
from redact.llms import (
    get_backend,
    RateLimiter,
    load_prompt,
    build_messages,
    generate_sample,
)
from redact.llms.base import LLMBackend
from redact.content_moderation import (
    InputPipeline,
    CategoryResult,
    generate_category_description,
    generate_seeds,
)
from redact.content_moderation.checker import build_quality_checker
from redact.dataset.io import get_existing_samples
from redact.content_moderation.paraphrase import paraphrase_sample
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
    sample_combination,
    apply_combination,
    is_noop,
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
    """Resolve backend — auto-select from model name if None."""
    if backend is None:
        backend = get_backend(model)
    return backend, RateLimiter()


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

    rate_limiter = RateLimiter()

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
    """Generate input prompts from constitution entries.

    # CONSTITUTION-TO-INPUT: High-level pipeline function.
    # Reads constitution CSVs and expands each entry's sample_description
    # into full realistic prompts using the specified template style.
    #
    # Future work: chain with generate_outputs() and generate_jailbreaks().

    For each constitution entry, uses the content moderation InputPipeline
    to generate full-length prompts from the short sample_description.
    Output is saved in standard content moderation format with constitution
    metadata preserved as extra columns.

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
    from redact.constitution.input_generation import ConstitutionInputPipeline

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
) -> pd.DataFrame:
    """Generate content moderation input samples.

    Wraps the full input pipeline: taxonomy loading, description/seed
    generation, sample generation with quality checking, and per-category
    CSV saving.

    Args:
        taxonomy: Taxonomy name (string) or pre-loaded taxonomy dict.
        samples_per_category: Target number of accepted samples per category.
        use_metaprompt: If True, use LLM to generate descriptions and seeds.
            If False, use taxonomy descriptions and hand-written seeds.
        seeds_name: Name of seeds JSON (only used when use_metaprompt=False).
        samples_per_request: Samples requested per LLM call.
        num_seeds: Number of seed prompts to generate (metaprompt mode).
        model: Model for generation.
        check_model: Model for quality checking. Defaults to same as model.
        backend: LLM backend. If None, creates from environment.
        base_url: API base URL (used when creating backend).
        num_categories: Limit to first N categories. None = all.
        dataset_dir: Where to save per-category CSVs. Defaults to
            ``redact/Datasets/``.
        fresh: If True, clear existing category CSVs before generating.
            Prevents old samples from inflating the prohibited set.
        verbose: Print progress.

    Returns:
        Merged DataFrame of all accepted samples.
    """
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

    prompt_config = load_prompt("content_moderation", "generation")
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
    use_paraphrase: bool = False,
    max_samples: int | None = None,
    backend: LLMBackend | None = None,
    base_url: str = "https://api.venice.ai/api/v1",
    dataset_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate model responses for content moderation input samples.

    Args:
        inputs: Input samples DataFrame. If None, loads from dataset_dir.
        model: Model for response generation.
        use_paraphrase: Apply fingerprint removal paraphrase.
        max_samples: Limit to first N samples. None = all.
        backend: LLM backend. If None, creates from environment.
        base_url: API base URL.
        dataset_dir: Where to find input CSVs (if inputs is None).
        output_path: Where to save output CSV. Defaults to
            ``Datasets/output_responses.csv``.
        verbose: Print progress.

    Returns:
        DataFrame with input-output pairs.
    """
    ds_dir = Path(dataset_dir) if dataset_dir else None

    if inputs is None:
        inputs = merge_all(ds_dir, accepted_only=True)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    if max_samples is not None:
        inputs = inputs.head(max_samples)

    backend, rate_limiter = _get_backend(backend, model)
    prompt_config = load_prompt("content_moderation", "output_generation")

    text_col = "sample" if "sample" in inputs.columns else "prompt"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Generate Output Responses")
        print(f"{'='*60}")
        print(f"Samples: {len(inputs)} | Model: {model} | Paraphrase: {use_paraphrase}")

    results = []
    for idx, (_, row) in enumerate(inputs.iterrows()):
        input_text = row[text_col]
        category = row.get("category", "unknown")

        if verbose:
            print(f"  [{idx+1}/{len(inputs)}] {category}: {input_text[:60]}...")

        messages = build_messages(
            prompt_config,
            input_prompt=input_text,
            Category=category,
        )
        response = generate_sample(backend, model, messages, rate_limiter)

        if use_paraphrase:
            response = paraphrase_sample(backend, model, response, rate_limiter)

        results.append({
            "input_id": row.get("id", ""),
            "input_prompt": input_text,
            "category": category,
            "output_response": response,
            "model": model,
            "paraphrased": use_paraphrase,
        })

    output_df = pd.DataFrame(results)

    # Save
    out_path = Path(output_path) if output_path else (_default_dataset_dir() / "output_responses.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    if verbose:
        print(f"\n  Saved {len(output_df)} output responses to {out_path}")

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
    model: str = "venice-uncensored",
    backend: LLMBackend | None = None,
    base_url: str = "https://api.venice.ai/api/v1",
    auto_generate_benign: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate jailbreak variants of input prompts using technique combinations.

    Each prompt receives a randomly sampled valid combination of techniques
    applied in series (e.g. framing → rot13 → indirect embedding). The sampler
    respects compatibility rules from combination_spec.json. Single-technique
    attacks are a natural subset when the sampler picks only one technique.

    Args:
        inputs: Input prompts DataFrame. If None, loads from Datasets/.
        output_path: Where to save the output CSV. Defaults to
            ``Datasets/jailbreaks.csv``.
        max_complexity: Max total complexity score across all selected techniques.
        max_obfuscations: Max number of obfuscation families per combination.
            Set to 1 with include_hacking/manipulation/requests=False for
            single-technique mode.
        seed: Random seed for reproducible combination sampling.
        pure_only: If True, exclude techniques that require LLM calls.
        entry_types: If set, filter inputs to these entry_type values.
        include_hacking: Allow hacking-layer techniques in combinations.
        include_manipulation: Allow manipulation-layer techniques (FSH/DAP).
        include_obfuscation: Allow obfuscation-layer techniques.
        include_requests: Allow request-layer techniques.
        model: Model for LLM-dependent techniques.
        backend: LLM backend. If None, creates from environment.
        base_url: API base URL.
        auto_generate_benign: Pre-load or generate benign data for FSH/DAP.
        verbose: Print progress.

    Returns:
        DataFrame of jailbreak samples with one row per input prompt.
    """
    out = Path(output_path) if output_path else _default_jailbreak_path()

    if inputs is None:
        inputs = merge_all(accepted_only=True)
        if inputs.empty:
            raise ValueError("No input samples found. Run generate_inputs() first.")

    # Normalize: content mod uses 'sample', jailbreak expects 'prompt'
    if "sample" in inputs.columns and "prompt" not in inputs.columns:
        inputs = inputs.rename(columns={"sample": "prompt"})

    if entry_types:
        inputs = inputs[inputs["entry_type"].isin(entry_types)].reset_index(drop=True)

    backend, rate_limiter = _get_backend(backend, model)

    # Build pool — respect include_* flags
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

    if verbose:
        print(f"\n{'='*60}")
        print(f"Generate Jailbreaks")
        print(f"{'='*60}")
        print(f"Pool: {len(pool)} techniques | Input: {len(inputs)} prompts | Model: {model}")

    # Pre-load benign data upfront if any manipulation technique is in pool
    has_manipulation = any(
        "fsh" in getattr(f, "families", []) or "dap" in getattr(f, "families", [])
        for f in pool
    )
    benign_data = None
    if auto_generate_benign and has_manipulation:
        benign_data = _ensure_benign_data(backend, model, rate_limiter, verbose=verbose)

    # Sample one valid combination per prompt and apply in series
    rng = _random.Random(seed)
    all_results = []
    for i, (_, row) in enumerate(inputs.iterrows()):
        if verbose and (i % 10 == 0 or i == len(inputs) - 1):
            print(f"  [{i + 1}/{len(inputs)}] sampling combination...")

        fn = sample_combination(
            rng, pool,
            max_complexity=max_complexity,
            max_obfuscations=max_obfuscations,
            include_hacking=include_hacking,
            include_manipulation=include_manipulation,
            include_obfuscation=include_obfuscation,
            include_requests=include_requests,
        )
        try:
            result, info, accepted, reasoning = apply_combination(
                fn, row["prompt"],
                backend=backend, model=model, rate_limiter=rate_limiter,
                benign_data=benign_data, auto_benign=False,
            )
        except Exception as e:
            if verbose:
                print(f"    Failed (combination={fn.__name__}): {e}")
            result = row["prompt"]
            info = f"ERROR: {e}"
            accepted = False
            reasoning = str(e)

        techniques = getattr(fn, "techniques", [] if fn.__name__ == "identity" else [fn])
        all_results.append({
            **row.to_dict(),
            "jailbreak": result,
            "technique": fn.__name__,
            "technique_info": info,
            "complexity": sum(getattr(t, "complexity", 0) for t in techniques),
            "num_techniques": len(techniques),
            "is_noop": is_noop(row["prompt"], result),
            "accepted": accepted,
            "reasoning": reasoning,
        })

    jailbreaks = pd.DataFrame(all_results)
    out.parent.mkdir(parents=True, exist_ok=True)
    jailbreaks.to_csv(out, index=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Jailbreak Summary")
        print(f"{'='*60}")
        print(f"  Total: {len(jailbreaks)} samples")
        n_noop = int(jailbreaks["is_noop"].sum()) if not jailbreaks.empty else 0
        n_accepted = int(jailbreaks["accepted"].sum()) if not jailbreaks.empty else 0
        n_rejected = len(jailbreaks) - n_accepted if not jailbreaks.empty else 0
        print(f"  Accepted: {n_accepted} | Rejected: {n_rejected}")
        print(f"  Transformed: {len(jailbreaks) - n_noop} | No-ops: {n_noop}")
        if not jailbreaks.empty:
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
