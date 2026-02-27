"""Example: Generate content moderation output (model responses).

Takes accepted input samples from the content moderation dataset and
generates model responses for each. Optionally paraphrases through
the fingerprint removal pipeline.

Pipeline:
1. Load accepted input samples from Datasets/{category}/samples.csv
2. For each sample, generate a model response
3. Optionally paraphrase the response (fingerprint removal)
4. Save input+output pairs to a separate output CSV

This example processes a SMALL number of samples for demonstration.

Usage:
    export VENICE_API_KEY=your_api_key_here
    python examples/02_generate_output_responses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from Redact_Library import Config, set_seed
from Redact_Library.LLMs import (
    APIBackend,
    RateLimiter,
    load_prompt,
    build_messages,
    generate_sample,
)
from Redact_Library.Content_Moderation.paraphrase import paraphrase_sample
from Redact_Library.Dataset_Functions import (
    merge_category_csvs,
    discover_categories,
)


# ── Configuration ─────────────────────────────────────────────────────────

BASE_URL = "https://api.venice.ai/api/v1"
GEN_MODEL = "venice-uncensored"

MAX_SAMPLES = 5        # Process only this many samples (for demo)
USE_PARAPHRASE = False  # Set True when paraphrase model is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Redact_Library" / "Datasets"
OUTPUT_PATH = DATASET_DIR / "output_responses.csv"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    set_seed(42)

    # 1. Load accepted input samples
    categories = discover_categories(DATASET_DIR)
    if not categories:
        print("No category datasets found. Run 01_generate_content_moderation.py first.")
        sys.exit(1)

    df = merge_category_csvs(categories, dataset_dir=DATASET_DIR, accepted_only=True)
    print(f"Loaded {len(df)} accepted samples from {len(categories)} categories")

    if df.empty:
        print("No accepted samples found. Run 01_generate_content_moderation.py first.")
        sys.exit(1)

    # Limit for demo
    df = df.head(MAX_SAMPLES)
    print(f"Processing {len(df)} samples (limited for demo)\n")

    # 2. Validate and set up LLM backend
    Config.validate([Config.VENICE_API_KEY])

    backend = APIBackend(
        api_key=Config.get(Config.VENICE_API_KEY),
        base_url=BASE_URL,
    )
    rate_limiter = RateLimiter()

    # 3. Load output generation prompt
    prompt_config = load_prompt("Content_Moderation", "output_generation")

    # 4. Generate responses
    results = []
    text_col = "sample" if "sample" in df.columns else "prompt"

    for idx, row in df.iterrows():
        input_text = row[text_col]
        category = row.get("category", "unknown")

        print(f"[{idx + 1}/{len(df)}] Category: {category}")
        print(f"  Input: {input_text[:80]}...")

        # Build messages from prompt template
        messages = build_messages(
            prompt_config,
            input_prompt=input_text,
            Category=category,
        )

        # Generate response
        response = generate_sample(
            backend, GEN_MODEL, messages, rate_limiter
        )

        # Optionally paraphrase (fingerprint removal)
        if USE_PARAPHRASE:
            response = paraphrase_sample(
                backend, GEN_MODEL, response, rate_limiter
            )

        print(f"  Output: {response[:80]}...\n")

        results.append({
            "input_id": row.get("id", ""),
            "input_prompt": input_text,
            "category": category,
            "output_response": response,
            "model": GEN_MODEL,
            "paraphrased": USE_PARAPHRASE,
        })

    # 5. Save results
    output_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(output_df)} input-output pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
