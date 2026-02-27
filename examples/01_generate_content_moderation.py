"""Example: Generate content moderation input samples.

Uses the high-level generate_inputs() pipeline function.

Two modes (toggle USE_METAPROMPT):

  Automated mode (USE_METAPROMPT=True):
    Step 1. category name -> LLM -> rich category description
    Step 2. category name + description -> LLM -> seed prompts
    Step 3. description + seeds -> LLM -> samples + quality check

  Simple mode (USE_METAPROMPT=False):
    - Category description: from taxonomy JSON (short, one-liner)
    - Seeds: from content_moderation_seeds.json (hand-written)
    - Generation: description + seeds -> LLM -> samples

The pipeline keeps generating until samples_per_category accepted
samples are reached. Each iteration requests samples_per_request
samples from the LLM.

Usage:
    export VENICE_API_KEY=your_api_key_here
    python examples/01_generate_content_moderation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Redact_Library import set_seed, generate_inputs


# -- Configuration -----------------------------------------------------------

SAMPLES_PER_CATEGORY = 15   # Target: keep generating until this many are accepted
SAMPLES_PER_REQUEST = 5     # Samples requested per LLM call
NUM_CATEGORIES = 2          # Categories to process (None = all)
NUM_SEEDS = 8               # Seed prompts to generate (metaprompt mode)

USE_METAPROMPT = True        # True = LLM generates descriptions + seeds
FRESH_RUN = True             # Clear existing category CSVs before generating

GEN_MODEL = "venice-uncensored"
BASE_URL = "https://api.venice.ai/api/v1"


# -- Main --------------------------------------------------------------------

def main():
    set_seed(42)

    inputs = generate_inputs(
        samples_per_category=SAMPLES_PER_CATEGORY,
        samples_per_request=SAMPLES_PER_REQUEST,
        num_categories=NUM_CATEGORIES,
        num_seeds=NUM_SEEDS,
        use_metaprompt=USE_METAPROMPT,
        model=GEN_MODEL,
        base_url=BASE_URL,
        fresh=FRESH_RUN,
    )

    print(f"\nDone. {len(inputs)} total accepted samples.")


if __name__ == "__main__":
    main()
