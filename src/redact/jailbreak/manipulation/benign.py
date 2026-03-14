"""Benign sample generation and loading for FSH/DAP manipulation.

Generated samples are saved in Data_cache/benign/.
Loading is lazy (explicit function call, not module-level import).

Ported from reference manipulation.py lines 40-289.
"""

import pandas as pd
from pathlib import Path

from Redact_Library.LLMs.base import LLMBackend
from Redact_Library.LLMs.calls import generate_sample
from Redact_Library.LLMs.wrappers import RateLimiter
from Redact_Library.LLMs.extraction import extract_structured_qa

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent  # Redact_Library/
from Redact_Library.LLMs.prompts import load_prompt, build_messages


# ---------------------------------------------------------------------------
# Category definitions (52 categories from reference)
# ---------------------------------------------------------------------------

BENIGN_CATEGORIES: list[tuple[str, str]] = [
    # Home & Daily Life
    ("Home & Daily Life", "Cooking & Baking"),
    ("Home & Daily Life", "Gardening & Plant Care"),
    ("Home & Daily Life", "Pet Care"),
    ("Home & Daily Life", "Home Organization"),
    ("Home & Daily Life", "Cleaning"),
    ("Home & Daily Life", "Home Maintenance & DIY Repairs"),
    ("Home & Daily Life", "Car Maintenance"),

    # Health & Wellness
    ("Health & Wellness", "Mindfulness & Meditation"),
    ("Health & Wellness", "Fitness & Workouts"),
    ("Health & Wellness", "Nutrition & Diet"),
    ("Health & Wellness", "First Aid"),
    ("Health & Wellness", "Sleep & Rest"),

    # Personal Development
    ("Personal Development", "Task Planning & Organization"),
    ("Personal Development", "Language Learning"),
    ("Personal Development", "Public Speaking"),
    ("Personal Development", "Time Management"),
    ("Personal Development", "Study Techniques & Learning Strategies"),

    # Work & Professional
    ("Work & Professional", "Email Writing"),
    ("Work & Professional", "Presentation Creation"),
    ("Work & Professional", "Programming & Technical Tasks"),
    ("Work & Professional", "Career Planning & Development"),
    ("Work & Professional", "Resume/CV Writing"),
    ("Work & Professional", "Interview Preparation"),
    ("Work & Professional", "Strategy & Business Planning"),
    ("Work & Professional", "Project Management"),

    # Finance & Shopping
    ("Finance & Shopping", "Personal Finance & Budgeting"),
    ("Finance & Shopping", "Product Comparisons"),
    ("Finance & Shopping", "Shopping Recommendations"),
    ("Finance & Shopping", "Financial Planning"),

    # Travel & Leisure
    ("Travel & Leisure", "Destination Selection"),
    ("Travel & Leisure", "Route Planning & Itineraries"),
    ("Travel & Leisure", "Travel Tips & Preparation"),
    ("Travel & Leisure", "Event Planning (parties, gatherings)"),

    # Entertainment & Hobbies
    ("Entertainment & Hobbies", "Book/Movie/Game Recommendations"),
    ("Entertainment & Hobbies", "Sports & Recreation Activities"),
    ("Entertainment & Hobbies", "Crafts & DIY Art"),
    ("Entertainment & Hobbies", "Photography"),
    ("Entertainment & Hobbies", "Music Learning & Appreciation"),

    # Creative & Writing
    ("Creative & Writing", "Creative Writing"),
    ("Creative & Writing", "Content Creation"),
    ("Creative & Writing", "Storytelling"),
    ("Creative & Writing", "Poetry & Literary Analysis"),

    # Education & Knowledge
    ("Education & Knowledge", "Homework Help (Math, Science, History)"),
    ("Education & Knowledge", "Research Assistance"),
    ("Education & Knowledge", "General Knowledge Questions"),
    ("Education & Knowledge", "How-to Explanations"),
    ("Education & Knowledge", "Historical & Factual Information"),
    ("Education & Knowledge", "Science Concepts"),

    # Social & Relationships
    ("Social & Relationships", "Social Skills (small talk, networking)"),
    ("Social & Relationships", "Communication Skills"),
    ("Social & Relationships", "Conversation Starters"),
    ("Social & Relationships", "Etiquette & Social Norms"),
    ("Social & Relationships", "Gift Ideas & Selection"),
    ("Social & Relationships", "Relationship Advice (general)"),

    # Technology & Tips
    ("Technology & Tips", "Tech Tips & Troubleshooting"),
    ("Technology & Tips", "Software Recommendations"),
    ("Technology & Tips", "Digital Organization"),
    ("Technology & Tips", "Productivity Tools"),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_DIR = _PACKAGE_DIR / "Prompts"


def generate_short_benign(
    category: tuple[str, str],
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    num_samples: int = 8,
    prompt_dir: str | Path = _DEFAULT_PROMPT_DIR,
) -> str:
    """Generate short (1-sentence answer) benign prompts for a category.

    Returns raw LLM text in structured **Prompt N:** format.
    """
    config = load_prompt("Jailbreak", "benign_generation", prompt_dir)
    messages = build_messages(
        config,
        sub_category=category[1],
        main_category=category[0],
        num_samples=str(num_samples),
        answer_style="Keep the answers concise and to the point. Make the answer be only 1 sentence long.",
    )
    return generate_sample(backend, model, messages, rate_limiter)


def generate_long_benign(
    category: tuple[str, str],
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    num_samples: int = 8,
    prompt_dir: str | Path = _DEFAULT_PROMPT_DIR,
) -> str:
    """Generate long (multi-sentence answer) benign prompts for a category.

    Returns raw LLM text in structured **Prompt N:** format.
    """
    config = load_prompt("Jailbreak", "benign_generation", prompt_dir)
    messages = build_messages(
        config,
        sub_category=category[1],
        main_category=category[0],
        num_samples=str(num_samples),
        answer_style="Make the answers detailed and multi-sentence.",
    )
    return generate_sample(backend, model, messages, rate_limiter)


def process_category(
    category: tuple[str, str],
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    num_samples: int = 8,
    prompt_dir: str | Path = _DEFAULT_PROMPT_DIR,
) -> list[dict]:
    """Generate both short and long benign samples for one category.

    Returns list of row dicts with keys:
    main_category, sub_category, prompt, answer, answer_type.
    """
    main_cat, sub_cat = category

    raw_short = generate_short_benign(
        category, backend, model, rate_limiter, num_samples, prompt_dir
    )
    pairs_short = extract_structured_qa(raw_short)

    raw_long = generate_long_benign(
        category, backend, model, rate_limiter, num_samples, prompt_dir
    )
    pairs_long = extract_structured_qa(raw_long)

    rows: list[dict] = []
    for pair in pairs_short:
        rows.append({
            "main_category": main_cat,
            "sub_category": sub_cat,
            "prompt": pair["prompt"],
            "answer": pair["answer"],
            "answer_type": "short",
        })
    for pair in pairs_long:
        rows.append({
            "main_category": main_cat,
            "sub_category": sub_cat,
            "prompt": pair["prompt"],
            "answer": pair["answer"],
            "answer_type": "long",
        })
    return rows


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_DEFAULT_BENIGN_PATH = _PACKAGE_DIR / "Data_cache" / "benign" / "benign_samples.csv"


def load_benign_data(
    path: str | Path = _DEFAULT_BENIGN_PATH,
) -> dict:
    """Load benign samples from CSV. Returns a dict with indexed views.

    Returns:
        {
            "short": list[dict],
            "long": list[dict],
            "by_subcat_short": dict[str, list[dict]],
            "by_subcat_long": dict[str, list[dict]],
            "all_subcategories": list[str],
        }

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Benign samples CSV not found at {path}. "
            "Run benign generation first."
        )
    df = pd.read_csv(path)

    short_df = df[df["answer_type"] == "short"]
    long_df = df[df["answer_type"] == "long"]

    short = short_df.to_dict("records")
    long_ = long_df.to_dict("records")

    by_subcat_short = (
        short_df.groupby("sub_category")
        .apply(lambda g: g.to_dict("records"))
        .to_dict()
    )
    by_subcat_long = (
        long_df.groupby("sub_category")
        .apply(lambda g: g.to_dict("records"))
        .to_dict()
    )

    return {
        "short": short,
        "long": long_,
        "by_subcat_short": by_subcat_short,
        "by_subcat_long": by_subcat_long,
        "all_subcategories": list(by_subcat_short.keys()),
    }
