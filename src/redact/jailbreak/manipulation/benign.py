"""Benign Q&A pairs for the FSH and DAP techniques, generated once and cached."""

import logging
from pathlib import Path

import pandas as pd

from redact import paths
from redact.llms.client import ModelClient
from redact.llms.prompting import build_messages, extract_structured_qa, load_prompt
from redact.llms.router import generate_sample

logger = logging.getLogger(__name__)


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


_ANSWER_STYLES = {
    "short": 'Keep the answers concise and to the point. Make the answer be only 1 sentence long.',
    "long": 'Make the answers detailed and multi-sentence.',
}


def generate_benign(
    category: tuple[str, str],
    client: ModelClient,
    num_samples: int = 8,
    answer_type: str = "short",
    prompt_dir: str | Path | None = None,
) -> str:
    """Generate benign prompts with ``answer_type`` ("short" or "long") answers.

    Returns raw LLM text in structured **Prompt N:** format.
    """
    config = load_prompt("jailbreak", "benign_generation", prompt_dir)
    messages = build_messages(
        config,
        sub_category=category[1],
        main_category=category[0],
        num_samples=str(num_samples),
        answer_style=_ANSWER_STYLES[answer_type],
    )
    return generate_sample(client, messages)


def process_category(
    category: tuple[str, str],
    client: ModelClient,
    num_samples: int = 8,
    prompt_dir: str | Path | None = None,
) -> list[dict]:
    """Generate both short and long benign samples for one category.

    Returns list of row dicts with keys:
    main_category, sub_category, prompt, answer, answer_type.
    """
    main_cat, sub_cat = category

    raw_short = generate_benign(category, client, num_samples, "short", prompt_dir)
    # A reply with any other number of pairs is rejected whole.
    pairs_short = extract_structured_qa(raw_short, expected_count=num_samples)

    raw_long = generate_benign(category, client, num_samples, "long", prompt_dir)
    pairs_long = extract_structured_qa(raw_long, expected_count=num_samples)

    rows: list[dict] = []
    for pair in pairs_short:
        rows.append({
            "main_category": main_cat,
            "sub_category": sub_cat,
            "prompt": pair["question"],
            "answer": pair["answer"],
            "answer_type": "short",
        })
    for pair in pairs_long:
        rows.append({
            "main_category": main_cat,
            "sub_category": sub_cat,
            "prompt": pair["question"],
            "answer": pair["answer"],
            "answer_type": "long",
        })
    return rows


def get_or_generate_benign_data(
    client: "ModelClient | None" = None,
    cache_path=None,
    verbose: bool = True,
) -> dict:
    """Load the benign data from ``cache_path``, or generate it with ``client``.

    Generated data is saved to ``cache_path``, which defaults to
    ``paths.benign_csv()``. Returns what :func:`load_benign_data` returns.

    Raises:
        ValueError: If generation is needed but no client was provided.
    """
    path = Path(cache_path) if cache_path else paths.benign_csv()

    if path.exists():
        if verbose:
            logger.info("Loading cached benign data from %s", path)
        return load_benign_data(path)

    if client is None:
        raise ValueError(
            "Benign data not found and no client provided for generation. "
            f"Either pre-generate benign data at {path} or pass a ModelClient."
        )

    if verbose:
        logger.info("Generating benign data (%d categories)...", len(BENIGN_CATEGORIES))

    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for i, category in enumerate(BENIGN_CATEGORIES):
        rows = process_category(category, client)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(path, index=False)
        if verbose:
            logger.info("[%d/%d] %s / %s → saved (%d rows total)",
                        i + 1, len(BENIGN_CATEGORIES), category[0], category[1], len(all_rows))

    df = pd.DataFrame(all_rows)
    if verbose:
        logger.info("Done. %d benign samples saved to %s", len(df), path)

    return load_benign_data(path)


def load_benign_data(
    path: str | Path | None = None,
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
    path = Path(path) if path else paths.benign_csv()
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

    by_subcat_short = {
        subcat: group.to_dict("records")
        for subcat, group in short_df.groupby("sub_category")
    }
    by_subcat_long = {
        subcat: group.to_dict("records")
        for subcat, group in long_df.groupby("sub_category")
    }

    return {
        "short": short,
        "long": long_,
        "by_subcat_short": by_subcat_short,
        "by_subcat_long": by_subcat_long,
        "all_subcategories": list(by_subcat_short.keys()),
    }
