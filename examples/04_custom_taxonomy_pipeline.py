"""Example: Custom taxonomy and technique integration.

Shows how to extend the library with:
1. Your own taxonomy JSON (custom harm categories)
2. Custom jailbreak technique functions
3. Integration with the dataset splitting and merging pipeline

This is the extensibility reference -- demonstrates the patterns
you need to plug your own logic into the REDACT pipeline.

Usage:
    python examples/04_custom_taxonomy_pipeline.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from Redact_Library import set_seed
from Redact_Library.Dataset_Functions import (
    load_taxonomy,
    get_categories,
    get_category_descriptions,
    iter_categories,
    filter_by_group,
    normalize_categories,
    deterministic_balanced_assign,
    split_by_functions,
)
from Redact_Library.Jailbreak import combine_techniques
from Redact_Library.Jailbreak.obfuscation.encoding import to_base64, to_rot13


# ======================================================================
# PART 1: Create and load a custom taxonomy
# ======================================================================

def demo_custom_taxonomy():
    """Create a custom taxonomy JSON and load it with the library."""

    print("=" * 60)
    print("PART 1: Custom Taxonomy")
    print("=" * 60)

    # Define your own taxonomy -- any JSON matching the schema works
    custom_taxonomy = {
        "name": "financial_harm",
        "description": "Categories for financial harm content moderation",
        "categories": {
            "Investment Fraud": {
                "description": "Fake investment schemes, pump-and-dump, Ponzi schemes",
                "subcategories": ["Ponzi Scheme", "Pump and Dump", "Fake ICO"],
            },
            "Identity Theft": {
                "description": "Techniques for stealing personal financial identity",
                "subcategories": ["Credit Card Fraud", "Account Takeover"],
            },
            "Money Laundering": {
                "description": "Methods to disguise illicit financial transactions",
                "subcategories": ["Layering", "Structuring", "Shell Companies"],
            },
            "Tax Evasion": {
                "description": "Illegal methods to avoid tax obligations",
                "subcategories": ["Offshore Hiding", "False Deductions"],
            },
        },
        "aliases": {
            "Scam": "Investment Fraud",
            "Phishing": "Identity Theft",
        },
        "groups": {
            "direct_theft": ["Identity Theft", "Investment Fraud"],
            "financial_crime": ["Money Laundering", "Tax Evasion"],
        },
    }

    # Save to a temp directory (in production, save to Dataset_Configs/taxonomy/)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "financial_harm.json"
        with open(path, "w") as f:
            json.dump(custom_taxonomy, f, indent=2)

        # Load it with the library
        taxonomy = load_taxonomy("financial_harm", config_dir=tmpdir)

    # Use taxonomy functions
    categories = get_categories(taxonomy)
    print(f"\n  Categories: {categories}")

    descriptions = get_category_descriptions(taxonomy)
    for name, desc in descriptions.items():
        print(f"  - {name}: {desc}")

    # Iterate for generation -- this is how you'd drive a generation loop
    print("\n  Generation loop:")
    for cat_name, cat_info in iter_categories(taxonomy):
        subcats = cat_info.get("subcategories", [])
        print(f"  - {cat_name}: {len(subcats)} subcategories")

    # Normalize aliases in a DataFrame
    df = pd.DataFrame({
        "text": ["How to run a scam", "Phishing email template"],
        "category": ["Scam", "Phishing"],
    })
    normalized = normalize_categories(df, taxonomy, column="category")
    print(f"\n  Alias normalization:")
    print(f"    Before: {df['category'].tolist()}")
    print(f"    After : {normalized['category'].tolist()}")

    # Filter by group
    full_df = pd.DataFrame({
        "text": ["text1", "text2", "text3", "text4"],
        "category": [
            "Investment Fraud", "Identity Theft",
            "Money Laundering", "Tax Evasion",
        ],
    })
    theft_only = filter_by_group(full_df, taxonomy, "direct_theft")
    print(f"\n  Group filter 'direct_theft': {theft_only['category'].tolist()}")

    return taxonomy


# ======================================================================
# PART 2: Custom technique functions
# ======================================================================

def demo_custom_techniques():
    """Register and use custom jailbreak technique functions."""

    print(f"\n{'='*60}")
    print("PART 2: Custom Technique Functions")
    print("=" * 60)

    # A custom technique follows the signature:
    #   (prompt: str, **kwargs) -> tuple[str, str]
    # Returns (transformed_text, additional_info)

    def to_pig_latin(prompt: str, **kwargs) -> tuple[str, str]:
        """Custom technique: convert to pig latin."""
        words = prompt.split()
        pig_words = []
        for word in words:
            if word[0].lower() in "aeiou":
                pig_words.append(word + "yay")
            else:
                pig_words.append(word[1:] + word[0] + "ay")
        return " ".join(pig_words), "pig_latin"

    def to_reverse_words(prompt: str, **kwargs) -> tuple[str, str]:
        """Custom technique: reverse each word."""
        words = prompt.split()
        reversed_words = [w[::-1] for w in words]
        return " ".join(reversed_words), "reverse_words"

    # Use standalone
    test_prompt = "How to pick a lock"
    result, info = to_pig_latin(test_prompt)
    print(f"\n  Pig Latin: '{test_prompt}' -> '{result}'")

    result, info = to_reverse_words(test_prompt)
    print(f"  Reverse:   '{test_prompt}' -> '{result}'")

    # Chain with built-in techniques using combine_techniques()
    combo = combine_techniques(to_pig_latin, to_base64)
    result, info = combo(test_prompt)
    print(f"\n  Combined ({combo.__name__}):")
    print(f"    '{test_prompt}' -> '{result[:50]}...'")
    print(f"    Info: {info}")

    return [to_pig_latin, to_reverse_words]


# ======================================================================
# PART 3: Integrate custom functions with dataset splitting
# ======================================================================

def demo_custom_splitting(custom_techniques):
    """Show how custom functions integrate with the splitting pipeline."""

    print(f"\n{'='*60}")
    print("PART 3: Custom Functions + Dataset Splitting")
    print("=" * 60)

    # Create a demo dataset
    df = pd.DataFrame({
        "id": [f"s{i}" for i in range(12)],
        "prompt": [f"Sample prompt {i}" for i in range(12)],
        "category": ["CatA"] * 4 + ["CatB"] * 4 + ["CatC"] * 4,
        "origin": ["handcrafted", "generated"] * 6,
    })

    # Build a technique registry in the same format as the library
    # Key: type name -> Value: getter function returning list of techniques
    all_techniques = custom_techniques + [to_base64, to_rot13]

    def get_custom_functions():
        return custom_techniques

    def get_builtin_functions():
        return [to_base64, to_rot13]

    custom_registry = {
        "custom": get_custom_functions,
        "encoding": get_builtin_functions,
    }

    # Split dataset by all functions
    splits = split_by_functions(df, custom_registry)

    print(f"\n  Dataset: {len(df)} rows, split across {len(splits)} techniques:")
    for func_name, split_df in splits.items():
        print(f"    {func_name:25s} -- {len(split_df)} rows, "
              f"categories: {split_df['category'].unique().tolist()}")

    # Apply each technique to its split
    print(f"\n  Applying techniques:")
    for func_name, split_df in splits.items():
        # Find the matching function
        func = next(f for f in all_techniques if f.__name__ == func_name)
        for _, row in split_df.head(1).iterrows():
            result, info = func(row["prompt"])
            print(f"    {func_name}: '{row['prompt']}' -> '{result[:40]}...'")

    # You can also use deterministic_balanced_assign directly
    print(f"\n  Direct balanced split (3 parts):")
    parts = deterministic_balanced_assign(df, num_splits=3)
    for i, part in enumerate(parts):
        cats = part["category"].value_counts().to_dict()
        print(f"    Part {i}: {len(part)} rows -- {cats}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    set_seed(42)
    print("REDACT Library -- Extensibility Demo\n")

    demo_custom_taxonomy()
    custom_techniques = demo_custom_techniques()
    demo_custom_splitting(custom_techniques)

    print(f"\n{'='*60}")
    print("Done! This example shows how to:")
    print("  1. Create and load custom taxonomies")
    print("  2. Write custom technique functions")
    print("  3. Chain custom + built-in techniques")
    print("  4. Build custom registries for dataset splitting")
    print("  5. Integrate with the full REDACT pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
