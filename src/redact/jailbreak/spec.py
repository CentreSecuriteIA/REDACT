"""Loading and applying combination_spec.json — the technique compatibility spec.

Split out of jailbreak/utils.py (which used to mix 7 unrelated concerns in
one file — see that module's docstring for the full breakdown and why this
split happened). This one owns: reading the spec JSON once, and tagging
technique functions with the metadata it describes.
"""

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

_SPEC_PATH = Path(__file__).parent.parent / "configs" / "jailbreak" / "combination_spec.json"


@lru_cache(maxsize=1)
def load_spec() -> dict:
    """Load and cache the combination_spec.json.

    Returns the full spec dict with families, techniques, layer_order, etc.
    """
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def tag_all_functions(funcs: list[Callable]) -> None:
    """Attach compatibility metadata from combination_spec.json to each function.

    Sets on each function object:
        fn.families          list[str]   — family membership
        fn.complexity        int         — complexity score (0-3)
        fn.requires_llm      bool        — whether an LLM backend is needed
        fn.layer             str         — "hacking" | "manipulation" | "obfuscation" | "requests"
        fn.within_layer_order int        — sort key within the obfuscation layer (0 = N/A)
        fn.encode_weight     str | None  — "light" | "heavy" | None (encode family only)

    Functions not found in the spec are skipped silently (no attributes set).
    Call once at jailbreak package import time.
    """
    spec = load_spec()
    families_spec = spec["families"]
    techniques = spec["techniques"]

    # Build layer → within_layer_order lookup from families
    family_layer: dict[str, str] = {name: d["layer"] for name, d in families_spec.items()}
    family_wlo: dict[str, int] = {
        name: d.get("within_layer_order", 0) for name, d in families_spec.items()
    }

    for fn in funcs:
        name = getattr(fn, "__name__", None)
        if name not in techniques:
            continue
        t = techniques[name]
        fn.families = t.get("families", [])
        fn.complexity = t.get("complexity", 1)
        fn.requires_llm = t.get("requires_llm", False)
        # Needs pre-generated benign Q&A data (FSH/DAP). Distinct from
        # requires_llm: the data is generated once up front, not per call — so
        # these aren't per-sample LLM techniques, but they still can't run in a
        # strictly no-LLM (pure_only) run.
        fn.requires_benign = t.get("requires_benign", False)
        fn.encode_weight = t.get("encode_weight", None)

        # Derive layer and within_layer_order from the first family
        primary_family = fn.families[0] if fn.families else None
        fn.layer = family_layer.get(primary_family, "obfuscation") if primary_family else "obfuscation"
        fn.within_layer_order = family_wlo.get(primary_family, 0) if primary_family else 0
