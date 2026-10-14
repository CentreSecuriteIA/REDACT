"""The units of a constitution run: what one call asks, and what its reply becomes."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..dataset import content_id, iter_categories
from ..llms.prompting import build_messages, load_prompt
from ..types import ALL_ENTRY_TYPES, EntryType
from .parse import parse_constitution, strip_end_marker

logger = logging.getLogger(__name__)

# The columns of every constitution CSV, in order.
_COLUMNS = [
    "sample_id",
    "constitution_category",
    "constitution_subcategory",
    "sample_description",
    "entry_type",
    "source_category",
    "source_group_tag",
    "model",
]

# The source category and group tag of the unit with no taxonomy category.
_GENERAL = "general"


@dataclass
class Unit:
    """One call: an entry type for a taxonomy category, or the general-benign call.

    ``entry_type`` also names the call's prompt under ``prompts/constitution/``;
    ``kwargs`` are the prompt's placeholder values.
    """

    source_category: str
    entry_type: str
    group_tag: str
    kwargs: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """The unit's ledger key, ``source_category::entry_type``."""
        return f"{self.source_category}::{self.entry_type}"

    @property
    def manifest_row(self) -> dict[str, str]:
        return {"source_category": self.source_category,
                "entry_type": self.entry_type, "status": "planned"}


def _prompt_kwargs(name: str, info: dict) -> dict[str, str]:
    """A taxonomy category's values for the constitution prompt."""
    subcats = info.get("subcategories", [])
    if isinstance(subcats, dict):
        subcats_str = ", ".join(subcats.keys())
    elif isinstance(subcats, list):
        subcats_str = ", ".join(subcats)
    else:
        subcats_str = str(subcats)
    return {
        "Category": name,
        "description": info.get("description", ""),
        "subcategories": subcats_str,
        "group_tag": info.get("group_tag", _GENERAL),
    }


def plan_units(
    taxonomy: dict,
    entry_types: Iterable[EntryType] | None = None,
    num_categories: int = 10,
    num_taxonomy_categories: int | None = None,
    include_standalone_benign: bool = False,
    standalone_benign_categories: int = 10,
) -> list[Unit]:
    """Every unit of a run, in the order they are generated.

    Each taxonomy category (the first ``num_taxonomy_categories``) gets one
    unit per entry type, asking for ``num_categories`` constitution
    categories; the general-benign unit, when included, comes last.
    """
    entry_types = ALL_ENTRY_TYPES if entry_types is None else list(entry_types)
    categories = list(iter_categories(taxonomy))[:num_taxonomy_categories]
    units = [
        Unit(name, entry_type.value, info.get("group_tag", _GENERAL),
             {**_prompt_kwargs(name, info), "num_categories": str(num_categories)})
        for name, info in categories
        for entry_type in entry_types
    ]
    if include_standalone_benign:
        units.append(Unit(_GENERAL, "general_benign", _GENERAL,
                          {"num_categories": str(standalone_benign_categories)}))
    return units


def messages_of(unit: Unit) -> list[dict]:
    """The messages of the unit's call."""
    return build_messages(load_prompt("constitution", unit.entry_type), **unit.kwargs)


def rows_of(unit: Unit, reply: str, model: str) -> list[dict]:
    """The rows of a reply, in ``_COLUMNS`` order; ``[]`` when none parses."""
    text, complete = strip_end_marker(reply)
    if not complete:
        logger.warning("Constitution reply for %s may be truncated (no [END] marker).",
                       unit.key)
    rows = [
        {
            "sample_id": content_id(entry.sample),
            "constitution_category": entry.category,
            "constitution_subcategory": entry.subcategory,
            "sample_description": entry.sample,
            "entry_type": unit.entry_type,
            "source_category": unit.source_category,
            "source_group_tag": unit.group_tag,
            "model": model,
        }
        for entry in parse_constitution(text)
    ]
    if not rows:
        logger.warning("No entries parsed for %s (reply of %d chars).",
                       unit.key, len(reply))
    return rows
