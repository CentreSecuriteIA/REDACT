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
    "source_taxonomy",
    "source_group_tag",
    "model",
]

# The group tag of a category that names none.
_GENERAL = "general"
# The prompt, and the file, of a benign-taxonomy category's unit.
GENERAL_BENIGN = "general_benign"
# The ``source_taxonomy`` of each side's rows.
_HARMFUL, _BENIGN = "harmful", "benign"


@dataclass
class Unit:
    """One call: an entry type for a taxonomy category, or the general-benign
    call for a benign-taxonomy category.

    ``prompt`` names the call's prompt under ``prompts/constitution/``, the
    entry type unless given; ``kwargs`` are its placeholder values.
    ``source_taxonomy`` says which taxonomy the category comes from.
    """

    source_category: str
    entry_type: str
    group_tag: str
    kwargs: dict[str, str] = field(default_factory=dict)
    prompt: str = ""
    source_taxonomy: str = _HARMFUL

    def __post_init__(self) -> None:
        self.prompt = self.prompt or self.entry_type

    @property
    def key(self) -> str:
        """The unit's ledger key, ``source_category::prompt``."""
        return f"{self.source_category}::{self.prompt}"

    @property
    def manifest_row(self) -> dict[str, str]:
        return {"source_category": self.source_category,
                "entry_type": self.entry_type, "prompt": self.prompt,
                "status": "planned"}


def file_of(row: dict) -> str:
    """The name of the CSV, beside ``merged.csv``, a constitution row goes to:
    its entry type's, or ``general_benign`` for a benign-taxonomy row."""
    return GENERAL_BENIGN if row["source_taxonomy"] == _BENIGN else row["entry_type"]


def key_of(row: dict) -> str:
    """The ledger key of the unit a constitution row came from: its
    ``source_category``, then the unit's prompt, which names its file too."""
    return f"{row['source_category']}::{file_of(row)}"


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
    include_general_benign: bool = False,
    general_benign_categories: int = 10,
    benign_taxonomy: dict | None = None,
) -> list[Unit]:
    """Every unit of a run, in the order they are generated.

    Each category of ``taxonomy`` gets one unit per entry type, asking for
    ``num_categories`` constitution categories. With
    ``include_general_benign``, each category of ``benign_taxonomy`` then
    gets one unit of the ``general_benign`` prompt, asking for
    ``general_benign_categories``, its rows labelled ``benign``. Each
    taxonomy gives its first ``num_taxonomy_categories`` categories.
    """
    if entry_types is None:
        entry_types = ALL_ENTRY_TYPES
    entry_types = list(dict.fromkeys(entry_types))
    categories = list(iter_categories(taxonomy))[:num_taxonomy_categories]
    units = [
        Unit(name, entry_type.value, info.get("group_tag", _GENERAL),
             {**_prompt_kwargs(name, info), "num_categories": str(num_categories)})
        for name, info in categories
        for entry_type in entry_types
    ]
    if include_general_benign:
        benign = list(iter_categories(benign_taxonomy or {}))[:num_taxonomy_categories]
        units += [
            Unit(name, EntryType.BENIGN.value, info.get("group_tag", _GENERAL),
                 {**_prompt_kwargs(name, info),
                  "num_categories": str(general_benign_categories)},
                 prompt=GENERAL_BENIGN, source_taxonomy=_BENIGN)
            for name, info in benign
        ]
    return units


def messages_of(unit: Unit, prompt_dir=None) -> list[dict]:
    """The messages of the unit's call, its prompt read through ``prompt_dir``."""
    return build_messages(
        load_prompt("constitution", unit.prompt, prompt_dir), **unit.kwargs)


def rows_of(unit: Unit, reply: str, model: str) -> list[dict]:
    """The rows of a reply, in ``_COLUMNS`` order.

    ``[]`` rejects the reply whole: one without its ``[END]`` marker, which
    stopped early, or one with no entry that parses.
    """
    text, complete = strip_end_marker(reply)
    if not complete:
        logger.warning("Constitution reply for %s rejected: no [END] marker.", unit.key)
        return []
    rows = [
        {
            "sample_id": content_id(entry.sample),
            "constitution_category": entry.category,
            "constitution_subcategory": entry.subcategory,
            "sample_description": entry.sample,
            "entry_type": unit.entry_type,
            "source_category": unit.source_category,
            "source_taxonomy": unit.source_taxonomy,
            "source_group_tag": unit.group_tag,
            "model": model,
        }
        for entry in parse_constitution(text)
    ]
    if not rows:
        logger.warning("No entries parsed for %s (reply of %d chars).",
                       unit.key, len(reply))
    return rows
