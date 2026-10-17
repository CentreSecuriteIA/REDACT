"""Benign Q&A pairs for the FSH and DAP techniques: generated on the worker pool
for each subcategory of a benign taxonomy, and cached.
"""

import logging
from pathlib import Path

import pandas as pd

from redact import paths
from redact.dataset import CommitBuffer, Ledger, Manifest, append_rows, resume_state
from redact.dataset.taxonomy import iter_categories, load_taxonomy
from redact.llm_pipeline import drive_units, extracted, pool_size
from redact.llms.client import ModelClient
from redact.llms.prompting import build_messages, extract_structured_qa, load_prompt

logger = logging.getLogger(__name__)

#: The packaged benign taxonomy, ``configs/taxonomy/benign.json``.
DEFAULT_TAXONOMY = "benign"
#: The answer types of a subcategory's pairs, each one unit.
ANSWER_TYPES = ("short", "long")
# A unit's key, and its ledger's.
_KEY_FIELDS = ("main_category", "sub_category", "answer_type")
# Generations of a unit before it is given up.
_ATTEMPTS = 3


def benign_categories(taxonomy: dict) -> list[tuple[str, str]]:
    """The ``(main, sub)`` pairs of a benign taxonomy, in file order, each once.

    A category's subcategories are a list, or a dict of which the keys are
    read; a category without any is its own subcategory.
    """
    pairs = [(main, str(sub))
             for main, info in iter_categories(taxonomy)
             for sub in info.get("subcategories") or [main]]
    return list(dict.fromkeys(pairs))


def _ledger(path: Path) -> Ledger:
    """The ledger of the benign data at ``path``: ``<name>.state.jsonl`` beside it."""
    return Ledger.sidecar(path, _KEY_FIELDS, dict.fromkeys(_KEY_FIELDS, str))


def generate_benign_data(
    categories: list[tuple[str, str]],
    client: ModelClient,
    out_path: str | Path,
    *,
    prompt_dir: str | Path | None = None,
    num_samples: int = 8,
    batch_size: int = 256,
    verbose: bool = True,
) -> None:
    """Generate the pairs ``out_path`` lacks for ``categories``, appended to it.

    A unit is one ``(main, sub)`` and answer type: one call asking for
    ``num_samples`` pairs. A reply with any other number of pairs is rejected
    whole and the call made again, up to three times; a unit rejected every
    time is acked ``rejected`` with no rows. ``pool_size(batch_size, client)``
    units are live, a call carries up to ``batch_size`` requests, and rows
    are appended every 32 units.

    The run always resumes: the ledger beside ``out_path``, keyed
    ``(main_category, sub_category, answer_type)``, says which units are
    done, and a file written before the ledger is taken over, so only the
    units it holds no rows of are generated.
    """
    out_path = Path(out_path)
    config = load_prompt("jailbreak", "benign_generation", prompt_dir)
    units = [(main, sub, answer_type)
             for main, sub in categories for answer_type in ANSWER_TYPES]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _ledger(out_path)
    manifest = Manifest.sidecar(out_path)
    done = resume_state(ledger, resume=True, artifact=out_path, manifest=manifest)
    manifest.write({**dict(zip(_KEY_FIELDS, unit)), "status": "planned"}
                   for unit in units)
    pending = [unit for unit in units if unit not in done]

    if verbose:
        logger.info("Benign data: %d units (%d subcategories x %d answer types), "
                    "%d to generate | Model: %s | Batch: %d", len(units),
                    len(categories), len(ANSWER_TYPES), len(pending), client.model,
                    batch_size)

    def _unit(unit: tuple[str, str, str]):
        main, sub, answer_type = unit
        messages = build_messages(
            config,
            sub_category=sub,
            main_category=main,
            num_samples=str(num_samples),
            answer_style=config["answer_styles"][answer_type],
        )
        return extracted(
            client, lambda: messages,
            lambda reply: extract_structured_qa(reply, expected_count=num_samples),
            max_attempts=_ATTEMPTS,
        )

    written = 0
    rejected: list[tuple[str, str, str]] = []

    def _on_done(i: int, value: tuple[list[dict], int]) -> None:
        nonlocal written
        pairs, attempts = value
        main, sub, answer_type = unit = pending[i]
        record = dict(zip(_KEY_FIELDS, unit))
        if not pairs:
            # Acked: un-acked, it would be generated, and paid for, again on
            # every resume.
            buf.add([], {**record, "status": "rejected", "attempts": attempts})
            rejected.append(unit)
            return
        buf.add([{"main_category": main, "sub_category": sub,
                  "prompt": pair["question"], "answer": pair["answer"],
                  "answer_type": answer_type} for pair in pairs], record)
        written += len(pairs)

    with CommitBuffer(append=lambda rows: append_rows(out_path, rows),
                      ledger=ledger) as buf:
        drive_units(
            ((i, _unit(unit)) for i, unit in enumerate(pending)),
            on_done=_on_done,
            pool_size=pool_size(batch_size, client),
            cap=batch_size,
            progress="benign" if verbose else None,
        )

    if rejected:
        logger.warning(
            "%d benign units rejected after %d attempt(s) each: %s. They are "
            "acked and skipped on resume; delete %s to generate them again (the "
            "pairs already in %s are kept).",
            len(rejected), _ATTEMPTS, ", ".join(" / ".join(u) for u in rejected),
            ledger.path.name, out_path.name,
        )
    if verbose:
        logger.info("Benign data: %d pairs generated this run, saved to %s",
                    written, out_path)


def get_or_generate_benign_data(
    client: ModelClient | None = None,
    cache_path=None,
    verbose: bool = True,
    *,
    taxonomy: dict | str = DEFAULT_TAXONOMY,
    taxonomy_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    batch_size: int = 256,
) -> dict:
    """The benign data at ``cache_path``, generated first with ``client``.

    With a client, the pairs of every subcategory of ``taxonomy`` that the
    cache does not hold yet are generated (:func:`generate_benign_data`).
    ``taxonomy`` is a taxonomy file's name under ``taxonomy_dir`` (the
    packaged ``configs/taxonomy`` by default), or a loaded taxonomy. Without
    a client, the cache is loaded as it is. ``cache_path`` defaults to
    ``paths.benign_csv()``.

    Returns:
        What :func:`load_benign_data` returns.

    Raises:
        ValueError: No cache and no client, a taxonomy without a category, or
            no pair in the cache after generation.
        FileNotFoundError: No such taxonomy file.
    """
    path = Path(cache_path) if cache_path else paths.benign_csv()

    if client is None:
        if not path.exists():
            raise ValueError(
                "Benign data not found and no client provided for generation. "
                f"Either pre-generate benign data at {path} or pass a ModelClient."
            )
        if verbose:
            logger.info("Loading cached benign data from %s", path)
        return load_benign_data(path)

    if isinstance(taxonomy, str):
        taxonomy = load_taxonomy(taxonomy, config_dir=taxonomy_dir)
    categories = benign_categories(taxonomy)
    if not categories:
        raise ValueError(
            f"The benign taxonomy {taxonomy.get('name', '')!r} has no category; "
            "write its categories, each with its subcategories.")
    generate_benign_data(categories, client, path, prompt_dir=prompt_dir,
                         batch_size=batch_size, verbose=verbose)
    if not path.exists():
        raise ValueError(
            f"No benign pairs at {path}: every unit's reply was rejected "
            f"{_ATTEMPTS} times, and the units are acked as rejected in "
            f"{_ledger(path).path.name}. Delete that file to generate them again.")
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
