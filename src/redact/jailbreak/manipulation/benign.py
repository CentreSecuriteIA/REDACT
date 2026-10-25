"""Benign Q&A pairs for the FSH and DAP techniques: generated on the worker pool
for each subcategory of a benign taxonomy, each pair checked, and cached.
"""

import logging
from collections.abc import Generator
from pathlib import Path

import pandas as pd

from redact import paths
from redact.content_moderation.feedback import NOT_ONE_PAIR, with_feedback
from redact.dataset import CommitBuffer, Ledger, Manifest, append_rows, resume_state
from redact.dataset.taxonomy import iter_categories, load_taxonomy
from redact.llm_pipeline import (
    CHECK_TEMPERATURE,
    LLMRequest,
    checked,
    drive_units,
    extracted,
    is_accepted,
    pool_size,
)
from redact.llms.client import ModelClient
from redact.llms.prompting import build_messages, extract_structured_qa, load_prompt

logger = logging.getLogger(__name__)

#: The packaged benign taxonomy, ``configs/taxonomy/benign.json``.
DEFAULT_TAXONOMY = "benign"
#: The answer types of a subcategory's pairs, each one unit.
ANSWER_TYPES = ("short", "long")
# A unit's key, and its ledger's.
_KEY_FIELDS = ("main_category", "sub_category", "answer_type")
# Generations of a unit's reply before it is given up, and of each pair: the
# reply's generation is a pair's first, then it is asked for alone again.
_ATTEMPTS = 3

#: A unit's pairs: ``{"question", "answer"}`` each.
Pairs = list[dict[str, str]]


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


def _category(main: str, sub: str) -> str:
    """The category a pair's check is shown: its subcategory under its main
    category, or the main category alone when it is its own subcategory."""
    return main if sub == main else f"{main} / {sub}"


def _one_pair(reply: str) -> Pairs:
    """The pair of a reply that holds exactly one, ``[]`` for any other."""
    return extract_structured_qa(reply, expected_count=1)


def generate_benign_data(
    categories: list[tuple[str, str]],
    client: ModelClient,
    out_path: str | Path,
    *,
    check: ModelClient | None = None,
    prompt_dir: str | Path | None = None,
    num_samples: int = 8,
    batch_size: int = 256,
    verbose: bool = True,
) -> None:
    """Generate the pairs ``out_path`` lacks for ``categories``, appended to it.

    A unit is one ``(main, sub)`` and answer type: one call asking for
    ``num_samples`` pairs. A reply with any other number of pairs is rejected
    whole and the call made again, up to three times; a unit rejected every
    time is acked ``rejected`` with no rows.

    Each pair of an accepted reply is then read by ``check`` (``client``
    when None) with ``jailbreak/benign_check`` (its ``category``,
    ``question`` and ``answer``), at
    :data:`~redact.llm_pipeline.CHECK_TEMPERATURE`. A rejected pair is
    asked for again alone (the generation prompt for one pair) with the
    verdict as feedback, and the new pair checked, through
    :func:`~redact.llm_pipeline.checked`: three generations of a pair in all,
    the reply's first; a reply that is not one pair is rejected without a
    check call (:data:`~redact.content_moderation.feedback.NOT_ONE_PAIR`,
    its feedback). A pair rejected every time is left out, its unit's ack
    counting it (``rejected_pairs``); a unit none of whose pairs is left is
    acked ``rejected`` with no rows, as one with no reply in the format.

    ``pool_size(batch_size, client, check)`` units are live, a call carries
    up to ``batch_size`` requests, and rows are appended every 32 units.

    The run always resumes: the ledger beside ``out_path``, keyed
    ``(main_category, sub_category, answer_type)``, says which units are
    done, and a file written before the ledger is taken over, so only the
    units it holds no rows of are generated.
    """
    out_path = Path(out_path)
    check = check if check is not None else client
    config = load_prompt("jailbreak", "benign_generation", prompt_dir)
    check_config = load_prompt("jailbreak", "benign_check", prompt_dir)
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
                    "%d to generate | Model: %s | Check: %s | Batch: %d", len(units),
                    len(categories), len(ANSWER_TYPES), len(pending), client.model,
                    check.model, batch_size)

    def _check_messages(category: str, pair: dict[str, str]) -> list[dict]:
        return build_messages(check_config, category=category,
                              question=pair["question"], answer=pair["answer"])

    def _checked_pair(pair: dict[str, str], ask_one: list[dict], category: str,
                      ) -> Generator[LLMRequest, str, tuple[dict | None, str]]:
        """Check ``pair``; ask for one pair again (``ask_one``) with the
        verdict until one is accepted, :data:`_ATTEMPTS` generations in all.
        Returns ``(pair, "")``, or ``(None, last verdict)``."""
        verdict = yield LLMRequest.for_client(
            check, _check_messages(category, pair), temperature=CHECK_TEMPERATURE)
        if is_accepted(verdict):
            return pair, ""
        reply, accepted, reasoning = yield from checked(
            client, lambda feedback: with_feedback(ask_one, feedback or verdict),
            check,
            lambda _original, reply: _check_messages(category, _one_pair(reply)[0]),
            max_attempts=_ATTEMPTS - 1,
            precheck=lambda reply: "" if _one_pair(reply) else NOT_ONE_PAIR,
            temperatures=(None, CHECK_TEMPERATURE))
        return (_one_pair(reply)[0], "") if accepted else (None, reasoning)

    def _unit(unit: tuple[str, str, str],
              ) -> Generator[LLMRequest, str, tuple[Pairs, int, int]]:
        """The unit's pairs that passed the check, the generations its reply
        took, and how many pairs the check left out."""
        main, sub, answer_type = unit

        def ask(count: int) -> list[dict]:
            return build_messages(
                config, sub_category=sub, main_category=main, num_samples=str(count),
                answer_style=config["answer_styles"][answer_type])

        messages = ask(num_samples)
        pairs, attempts = yield from extracted(
            client, lambda: messages,
            lambda reply: extract_structured_qa(reply, expected_count=num_samples),
            max_attempts=_ATTEMPTS,
        )
        kept, ask_one, category = [], ask(1), _category(main, sub)
        for pair in pairs:
            pair, _verdict = yield from _checked_pair(pair, ask_one, category)
            if pair is not None:
                kept.append(pair)
        return kept, attempts, len(pairs) - len(kept)

    written = dropped = 0
    rejected: list[tuple[str, str, str]] = []

    def _on_done(i: int, value: tuple[Pairs, int, int]) -> None:
        nonlocal written, dropped
        pairs, attempts, left_out = value
        main, sub, answer_type = unit = pending[i]
        record = dict(zip(_KEY_FIELDS, unit))
        if left_out:
            record["rejected_pairs"] = left_out
            dropped += left_out
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
            pool_size=pool_size(batch_size, client, check),
            cap=batch_size,
            progress="benign" if verbose else None,
        )

    if rejected:
        logger.warning(
            "%d benign units rejected: no reply held %d pairs in %d calls, or "
            "the check rejected each pair %d times: %s. They are acked and "
            "skipped on resume; delete %s to generate them again (the pairs "
            "already in %s are kept).",
            len(rejected), num_samples, _ATTEMPTS, _ATTEMPTS,
            ", ".join(" / ".join(u) for u in rejected), ledger.path.name,
            out_path.name,
        )
    if verbose:
        logger.info("Benign data: %d pairs generated this run, saved to %s; %d "
                    "left out, rejected by the check %d times each", written,
                    out_path, dropped, _ATTEMPTS)


def get_or_generate_benign_data(
    client: ModelClient | None = None,
    cache_path=None,
    verbose: bool = True,
    *,
    check: ModelClient | None = None,
    taxonomy: dict | str = DEFAULT_TAXONOMY,
    taxonomy_dir: str | Path | None = None,
    prompt_dir: str | Path | None = None,
    batch_size: int = 256,
) -> dict:
    """The benign data at ``cache_path``, generated first with ``client``.

    With a client, the pairs of every subcategory of ``taxonomy`` that the
    cache does not hold yet are generated, each pair checked by ``check``
    (``client`` when None; :func:`generate_benign_data`). ``taxonomy`` is a
    taxonomy file's name under ``taxonomy_dir`` (the packaged
    ``configs/taxonomy`` by default), or a loaded taxonomy. Without a client,
    the cache is loaded as it is, unchecked. ``cache_path`` defaults to
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
    generate_benign_data(categories, client, path, check=check, prompt_dir=prompt_dir,
                         batch_size=batch_size, verbose=verbose)
    if not path.exists():
        raise ValueError(
            f"No benign pairs at {path}: every unit was rejected (no reply in "
            f"the format in {_ATTEMPTS} calls, or every pair rejected by the "
            f"check {_ATTEMPTS} times), and the units are acked as rejected in "
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
