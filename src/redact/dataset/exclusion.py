"""The rows another run holds, which an eval build leaves out: by id, by text
up to case, spacing and punctuation, and by constitution entry."""

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from redact import paths

from .identity import content_id, normalised_id
from .rows import _CHUNK_ROWS, _has_bytes, accepted_mask

#: Why a row is left out, in the order a row's reasons are taken.
REASONS = ("exact", "normalised", "descriptions")

# The columns that hold a row's text in a stage file, a dataset or an eval file.
_TEXT_COLUMNS = ("sample", "jailbreak", "output_response", "text")
# The columns that hold a constitution entry's description.
_DESCRIPTION_COLUMNS = ("sample_description", "source_sample_description")


def exclusion_files(given: Iterable[str | Path]) -> list[Path]:
    """The CSVs that ``given`` names: a file as it is, a run's folder as its
    ``Datasets/*.csv`` and its constitution's ``merged.csv``."""
    files: list[Path] = []
    for path in map(Path, given):
        if path.is_dir():
            constitution = paths.constitution_merged_csv(path)
            files += sorted(paths.datasets(path).glob("*.csv"))
            files += [constitution] if constitution.exists() else []
        else:
            files.append(path)
    return files


class Exclusion:
    """What the files of another run hold, and the eval rows it leaves out.

    A row is left out when its ``sample_id`` is a row's of those files
    (``exact``), when its text is one of theirs up to case, spacing and
    punctuation (``normalised``), or when the input it comes from is left
    out, its constitution entry being one of theirs (``descriptions``); a
    row derived from a left-out input is left out for its input's reason.
    Only the accepted rows of a file with an ``accepted`` column count.

    Attributes:
        counts: The rows left out so far, by reason.
    """

    def __init__(self, files: Iterable[Path]):
        self.ids: set[str] = set()
        self.normalised: set[str] = set()
        self.descriptions: set[str] = set()
        for path in files:
            self._read(path)
        self.normalised.discard(normalised_id(""))
        self.inputs: dict[str, str] = {}      # a left-out input's id: its reason
        self.counts = dict.fromkeys(REASONS, 0)

    def _read(self, path: Path) -> None:
        """Add the ids, texts and descriptions of the CSV at ``path``, read
        in chunks."""
        if not _has_bytes(path):
            return
        header = pd.read_csv(path, nrows=0).columns
        wanted = [c for c in ("sample_id", "accepted", *_TEXT_COLUMNS,
                              *_DESCRIPTION_COLUMNS) if c in header]
        if not wanted:
            return
        with pd.read_csv(path, usecols=wanted, dtype=str,
                         chunksize=_CHUNK_ROWS) as chunks:
            for chunk in chunks:
                if "accepted" in chunk.columns:
                    chunk = chunk[accepted_mask(chunk["accepted"])]
                if "sample_id" in chunk.columns:
                    self.ids.update(chunk["sample_id"].dropna())
                for column in _TEXT_COLUMNS:
                    if column in chunk.columns:
                        self.normalised.update(chunk[column].dropna().map(normalised_id))
                for column in _DESCRIPTION_COLUMNS:
                    if column in chunk.columns:
                        self.descriptions.update(chunk[column].dropna().map(content_id))

    def read_inputs(self, inputs: Path | pd.DataFrame) -> None:
        """Note which of the eval inputs are left out, and why, so the rows
        derived from them are too."""
        for chunk in _chunks(inputs):
            if "sample_id" not in chunk.columns:
                continue
            reasons = self._reasons(chunk, "sample")
            if "source_sample_description" in chunk.columns:
                described = chunk["source_sample_description"].fillna("").astype(str)
                held = described.ne("") & described.map(content_id).isin(
                    self.descriptions)
                reasons = reasons.mask(reasons.isna() & held, "descriptions")
            left_out = reasons.dropna()
            ids = chunk.loc[left_out.index, "sample_id"].astype(str)
            self.inputs.update(zip(ids, left_out))

    def keeps(self, text: str, origin: str) -> Callable[[pd.DataFrame], pd.Series]:
        """A filter of chunks whose text is in column ``text`` and whose input's
        id is in column ``origin``: the rows it keeps, counting the others."""
        def kept(chunk: pd.DataFrame) -> pd.Series:
            reasons = self._reasons(chunk, text)
            if origin in chunk.columns:
                reasons = reasons.fillna(chunk[origin].astype(str).map(self.inputs))
            for reason, count in reasons.value_counts().items():
                self.counts[reason] += int(count)
            return reasons.isna()
        return kept

    def _reasons(self, chunk: pd.DataFrame, text: str) -> pd.Series:
        """Each row's reason to be left out by its own id or text, else NA."""
        reasons = pd.Series(pd.NA, index=chunk.index, dtype=object)
        if text in chunk.columns:
            texts = chunk[text].dropna().astype(str)
            held = texts.map(normalised_id).isin(self.normalised)
            reasons.loc[held[held].index] = "normalised"
        if "sample_id" in chunk.columns:
            reasons.loc[chunk["sample_id"].astype(str).isin(self.ids)] = "exact"
        return reasons


def _chunks(source: Path | pd.DataFrame) -> Iterable[pd.DataFrame]:
    """The rows of a file, read as text in chunks, or a frame whole."""
    if isinstance(source, pd.DataFrame):
        yield source
    elif _has_bytes(source):
        with pd.read_csv(source, dtype=str, chunksize=_CHUNK_ROWS) as chunks:
            yield from chunks
