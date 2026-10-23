"""The variants table of a jailbreak kind: which template variant its written
rows drew, counted per technique.

A template technique names the variant it drew in its info, under its
pool's ``info_key`` (:class:`~redact.jailbreak.directives.PoolRef`):
``variant=<name>``, or ``style=<name>`` for continuation; structural's
``to_xml`` and ``to_markdown`` name none and are not counted. A row's
``technique_info`` joins its techniques' infos in their order, and a
framing and a request technique both write ``variant=``, so a variant is
told apart by position: the n-th ``variant=`` of a row belongs to the n-th
of its techniques that writes one.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from redact.dataset import has_rows
from redact.dataset.planfile import variants_path

from .directives import enabled, read_pools

#: The columns of the variants table.
COLUMNS = ["technique", "variant", "rows"]
#: The rows read at a time, as the dataset's readers do.
_CHUNK_ROWS = 50_000


def variants_of(
    technique, info, registry: dict[str, Callable],
) -> list[tuple[str, str]]:
    """The ``(technique, variant)`` pairs one row drew, from its
    ``technique`` (names joined by ``+``, in the order they ran) and its
    ``technique_info``.

    For each key that names a variant, its values in the info go, in order,
    to the row's techniques whose pool writes that key, less those the info
    notes as ``noop=<name>``. A row whose values and techniques do not pair
    up gives none: a rejected row, whose info is its rejection's, or an
    error row.
    """
    pairs = [part.split("=", 1) for part in str(info).split(";") if "=" in part]
    noops = {value for key, value in pairs if key == "noop"}
    writers: dict[str, list[str]] = {}
    for name in str(technique).split("+"):
        ref = getattr(registry.get(name), "variant_pool", None)
        if ref is not None and ref.info_key and name not in noops:
            writers.setdefault(ref.info_key, []).append(name)
    drawn = []
    for key, names in writers.items():
        values = [value for k, value in pairs if k == key]
        if len(values) != len(names):
            return []
        drawn += zip(names, values)
    return drawn


def write_variants(
    out_path, manifest_path, pool: list[Callable], registry: dict[str, Callable],
) -> Path:
    """Write ``<stem>.variants.csv`` beside the manifest at ``manifest_path``,
    counting the rows of ``out_path`` (read in chunks) per technique and
    variant (:func:`variants_of`; ``registry`` is every technique, so a row
    of a kept plan made with another pool counts).

    One row per variant on (weight above 0) of each technique of ``pool``
    whose info names its variant, with its count, 0 when no row drew it;
    then any other technique and variant a row drew. Columns
    :data:`COLUMNS`. Returns the table's path.
    """
    counts: Counter = Counter()
    if has_rows(out_path):
        with pd.read_csv(out_path, usecols=["technique", "technique_info"], dtype=str,
                         keep_default_na=False, chunksize=_CHUNK_ROWS) as chunks:
            for chunk in chunks:
                for technique, info in zip(chunk["technique"], chunk["technique_info"]):
                    counts.update(variants_of(technique, info, registry))

    table: dict[tuple[str, str], int] = {}
    pools: dict[str, dict] = {}
    for fn in pool:
        ref = getattr(fn, "variant_pool", None)
        if ref is None or not ref.info_key:
            continue
        if ref.file not in pools:
            pools[ref.file] = read_pools(ref.file)
        for variant in enabled(pools[ref.file][ref.directive]):
            table[(fn.__name__, variant["name"])] = 0
    for key, n in counts.items():
        table[key] = n

    out = variants_path(manifest_path)
    pd.DataFrame([{"technique": name, "variant": variant, "rows": n}
                  for (name, variant), n in table.items()],
                 columns=COLUMNS).to_csv(out, index=False)
    return out
