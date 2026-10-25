"""Re-exports of the recipe loaders, the driver and the stage table, for one release.

Importing this module warns (``DeprecationWarning``): import them from
:mod:`redact`, or from :mod:`redact.recipe` and :mod:`redact.driver`.
``manifest_path``, ``write_manifest`` and ``read_manifest`` are the former
names of ``record_path``, ``write_record`` and ``read_record``.
"""

import warnings

from redact.driver import read_record, record_path, run_pipeline, write_record
from redact.recipe import load_params, load_recipe
from redact.stages import STAGES

manifest_path = record_path
write_manifest = write_record
read_manifest = read_record

warnings.warn(
    "redact.runconfig is deprecated: import run_pipeline, load_recipe, "
    "load_params, STAGES, read_record and write_record from redact (record_path "
    "from redact.driver). The former module is read for this release.",
    DeprecationWarning, stacklevel=2)

__all__ = [
    "STAGES",
    "load_params",
    "load_recipe",
    "manifest_path",
    "read_manifest",
    "read_record",
    "record_path",
    "run_pipeline",
    "write_manifest",
    "write_record",
]
