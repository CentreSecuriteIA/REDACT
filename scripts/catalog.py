"""Write the technique catalog, ``docs/catalog.json`` and ``docs/catalog.md``,
from the code.

    python scripts/catalog.py [--out-dir DIR] [--check]

The same as ``PYTHONPATH=src py -m redact.jailbreak.catalog``: every
registered jailbreak technique with its tags, variants, prompts, pools and
the techniques it combines with, the families and their matrix, the layers,
the pools, the prompt inventory and the combination sizes each packaged
kind plans. ``--check`` writes nothing and exits 1 when a file is stale;
``tests/jailbreak/test_catalog.py`` fails then too.
"""

import sys

from redact.jailbreak.catalog import main

if __name__ == "__main__":
    sys.exit(main())
