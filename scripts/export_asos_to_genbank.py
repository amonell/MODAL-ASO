#!/usr/bin/env python3
"""CLI wrapper for exporting routed ASO features to GenBank."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aso_agents.genbank_export import export_to_genbank, main


if __name__ == "__main__":
    main()
