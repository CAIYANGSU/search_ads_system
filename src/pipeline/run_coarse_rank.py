"""Command-line entry point for the coarse-ranking pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.ranking.coarse_rank import main  # noqa: E402


if __name__ == "__main__":
    main()
