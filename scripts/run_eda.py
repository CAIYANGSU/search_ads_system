"""Produce EDA artifacts for canonical click-conversion data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.data.eda import run_eda  # noqa: E402
from search_ads_system.data.interfaces import parse_preprocess_config  # noqa: E402


def main() -> None:
    """Execute EDA from command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate EDA statistics for unified Criteo data.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_preprocess_config(load_yaml_config(config_path), config_path)
    summary = run_eda(
        config.outputs.unified_data,
        config.outputs.eda_summary,
        config.outputs.eda_categories,
        config.eda,
        config.dataset.chunk_size,
    )
    logging.getLogger(__name__).info(
        "EDA complete: rows=%s conversion_rate=%.6f", summary["row_count"], summary["conversion_rate"]
    )


if __name__ == "__main__":
    main()
