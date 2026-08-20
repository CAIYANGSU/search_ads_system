"""Convert raw Criteo conversion data into the canonical product-ad event schema."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.data.conversion import convert_criteo_to_unified  # noqa: E402
from search_ads_system.data.interfaces import parse_preprocess_config  # noqa: E402


def main() -> None:
    """Execute canonical Criteo conversion from command-line arguments."""

    parser = argparse.ArgumentParser(description="Convert Criteo data to the unified click-conversion schema.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated CSV parts.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_preprocess_config(load_yaml_config(config_path), config_path)
    result = convert_criteo_to_unified(
        config.dataset, config.outputs.unified_data, overwrite=args.overwrite
    )
    logging.getLogger(__name__).info(
        "Conversion complete: rows=%s parts=%s output=%s",
        result.rows_written,
        result.parts_written,
        result.output_directory,
    )


if __name__ == "__main__":
    main()
