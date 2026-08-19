"""Build model-agnostic features from canonical click-conversion data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.data.features import build_features  # noqa: E402
from search_ads_system.data.interfaces import parse_preprocess_config  # noqa: E402


def main() -> None:
    """Execute feature generation from command-line arguments."""

    parser = argparse.ArgumentParser(description="Build features from unified Criteo data.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated CSV parts.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_preprocess_config(load_yaml_config(config_path), config_path)
    result = build_features(
        config.outputs.unified_data,
        config.outputs.feature_data,
        config.outputs.feature_metadata,
        config.features,
        config.dataset.chunk_size,
        overwrite=args.overwrite,
    )
    logging.getLogger(__name__).info(
        "Feature generation complete: rows=%s parts=%s output=%s",
        result.rows_written,
        result.parts_written,
        result.output_directory,
    )


if __name__ == "__main__":
    main()
