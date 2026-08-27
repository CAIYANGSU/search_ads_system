"""Calibrate a fixed Attribution ESMM checkpoint using Future-A only."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.evaluation.final_holdout import future_b_opened_warning  # noqa: E402
from search_ads_system.ranking.attribution_calibration_pipeline import (  # noqa: E402
    calibrate_and_evaluate,
    generate_prediction_artifacts,
    parse_attribution_calibration_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Attribution ESMM on Future-A only; Future-B is never read.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "predict", "calibrate", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated calibration prediction parts.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    if warning := future_b_opened_warning(config_path): logging.warning(warning)
    config = parse_attribution_calibration_config(load_yaml_config(config_path), config_path)
    if args.stage == "sanity":
        generate_prediction_artifacts(config, artifact_suffix="sanity", max_rows=config.sanity_max_rows, overwrite=args.overwrite)
        calibrate_and_evaluate(config, artifact_suffix="sanity")
    elif args.stage == "predict":
        generate_prediction_artifacts(config, overwrite=args.overwrite)
    elif args.stage == "calibrate":
        calibrate_and_evaluate(config)
    else:
        generate_prediction_artifacts(config, overwrite=args.overwrite)
        calibrate_and_evaluate(config)


if __name__ == "__main__":
    main()
