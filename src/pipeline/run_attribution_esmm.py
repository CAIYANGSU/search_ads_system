"""Run Attribution-only CTR/CVR baselines and standard ESMM.

Future-B is intentionally not a command-line option or a runner input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.ranking.attribution_esmm_pipeline import (  # noqa: E402
    evaluate_checkpoints,
    parse_attribution_esmm_config,
    sanity_config,
    train_all_models,
    write_metrics_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Attribution-only CTR/CVR/ESMM baselines; never reads Future-B.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_attribution_esmm_config(load_yaml_config(config_path), config_path)
    if args.stage == "sanity":
        active = sanity_config(config)
        training = train_all_models(active, artifact_suffix="sanity")
        report = evaluate_checkpoints(active, artifact_suffix="sanity")
        report["training"] = training
        write_metrics_report(active, report, artifact_suffix="sanity")
    elif args.stage == "train":
        train_all_models(config)
    elif args.stage == "evaluate":
        write_metrics_report(config, evaluate_checkpoints(config))
    else:
        training = train_all_models(config)
        report = evaluate_checkpoints(config)
        report["training"] = training
        write_metrics_report(config, report)


if __name__ == "__main__":
    main()
