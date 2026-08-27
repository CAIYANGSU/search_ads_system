"""Build, split, and audit the Attribution-only impression data contract."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.data.attribution import (  # noqa: E402
    build_attribution_audit,
    build_attribution_impressions,
    split_attribution_temporally,
)
from search_ads_system.data.interfaces import parse_attribution_preprocess_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Attribution impressions; never trains an ESMM model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("build", "split", "audit", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated Attribution CSV parts.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_attribution_preprocess_config(load_yaml_config(config_path), config_path)
    if args.stage in {"build", "all"}:
        build_attribution_impressions(config, overwrite=args.overwrite)
    if args.stage in {"split", "all"}:
        split_attribution_temporally(config, overwrite=args.overwrite)
    if args.stage in {"audit", "all"}:
        build_attribution_audit(config)


if __name__ == "__main__":
    main()
