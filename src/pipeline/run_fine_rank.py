"""Command-line entry point for DCNv2 multi-task fine ranking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.ranking.fine_rank import parse_fine_rank_config, run_fine_rank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train, evaluate, or infer DCNv2 fine ranking.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("build_dataset", "train", "evaluate", "infer", "all"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    result = run_fine_rank(parse_fine_rank_config(load_yaml_config(config_path), config_path), stage=args.stage)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
