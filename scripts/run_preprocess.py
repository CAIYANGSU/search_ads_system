"""Run a schema-driven inspection; this command never trains a model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.data.interfaces import parse_preprocess_config  # noqa: E402
from search_ads_system.data.processing import run_preprocessing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a configured ads dataset schema.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_path = args.config.resolve()
    raw_config = load_yaml_config(config_path)
    config = parse_preprocess_config(raw_config, config_path)
    report = run_preprocessing(config)

    config.outputs.schema_report.parent.mkdir(parents=True, exist_ok=True)
    with config.outputs.schema_report.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSchema report written to: {config.outputs.schema_report}")


if __name__ == "__main__":
    main()
