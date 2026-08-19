"""Run schema-driven preprocessing checks; this command never trains a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.data.interfaces import parse_preprocess_config  # noqa: E402
from search_ads_system.data.processing import run_preprocessing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a configured ads dataset schema.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    config = parse_preprocess_config(raw_config, config_path)
    report = run_preprocessing(config)

    config.schema_report_path.parent.mkdir(parents=True, exist_ok=True)
    with config.schema_report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSchema report written to: {config.schema_report_path}")


if __name__ == "__main__":
    main()
