"""Run the strict Search Conversion Fine Rank credibility audit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from search_ads_system.common.config import load_yaml_config
from search_ads_system.ranking.fine_rank_multitask import parse_fine_rank_multitask_config
from search_ads_system.ranking.fine_rank_multitask_audit import (
    parse_fine_rank_multitask_audit_config, run_fine_rank_multitask_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Conversion Fine Rank leakage and memorization audit")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "all"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = args.config.resolve(); raw = load_yaml_config(path)
    model_config = parse_fine_rank_multitask_config(raw, path, stage=args.stage)
    audit_config = parse_fine_rank_multitask_audit_config(raw, path, stage=args.stage)
    # Audit batch sizing and epoch policy are independent from the selected
    # model's experiment settings, while feature/hash/data contracts remain identical.
    model_config = replace(model_config, batch_size=audit_config.batch_size, epochs=audit_config.epochs, patience=audit_config.patience)
    print(json.dumps(run_fine_rank_multitask_audit(model_config, audit_config, stage=args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
