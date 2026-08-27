"""Final one-way Future-B holdout runner; all is the only reading stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from search_ads_system.common.config import load_yaml_config
from search_ads_system.evaluation.final_holdout import run_final_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen final Future-B holdout evaluation")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "all"), required=True)
    args = parser.parse_args(); path = args.config.resolve()
    print(json.dumps(run_final_holdout(load_yaml_config(path), path, stage=args.stage), indent=2, sort_keys=True))


if __name__ == "__main__": main()
