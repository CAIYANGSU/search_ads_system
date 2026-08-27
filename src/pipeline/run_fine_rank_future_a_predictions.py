"""Materialize standalone Search Conversion Future-A prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.ranking.fine_rank_inference import write_future_a_predictions
from search_ads_system.ranking.fine_rank_multitask import parse_fine_rank_multitask_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Future-A-only Search Conversion DCNv2 inference")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "all"), required=True)
    args = parser.parse_args(); path = args.config.resolve(); raw = load_yaml_config(path)
    # Architecture comes from the checkpoint; stage only limits rows for sanity.
    config = parse_fine_rank_multitask_config(raw, path, stage="all")
    options = raw.get("fine_rank_multitask", {}).get("predictions", {})
    if not isinstance(options, dict): raise ValueError("fine_rank_multitask.predictions must be a mapping")
    root = path.parent; output = resolve_path(str(options.get("output_dir", "outputs/fine_rank/predictions/future_a_predictions")), root)
    max_rows = options.get("sanity", {}).get("max_rows") if args.stage == "sanity" else options.get("max_rows")
    print(json.dumps(write_future_a_predictions(config, checkpoint_path=resolve_path(str(options.get("checkpoint", "outputs/fine_rank/models/dcnv2.pt")), root), output_dir=output, max_rows=None if max_rows is None else int(max_rows)), indent=2, sort_keys=True))


if __name__ == "__main__": main()
