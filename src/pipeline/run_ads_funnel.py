"""Run the formal full-data Recall Top1000 -> Coarse Top100 funnel."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from search_ads_system.common.config import load_yaml_config  # noqa: E402
from search_ads_system.evaluation.final_holdout import future_b_opened_warning  # noqa: E402
from search_ads_system.ranking.coarse_rank import parse_coarse_rank_config, run_coarse_rank  # noqa: E402
from search_ads_system.recall.itemcf_recall import (  # noqa: E402
    generate_itemcf_candidates, load_interactions as load_itemcf, parse_itemcf_config, write_candidates as write_itemcf,
)
from search_ads_system.recall.popularity_recall import (  # noqa: E402
    generate_popularity_candidates, parse_popularity_config, write_candidates as write_popularity,
)
from search_ads_system.recall.rrf_fusion import fuse_and_write_candidates, parse_rrf_config  # noqa: E402
from search_ads_system.recall.two_tower_recall import parse_two_tower_config, run_two_tower_recall  # noqa: E402


def run_formal_funnel(raw: dict, config_path: Path, stage: str = "all") -> dict[str, object]:
    """Build all three recall routes, fixed RRF Top1000, then Coarse Top100."""
    result: dict[str, object] = {}
    if stage in {"recall", "all"}:
        item = parse_itemcf_config(raw, config_path)
        write_itemcf(generate_itemcf_candidates(load_itemcf(item), item), item.output_path)
        popularity = parse_popularity_config(raw, config_path)
        write_popularity(generate_popularity_candidates(popularity), popularity.output_path)
        two_tower = parse_two_tower_config(raw, config_path)
        run_two_tower_recall(two_tower)
        rrf = parse_rrf_config(raw, config_path)
        rows = fuse_and_write_candidates(rrf)
        result["recall"] = {"rrf_output": str(rrf.output_path), "rrf_rows": rows, "top_k_per_user": rrf.top_k_per_user, "rrf_k": rrf.k, "weights": dict(rrf.weights or {})}
    if stage in {"coarse", "all"}:
        coarse = parse_coarse_rank_config(raw, config_path)
        result["coarse"] = run_coarse_rank(coarse)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal multi-recall Top1000 -> Coarse Top100 funnel.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("recall", "coarse", "all"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    if warning := future_b_opened_warning(config_path): logging.warning(warning)
    print(json.dumps(run_formal_funnel(load_yaml_config(config_path), config_path, args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
