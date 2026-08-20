"""Streaming global-popularity recall for product advertisements."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from search_ads_system.common.config import load_yaml_config, resolve_path

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ("candidate_ad_id", "popularity_score", "rank")


@dataclass(frozen=True)
class PopularityRecallConfig:
    """Configuration for streaming global-ad popularity candidate generation."""

    input_path: Path
    output_path: Path
    top_k: int = 200
    click_weight: float = 1.0
    conversion_weight: float = 3.0
    chunk_size: int = 200_000
    user_id_column: str = "user_id"
    product_id_column: str = "product_id"
    conversion_label_column: str = "conversion_label"


def parse_popularity_config(raw_config: Mapping[str, Any], config_path: Path) -> PopularityRecallConfig:
    """Read and validate ``recall.popularity`` from the project configuration."""

    try:
        paths = raw_config["paths"]
        recall = raw_config["recall"]
    except KeyError as error:
        raise ValueError("Configuration must define paths and recall") from error
    if not isinstance(paths, Mapping) or not isinstance(recall, Mapping):
        raise ValueError("paths and recall configuration must be mappings")
    options = recall.get("popularity", {})
    if not isinstance(options, Mapping):
        raise ValueError("recall.popularity configuration must be a mapping")
    root = config_path.parent.resolve()
    output_root = resolve_path(str(paths["outputs_dir"]), root)
    config = PopularityRecallConfig(
        input_path=resolve_path(str(options.get("input_path", paths["unified_data"])), root),
        output_path=resolve_path(
            str(options.get("output_path", "outputs/recall_candidates/popularity_topk.csv")), root
        ),
        top_k=int(options.get("top_k", recall.get("top_k", 200))),
        click_weight=float(options.get("click_weight", 1.0)),
        conversion_weight=float(options.get("conversion_weight", 3.0)),
        chunk_size=int(options.get("chunk_size", 200_000)),
        user_id_column=str(options.get("user_id_column", "user_id")),
        product_id_column=str(options.get("product_id_column", "product_id")),
        conversion_label_column=str(options.get("conversion_label_column", "conversion_label")),
    )
    try:
        config.output_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError("recall.popularity.output_path must be within paths.outputs_dir") from error
    _validate_config(config)
    return config


def generate_popularity_candidates(config: PopularityRecallConfig) -> pd.DataFrame:
    """Aggregate weighted interactions in chunks and return deterministic global TopK."""

    scores: dict[str, float] = {}
    processed_rows = 0
    chunk_number = 0
    use_columns = [config.user_id_column, config.product_id_column, config.conversion_label_column]
    for input_file in _input_csv_files(config.input_path):
        for chunk in pd.read_csv(input_file, usecols=use_columns, chunksize=config.chunk_size, low_memory=False):
            chunk_number += 1
            LOGGER.info("Reading chunk %s from %s", chunk_number, input_file)
            processed_rows += len(chunk)
            chunk_scores = _score_chunk(chunk, config)
            for ad_id, score in chunk_scores.items():
                scores[ad_id] = scores.get(ad_id, 0.0) + float(score)
            LOGGER.info("Processed rows: %s; Unique ads: %s", processed_rows, len(scores))

    if not scores:
        LOGGER.info("Top popularity ads generated: 0")
        return pd.DataFrame(columns=OUTPUT_COLUMNS).astype(
            {"candidate_ad_id": "string", "popularity_score": "float64", "rank": "int64"}
        )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: config.top_k]
    candidates = pd.DataFrame(
        {
            "candidate_ad_id": [ad_id for ad_id, _ in ranked],
            "popularity_score": [score for _, score in ranked],
            "rank": range(1, len(ranked) + 1),
        },
        columns=OUTPUT_COLUMNS,
    ).astype({"candidate_ad_id": "string", "popularity_score": "float64", "rank": "int64"})
    LOGGER.info("Top popularity ads generated: %s", len(candidates))
    return candidates


def write_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    """Atomically write the documented global-popularity candidate schema."""

    if tuple(candidates.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Candidate output columns must be {OUTPUT_COLUMNS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    candidates.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    LOGGER.info("Wrote %s popularity candidates to %s", len(candidates), output_path)


def _score_chunk(chunk: pd.DataFrame, config: PopularityRecallConfig) -> dict[str, float]:
    """Calculate weighted score contributions for a single input chunk."""

    ads = chunk[config.product_id_column].astype("string").str.strip()
    labels = pd.to_numeric(chunk[config.conversion_label_column], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("conversion_label must be binary")
    valid_ads = ads.notna() & ads.ne("")
    weighted = pd.Series(config.click_weight, index=chunk.index, dtype="float64")
    weighted.loc[labels == 1] = config.conversion_weight
    grouped = pd.DataFrame({"ad_id": ads.loc[valid_ads], "weight": weighted.loc[valid_ads]}).groupby(
        "ad_id", sort=False
    )["weight"].sum()
    return {str(ad_id): float(score) for ad_id, score in grouped.items()}


def _input_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Popularity interaction input does not exist: {input_path}")
    files = sorted(input_path.glob("part-*.csv")) or sorted(
        path for path in input_path.glob("*.csv") if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No CSV interaction files found in: {input_path}")
    return files


def _validate_config(config: PopularityRecallConfig) -> None:
    if config.top_k <= 0 or config.chunk_size <= 0:
        raise ValueError("recall.popularity.top_k and chunk_size must be greater than zero")
    if config.click_weight < 0 or config.conversion_weight < 0:
        raise ValueError("recall.popularity weights must be non-negative")
    if not all((config.user_id_column, config.product_id_column, config.conversion_label_column)):
        raise ValueError("Popularity input column names must not be empty")


def main() -> None:
    """Run popularity recall as a standalone command."""

    parser = argparse.ArgumentParser(description="Generate global product-ad popularity candidates.")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_popularity_config(load_yaml_config(config_path), config_path)
    write_candidates(generate_popularity_candidates(config), config.output_path)


if __name__ == "__main__":
    main()
