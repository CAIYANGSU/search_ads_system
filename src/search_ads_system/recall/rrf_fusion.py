"""Streaming Reciprocal Rank Fusion for recall candidate sources.

Popularity recall is a global list without a ``user_id`` column.  It is
therefore applied to every user present in either personalised recall source.
"""

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
OUTPUT_COLUMNS = ("user_id", "candidate_ad_id", "rrf_score", "source_count")
_PERSONALISED_SOURCES = ("itemcf", "two_tower")
_SOURCE_BITS = {"itemcf": 1, "two_tower": 2, "popularity": 4}


@dataclass(frozen=True)
class RRFFusionConfig:
    """Configuration for a streaming multi-source RRF fusion run."""

    itemcf_path: Path
    two_tower_path: Path
    popularity_path: Path
    output_path: Path
    k: int = 60
    weights: Mapping[str, float] | None = None
    top_k_per_user: int = 200
    chunk_size: int = 200_000

    def weight_for(self, source: str) -> float:
        """Return the configured RRF weight for one source."""

        return float((self.weights or {})[source])


def parse_rrf_config(raw_config: Mapping[str, Any], config_path: Path) -> RRFFusionConfig:
    """Read and validate the ``recall.rrf`` section from project configuration.

    ``rrf`` at the configuration root is also accepted for compatibility with
    compact, standalone fusion configurations.
    """

    try:
        paths = raw_config["paths"]
    except KeyError as error:
        raise ValueError("Configuration must define paths") from error
    if not isinstance(paths, Mapping):
        raise ValueError("paths configuration must be a mapping")
    recall = raw_config.get("recall", {})
    if not isinstance(recall, Mapping):
        raise ValueError("recall configuration must be a mapping")
    options = recall.get("rrf", raw_config.get("rrf", {}))
    if not isinstance(options, Mapping):
        raise ValueError("recall.rrf configuration must be a mapping")
    raw_weights = options.get("weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("recall.rrf.weights must be a mapping")

    root = config_path.parent.resolve()
    output_root = resolve_path(str(paths["outputs_dir"]), root)
    source_paths = {
        "itemcf": options.get("itemcf_path", "outputs/recall_candidates/itemcf_topk.csv"),
        "two_tower": options.get("two_tower_path", "outputs/recall_candidates/two_tower_topk.csv"),
        "popularity": options.get("popularity_path", "outputs/recall_candidates/popularity_topk.csv"),
    }
    config = RRFFusionConfig(
        itemcf_path=resolve_path(str(source_paths["itemcf"]), root),
        two_tower_path=resolve_path(str(source_paths["two_tower"]), root),
        popularity_path=resolve_path(str(source_paths["popularity"]), root),
        output_path=resolve_path(
            str(options.get("output_path", "outputs/recall_candidates/fused_candidates.csv")), root
        ),
        k=int(options.get("k", 60)),
        weights={
            "itemcf": float(raw_weights.get("itemcf", 1.0)),
            "two_tower": float(raw_weights.get("two_tower", 1.0)),
            "popularity": float(raw_weights.get("popularity", 0.5)),
        },
        top_k_per_user=int(options.get("top_k_per_user", 200)),
        chunk_size=int(options.get("chunk_size", 200_000)),
    )
    try:
        config.output_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError("recall.rrf.output_path must be within paths.outputs_dir") from error
    _validate_config(config)
    return config


def fuse_recall_candidates(config: RRFFusionConfig) -> pd.DataFrame:
    """Fuse ItemCF, Two Tower, and global popularity candidates using RRF.

    Input CSVs are consumed in chunks.  The in-memory accumulator contains
    only the merged candidate keys and their source bit masks, never an input
    file's full raw rows.
    """

    _validate_config(config)
    accumulator: dict[tuple[str, str], list[float | int]] = {}
    users: set[str] = set()
    for source in _PERSONALISED_SOURCES:
        path = getattr(config, f"{source}_path")
        rows = _load_personalised_source(path, source, config, accumulator, users)
        LOGGER.info("Loaded %s candidates: %s", source.replace("_", " "), rows)

    popularity_rows = _load_popularity_source(config, accumulator, users)
    LOGGER.info("Loaded popularity candidates: %s", popularity_rows)
    fused = _select_top_k_per_user(accumulator, config.top_k_per_user)
    LOGGER.info("Unique users: %s", len(users))
    LOGGER.info("Unique candidates: %s", len({candidate_id for _, candidate_id in accumulator}))
    LOGGER.info("Generated fused candidates: %s", len(fused))
    return fused


def write_fused_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    """Atomically write fused candidates using the documented output schema."""

    if tuple(candidates.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Fused output columns must be {OUTPUT_COLUMNS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    candidates.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    LOGGER.info("Wrote %s fused candidates to %s", len(candidates), output_path)


def _load_personalised_source(
    path: Path,
    source: str,
    config: RRFFusionConfig,
    accumulator: dict[tuple[str, str], list[float | int]],
    users: set[str],
) -> int:
    required_columns = ["user_id", "candidate_ad_id", "rank"]
    rows = 0
    try:
        reader = pd.read_csv(path, usecols=required_columns, chunksize=config.chunk_size, low_memory=False)
        for chunk in reader:
            rows += len(chunk)
            for user_id, candidate_id, rank in _normalise_rows(chunk, source):
                users.add(user_id)
                _add_rrf_score(accumulator, user_id, candidate_id, source, rank, config)
    except ValueError as error:
        raise ValueError(f"{source} candidates at {path} are missing required columns: {error}") from error
    return rows


def _load_popularity_source(
    config: RRFFusionConfig,
    accumulator: dict[tuple[str, str], list[float | int]],
    users: set[str],
) -> int:
    required_columns = ["candidate_ad_id", "rank"]
    rows = 0
    try:
        reader = pd.read_csv(config.popularity_path, usecols=required_columns, chunksize=config.chunk_size, low_memory=False)
        for chunk in reader:
            rows += len(chunk)
            for candidate_id, rank in _normalise_popularity_rows(chunk):
                for user_id in users:
                    _add_rrf_score(accumulator, user_id, candidate_id, "popularity", rank, config)
    except ValueError as error:
        raise ValueError(
            f"popularity candidates at {config.popularity_path} are missing required columns: {error}"
        ) from error
    return rows


def _normalise_rows(chunk: pd.DataFrame, source: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for user_id, candidate_id, rank in chunk.loc[:, ["user_id", "candidate_ad_id", "rank"]].itertuples(index=False):
        user = _normalise_id(user_id)
        candidate = _normalise_id(candidate_id)
        if user is None or candidate is None:
            continue
        rows.append((user, candidate, _normalise_rank(rank, source)))
    return rows


def _normalise_popularity_rows(chunk: pd.DataFrame) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for candidate_id, rank in chunk.loc[:, ["candidate_ad_id", "rank"]].itertuples(index=False):
        candidate = _normalise_id(candidate_id)
        if candidate is not None:
            rows.append((candidate, _normalise_rank(rank, "popularity")))
    return rows


def _normalise_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalised = str(value).strip()
    return normalised or None


def _normalise_rank(value: object, source: str) -> int:
    try:
        numeric_rank = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} rank must be a positive integer: {value!r}") from error
    if not numeric_rank.is_integer() or numeric_rank <= 0:
        raise ValueError(f"{source} rank must be a positive integer: {value!r}")
    return int(numeric_rank)


def _add_rrf_score(
    accumulator: dict[tuple[str, str], list[float | int]],
    user_id: str,
    candidate_id: str,
    source: str,
    rank: int,
    config: RRFFusionConfig,
) -> None:
    key = (user_id, candidate_id)
    score, source_mask = accumulator.get(key, [0.0, 0])
    bit = _SOURCE_BITS[source]
    # A source normally emits each pair only once.  Ignore repeated records so
    # source_count remains a count of routes rather than rows.
    if not int(source_mask) & bit:
        score = float(score) + config.weight_for(source) / (config.k + rank)
        source_mask = int(source_mask) | bit
        accumulator[key] = [score, source_mask]


def _select_top_k_per_user(
    accumulator: Mapping[tuple[str, str], list[float | int]], top_k: int
) -> pd.DataFrame:
    by_user: dict[str, list[tuple[str, float, int]]] = {}
    for (user_id, candidate_id), (score, source_mask) in accumulator.items():
        by_user.setdefault(user_id, []).append((candidate_id, float(score), int(source_mask).bit_count()))

    rows: list[tuple[str, str, float, int]] = []
    for user_id in sorted(by_user):
        ranked = sorted(by_user[user_id], key=lambda row: (-row[1], row[0]))[:top_k]
        rows.extend((user_id, candidate_id, score, source_count) for candidate_id, score, source_count in ranked)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).astype(
        {"user_id": "string", "candidate_ad_id": "string", "rrf_score": "float64", "source_count": "int64"}
    )


def _validate_config(config: RRFFusionConfig) -> None:
    if config.k < 0:
        raise ValueError("recall.rrf.k must be non-negative")
    if config.top_k_per_user <= 0 or config.chunk_size <= 0:
        raise ValueError("recall.rrf.top_k_per_user and chunk_size must be greater than zero")
    if config.weights is None or set(config.weights) != set(_SOURCE_BITS):
        raise ValueError("recall.rrf.weights must define itemcf, two_tower, and popularity")
    if any(weight < 0 for weight in config.weights.values()):
        raise ValueError("recall.rrf.weights must be non-negative")


def main() -> None:
    """Run RRF recall fusion as a standalone command."""

    parser = argparse.ArgumentParser(description="Fuse recall candidate sources using Reciprocal Rank Fusion.")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    config = parse_rrf_config(load_yaml_config(config_path), config_path)
    write_fused_candidates(fuse_recall_candidates(config), config.output_path)


if __name__ == "__main__":
    main()
