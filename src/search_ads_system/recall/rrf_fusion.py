"""Memory-bounded Reciprocal Rank Fusion for recall candidate sources.

Popularity recall is a global list without a ``user_id`` column.  It is
applied to each selected user from the personalised recall sources.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import logging
from collections.abc import Iterator, Mapping
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
    """Configuration for a memory-bounded multi-source RRF fusion run."""

    itemcf_path: Path
    two_tower_path: Path
    popularity_path: Path
    output_path: Path
    k: int = 60
    weights: Mapping[str, float] | None = None
    top_k_per_user: int = 200
    max_users: int = 500_000
    chunk_size: int = 200_000

    def weight_for(self, source: str) -> float:
        return float((self.weights or {})[source])


def parse_rrf_config(raw_config: Mapping[str, Any], config_path: Path) -> RRFFusionConfig:
    """Read and validate the ``recall.rrf`` configuration section."""

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
    config = RRFFusionConfig(
        itemcf_path=resolve_path(
            str(options.get("itemcf_path", "outputs/recall_candidates/itemcf_topk.csv")), root
        ),
        two_tower_path=resolve_path(
            str(options.get("two_tower_path", "outputs/recall_candidates/two_tower_topk.csv")), root
        ),
        popularity_path=resolve_path(
            str(options.get("popularity_path", "outputs/recall_candidates/popularity_topk.csv")), root
        ),
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
        max_users=int(options.get("max_users", 500_000)),
        chunk_size=int(options.get("chunk_size", 200_000)),
    )
    try:
        config.output_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError("recall.rrf.output_path must be within paths.outputs_dir") from error
    _validate_config(config)
    return config


def fuse_recall_candidates(config: RRFFusionConfig) -> pd.DataFrame:
    """Return fused candidates as a dataframe for small programmatic workloads.

    Production should call :func:`fuse_and_write_candidates`, which writes a
    completed user immediately instead of retaining the entire output.
    """

    rows = list(_iter_fused_rows(config))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).astype(
        {"user_id": "string", "candidate_ad_id": "string", "rrf_score": "float64", "source_count": "int64"}
    )


def fuse_and_write_candidates(config: RRFFusionConfig) -> int:
    """Fuse selected users and atomically write rows without global candidate state."""

    _validate_config(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config.output_path.with_suffix(config.output_path.suffix + ".tmp")
    row_count = 0
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)
        for row in _iter_fused_rows(config):
            writer.writerow(row)
            row_count += 1
    temporary_path.replace(config.output_path)
    LOGGER.info("Wrote %s fused candidates to %s", row_count, config.output_path)
    return row_count


def write_fused_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    """Atomically write a small dataframe using the documented output schema."""

    if tuple(candidates.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Fused output columns must be {OUTPUT_COLUMNS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    candidates.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def _iter_fused_rows(config: RRFFusionConfig) -> Iterator[tuple[str, str, float, int]]:
    """Yield final Top-K rows one user at a time.

    Existing ItemCF and Two Tower outputs are sorted by ``user_id``.  This
    lets fusion retain no more than one user's candidates plus popularity.
    """

    _validate_config(config)
    selected_users, total_users, source_row_counts = _select_users(config)
    LOGGER.info("Selected users=%s/%s", len(selected_users), total_users)
    for source in _PERSONALISED_SOURCES:
        LOGGER.info("Loaded %s candidates: %s", source.replace("_", " "), source_row_counts[source])

    popularity = _load_popularity_candidates(config)
    LOGGER.info("Loaded popularity candidates: %s", len(popularity))
    LOGGER.info("Unique users: %s", len(selected_users))

    grouped_sources = [
        _iter_selected_user_groups(getattr(config, f"{source}_path"), source, selected_users, config)
        for source in _PERSONALISED_SOURCES
    ]
    heap: list[tuple[str, int, list[tuple[str, int]], Iterator[tuple[str, list[tuple[str, int]]]]]] = []
    for source_index, groups in enumerate(grouped_sources):
        try:
            user_id, candidates = next(groups)
        except StopIteration:
            continue
        heapq.heappush(heap, (user_id, source_index, candidates, groups))

    unique_candidates: set[str] = set()
    generated_rows = 0
    while heap:
        user_id = heap[0][0]
        source_candidates: dict[str, list[tuple[str, int]]] = {}
        while heap and heap[0][0] == user_id:
            _, source_index, candidates, groups = heapq.heappop(heap)
            source_candidates[_PERSONALISED_SOURCES[source_index]] = candidates
            try:
                next_user_id, next_candidates = next(groups)
            except StopIteration:
                continue
            heapq.heappush(heap, (next_user_id, source_index, next_candidates, groups))
        for row in _fuse_one_user(user_id, source_candidates, popularity, config):
            unique_candidates.add(row[1])
            generated_rows += 1
            yield row
    LOGGER.info("Unique candidates: %s", len(unique_candidates))
    LOGGER.info("Generated fused candidates: %s", generated_rows)


def _select_users(config: RRFFusionConfig) -> tuple[set[str], int, dict[str, int]]:
    """Scan user IDs in chunks and deterministically choose at most max_users."""

    all_users: set[str] = set()
    source_row_counts: dict[str, int] = {}
    for source in _PERSONALISED_SOURCES:
        path = getattr(config, f"{source}_path")
        row_count = 0
        try:
            for chunk in pd.read_csv(path, usecols=["user_id"], chunksize=config.chunk_size, low_memory=False):
                row_count += len(chunk)
                users = chunk["user_id"].astype("string").str.strip().dropna()
                all_users.update(str(user_id) for user_id in users if user_id)
        except ValueError as error:
            raise ValueError(f"{source} candidates at {path} are missing user_id: {error}") from error
        source_row_counts[source] = row_count
    total_users = len(all_users)
    return set(sorted(all_users)[: config.max_users]), total_users, source_row_counts


def _load_popularity_candidates(config: RRFFusionConfig) -> list[tuple[str, int]]:
    """Read the small global popularity list in chunks."""

    candidates: list[tuple[str, int]] = []
    try:
        reader = pd.read_csv(
            config.popularity_path,
            usecols=["candidate_ad_id", "rank"],
            chunksize=config.chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            for candidate_id, rank in chunk.itertuples(index=False):
                candidate = _normalise_id(candidate_id)
                if candidate is not None:
                    candidates.append((candidate, _normalise_rank(rank, "popularity")))
    except ValueError as error:
        raise ValueError(
            f"popularity candidates at {config.popularity_path} are missing required columns: {error}"
        ) from error
    return candidates


def _iter_selected_user_groups(
    path: Path,
    source: str,
    selected_users: set[str],
    config: RRFFusionConfig,
) -> Iterator[tuple[str, list[tuple[str, int]]]]:
    """Yield one selected, contiguous user group at a time from a CSV source."""

    current_user: str | None = None
    current_candidates: list[tuple[str, int]] = []
    previous_user: str | None = None
    try:
        reader = pd.read_csv(
            path,
            usecols=["user_id", "candidate_ad_id", "rank"],
            chunksize=config.chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            for raw_user_id, raw_candidate_id, raw_rank in chunk.itertuples(index=False):
                user_id = _normalise_id(raw_user_id)
                candidate_id = _normalise_id(raw_candidate_id)
                if user_id is None or candidate_id is None or user_id not in selected_users:
                    continue
                if current_user is None:
                    current_user = user_id
                elif user_id != current_user:
                    if previous_user is not None and current_user <= previous_user:
                        raise ValueError(f"{source} candidates must be grouped by ascending user_id for streaming fusion")
                    yield current_user, current_candidates
                    previous_user = current_user
                    current_user = user_id
                    current_candidates = []
                current_candidates.append((candidate_id, _normalise_rank(raw_rank, source)))
    except ValueError as error:
        if "must be grouped" in str(error):
            raise
        raise ValueError(f"{source} candidates at {path} are missing required columns: {error}") from error
    if current_user is not None:
        if previous_user is not None and current_user <= previous_user:
            raise ValueError(f"{source} candidates must be grouped by ascending user_id for streaming fusion")
        yield current_user, current_candidates


def _fuse_one_user(
    user_id: str,
    source_candidates: Mapping[str, list[tuple[str, int]]],
    popularity: list[tuple[str, int]],
    config: RRFFusionConfig,
) -> list[tuple[str, str, float, int]]:
    scores: dict[str, list[float | int]] = {}
    for candidate_id, rank in popularity:
        _add_rrf_score(scores, candidate_id, "popularity", rank, config)
    for source, candidates in source_candidates.items():
        for candidate_id, rank in candidates:
            _add_rrf_score(scores, candidate_id, source, rank, config)
    ranked = sorted(
        ((candidate_id, float(score), int(mask).bit_count()) for candidate_id, (score, mask) in scores.items()),
        key=lambda row: (-row[1], row[0]),
    )[: config.top_k_per_user]
    return [(user_id, candidate_id, score, source_count) for candidate_id, score, source_count in ranked]


def _add_rrf_score(
    scores: dict[str, list[float | int]], candidate_id: str, source: str, rank: int, config: RRFFusionConfig
) -> None:
    score, source_mask = scores.get(candidate_id, [0.0, 0])
    bit = _SOURCE_BITS[source]
    if not int(source_mask) & bit:
        scores[candidate_id] = [
            float(score) + config.weight_for(source) / (config.k + rank),
            int(source_mask) | bit,
        ]


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


def _validate_config(config: RRFFusionConfig) -> None:
    if config.k < 0:
        raise ValueError("recall.rrf.k must be non-negative")
    if config.top_k_per_user <= 0 or config.max_users <= 0 or config.chunk_size <= 0:
        raise ValueError("recall.rrf.top_k_per_user, max_users, and chunk_size must be greater than zero")
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
    fuse_and_write_candidates(parse_rrf_config(load_yaml_config(config_path), config_path))


if __name__ == "__main__":
    main()
