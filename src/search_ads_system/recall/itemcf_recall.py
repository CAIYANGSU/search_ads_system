"""Sparse ItemCF candidate generation for product-ad interactions.

The module intentionally covers only the ItemCF recall route.  It does not
build ANN indices, fuse recall routes, or rank candidates.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from search_ads_system.common.config import load_yaml_config, resolve_path

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ("user_id", "candidate_ad_id", "itemcf_score", "rank")


@dataclass(frozen=True)
class ItemCFRecallConfig:
    """Configuration for one offline ItemCF candidate-generation run."""

    input_path: Path
    output_path: Path
    user_id_column: str
    item_id_column: str
    interaction_label_column: str
    top_k: int
    label_weights: Mapping[str, float]
    default_interaction_weight: float
    interaction_aggregation: str
    similarity: str
    input_chunk_size: int


def parse_itemcf_config(raw_config: Mapping[str, Any], config_path: Path) -> ItemCFRecallConfig:
    """Read and validate the ``recall.itemcf`` part of a project config."""

    try:
        recall_config = raw_config["recall"]
        paths_config = raw_config["paths"]
    except KeyError as error:
        raise ValueError("Configuration must define paths and recall") from error
    if not isinstance(recall_config, Mapping):
        raise ValueError("recall configuration must be a mapping")

    itemcf = recall_config.get("itemcf", {})
    if not isinstance(itemcf, Mapping):
        raise ValueError("recall.itemcf configuration must be a mapping")
    config_directory = config_path.parent.resolve()
    output_root = resolve_path(str(paths_config["outputs_dir"]), config_directory)
    input_path = resolve_path(str(itemcf.get("input_path", paths_config["unified_data"])), config_directory)
    output_path = resolve_path(
        str(itemcf.get("output_path", "outputs/recall_candidates/itemcf_topk.csv")), config_directory
    )
    try:
        output_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError("recall.itemcf.output_path must be within paths.outputs_dir") from error

    raw_weights = itemcf.get("interaction_label_weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("recall.itemcf.interaction_label_weights must be a mapping")
    label_weights = {_normalise_label_key(key): float(value) for key, value in raw_weights.items()}
    if any(weight < 0 for weight in label_weights.values()):
        raise ValueError("recall.itemcf.interaction_label_weights must be non-negative")

    config = ItemCFRecallConfig(
        input_path=input_path,
        output_path=output_path,
        user_id_column=str(itemcf.get("user_id_column", "user_id")),
        item_id_column=str(itemcf.get("item_id_column", "ad_id")),
        interaction_label_column=str(itemcf.get("interaction_label_column", "interaction_label")),
        top_k=int(itemcf.get("top_k", recall_config.get("top_k", 200))),
        label_weights=label_weights,
        default_interaction_weight=float(itemcf.get("default_interaction_weight", 1.0)),
        interaction_aggregation=str(itemcf.get("interaction_aggregation", "sum")).lower(),
        similarity=str(itemcf.get("similarity", "cosine")).lower(),
        input_chunk_size=int(itemcf.get("input_chunk_size", 200_000)),
    )
    _validate_config(config)
    return config


def load_interactions(config: ItemCFRecallConfig) -> pd.DataFrame:
    """Load only the three configured interaction fields from CSV input.

    ``input_path`` can be one CSV file or a directory of CSV parts.  Directory
    files are read in name order, so pipeline results remain reproducible.
    """

    required_columns = [config.user_id_column, config.item_id_column, config.interaction_label_column]
    files = _input_csv_files(config.input_path)
    frames: list[pd.DataFrame] = []
    for file_path in files:
        LOGGER.info("Reading interactions from %s", file_path)
        try:
            reader: Iterable[pd.DataFrame] = pd.read_csv(
                file_path,
                usecols=required_columns,
                chunksize=config.input_chunk_size,
                low_memory=False,
            )
            frames.extend(reader)
        except ValueError as error:
            raise ValueError(f"Interaction input {file_path} is missing a required column: {error}") from error
    if not frames:
        return pd.DataFrame(columns=required_columns)
    return pd.concat(frames, ignore_index=True)


def prepare_interactions(
    interactions: pd.DataFrame,
    config: ItemCFRecallConfig,
) -> pd.DataFrame:
    """Validate, weight, and aggregate duplicate user-item interaction records."""

    required_columns = {config.user_id_column, config.item_id_column, config.interaction_label_column}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        raise ValueError(f"Interaction data is missing required columns: {sorted(missing_columns)}")

    prepared = interactions.loc[
        :, [config.user_id_column, config.item_id_column, config.interaction_label_column]
    ].copy()
    prepared.columns = ["user_id", "item_id", "interaction_label"]
    raw_interaction_count = len(prepared)
    prepared["user_id"] = prepared["user_id"].astype("string").str.strip()
    prepared["item_id"] = prepared["item_id"].astype("string").str.strip()
    has_required_ids = (
        prepared[["user_id", "item_id"]].notna().all(axis=1)
        & prepared[["user_id", "item_id"]].ne("").all(axis=1)
    )
    dropped_interaction_count = int((~has_required_ids).sum())
    prepared = prepared.loc[has_required_ids].copy()
    LOGGER.info(
        "Interaction ID cleaning: raw_interactions=%s dropped_interactions=%s remaining_interactions=%s",
        raw_interaction_count,
        dropped_interaction_count,
        len(prepared),
    )

    label_keys = prepared["interaction_label"].map(_normalise_label_key)
    prepared["interaction_weight"] = label_keys.map(config.label_weights).fillna(
        config.default_interaction_weight
    )
    prepared["interaction_weight"] = pd.to_numeric(prepared["interaction_weight"], errors="raise")
    prepared = prepared.loc[prepared["interaction_weight"] > 0, ["user_id", "item_id", "interaction_weight"]]
    if prepared.empty:
        return prepared.reset_index(drop=True)

    aggregated = (
        prepared.groupby(["user_id", "item_id"], as_index=False, sort=True)["interaction_weight"]
        .agg(config.interaction_aggregation)
        .sort_values(["user_id", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    return aggregated


def build_user_item_matrix(interactions: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build a weighted CSR user-by-item interaction matrix."""

    required_columns = {"user_id", "item_id", "interaction_weight"}
    if missing_columns := required_columns - set(interactions.columns):
        raise ValueError(f"Prepared interactions are missing columns: {sorted(missing_columns)}")
    if interactions.empty:
        return sparse.csr_matrix((0, 0), dtype=np.float64), np.array([], dtype=str), np.array([], dtype=str)

    user_codes, user_ids = pd.factorize(interactions["user_id"], sort=True)
    item_codes, item_ids = pd.factorize(interactions["item_id"], sort=True)
    matrix = sparse.coo_matrix(
        (
            interactions["interaction_weight"].to_numpy(dtype=np.float64),
            (user_codes, item_codes),
        ),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, user_ids.astype(str).to_numpy(), item_ids.astype(str).to_numpy()


def compute_item_similarity(matrix: sparse.csr_matrix, similarity: str = "cosine") -> sparse.csr_matrix:
    """Compute sparse item-item cosine similarity from a user-item matrix."""

    if similarity != "cosine":
        raise ValueError(f"Unsupported ItemCF similarity: {similarity}. Only cosine is currently supported.")
    if matrix.shape[1] == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    cooccurrence = (matrix.T @ matrix).tocsr()
    norms = np.sqrt(cooccurrence.diagonal())
    inverse_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    similarity_matrix = (
        sparse.diags(inverse_norms) @ cooccurrence @ sparse.diags(inverse_norms)
    ).tocsr()
    similarity_matrix.setdiag(0.0)
    similarity_matrix.eliminate_zeros()
    return similarity_matrix


def recall_top_k(
    user_item_matrix: sparse.csr_matrix,
    item_similarity: sparse.csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """Score unseen items from each user's history and return deterministic Top-K rows."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if user_item_matrix.shape != (len(user_ids), len(item_ids)):
        raise ValueError("user_item_matrix dimensions do not match supplied user_ids and item_ids")
    if item_similarity.shape != (len(item_ids), len(item_ids)):
        raise ValueError("item_similarity dimensions do not match supplied item_ids")

    rows: list[tuple[str, str, float, int]] = []
    for user_index, user_id in enumerate(user_ids):
        score_row = (user_item_matrix.getrow(user_index) @ item_similarity).tocsr()
        if score_row.nnz == 0:
            continue
        seen_items = user_item_matrix.indices[
            user_item_matrix.indptr[user_index] : user_item_matrix.indptr[user_index + 1]
        ]
        candidate_indices = score_row.indices
        candidate_scores = score_row.data
        unseen = ~np.isin(candidate_indices, seen_items, assume_unique=True)
        candidate_indices = candidate_indices[unseen]
        candidate_scores = candidate_scores[unseen]
        if not len(candidate_indices):
            continue

        # Item ID is the deterministic secondary key when ItemCF scores tie.
        order = np.lexsort((item_ids[candidate_indices], -candidate_scores))[:top_k]
        for rank, candidate_position in enumerate(order, start=1):
            item_index = candidate_indices[candidate_position]
            rows.append(
                (str(user_id), str(item_ids[item_index]), float(candidate_scores[candidate_position]), rank)
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def generate_itemcf_candidates(
    interactions: pd.DataFrame,
    config: ItemCFRecallConfig,
) -> pd.DataFrame:
    """Run the complete ItemCF flow from interaction records to candidates."""

    prepared = prepare_interactions(interactions, config)
    LOGGER.info(
        "Prepared %s user-item interactions from %s raw records", len(prepared), len(interactions)
    )
    matrix, user_ids, item_ids = build_user_item_matrix(prepared)
    LOGGER.info("Built user-item matrix with users=%s items=%s nnz=%s", *matrix.shape, matrix.nnz)
    similarity = compute_item_similarity(matrix, config.similarity)
    LOGGER.info("Built item similarity matrix with nnz=%s", similarity.nnz)
    candidates = recall_top_k(matrix, similarity, user_ids, item_ids, config.top_k)
    return candidates.astype(
        {"user_id": "string", "candidate_ad_id": "string", "itemcf_score": "float64", "rank": "int64"}
    )


def write_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    """Atomically write ItemCF candidates using the documented CSV schema."""

    if tuple(candidates.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Candidate output columns must be {OUTPUT_COLUMNS}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    candidates.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    LOGGER.info("Wrote %s ItemCF candidates to %s", len(candidates), output_path)


def _input_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"ItemCF interaction input does not exist: {input_path}")
    part_files = sorted(input_path.glob("part-*.csv"))
    files = part_files or sorted(path for path in input_path.glob("*.csv") if not path.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No CSV interaction files found in: {input_path}")
    return files


def _normalise_label_key(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


def _validate_config(config: ItemCFRecallConfig) -> None:
    if config.top_k <= 0:
        raise ValueError("recall.itemcf.top_k must be greater than zero")
    if config.input_chunk_size <= 0:
        raise ValueError("recall.itemcf.input_chunk_size must be greater than zero")
    if config.default_interaction_weight < 0:
        raise ValueError("recall.itemcf.default_interaction_weight must be non-negative")
    if config.interaction_aggregation not in {"sum", "max"}:
        raise ValueError("recall.itemcf.interaction_aggregation must be 'sum' or 'max'")
    if config.similarity != "cosine":
        raise ValueError("recall.itemcf.similarity must be 'cosine'")
    if not all((config.user_id_column, config.item_id_column, config.interaction_label_column)):
        raise ValueError("ItemCF input column names must not be empty")


def main() -> None:
    """Run ItemCF candidate generation as a standalone command."""

    parser = argparse.ArgumentParser(description="Generate product-ad candidates with ItemCF.")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", type=Path, default=project_root / "config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_path = args.config.resolve()
    config = parse_itemcf_config(load_yaml_config(config_path), config_path)
    interactions = load_interactions(config)
    candidates = generate_itemcf_candidates(interactions, config)
    write_candidates(candidates, config.output_path)


if __name__ == "__main__":
    main()
