"""Tests for offline Two Tower and FAISS recall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from search_ads_system.recall.faiss_index import build_faiss_index, search_faiss_index
from search_ads_system.recall.two_tower_recall import (
    OUTPUT_COLUMNS,
    TwoTowerModel,
    TwoTowerRecallConfig,
    generate_two_tower_candidates,
    prepare_training_data,
    write_candidates,
)


def _config(tmp_path) -> TwoTowerRecallConfig:
    return TwoTowerRecallConfig(
        input_path=tmp_path / "input.csv", output_path=tmp_path / "two_tower_topk.csv",
        index_path=tmp_path / "faiss_ad_index", checkpoint_path=tmp_path / "checkpoint.pt", top_k=2,
        batch_size=2, inference_batch_size=2,
    )


def test_two_tower_forward_and_embedding_shapes() -> None:
    model = TwoTowerModel(num_users=3, num_ads=4, embedding_dim=64)
    user_vectors, ad_vectors = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
    assert user_vectors.shape == (2, 64)
    assert ad_vectors.shape == (2, 64)
    assert torch.allclose(user_vectors.norm(dim=1), torch.ones(2), atol=1e-5)


def test_faiss_index_can_be_built_and_searched() -> None:
    pytest.importorskip("faiss")
    index = build_faiss_index(np.eye(3, dtype=np.float32))
    scores, positions = search_faiss_index(index, np.array([[1, 0, 0]], dtype=np.float32), top_k=2)
    assert index.ntotal == 3
    assert positions.tolist() == [[0, 1]]
    assert scores[0, 0] == 1.0


def test_candidate_output_schema_and_seen_ad_filtering(tmp_path) -> None:
    pytest.importorskip("faiss")
    config = _config(tmp_path)
    interactions = pd.DataFrame(
        [("u1", "a1", 0), ("u1", "a2", 1), ("u2", "a2", 0)],
        columns=["user_id", "product_id", "conversion_label"],
    )
    _, _, _, users, ads, histories, stats = prepare_training_data(interactions, config)
    assert stats["positive_samples"] == 1
    assert stats["negative_samples"] == 5
    index = build_faiss_index(np.eye(3, 64, dtype=np.float32))
    candidates = generate_two_tower_candidates(users, np.eye(2, 64, dtype=np.float32), ads, index, histories, 2)
    assert tuple(candidates.columns) == OUTPUT_COLUMNS
    assert not set(candidates.loc[candidates["user_id"] == "u1", "candidate_ad_id"]).intersection({"a1", "a2"})
    write_candidates(candidates, config.output_path)
    assert pd.read_csv(config.output_path).columns.tolist() == list(OUTPUT_COLUMNS)
