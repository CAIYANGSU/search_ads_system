"""Tests for offline Two Tower and FAISS recall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import search_ads_system.recall.two_tower_recall as two_tower_recall

from search_ads_system.recall.faiss_index import (
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
    search_faiss_index,
)
from search_ads_system.recall.two_tower_recall import (
    OUTPUT_COLUMNS,
    TwoTowerModel,
    TwoTowerRecallConfig,
    generate_two_tower_candidates,
    load_checkpoint,
    prepare_training_data,
    run_two_tower_recall,
    save_checkpoint,
    stream_two_tower_candidates,
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


def test_existing_checkpoint_is_loaded_without_retraining(tmp_path) -> None:
    config = _config(tmp_path)
    model = TwoTowerModel(num_users=2, num_ads=3, embedding_dim=64)
    user_ids = np.asarray(["u1", "u2"])
    product_ids = np.asarray(["a1", "a2", "a3"])
    expected_users, expected_ads = model(torch.tensor([0]), torch.tensor([1]))
    save_checkpoint(model, config, user_ids, product_ids)
    restored = load_checkpoint(config, user_ids, product_ids, torch.device("cpu"))
    actual_users, actual_ads = restored(torch.tensor([0]), torch.tensor([1]))
    assert torch.allclose(actual_users, expected_users)
    assert torch.allclose(actual_ads, expected_ads)


def test_train_switch_defaults_to_inference(tmp_path) -> None:
    config = _config(tmp_path)
    assert config.train is False


def test_streaming_retrieval_writes_each_batch_without_candidate_dataframe(tmp_path, monkeypatch) -> None:
    class FakeIndex:
        ntotal = 3

    def fake_search(index, user_embeddings, top_k):
        assert index.ntotal == 3
        assert top_k == 2
        return (
            np.tile(np.asarray([0.9, 0.8], dtype=np.float32), (len(user_embeddings), 1)),
            np.tile(np.asarray([0, 1], dtype=np.int64), (len(user_embeddings), 1)),
        )

    monkeypatch.setattr(two_tower_recall, "search_faiss_index", fake_search)
    output_path = tmp_path / "two_tower_topk.csv"
    rows_written = stream_two_tower_candidates(
        TwoTowerModel(num_users=3, num_ads=3),
        np.asarray(["u1", "u2", "u3"]),
        np.asarray(["a1", "a2", "a3"]),
        FakeIndex(),
        output_path,
        top_k=2,
        search_batch_size=2,
        device=torch.device("cpu"),
    )
    written = pd.read_csv(output_path)
    assert rows_written == 6
    assert written.columns.tolist() == list(OUTPUT_COLUMNS)
    assert written.groupby("user_id")["rank"].apply(list).to_dict() == {
        "u1": [1, 2], "u2": [1, 2], "u3": [1, 2]
    }


def test_inference_mode_uses_checkpoint_and_existing_index_without_loading_interactions(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    model = TwoTowerModel(num_users=2, num_ads=3)
    user_ids = np.asarray(["u1", "u2"])
    product_ids = np.asarray(["a1", "a2", "a3"])
    save_checkpoint(model, config, user_ids, product_ids)
    calls = []

    def interactions_must_not_be_loaded(_config):
        raise AssertionError("inference mode must not load interactions")

    def existing_index(_model, actual_product_ids, _config, _device):
        calls.append("load_index")
        return object(), actual_product_ids

    def streamed(_model, actual_user_ids, actual_product_ids, _index, _output, **kwargs):
        calls.append((actual_user_ids.tolist(), actual_product_ids.tolist(), kwargs["search_batch_size"]))
        return 4

    monkeypatch.setattr(two_tower_recall, "load_interactions", interactions_must_not_be_loaded)
    monkeypatch.setattr(two_tower_recall, "load_or_build_faiss_index", existing_index)
    monkeypatch.setattr(two_tower_recall, "stream_two_tower_candidates", streamed)
    result = run_two_tower_recall(config)
    assert result.empty
    assert calls == ["load_index", (["u1", "u2"], ["a1", "a2", "a3"], 10_000)]


def test_faiss_index_can_be_built_and_searched() -> None:
    pytest.importorskip("faiss")
    index = build_faiss_index(np.eye(3, dtype=np.float32), "hnsw", hnsw_m=32, ef_construction=200, ef_search=64)
    scores, positions = search_faiss_index(
        index, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), top_k=2
    )
    assert index.ntotal == 3
    assert positions.shape == (2, 2)
    assert scores.shape == (2, 2)
    assert positions[0, 0] == 0
    assert scores[0, 0] == 1.0


def test_faiss_save_load_preserves_search_results(tmp_path) -> None:
    pytest.importorskip("faiss")
    embeddings = np.eye(4, dtype=np.float32)
    product_ids = np.asarray(["a1", "a2", "a3", "a4"])
    queries = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    index = build_faiss_index(embeddings, "hnsw", hnsw_m=32, ef_construction=200, ef_search=64)
    before_scores, before_positions = search_faiss_index(index, queries, top_k=2)
    path = tmp_path / "faiss_ad_index"
    save_faiss_index(index, product_ids, path)
    restored, restored_ids = load_faiss_index(path)
    after_scores, after_positions = search_faiss_index(restored, queries, top_k=2)
    assert restored_ids.tolist() == product_ids.tolist()
    assert np.array_equal(after_positions, before_positions)
    assert np.allclose(after_scores, before_scores)


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
