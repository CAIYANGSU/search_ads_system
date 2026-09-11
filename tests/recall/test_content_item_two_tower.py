"""Tests for the strict-temporal, history-free content Item Tower ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import search_ads_system.recall.content_item_two_tower as module
from search_ads_system.recall.content_item_two_tower import (
    OOV_INDEX,
    PAD_INDEX,
    CatalogFeatures,
    ContentItemTwoTowerConfig,
    ContentItemTwoTowerModel,
    evaluate_content_item_candidates,
    extract_catalog_embeddings,
    fingerprint_mismatch_reasons,
    prepare_content_item_data,
    preprocess_prices,
    stream_content_item_candidates,
    train_variant,
)


def _write_past(path: Path, rows: list[dict[str, object]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "part-00000.csv", index=False)
    return path


def _base_rows() -> list[dict[str, object]]:
    return [
        {
            "user_id": "u1", "product_id": "p1", "conversion_label": 0, "click_timestamp": 10,
            "product_brand": "b1", "product_category_1": "c1", "product_category_2": "c2",
            "product_category_3": "c3", "product_category_4": "c4", "partner_id": "s1",
            "product_price": 10.0,
        },
        {
            "user_id": "u2", "product_id": "p2", "conversion_label": 1, "click_timestamp": 20,
            "product_brand": "b2", "product_category_1": "d1", "product_category_2": "d2",
            "product_category_3": "d3", "product_category_4": "d4", "partner_id": "s2",
            "product_price": 20.0,
        },
        {
            "user_id": "u1", "product_id": "p2", "conversion_label": 0, "click_timestamp": 30,
            "product_brand": "b2", "product_category_1": "d1", "product_category_2": "d2",
            "product_category_3": "d3", "product_category_4": "d4", "partner_id": "s2",
            "product_price": 21.0,
        },
    ]


def _config(tmp_path: Path, *, external: Path | None = None) -> ContentItemTwoTowerConfig:
    past = _write_past(tmp_path / "past", _base_rows())
    future = _write_past(
        tmp_path / "future_a",
        [
            {"user_id": "u1", "product_id": "p2", "conversion_label": 0, "click_timestamp": 110},
            {"user_id": "u2", "product_id": "p1", "conversion_label": 0, "click_timestamp": 111},
        ],
    )
    return ContentItemTwoTowerConfig(
        input_path=past,
        future_path=future,
        output_dir=tmp_path / "run",
        split_timestamp=100,
        product_catalog_path=external,
        product_catalog_as_of_timestamp=90 if external else None,
        embedding_dim=4,
        feature_embedding_dim=3,
        price_embedding_dim=2,
        hidden_dims=(8,),
        batch_size=2,
        epochs=1,
        negative_samples=1,
        top_k=1,
        search_batch_size=1,
        inference_batch_size=2,
        input_chunk_size=2,
        log_every_rows=100,
        faiss_index_type="flat",
    )


def test_pad_and_oov_are_reserved_and_unseen_product_uses_oov(tmp_path: Path) -> None:
    external = tmp_path / "catalog.csv"
    pd.DataFrame([
        {"product_id": "cold", "product_brand": "b1", "product_price": 12.0}
    ]).to_csv(external, index=False)
    config = _config(tmp_path, external=external)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    cold = data.product_to_catalog_position["cold"]
    assert PAD_INDEX == 0 and OOV_INDEX == 1 and PAD_INDEX != OOV_INDEX
    assert data.product_vocab[:2] == ("__PAD__", "__OOV__")
    assert data.catalog.product_id_indices[cold] == OOV_INDEX
    assert all(vocab[:2] == ("__PAD__", "__OOV__") for vocab in data.categorical_vocabs.values())


def test_different_oov_products_use_side_information(tmp_path: Path) -> None:
    external = tmp_path / "catalog.csv"
    pd.DataFrame([
        {"product_id": "cold-a", "product_brand": "b1", "product_category_1": "c1", "partner_id": "s1", "product_price": 11.0},
        {"product_id": "cold-b", "product_brand": "b2", "product_category_1": "d1", "partner_id": "s2", "product_price": 19.0},
    ]).to_csv(external, index=False)
    config = _config(tmp_path, external=external)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    model = ContentItemTwoTowerModel(data, "content_item")
    positions = torch.tensor([data.product_to_catalog_position["cold-a"], data.product_to_catalog_position["cold-b"]])
    vectors = module._encode_catalog_positions(model, data.catalog, positions, torch.device("cpu"))
    assert (data.catalog.product_id_indices[positions.numpy()] == OOV_INDEX).all()
    assert not torch.allclose(vectors[0], vectors[1])


def test_price_nan_inf_and_negative_are_explicitly_sanitized() -> None:
    values, counts, valid = preprocess_prices(pd.Series([np.nan, np.inf, -np.inf, -3.0, 9.0, "bad"]))
    assert np.isfinite(values).all()
    assert values.tolist()[:4] == [0.0, 0.0, 0.0, 0.0]
    assert counts == {"nan_or_non_numeric": 2, "positive_inf": 1, "negative_inf": 1, "negative": 1}
    assert valid.tolist() == [False, False, False, True, True, False]


def test_unordered_catalogue_uses_latest_value_before_cutoff(tmp_path: Path) -> None:
    rows = _base_rows() + [
        {**_base_rows()[0], "product_brand": "latest", "product_price": 15.0, "click_timestamp": 80},
        {**_base_rows()[0], "product_brand": "future", "product_price": 99.0, "click_timestamp": 101},
        {**_base_rows()[0], "product_brand": "middle", "product_price": 12.0, "click_timestamp": 50},
    ]
    config = _config(tmp_path)
    _write_past(config.input_path, rows)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    position = data.product_to_catalog_position["p1"]
    brand_index = data.catalog.categorical_indices["product_brand"][position]
    assert data.categorical_vocabs["product_brand"][brand_index] == "latest"
    expected = (np.log1p(15.0) - data.catalog.price_mean) / data.catalog.price_std
    assert data.catalog.normalized_log_prices[position] == pytest.approx(expected)


def test_past_only_catalogue_does_not_claim_cold_retrieval(tmp_path: Path) -> None:
    config = _config(tmp_path)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    assert data.metadata["catalog_mode"] == "past_only"
    assert data.metadata["searchable_cold_products"] == 0


def test_external_catalogue_cold_item_is_encoded(tmp_path: Path) -> None:
    external = tmp_path / "catalog.csv"
    pd.DataFrame([{"product_id": "cold", "product_brand": "b1", "product_price": 12.0}]).to_csv(external, index=False)
    config = _config(tmp_path, external=external)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    model = ContentItemTwoTowerModel(data, "content_item")
    position = data.product_to_catalog_position["cold"]
    vector = module._encode_catalog_positions(model, data.catalog, torch.tensor([position]), torch.device("cpu"))
    assert data.metadata["searchable_cold_products"] == 1
    assert data.catalog.product_id_indices[position] == OOV_INDEX
    assert torch.isfinite(vector).all() and vector.norm(dim=1).item() == pytest.approx(1.0, abs=1e-5)


def test_fingerprint_change_forces_rebuild_reason() -> None:
    expected = {
        "checkpoint_fingerprint": "new", "model_config_fingerprint": "m", "item_vocab_fingerprint": "i",
        "catalogue_fingerprint": "c", "embedding_dim": 4, "normalized": True,
    }
    actual = {**expected, "checkpoint_fingerprint": "old", "catalogue_fingerprint": "old-c"}
    assert fingerprint_mismatch_reasons(expected, actual) == ["checkpoint_fingerprint", "catalogue_fingerprint"]


def test_changed_fingerprint_rebuilds_existing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    model = ContentItemTwoTowerModel(data, "id_only")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    index_path = tmp_path / "index"
    index_path.write_bytes(b"old-index")
    expected = module._expected_index_metadata(model, data, "new-checkpoint")
    old = {**expected, "checkpoint_fingerprint": "old-checkpoint"}
    calls: list[str] = []

    class Index:
        ntotal = len(data.catalog.product_ids)

    monkeypatch.setattr(module, "load_faiss_index_metadata", lambda _path: old)
    monkeypatch.setattr(module, "extract_catalog_embeddings", lambda *_args: np.eye(len(data.catalog.product_ids), config.embedding_dim, dtype=np.float32))
    monkeypatch.setattr(module, "build_faiss_index", lambda *_args, **_kwargs: calls.append("build") or Index())
    monkeypatch.setattr(module, "save_faiss_index", lambda *_args, **_kwargs: calls.append("save"))
    _, _, rebuilt, reasons = module.load_or_build_fingerprinted_index(
        model, data, checkpoint, "new-checkpoint", index_path, torch.device("cpu")
    )
    assert rebuilt is True and reasons == ["checkpoint_fingerprint"]
    assert calls == ["build", "save"]


def test_streaming_retrieval_schema_and_past_seen_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    model = ContentItemTwoTowerModel(data, "id_only")
    p1 = data.product_to_catalog_position["p1"]
    p2 = data.product_to_catalog_position["p2"]

    class Index:
        ntotal = 2

    def fake_search(_index: object, queries: np.ndarray, _top_k: int) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray([[p1, p2]] if len(queries) == 1 else [[p1, p2], [p2, p1]], dtype=np.int64)
        return np.asarray([[1.0, 0.5]] * len(queries), dtype=np.float32), positions[: len(queries)]

    monkeypatch.setattr(module, "search_faiss_index", fake_search)
    output = tmp_path / "candidates.csv"
    result = stream_content_item_candidates(model, data, Index(), output, torch.device("cpu"))
    frame = pd.read_csv(output)
    assert frame.columns.tolist() == list(module.OUTPUT_COLUMNS)
    assert result["candidate_rows_written"] == 1
    assert frame.loc[frame.user_id == "u1", "candidate_ad_id"].tolist() == []  # u1 saw both Past products
    assert frame.loc[frame.user_id == "u2", "candidate_ad_id"].tolist() == ["p1"]


def test_catalog_coverage_uses_searchable_catalog_denominator(tmp_path: Path) -> None:
    config = _config(tmp_path)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame([("u1", "p2", 0.9, 1)], columns=module.OUTPUT_COLUMNS).to_csv(candidates, index=False)
    metrics = evaluate_content_item_candidates(candidates, config.future_path, data)
    assert metrics["catalog_coverage"] == 0.5
    assert metrics["unique_recalled_items"] == 1
    assert metrics["cold_recall@100"] is None


def test_training_and_embedding_export_are_finite_and_history_free(tmp_path: Path) -> None:
    config = _config(tmp_path)
    data = prepare_content_item_data(config, tmp_path / "state.sqlite")
    model, losses, _ = train_variant(data, "content_item", torch.device("cpu"))
    embeddings = extract_catalog_embeddings(model, data.catalog, 2, torch.device("cpu"))
    assert len(losses) == 1 and np.isfinite(losses).all()
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert not hasattr(model, "history_indices")
    checkpoint = tmp_path / "content-item.pt"
    fingerprint = module.save_variant_checkpoint(model, data, checkpoint, losses)
    restored, payload, restored_fingerprint = module.load_variant_checkpoint(
        data, "content_item", checkpoint, torch.device("cpu")
    )
    assert fingerprint == restored_fingerprint
    assert payload["product_vocab"][:2] == ["__PAD__", "__OOV__"]
    assert payload["categorical_vocabs"]["product_brand"][:2] == ["__PAD__", "__OOV__"]
    assert not hasattr(restored, "history_indices")
