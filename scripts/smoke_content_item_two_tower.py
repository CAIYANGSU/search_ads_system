"""Small strict-temporal smoke test for the history-free Item Tower ablation.

When FAISS is unavailable this script uses an exact NumPy inner-product index;
all model, streaming-output, seen-filter, and evaluation paths remain the same.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import search_ads_system.recall.content_item_two_tower as content_item  # noqa: E402


class ExactNumpyIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.ntotal = len(vectors)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        return np.take_along_axis(scores, order, axis=1).astype(np.float32), order.astype(np.int64)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="content-item-smoke-") as temporary:
        root = Path(temporary)
        past, future = root / "past", root / "future_a"
        past.mkdir(); future.mkdir()
        past_rows = []
        future_rows = []
        users = 1_000
        categories = 50
        for number in range(users):
            category = number % categories
            side = (number // categories) % 2
            product = f"p-{category:02d}-{side}"
            target = f"p-{category:02d}-{1-side}"
            common = {
                "product_brand": f"brand-{category % 10}",
                "product_category_1": f"category-{category}",
                "product_category_2": f"department-{category % 5}",
                "product_category_3": f"vertical-{category % 3}",
                "product_category_4": f"leaf-{category}",
                "partner_id": f"partner-{category % 4}",
                "product_price": float(10 + category),
            }
            past_rows.append({"user_id": f"u-{number:04d}", "product_id": product, "conversion_label": number % 7 == 0, "click_timestamp": 10 + number % 80, **common})
            future_rows.append({"user_id": f"u-{number:04d}", "product_id": target, "conversion_label": 0, "click_timestamp": 110 + number % 10})
        pd.DataFrame(past_rows).to_csv(past / "part-00000.csv", index=False)
        pd.DataFrame(future_rows).to_csv(future / "part-00000.csv", index=False)
        config = content_item.ContentItemTwoTowerConfig(
            input_path=past, future_path=future, output_dir=root / "run", split_timestamp=100,
            embedding_dim=16, feature_embedding_dim=8, price_embedding_dim=4, hidden_dims=(32,),
            batch_size=256, epochs=1, negative_samples=5, max_train_rows=10_000,
            top_k=50, search_batch_size=200, inference_batch_size=100,
            input_chunk_size=250, log_every_rows=500, device="cpu", faiss_index_type="flat",
        )
        data = content_item.prepare_content_item_data(config, root / "state.sqlite")
        report = {"temporal_boundary_strict": True, "search_backend": "exact_numpy_fallback", "models": {}}
        original_search = content_item.search_faiss_index
        try:
            content_item.search_faiss_index = lambda index, queries, top_k: index.search(queries, top_k)
            for variant in ("id_only", "content_item"):
                model, losses, train_stats = content_item.train_variant(data, variant, torch.device("cpu"))
                embeddings = content_item.extract_catalog_embeddings(model, data.catalog, 100, torch.device("cpu"))
                output = root / f"{variant}.csv"
                retrieval = content_item.stream_content_item_candidates(
                    model, data, ExactNumpyIndex(embeddings), output, torch.device("cpu")
                )
                metrics = content_item.evaluate_content_item_candidates(output, future, data)
                report["models"][variant] = {
                    "loss": losses[-1], "embedding_norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
                    "embedding_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
                    "candidate_columns": pd.read_csv(output, nrows=0).columns.tolist(),
                    "train_time": train_stats["train_time"], "retrieval_time": retrieval["retrieval_time"],
                    "peak_rss_mb": retrieval["peak_rss_mb"], "recall@50": metrics["recall@50"],
                    "warm_recall@50": metrics["warm_recall@50"], "catalog_coverage": metrics["catalog_coverage"],
                }
        finally:
            content_item.search_faiss_index = original_search
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
