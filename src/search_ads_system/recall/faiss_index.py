"""FAISS helpers for normalized product-ad embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as error:  # pragma: no cover - depends on optional local runtime
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = error
else:
    _FAISS_IMPORT_ERROR = None


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError(
            "FAISS is required for Two Tower recall. Install the project's faiss-cpu dependency."
        ) from _FAISS_IMPORT_ERROR


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return float32 L2-normalized vectors suitable for cosine inner product."""

    vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32)).copy()
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if not len(vectors):
        return vectors
    _require_faiss()
    faiss.normalize_L2(vectors)
    return vectors


def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """Build a cosine-similarity FAISS index from advertisement embeddings.

    ``flat`` is exact ``IndexFlatIP``. ``hnsw`` uses HNSW with inner product;
    both are cosine indexes because vectors are normalized before insertion.
    """

    _require_faiss()
    vectors = normalize_embeddings(embeddings)
    if vectors.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")
    kind = index_type.lower()
    if kind == "flat":
        index: faiss.Index = faiss.IndexFlatIP(vectors.shape[1])
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
    else:
        raise ValueError("index_type must be 'flat' or 'hnsw'")
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, product_ids: np.ndarray, path: Path) -> None:
    """Persist an index and its position-to-original-product-ID mapping."""

    _require_faiss()
    identifiers = np.asarray(product_ids, dtype=str)
    if index.ntotal != len(identifiers):
        raise ValueError("product_ids length must match the number of indexed vectors")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    faiss.write_index(index, str(temporary_path))
    temporary_path.replace(path)
    metadata_path = path.with_name(path.name + ".metadata.json")
    metadata_path.write_text(
        json.dumps({"product_ids": identifiers.tolist()}, ensure_ascii=False), encoding="utf-8"
    )


def load_faiss_index(path: Path) -> tuple[faiss.Index, np.ndarray]:
    """Load an index persisted by :func:`save_faiss_index`."""

    _require_faiss()
    index = faiss.read_index(str(path))
    metadata_path = path.with_name(path.name + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    product_ids = np.asarray(metadata["product_ids"], dtype=str)
    if index.ntotal != len(product_ids):
        raise ValueError("FAISS index and product ID metadata have different sizes")
    return index, product_ids


def search_faiss_index(index: faiss.Index, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search normalized query embeddings and return scores and positions."""

    _require_faiss()
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not index.ntotal:
        empty = np.empty((len(query_embeddings), 0), dtype=np.float32)
        return empty, np.empty((len(query_embeddings), 0), dtype=np.int64)
    queries = normalize_embeddings(query_embeddings)
    return index.search(queries, min(top_k, index.ntotal))
