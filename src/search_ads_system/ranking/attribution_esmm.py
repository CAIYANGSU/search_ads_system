"""Leakage-safe, Attribution-only neural baselines and standard ESMM head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CATEGORICAL_FEATURES: tuple[str, ...] = (
    "user_id", "campaign_id", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
)
DENSE_FEATURES: tuple[str, ...] = ("time_since_last_click_log1p_z", "time_since_last_click_missing")
FORBIDDEN_MODEL_FEATURES: frozenset[str] = frozenset({
    "click", "conversion", "click_and_conversion", "conversion_timestamp", "conversion_id", "attribution",
    "click_pos", "click_nb", "cost", "cpo", "timestamp", "event_id", "source_row_number",
})


def validate_feature_contract(features: Sequence[str]) -> None:
    """Reject labels, post-event attribution fields, and accounting fields."""

    forbidden = sorted(set(features) & FORBIDDEN_MODEL_FEATURES)
    if forbidden:
        raise ValueError(f"Attribution ESMM feature contract contains forbidden fields: {forbidden}")
    expected = set(CATEGORICAL_FEATURES) | {"time_since_last_click"}
    unknown = sorted(set(features) - expected)
    if unknown:
        raise ValueError(f"Attribution ESMM feature contract contains unknown fields: {unknown}")


def stable_hash_series(values: pd.Series, bucket_size: int) -> np.ndarray:
    """Return process-independent categorical hash IDs without building a vocabulary."""

    if bucket_size <= 1:
        raise ValueError("hash bucket size must be greater than one")
    normalized = values.astype("string").fillna("__MISSING__")
    hashed = pd.util.hash_pandas_object(normalized, index=False, categorize=True).to_numpy(dtype=np.uint64)
    return (hashed % np.uint64(bucket_size)).astype(np.int64, copy=False)


class _HashedBackbone(nn.Module):
    """Shared hash-embedding representation for all Attribution baselines."""

    def __init__(self, bucket_sizes: Sequence[int], embedding_dim: int, dense_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        if len(bucket_sizes) != len(CATEGORICAL_FEATURES) or any(size <= 1 for size in bucket_sizes):
            raise ValueError("One hash bucket size greater than one is required for every categorical feature")
        if embedding_dim <= 0 or dense_dim <= 0:
            raise ValueError("embedding_dim and dense_dim must be positive")
        self.bucket_sizes = tuple(int(size) for size in bucket_sizes)
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in self.bucket_sizes)
        input_dim = len(self.embeddings) * embedding_dim + dense_dim
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            if int(hidden) <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
            previous = int(hidden)
        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = previous
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sparse: Tensor, dense: Tensor) -> Tensor:
        if sparse.ndim != 2 or sparse.shape[1] != len(self.embeddings):
            raise ValueError(f"sparse must have shape [batch, {len(self.embeddings)}]")
        if dense.ndim != 2 or dense.shape[1] != len(DENSE_FEATURES):
            raise ValueError(f"dense must have shape [batch, {len(DENSE_FEATURES)}]")
        if not torch.isfinite(dense).all():
            raise FloatingPointError("Attribution dense features contain non-finite values")
        embedded = [embedding(sparse[:, index].remainder(embedding.num_embeddings)) for index, embedding in enumerate(self.embeddings)]
        representation = self.network(torch.cat((dense, *embedded), dim=1))
        if not torch.isfinite(representation).all():
            raise FloatingPointError("Attribution shared representation became non-finite")
        return representation


def _tower(input_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        if int(hidden) <= 0:
            raise ValueError("tower hidden dimensions must be positive")
        layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
        previous = int(hidden)
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


class AttributionSingleTask(nn.Module):
    """Shared baseline architecture for single-task CTR and clicked-only CVR."""

    def __init__(self, bucket_sizes: Sequence[int], embedding_dim: int, shared_hidden_dims: Sequence[int], tower_hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.backbone = _HashedBackbone(bucket_sizes, embedding_dim, len(DENSE_FEATURES), shared_hidden_dims)
        self.head = _tower(self.backbone.output_dim, tower_hidden_dims)

    def forward(self, sparse: Tensor, dense: Tensor) -> Tensor:
        logits = self.head(self.backbone(sparse, dense)).squeeze(1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Attribution single-task model produced non-finite logits")
        return logits


class AttributionESMM(nn.Module):
    """Standard ESMM: pCTCVR is explicitly constrained to pCTR * pCVR."""

    def __init__(
        self,
        bucket_sizes: Sequence[int],
        embedding_dim: int,
        shared_hidden_dims: Sequence[int],
        ctr_hidden_dims: Sequence[int],
        cvr_hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.backbone = _HashedBackbone(bucket_sizes, embedding_dim, len(DENSE_FEATURES), shared_hidden_dims)
        self.ctr_tower = _tower(self.backbone.output_dim, ctr_hidden_dims)
        self.cvr_tower = _tower(self.backbone.output_dim, cvr_hidden_dims)

    def forward(self, sparse: Tensor, dense: Tensor) -> dict[str, Tensor]:
        representation = self.backbone(sparse, dense)
        ctr_logit = self.ctr_tower(representation).squeeze(1)
        cvr_logit = self.cvr_tower(representation).squeeze(1)
        pctr = torch.sigmoid(ctr_logit)
        pcvr = torch.sigmoid(cvr_logit)
        pctcvr = pctr * pcvr
        if not all(torch.isfinite(value).all() for value in (ctr_logit, cvr_logit, pctr, pcvr, pctcvr)):
            raise FloatingPointError("Attribution ESMM produced non-finite values")
        return {"ctr_logit": ctr_logit, "cvr_logit": cvr_logit, "pctr": pctr, "pcvr": pcvr, "pctcvr": pctcvr}


def esmm_loss(outputs: Mapping[str, Tensor], click: Tensor, click_and_conversion: Tensor, lambda_ctcvr: float = 1.0, eps: float = 1e-7) -> dict[str, Tensor]:
    """Numerically guarded impression-space CTR + CTCVR ESMM objective.

    The forward pass may be under CUDA AMP.  Probability BCE is explicitly
    outside autocast because PyTorch rejects it in reduced precision.  CTCVR
    is still the standard ESMM product, recomputed from its two logits in
    float32; no independent CTCVR head/logit is introduced.
    """

    if lambda_ctcvr < 0.0 or not 0.0 < eps < 0.5:
        raise ValueError("lambda_ctcvr must be non-negative and eps must be in (0, 0.5)")
    device_type = outputs["ctr_logit"].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        ctr_logit = outputs["ctr_logit"].float()
        cvr_logit = outputs["cvr_logit"].float()
        pctr = torch.sigmoid(ctr_logit)
        pcvr = torch.sigmoid(cvr_logit)
        pctcvr = pctr * pcvr
        ctr = F.binary_cross_entropy_with_logits(ctr_logit, click.float())
        ctcvr = F.binary_cross_entropy(
            pctcvr.clamp(min=eps, max=1.0 - eps), click_and_conversion.float()
        )
    total = ctr + float(lambda_ctcvr) * ctcvr
    probabilities = {"pCTR": pctr, "pCVR": pcvr, "pCTCVR": pctcvr}
    if any(not torch.isfinite(value).all() for value in (total, ctr, ctcvr, *probabilities.values())):
        raise FloatingPointError("Attribution ESMM loss/probability is non-finite")
    if any(((value < 0.0) | (value > 1.0)).any() for value in probabilities.values()):
        raise FloatingPointError("Attribution ESMM probability fell outside [0, 1]")
    if not torch.equal(pctcvr, pctr * pcvr):
        raise AssertionError("Attribution ESMM must keep pCTCVR equal to pCTR * pCVR")
    return {"total": total, "ctr": ctr, "ctcvr": ctcvr}
