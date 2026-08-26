"""PyTorch DCNv2 backbone used by click-conditioned fine ranking.

The module deliberately exposes logits for the CVR task.  A sigmoid is only
applied at prediction time so training can use the numerically stable
``BCEWithLogitsLoss``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class CrossNetworkV2(nn.Module):
    """Vector-parameterised DCNv2 cross layers."""

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        if input_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim and num_layers must be positive")
        self.layers = nn.ModuleList(nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_layers))

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class DCNv2MultiTask(nn.Module):
    """Shared DCNv2 + deep tower with pCVR and log-value heads."""

    def __init__(
        self,
        *,
        dense_dim: int,
        sparse_bucket_sizes: Sequence[int],
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 128, 64),
        num_cross_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dense_dim <= 0 or embedding_dim <= 0 or not sparse_bucket_sizes:
            raise ValueError("dense_dim, embedding_dim, and sparse_bucket_sizes must be non-empty/positive")
        if any(size <= 1 for size in sparse_bucket_sizes):
            raise ValueError("all sparse bucket sizes must be greater than one")
        self.dense_dim = dense_dim
        self.sparse_bucket_sizes = tuple(int(size) for size in sparse_bucket_sizes)
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in self.sparse_bucket_sizes)
        input_dim = dense_dim + len(self.embeddings) * embedding_dim
        self.cross = CrossNetworkV2(input_dim, num_cross_layers)
        deep_layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            if hidden <= 0:
                raise ValueError("hidden dimensions must be positive")
            deep_layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
            if dropout:
                deep_layers.append(nn.Dropout(dropout))
            previous = int(hidden)
        self.deep = nn.Sequential(*deep_layers) if deep_layers else nn.Identity()
        representation_dim = input_dim + previous
        self.cvr_head = nn.Linear(representation_dim, 1)
        self.value_head = nn.Linear(representation_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, dense: Tensor, sparse: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(conversion_logits, predicted_log_value)`` as 1-D tensors."""
        if dense.ndim != 2 or dense.shape[1] != self.dense_dim:
            raise ValueError(f"dense must have shape [batch, {self.dense_dim}]")
        if sparse.ndim != 2 or sparse.shape[1] != len(self.embeddings):
            raise ValueError(f"sparse must have shape [batch, {len(self.embeddings)}]")
        embedded = [embedding(sparse[:, index].remainder(embedding.num_embeddings)) for index, embedding in enumerate(self.embeddings)]
        x = torch.cat((dense, *embedded), dim=1)
        shared = torch.cat((self.cross(x), self.deep(x)), dim=1)
        return self.cvr_head(shared).squeeze(1), self.value_head(shared).squeeze(1)

    @torch.no_grad()
    def predict(self, dense: Tensor, sparse: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Backward-compatible prediction using a bounded raw log-value head."""
        probability, _, value, expected = self.predict_with_log(dense, sparse)
        return probability, value, expected

    @torch.no_grad()
    def predict_with_log(self, dense: Tensor, sparse: Tensor, *, value_mean: float = 0.0, value_std: float = 1.0, prediction_log_min: float = 0.0, prediction_log_max: float = 20.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode a normalized value head with a finite, non-negative clamp.

        ``prediction_log_max`` is intentionally applied before ``expm1``.  A
        non-finite model output is a numerical failure, not a missing target,
        and is raised rather than silently filtered from metrics.
        """
        logits, log_value = self(dense, sparse)
        if not torch.isfinite(logits).all() or not torch.isfinite(log_value).all():
            raise FloatingPointError("Fine-rank model produced non-finite logits")
        probability = torch.sigmoid(logits)
        predicted_log_value = (log_value * float(value_std) + float(value_mean)).clamp(float(prediction_log_min), float(prediction_log_max))
        value = torch.expm1(predicted_log_value)
        expected = probability * value
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise FloatingPointError("Fine-rank value decoding produced non-finite predictions")
        return probability, predicted_log_value, value, expected
