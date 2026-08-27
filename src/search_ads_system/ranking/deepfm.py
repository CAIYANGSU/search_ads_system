"""DeepFM backbone for clicked-interaction conversion/value fine ranking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class DeepFMMultiTask(nn.Module):
    """Wide + second-order FM + deep MLP with separate CVR/value heads."""

    def __init__(
        self, *, dense_dim: int, sparse_bucket_sizes: Sequence[int], embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 128, 64), dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dense_dim <= 0 or embedding_dim <= 0 or not sparse_bucket_sizes or any(size <= 1 for size in sparse_bucket_sizes):
            raise ValueError("DeepFM dimensions and sparse bucket sizes must be positive")
        self.dense_dim = int(dense_dim)
        self.sparse_bucket_sizes = tuple(int(size) for size in sparse_bucket_sizes)
        self.embedding_dim = int(embedding_dim)
        # Linear embeddings provide the wide component; vector embeddings feed
        # both the actual second-order FM term and the deep MLP.
        self.linear_dense = nn.Linear(dense_dim, 1)
        self.linear_embeddings = nn.ModuleList(nn.Embedding(size, 1) for size in self.sparse_bucket_sizes)
        self.feature_embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in self.sparse_bucket_sizes)
        deep_layers: list[nn.Module] = []
        previous = dense_dim + len(self.feature_embeddings) * embedding_dim
        for hidden in hidden_dims:
            deep_layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
            if dropout:
                deep_layers.append(nn.Dropout(dropout))
            previous = int(hidden)
        self.deep = nn.Sequential(*deep_layers) if deep_layers else nn.Identity()
        self.cvr_head = nn.Linear(previous + 2, 1)
        self.value_head = nn.Linear(previous + 2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (*self.linear_embeddings, *self.feature_embeddings):
            nn.init.normal_(embedding.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, dense: Tensor, sparse: Tensor) -> tuple[Tensor, Tensor]:
        if dense.ndim != 2 or dense.shape[1] != self.dense_dim:
            raise ValueError(f"dense must have shape [batch, {self.dense_dim}]")
        if sparse.ndim != 2 or sparse.shape[1] != len(self.feature_embeddings):
            raise ValueError(f"sparse must have shape [batch, {len(self.feature_embeddings)}]")
        vectors = [embedding(sparse[:, index].remainder(embedding.num_embeddings)) for index, embedding in enumerate(self.feature_embeddings)]
        stacked = torch.stack(vectors, dim=1)
        # 1/2 ((sum v)^2 - sum(v^2)) is the FM second-order interaction.
        fm = 0.5 * ((stacked.sum(dim=1).square() - stacked.square().sum(dim=1)).sum(dim=1, keepdim=True))
        wide = self.linear_dense(dense) + torch.stack(
            [embedding(sparse[:, index].remainder(embedding.num_embeddings)).squeeze(1) for index, embedding in enumerate(self.linear_embeddings)], dim=1
        ).sum(dim=1, keepdim=True)
        deep = self.deep(torch.cat((dense, stacked.flatten(1)), dim=1))
        shared = torch.cat((wide, fm, deep), dim=1)
        return self.cvr_head(shared).squeeze(1), self.value_head(shared).squeeze(1)
