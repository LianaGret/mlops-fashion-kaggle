from __future__ import annotations

import torch
import torch.nn as nn


class CollaborativeFilteringModel(nn.Module):
    def __init__(
        self,
        num_customers: int,
        num_articles: int,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.customer_embedding = nn.Embedding(num_customers, embedding_dim)
        self.article_embedding = nn.Embedding(num_articles, embedding_dim)

        self.customer_bias = nn.Embedding(num_customers, 1)
        self.article_bias = nn.Embedding(num_articles, 1)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.customer_embedding.weight)
        nn.init.xavier_uniform_(self.article_embedding.weight)
        nn.init.zeros_(self.customer_bias.weight)
        nn.init.zeros_(self.article_bias.weight)

    def forward(
        self,
        customer_idx: torch.Tensor,
        article_idx: torch.Tensor,
    ) -> torch.Tensor:
        customer_emb = self.dropout(self.customer_embedding(customer_idx))
        article_emb = self.dropout(self.article_embedding(article_idx))

        customer_b = self.customer_bias(customer_idx).squeeze(-1)
        article_b = self.article_bias(article_idx).squeeze(-1)

        dot_product = (customer_emb * article_emb).sum(dim=-1)
        score = dot_product + customer_b + article_b

        return score


class ContentBasedModel(nn.Module):
    def __init__(
        self,
        num_articles: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_history_length: int = 50,
    ) -> None:
        super().__init__()

        self.article_embedding = nn.Embedding(num_articles + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.article_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        article_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = history.shape
        device = history.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        history_emb = self.article_embedding(history) + self.position_embedding(positions)
        history_emb = self.dropout(history_emb)

        padding_mask = history_mask == 0
        encoded = self.transformer(history_emb, src_key_padding_mask=padding_mask)

        history_lengths = history_mask.sum(dim=-1).long() - 1
        history_lengths = history_lengths.clamp(min=0)
        batch_indices = torch.arange(batch_size, device=device)
        customer_repr = encoded[batch_indices, history_lengths]

        customer_repr = self.output_projection(customer_repr)

        target_emb = self.article_embedding(article_idx)

        score = (customer_repr * target_emb).sum(dim=-1)

        return score


class HybridRecommender(nn.Module):
    def __init__(
        self,
        num_customers: int,
        num_articles: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_history_length: int = 50,
    ) -> None:
        super().__init__()

        self.collab_model = CollaborativeFilteringModel(
            num_customers=num_customers,
            num_articles=num_articles,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.content_model = ContentBasedModel(
            num_articles=num_articles,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_history_length=max_history_length,
        )

        self.gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        customer_idx: torch.Tensor,
        article_idx: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        collab_score = self.collab_model(customer_idx, article_idx)
        content_score = self.content_model(history, history_mask, article_idx)

        scores = torch.stack([collab_score, content_score], dim=-1)
        gate_weights = self.gate(scores)

        combined_score = (scores * gate_weights).sum(dim=-1)

        return combined_score
