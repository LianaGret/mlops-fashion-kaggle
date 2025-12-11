from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VisualEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 2,
    ) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        backbone_dim = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            layers_to_unfreeze = [
                self.backbone.layer4,
                self.backbone.layer3,
                self.backbone.layer2,
                self.backbone.layer1,
            ][:unfreeze_layers]

            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=-1)


class HistoryAggregator(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history_embeddings.shape[0]

        query = self.query.expand(batch_size, -1, -1)

        key_padding_mask = history_mask == 0

        attended, _ = self.attention(
            query=query,
            key=history_embeddings,
            value=history_embeddings,
            key_padding_mask=key_padding_mask,
        )

        output = self.norm(attended.squeeze(1))
        return self.dropout(output)


class VisualRecommender(nn.Module):
    def __init__(
        self,
        num_customers: int,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_customer_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_customer_embedding = use_customer_embedding

        self.visual_encoder = VisualEncoder(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_layers=unfreeze_layers,
        )

        self.history_aggregator = HistoryAggregator(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        if use_customer_embedding:
            self.customer_embedding = nn.Embedding(num_customers, embedding_dim)
            nn.init.xavier_uniform_(self.customer_embedding.weight)
            self.combiner = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.score_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            batch_size, seq_len, C, H, W = images.shape
            images_flat = images.view(batch_size * seq_len, C, H, W)
            embeddings_flat = self.visual_encoder(images_flat)
            return embeddings_flat.view(batch_size, seq_len, -1)
        else:
            return self.visual_encoder(images)

    def get_customer_preference(
        self,
        customer_idx: torch.Tensor,
        history_images: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_embeddings = self.encode_images(history_images)

        visual_preference = self.history_aggregator(history_embeddings, history_mask)

        if self.use_customer_embedding:
            customer_emb = self.customer_embedding(customer_idx)
            combined = torch.cat([visual_preference, customer_emb], dim=-1)
            return self.combiner(combined)
        else:
            return visual_preference

    def forward(
        self,
        customer_idx: torch.Tensor,
        target_image: torch.Tensor,
        history_images: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        customer_pref = self.get_customer_preference(customer_idx, history_images, history_mask)

        target_emb = self.encode_images(target_image)

        combined = torch.cat([customer_pref, target_emb], dim=-1)
        scores = self.score_head(combined).squeeze(-1)

        return scores

    def score_articles(
        self,
        customer_idx: torch.Tensor,
        history_images: torch.Tensor,
        history_mask: torch.Tensor,
        article_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        customer_pref = self.get_customer_preference(customer_idx, history_images, history_mask)

        batch_size = customer_pref.shape[0]
        num_articles = article_embeddings.shape[0]

        customer_expanded = customer_pref.unsqueeze(1).expand(-1, num_articles, -1)
        articles_expanded = article_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([customer_expanded, articles_expanded], dim=-1)
        scores = self.score_head(combined).squeeze(-1)

        return scores
