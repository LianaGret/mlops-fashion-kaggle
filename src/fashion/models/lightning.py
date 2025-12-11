from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig

from fashion.models.architectures import (
    CollaborativeFilteringModel,
    ContentBasedModel,
    HybridRecommender,
)


class FashionRecommender(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = self._build_model()
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.example_input_array = self._create_example_input()

    def _build_model(self) -> nn.Module:
        model_type = self.config.get("type", "hybrid")
        num_customers = self.config.get("num_customers", 100000)
        num_articles = self.config.get("num_articles", 100000)
        embedding_dim = self.config.get("embedding_dim", 64)
        dropout = self.config.get("dropout", 0.1)

        if model_type == "collaborative":
            return CollaborativeFilteringModel(
                num_customers=num_customers,
                num_articles=num_articles,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif model_type == "content":
            return ContentBasedModel(
                num_articles=num_articles,
                embedding_dim=embedding_dim,
                num_heads=self.config.get("num_heads", 4),
                num_layers=self.config.get("num_layers", 2),
                dropout=dropout,
                max_history_length=self.config.get("max_history_length", 50),
            )
        else:
            return HybridRecommender(
                num_customers=num_customers,
                num_articles=num_articles,
                embedding_dim=embedding_dim,
                num_heads=self.config.get("num_heads", 4),
                num_layers=self.config.get("num_layers", 2),
                dropout=dropout,
                max_history_length=self.config.get("max_history_length", 50),
            )

    def _create_example_input(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        max_history = self.config.get("max_history_length", 50)

        return {
            "customer_idx": torch.zeros(batch_size, dtype=torch.long),
            "article_idx": torch.zeros(batch_size, dtype=torch.long),
            "history": torch.zeros(batch_size, max_history, dtype=torch.long),
            "history_mask": torch.ones(batch_size, max_history, dtype=torch.float),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        model_type = self.config.get("type", "hybrid")

        if model_type == "collaborative":
            return self.model(batch["customer_idx"], batch["article_idx"])
        elif model_type == "content":
            return self.model(batch["history"], batch["history_mask"], batch["article_idx"])
        else:
            return self.model(
                batch["customer_idx"],
                batch["article_idx"],
                batch["history"],
                batch["history_mask"],
            )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"])

        predictions = torch.sigmoid(scores) > 0.5
        accuracy = (predictions == batch["label"]).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"])

        predictions = torch.sigmoid(scores) > 0.5
        accuracy = (predictions == batch["label"]).float().mean()

        precision, recall = self._compute_precision_recall(scores, batch["label"])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"])

        predictions = torch.sigmoid(scores) > 0.5
        accuracy = (predictions == batch["label"]).float().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        return loss

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> list[dict[str, Any]]:
        scores = self(batch)
        probabilities = torch.sigmoid(scores)

        results = []
        for i in range(len(scores)):
            results.append(
                {
                    "customer_idx": batch["customer_idx"][i].item(),
                    "article_idx": batch["article_idx"][i].item(),
                    "score": scores[i].item(),
                    "probability": probabilities[i].item(),
                }
            )

        return results

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.get("scheduler_t0", 10),
            T_mult=self.config.get("scheduler_t_mult", 2),
            eta_min=self.config.get("min_lr", 1e-6),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _compute_precision_recall(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.sigmoid(scores) > 0.5

        true_positives = ((predictions == 1) & (labels == 1)).float().sum()
        predicted_positives = predictions.float().sum()
        actual_positives = labels.sum()

        precision = (
            true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
        )
        recall = true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0)

        return precision, recall
