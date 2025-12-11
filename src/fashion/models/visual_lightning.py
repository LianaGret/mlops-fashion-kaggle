from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig

from fashion.models.metrics import MAP12Metric
from fashion.models.visual_model import VisualRecommender


class VisualFashionRecommender(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = self._build_model()
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_map12 = MAP12Metric(k=12)
        self.val_map12 = MAP12Metric(k=12)

    def _build_model(self) -> nn.Module:
        return VisualRecommender(
            num_customers=self.config.get("num_customers", 100000),
            embedding_dim=self.config.get("embedding_dim", 128),
            pretrained=self.config.get("pretrained", True),
            freeze_backbone=self.config.get("freeze_backbone", True),
            unfreeze_layers=self.config.get("unfreeze_layers", 2),
            num_heads=self.config.get("num_heads", 4),
            dropout=self.config.get("dropout", 0.1),
            use_customer_embedding=self.config.get("use_customer_embedding", True),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            customer_idx=batch["customer_idx"],
            target_image=batch["target_image"],
            history_images=batch["history_images"],
            history_mask=batch["history_mask"],
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"])

        predictions = torch.sigmoid(scores) > 0.5
        accuracy = (predictions == batch["label"]).float().mean()

        self.train_map12.update(
            batch["customer_idx"],
            batch["article_idx"],
            scores,
            batch["label"],
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        map12 = self.train_map12.compute()
        self.log("train/map12", map12, prog_bar=True)
        self.train_map12.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"])

        predictions = torch.sigmoid(scores) > 0.5
        accuracy = (predictions == batch["label"]).float().mean()

        true_positives = ((predictions == 1) & (batch["label"] == 1)).float().sum()
        predicted_positives = predictions.float().sum()
        actual_positives = batch["label"].sum()

        precision = (
            true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
        )
        recall = true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0)

        self.val_map12.update(
            batch["customer_idx"],
            batch["article_idx"],
            scores,
            batch["label"],
        )

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("val/precision", precision, on_step=False, on_epoch=True)
        self.log("val/recall", recall, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        map12 = self.val_map12.compute()
        self.log("val/map12", map12, prog_bar=True)
        self.val_map12.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        scores = self(batch)
        loss = self.loss_fn(scores, batch["label"])

        predictions = torch.sigmoid(scores) > 0.5
        accuracy = (predictions == batch["label"]).float().mean()

        self.log("test/loss", loss)
        self.log("test/accuracy", accuracy)

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
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "visual_encoder.backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.config.get("backbone_lr", 1e-5)},
                {"params": head_params, "lr": self.config.get("learning_rate", 1e-3)},
            ],
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.get("scheduler_t0", 5),
            T_mult=self.config.get("scheduler_t_mult", 2),
            eta_min=self.config.get("min_lr", 1e-7),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        if self.logger:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.logger.experiment.config.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "frozen_parameters": total_params - trainable_params,
                }
            )
