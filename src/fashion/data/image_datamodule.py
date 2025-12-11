from __future__ import annotations

import contextlib
from pathlib import Path

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from fashion.data.image_dataset import ImageFashionDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class ImageFashionDataModule(L.LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        self.data_dir = Path(config.get("data_dir", DATA_DIR))
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.max_history_length = config.get("max_history_length", 10)
        self.negative_samples = config.get("negative_samples", 4)
        self.image_size = config.get("image_size", 224)
        self.val_split = config.get("val_split", 0.1)
        self.test_split = config.get("test_split", 0.1)
        self.sample_fraction = config.get("sample_fraction", 0.1)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        raw_dir = self.data_dir / "raw"

        required_files = [
            raw_dir / "articles.csv",
            raw_dir / "customers.csv",
            raw_dir / "transactions_train.csv",
        ]

        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            import dvc.api

            from fashion.console import print_info

            print_info("Pulling data from DVC remote...")
            for file_path in missing_files:
                dvc_path = str(file_path.relative_to(PROJECT_ROOT))
                with contextlib.suppress(Exception):
                    dvc.api.pull(dvc_path)

    def setup(self, stage: str | None = None) -> None:
        raw_dir = self.data_dir / "raw"

        if stage in ("fit", "validate", None):
            full_dataset = ImageFashionDataset(
                transactions_path=raw_dir / "transactions_train.csv",
                articles_path=raw_dir / "articles.csv",
                customers_path=raw_dir / "customers.csv",
                images_dir=raw_dir / "images",
                max_history_length=self.max_history_length,
                negative_samples=self.negative_samples,
                image_size=self.image_size,
                augment=True,
                sample_fraction=self.sample_fraction,
            )

            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            test_size = int(total_size * self.test_split)
            train_size = total_size - val_size - test_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
            )

            self.num_articles = full_dataset.num_articles
            self.num_customers = full_dataset.num_customers
            self.article_to_idx = full_dataset.article_to_idx
            self.idx_to_article = full_dataset.idx_to_article

        if stage == "test" and self.test_dataset is None:
            self.setup(stage="fit")

    def train_dataloader(self) -> DataLoader:
        import torch

        pin_memory = not torch.backends.mps.is_available()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        import torch

        pin_memory = not torch.backends.mps.is_available()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        import torch

        pin_memory = not torch.backends.mps.is_available()

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
