from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Self

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fashion.data.dataset import FashionDataset, InferenceDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class FashionDataModule(L.LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.data_dir = Path(config.get("data_dir", DATA_DIR))
        self.batch_size = config.get("batch_size", 64)
        self.num_workers = config.get("num_workers", 4)
        self.max_history_length = config.get("max_history_length", 50)
        self.negative_samples = config.get("negative_samples", 4)
        self.val_split = config.get("val_split", 0.1)
        self.test_split = config.get("test_split", 0.1)

        self.train_dataset: FashionDataset | None = None
        self.val_dataset: FashionDataset | None = None
        self.test_dataset: FashionDataset | None = None
        self.predict_dataset: InferenceDataset | None = None

    @classmethod
    def for_inference(
        cls,
        input_path: Path | None = None,
        batch_size: int = 32,
    ) -> Self:
        config = DictConfig(
            {
                "data_dir": str(DATA_DIR),
                "batch_size": batch_size,
                "num_workers": 2,
                "max_history_length": 50,
                "input_path": str(input_path) if input_path else None,
            }
        )
        return cls(config)

    def prepare_data(self) -> None:
        import dvc.api

        raw_dir = self.data_dir / "raw"

        required_files = [
            raw_dir / "articles.csv",
            raw_dir / "customers.csv",
            raw_dir / "transactions_train.csv",
        ]

        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            from fashion.console import print_info

            print_info("Pulling data from DVC remote...")
            for file_path in missing_files:
                dvc_path = str(file_path.relative_to(PROJECT_ROOT))
                with contextlib.suppress(Exception):
                    dvc.api.pull(dvc_path)

    def setup(self, stage: str | None = None) -> None:
        raw_dir = self.data_dir / "raw"

        transactions_path = raw_dir / "transactions_train.csv"
        articles_path = raw_dir / "articles.csv"
        customers_path = raw_dir / "customers.csv"

        if stage in ("fit", "validate", None):
            full_dataset = FashionDataset(
                transactions_path=transactions_path,
                articles_path=articles_path,
                customers_path=customers_path,
                max_history_length=self.max_history_length,
                negative_samples=self.negative_samples,
            )

            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            test_size = int(total_size * self.test_split)
            train_size = total_size - val_size - test_size

            from torch.utils.data import random_split

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
            )

            self.num_articles = full_dataset.num_articles
            self.num_customers = full_dataset.num_customers

        if stage == "test" and self.test_dataset is None:
            self.setup(stage="fit")

        if stage == "predict":
            input_path = self.config.get("input_path")
            if input_path:
                import pandas as pd

                input_df = pd.read_csv(input_path)
                customer_ids = input_df["customer_id"].unique().tolist()
            else:
                import pandas as pd

                customers_df = pd.read_csv(customers_path)
                customer_ids = customers_df["customer_id"].head(100).tolist()

            self.predict_dataset = InferenceDataset(
                customer_ids=customer_ids,
                articles_path=articles_path,
                customers_path=customers_path,
                max_history_length=self.max_history_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
