from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFashionDataset(Dataset):
    def __init__(
        self,
        transactions_path: Path,
        articles_path: Path,
        customers_path: Path,
        images_dir: Path,
        max_history_length: int = 10,
        negative_samples: int = 4,
        image_size: int = 224,
        augment: bool = True,
        sample_fraction: float = 0.1,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.max_history_length = max_history_length
        self.negative_samples = negative_samples
        self.image_size = image_size

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size + 32, image_size + 32)),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.transactions = pd.read_csv(transactions_path)
        self.articles = pd.read_csv(articles_path)
        self.customers = pd.read_csv(customers_path)

        self.sample_fraction = sample_fraction
        self._preprocess()

    def _get_image_path(self, article_id: int) -> Path:
        article_str = str(article_id).zfill(10)
        prefix = article_str[:3]
        return self.images_dir / prefix / f"{article_str}.jpg"

    def _preprocess(self) -> None:
        valid_articles = []
        for article_id in self.articles["article_id"].unique():
            if self._get_image_path(article_id).exists():
                valid_articles.append(article_id)

        self.article_to_idx = {aid: idx for idx, aid in enumerate(valid_articles)}
        self.idx_to_article = {idx: aid for aid, idx in self.article_to_idx.items()}
        self.num_articles = len(self.article_to_idx)

        self.transactions = self.transactions[
            self.transactions["article_id"].isin(self.article_to_idx.keys())
        ]

        unique_customers = self.transactions["customer_id"].unique()
        if self.sample_fraction < 1.0:
            n_sample = max(1000, int(len(unique_customers) * self.sample_fraction))
            sampled_customers = np.random.choice(
                unique_customers, size=min(n_sample, len(unique_customers)), replace=False
            )
            self.transactions = self.transactions[
                self.transactions["customer_id"].isin(sampled_customers)
            ]

        self.customer_to_idx = {
            cid: idx for idx, cid in enumerate(self.transactions["customer_id"].unique())
        }
        self.num_customers = len(self.customer_to_idx)

        self.customer_histories: dict[int, list[int]] = {}
        for customer_id, group in self.transactions.groupby("customer_id"):
            if customer_id in self.customer_to_idx:
                customer_idx = self.customer_to_idx[customer_id]
                article_indices = [
                    self.article_to_idx[aid]
                    for aid in group["article_id"]
                    if aid in self.article_to_idx
                ]
                self.customer_histories[customer_idx] = article_indices[-self.max_history_length :]

        self.samples = self._create_samples()

    def _create_samples(self) -> list[tuple[int, int, int]]:
        samples = []

        for customer_idx, history in self.customer_histories.items():
            if len(history) < 2:
                continue

            for i in range(1, len(history)):
                target_article = history[i]
                samples.append((customer_idx, target_article, 1))

                for _ in range(self.negative_samples):
                    neg_article = np.random.randint(0, self.num_articles)
                    while neg_article in history:
                        neg_article = np.random.randint(0, self.num_articles)
                    samples.append((customer_idx, neg_article, 0))

        return samples

    def _load_image(self, article_idx: int) -> torch.Tensor:
        article_id = self.idx_to_article[article_idx]
        image_path = self._get_image_path(article_id)

        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        customer_idx, article_idx, label = self.samples[idx]

        target_image = self._load_image(article_idx)

        history = self.customer_histories.get(customer_idx, [])
        history_images = []
        for hist_idx in history[-self.max_history_length :]:
            history_images.append(self._load_image(hist_idx))

        while len(history_images) < self.max_history_length:
            history_images.append(torch.zeros(3, self.image_size, self.image_size))

        history_tensor = torch.stack(history_images)
        history_mask = torch.tensor(
            [1.0] * min(len(history), self.max_history_length)
            + [0.0] * (self.max_history_length - min(len(history), self.max_history_length)),
            dtype=torch.float,
        )

        return {
            "customer_idx": torch.tensor(customer_idx, dtype=torch.long),
            "article_idx": torch.tensor(article_idx, dtype=torch.long),
            "target_image": target_image,
            "history_images": history_tensor,
            "history_mask": history_mask,
            "label": torch.tensor(label, dtype=torch.float),
        }
