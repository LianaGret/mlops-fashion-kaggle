from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FashionDataset(Dataset):
    def __init__(
        self,
        transactions_path: Path,
        articles_path: Path,
        customers_path: Path,
        max_history_length: int = 50,
        negative_samples: int = 4,
    ) -> None:
        self.max_history_length = max_history_length
        self.negative_samples = negative_samples

        self.transactions = pd.read_csv(transactions_path)
        self.articles = pd.read_csv(articles_path)
        self.customers = pd.read_csv(customers_path)

        self._preprocess()

    def _preprocess(self) -> None:
        self.article_to_idx = {
            aid: idx for idx, aid in enumerate(self.articles["article_id"].unique())
        }
        self.customer_to_idx = {
            cid: idx for idx, cid in enumerate(self.customers["customer_id"].unique())
        }

        self.num_articles = len(self.article_to_idx)
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        customer_idx, article_idx, label = self.samples[idx]

        history = self.customer_histories.get(customer_idx, [])
        history_padded = self._pad_history(history)

        return {
            "customer_idx": torch.tensor(customer_idx, dtype=torch.long),
            "article_idx": torch.tensor(article_idx, dtype=torch.long),
            "history": torch.tensor(history_padded, dtype=torch.long),
            "history_mask": torch.tensor(
                [1] * len(history) + [0] * (self.max_history_length - len(history)),
                dtype=torch.float,
            ),
            "label": torch.tensor(label, dtype=torch.float),
        }

    def _pad_history(self, history: list[int]) -> list[int]:
        if len(history) >= self.max_history_length:
            return history[-self.max_history_length :]
        return history + [0] * (self.max_history_length - len(history))


class InferenceDataset(Dataset):
    def __init__(
        self,
        customer_ids: list[str],
        articles_path: Path,
        customers_path: Path,
        customer_histories: dict[str, list[str]] | None = None,
        max_history_length: int = 50,
    ) -> None:
        self.customer_ids = customer_ids
        self.max_history_length = max_history_length

        self.articles = pd.read_csv(articles_path)
        self.customers = pd.read_csv(customers_path)

        self.article_to_idx = {
            aid: idx for idx, aid in enumerate(self.articles["article_id"].unique())
        }
        self.customer_to_idx = {cid: idx for idx, cid in enumerate(customer_ids)}

        self.customer_histories = customer_histories or {}

    def __len__(self) -> int:
        return len(self.customer_ids) * len(self.article_to_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        customer_local_idx = idx // len(self.article_to_idx)
        article_idx = idx % len(self.article_to_idx)

        customer_id = self.customer_ids[customer_local_idx]
        customer_idx = self.customer_to_idx.get(customer_id, 0)

        history_ids = self.customer_histories.get(customer_id, [])
        history = [self.article_to_idx.get(aid, 0) for aid in history_ids]
        history_padded = self._pad_history(history)

        return {
            "customer_idx": torch.tensor(customer_idx, dtype=torch.long),
            "article_idx": torch.tensor(article_idx, dtype=torch.long),
            "history": torch.tensor(history_padded, dtype=torch.long),
            "history_mask": torch.tensor(
                [1] * len(history) + [0] * (self.max_history_length - len(history)),
                dtype=torch.float,
            ),
        }

    def _pad_history(self, history: list[int]) -> list[int]:
        if len(history) >= self.max_history_length:
            return history[-self.max_history_length :]
        return history + [0] * (self.max_history_length - len(history))
