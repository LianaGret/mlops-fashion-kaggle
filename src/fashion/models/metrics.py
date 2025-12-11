from __future__ import annotations

import torch


def compute_ap_at_k(predictions: list[int], ground_truth: set[int], k: int = 12) -> float:
    if not ground_truth:
        return 0.0

    predictions = predictions[:k]
    hits = 0
    precision_sum = 0.0

    for i, pred in enumerate(predictions):
        if pred in ground_truth:
            hits += 1
            precision_at_i = hits / (i + 1)
            precision_sum += precision_at_i

    denominator = min(k, len(ground_truth))
    if denominator == 0:
        return 0.0

    return precision_sum / denominator


def compute_map_at_k(
    all_predictions: list[list[int]],
    all_ground_truth: list[set[int]],
    k: int = 12,
) -> float:
    if len(all_predictions) != len(all_ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    ap_scores = []
    for preds, gt in zip(all_predictions, all_ground_truth, strict=True):
        if gt:
            ap_scores.append(compute_ap_at_k(preds, gt, k))

    if not ap_scores:
        return 0.0

    return sum(ap_scores) / len(ap_scores)


class MAP12Metric:
    def __init__(self, k: int = 12) -> None:
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.all_predictions: list[list[int]] = []
        self.all_ground_truth: list[set[int]] = []
        self.all_scores: list[list[tuple[int, float]]] = []
        self.customer_scores: dict[int, list[tuple[int, float]]] = {}
        self.customer_ground_truth: dict[int, set[int]] = {}

    def update(
        self,
        customer_indices: torch.Tensor,
        article_indices: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        customer_indices = customer_indices.cpu().numpy()
        article_indices = article_indices.cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.cpu().numpy()

        for cust_idx, art_idx, score, label in zip(
            customer_indices, article_indices, scores, labels, strict=True
        ):
            cust_idx = int(cust_idx)
            art_idx = int(art_idx)

            if cust_idx not in self.customer_scores:
                self.customer_scores[cust_idx] = []
                self.customer_ground_truth[cust_idx] = set()

            self.customer_scores[cust_idx].append((art_idx, float(score)))

            if label > 0.5:
                self.customer_ground_truth[cust_idx].add(art_idx)

    def update_rankings(
        self,
        customer_idx: int,
        ranked_articles: list[int],
        ground_truth: set[int] | list[int],
    ) -> None:
        self.all_predictions.append(ranked_articles[: self.k])
        gt_set = set(ground_truth) if isinstance(ground_truth, list) else ground_truth
        self.all_ground_truth.append(gt_set)

    def compute(self) -> float:
        if self.all_predictions:
            return compute_map_at_k(self.all_predictions, self.all_ground_truth, self.k)

        if not self.customer_scores:
            return 0.0

        predictions = []
        ground_truths = []

        for cust_idx in self.customer_scores:
            gt = self.customer_ground_truth.get(cust_idx, set())
            if not gt:
                continue

            sorted_articles = sorted(
                self.customer_scores[cust_idx], key=lambda x: x[1], reverse=True
            )
            ranked = [art_idx for art_idx, _ in sorted_articles[: self.k]]

            predictions.append(ranked)
            ground_truths.append(gt)

        return compute_map_at_k(predictions, ground_truths, self.k)
