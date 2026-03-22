from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    聚类 ACC:
    先通过匈牙利算法做最佳标签匹配，再计算准确率。
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    assert y_pred.size == y_true.size

    d = max(y_pred.max(), y_true.max()) + 1
    weight = np.zeros((d, d), dtype=np.int64)

    for yp, yt in zip(y_pred, y_true):
        weight[yp, yt] += 1

    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    matched = weight[row_ind, col_ind].sum()
    return matched / y_true.size


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return {
        "ACC": float(acc),
        "NMI": float(nmi),
        "ARI": float(ari),
    }