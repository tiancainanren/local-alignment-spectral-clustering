from __future__ import annotations

import numpy as np

from global_baseline.ot_utils import cosine_cost_matrix, sinkhorn_ot_distance


def compute_pairwise_ot_distance_matrix(
    patch_features: np.ndarray,
    reg: float = 0.05,
    verbose: bool = True,
) -> np.ndarray:
    """
    计算所有图像两两之间的 OT 距离矩阵。

    Args:
        patch_features: (N, M, D)
        reg: Sinkhorn regularization
        verbose: whether to print progress

    Returns:
        dist_matrix: (N, N)
    """
    n, m, d = patch_features.shape
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    total_pairs = n * (n + 1) // 2
    pair_count = 0

    for i in range(n):
        Pi = patch_features[i]  # (M, D)
        for j in range(i, n):
            Pj = patch_features[j]  # (M, D)

            cost_matrix = cosine_cost_matrix(Pi, Pj)
            _, ot_dist = sinkhorn_ot_distance(cost_matrix, reg=reg)

            dist_matrix[i, j] = ot_dist
            dist_matrix[j, i] = ot_dist

            pair_count += 1
            if verbose and pair_count % 2000 == 0:
                print(f"Processed {pair_count}/{total_pairs} pairs...")

    return dist_matrix


def ot_distance_to_affinity(
    dist_matrix: np.ndarray,
    gamma: float = 10.0,
    knn_k: int | None = 10,
) -> np.ndarray:
    """
    将 OT 距离矩阵转为 affinity matrix:
        A = exp(-gamma * dist^2)

    Args:
        dist_matrix: (N, N)
        gamma: RBF strength
        knn_k: keep top-k neighbors

    Returns:
        affinity: (N, N)
    """
    affinity = np.exp(-gamma * (dist_matrix ** 2))
    np.fill_diagonal(affinity, 0.0)

    if knn_k is not None:
        affinity = keep_topk_neighbors(affinity, knn_k)

    affinity = 0.5 * (affinity + affinity.T)
    return affinity


def keep_topk_neighbors(matrix: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive.")

    n = matrix.shape[0]
    out = np.zeros_like(matrix)

    for i in range(n):
        row = matrix[i]
        topk_idx = np.argpartition(row, -k)[-k:]
        out[i, topk_idx] = row[topk_idx]

    return out