from __future__ import annotations

import numpy as np

from ot_utils import sinkhorn_ot_distance


def select_topk_patches_by_global_similarity(
    patch_features: np.ndarray,
    global_features: np.ndarray,
    topk: int,
) -> np.ndarray:
    """
    根据 patch 与自身全局特征的 cosine similarity，保留 top-k patch。

    Args:
        patch_features:  (N, M, D)
        global_features: (N, D)
        topk: 保留的 patch 数量

    Returns:
        selected_patch_features: (N, topk, D)
    """
    n, m, d = patch_features.shape
    if topk <= 0 or topk > m:
        raise ValueError(f"topk must be in [1, {m}], got {topk}")

    selected = np.zeros((n, topk, d), dtype=patch_features.dtype)

    for i in range(n):
        patches = patch_features[i]          # (M, D)
        g = global_features[i]               # (D,)
        scores = patches @ g                 # (M,)  已归一化时即 cosine
        topk_idx = np.argsort(scores)[-topk:]
        selected[i] = patches[topk_idx]

    return selected


def hybrid_cost_matrix(
    patch_features_a: np.ndarray,
    patch_features_b: np.ndarray,
    global_feature_a: np.ndarray,
    global_feature_b: np.ndarray,
    theta: float = 0.8,
) -> np.ndarray:
    """
    构造 hybrid OT cost:

        C* = 1 - [ theta * cos(local, local) + (1-theta) * cos(global, global) ]

    Args:
        patch_features_a: (M, D)
        patch_features_b: (N, D)
        global_feature_a: (D,)
        global_feature_b: (D,)
        theta: 局部项权重

    Returns:
        cost_matrix: (M, N)
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0, 1], got {theta}")

    local_sim = patch_features_a @ patch_features_b.T           # (M, N)
    global_sim = float(global_feature_a @ global_feature_b)     # scalar

    hybrid_sim = theta * local_sim + (1.0 - theta) * global_sim
    cost = 1.0 - hybrid_sim
    return cost.astype(np.float64)


def compute_pairwise_hybrid_ot_distance_matrix(
    patch_features: np.ndarray,
    global_features: np.ndarray,
    reg: float = 0.05,
    theta: float = 0.8,
    verbose: bool = True,
) -> np.ndarray:
    """
    计算所有图像两两之间的 hybrid OT 距离矩阵。

    Args:
        patch_features:  (N, M, D)
        global_features: (N, D)
        reg: Sinkhorn regularization
        theta: hybrid cost 中局部项权重

    Returns:
        dist_matrix: (N, N)
    """
    n, m, d = patch_features.shape
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    total_pairs = n * (n + 1) // 2
    pair_count = 0

    for i in range(n):
        Pi = patch_features[i]
        gi = global_features[i]

        for j in range(i, n):
            Pj = patch_features[j]
            gj = global_features[j]

            cost_matrix = hybrid_cost_matrix(
                patch_features_a=Pi,
                patch_features_b=Pj,
                global_feature_a=gi,
                global_feature_b=gj,
                theta=theta,
            )

            _, ot_dist = sinkhorn_ot_distance(cost_matrix, reg=reg)

            dist_matrix[i, j] = ot_dist
            dist_matrix[j, i] = ot_dist

            pair_count += 1
            if verbose and pair_count % 2000 == 0:
                print(f"Processed {pair_count}/{total_pairs} pairs...")

    return dist_matrix


def hybrid_ot_distance_to_affinity(
    dist_matrix: np.ndarray,
    gamma: float = 10.0,
    knn_k: int | None = 10,
) -> np.ndarray:
    """
    将 hybrid OT 距离转为 affinity matrix:
        A = exp(-gamma * dist^2)
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