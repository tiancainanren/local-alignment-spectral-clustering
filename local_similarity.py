from __future__ import annotations

import numpy as np


def compute_pairwise_local_similarity_avg(
    patch_features: np.ndarray,
) -> np.ndarray:
    """
    基于 patch-patch cosine similarity 的全部平均，计算图像两两相似度。

    Args:
        patch_features: (N, M, D)
            N: 图像数
            M: 每张图的 patch 数
            D: 特征维度
        这里假设 patch 特征已经 L2 normalize 过。

    Returns:
        sim_matrix: (N, N)
    """
    n, m, d = patch_features.shape
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        Pi = patch_features[i]  # (M, D)
        for j in range(i, n):
            Pj = patch_features[j]  # (M, D)

            # patch-patch cosine similarity matrix: (M, M)
            local_sim = Pi @ Pj.T
            sim_ij = local_sim.mean()

            sim_matrix[i, j] = sim_ij
            sim_matrix[j, i] = sim_ij

    return sim_matrix


def compute_pairwise_local_similarity_maxavg(
    patch_features: np.ndarray,
) -> np.ndarray:
    """
    基于 row-max 再平均的局部相似度:
        Sim(i,j) = mean_m max_n cos(p_im, p_jn)
    """
    n, m, d = patch_features.shape
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        Pi = patch_features[i]
        for j in range(i, n):
            Pj = patch_features[j]

            local_sim = Pi @ Pj.T  # (M, M)
            sim_ij = local_sim.max(axis=1).mean()

            sim_matrix[i, j] = sim_ij
            sim_matrix[j, i] = sim_ij

    return sim_matrix


def local_similarity_to_affinity(
    sim_matrix: np.ndarray,
    knn_k: int | None = 10,
) -> np.ndarray:
    """
    将局部相似度矩阵映射到 [0, 1] 并构造 affinity matrix。
    这里直接用:
        affinity = (sim + 1) / 2

    然后可选保留 top-k 邻居，并对称化。
    """
    affinity = (sim_matrix + 1.0) / 2.0
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