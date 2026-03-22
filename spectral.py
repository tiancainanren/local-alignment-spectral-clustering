from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity


def build_affinity_matrix(
    features: np.ndarray,
    mode: Literal["cosine_shift", "rbf_from_cosine_distance"] = "cosine_shift",
    gamma: float = 5.0,
    knn_k: int | None = 10,
) -> np.ndarray:
    """
    从全局特征构造 affinity matrix。

    mode:
        - cosine_shift:
            A = (cosine + 1) / 2, 映射到 [0,1]
        - rbf_from_cosine_distance:
            dist = 1 - cosine
            A = exp(-gamma * dist^2)

    knn_k:
        若不为 None，则每行仅保留 top-k 邻居，构建稀疏图。
    """
    sim = cosine_similarity(features)  # [-1, 1]

    if mode == "cosine_shift":
        affinity = (sim + 1.0) / 2.0
    elif mode == "rbf_from_cosine_distance":
        dist = 1.0 - sim
        affinity = np.exp(-gamma * (dist ** 2))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    np.fill_diagonal(affinity, 0.0)

    if knn_k is not None:
        affinity = keep_topk_neighbors(affinity, k=knn_k)

    affinity = symmetrize_matrix(affinity)
    return affinity


def keep_topk_neighbors(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    每行只保留 top-k 最大值，其余置 0。
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    n = matrix.shape[0]
    out = np.zeros_like(matrix)

    for i in range(n):
        row = matrix[i]
        # 取前 k 个最大值索引
        topk_idx = np.argpartition(row, -k)[-k:]
        out[i, topk_idx] = row[topk_idx]

    return out


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def run_spectral_clustering(
    affinity: np.ndarray,
    num_clusters: int,
    random_state: int = 42,
    assign_labels: str = "kmeans",
) -> np.ndarray:
    """
    在预计算好的 affinity matrix 上做谱聚类。
    """
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",
        assign_labels=assign_labels,
        random_state=random_state,
    )
    pred_labels = clustering.fit_predict(affinity)
    return pred_labels