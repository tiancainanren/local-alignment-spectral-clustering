from __future__ import annotations

import numpy as np

try:
    import ot
except ImportError as exc:
    raise ImportError("Please install POT first: pip install POT") from exc


def uniform_weights(num_points: int) -> np.ndarray:
    """
    生成均匀分布权重。
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    return np.ones(num_points, dtype=np.float64) / num_points


def sinkhorn_ot_distance(
    cost_matrix: np.ndarray,
    reg: float = 0.05,
) -> tuple[np.ndarray, float]:
    """
    计算熵正则 OT plan 和 OT distance。

    Args:
        cost_matrix: (M, M) 或 (M, N)
        reg: Sinkhorn entropic regularization strength

    Returns:
        transport_plan: (M, N)
        ot_distance: scalar
    """
    a = uniform_weights(cost_matrix.shape[0])
    b = uniform_weights(cost_matrix.shape[1])

    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=reg)
    ot_distance = float(np.sum(transport_plan * cost_matrix))
    return transport_plan, ot_distance


def cosine_cost_matrix(
    patch_features_a: np.ndarray,
    patch_features_b: np.ndarray,
) -> np.ndarray:
    """
    基于 cosine similarity 构造 OT cost matrix:
        C = 1 - cosine

    假设输入特征已经做过 L2 normalize。

    Args:
        patch_features_a: (M, D)
        patch_features_b: (N, D)

    Returns:
        cost_matrix: (M, N)
    """
    sim = patch_features_a @ patch_features_b.T
    cost = 1.0 - sim
    return cost.astype(np.float64)