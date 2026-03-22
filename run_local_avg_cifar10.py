from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from backbone import extract_patch_features, load_clip_model
from datasets.cifar10_subset import CIFAR10Subset, CIFARSubsetConfig
from evaluate import evaluate_clustering
from local_similarity import (
    compute_pairwise_local_similarity_avg,
    compute_pairwise_local_similarity_maxavg,
    local_similarity_to_affinity,
)
from patch_extract import extract_grid_patches
from spectral import run_spectral_clustering


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Local patch baseline without OT")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Selected CIFAR-10 classes, e.g. 0 1 2 3",
    )
    parser.add_argument("--samples_per_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--resize_size", type=int, default=224)

    parser.add_argument(
        "--sim_mode",
        type=str,
        default="avg",
        choices=["avg", "maxavg"],
        help="How to aggregate patch-patch similarities",
    )
    parser.add_argument("--knn_k", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) 数据集
    ds_config = CIFARSubsetConfig(
        root=args.data_root,
        train=args.train,
        download=True,
        selected_classes=args.classes,
        samples_per_class=args.samples_per_class,
        random_seed=args.seed,
    )
    dataset = CIFAR10Subset(ds_config)

    images = [dataset[i][0] for i in range(len(dataset))]
    labels = np.array([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)

    print(f"Loaded {len(images)} images.")
    print(f"Selected original classes: {args.classes}")

    # 2) 模型
    model, preprocess = load_clip_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    print(f"Loaded CLIP model: {args.model_name} ({args.pretrained}) on {args.device}")

    # 3) 提取规则网格 patches
    patch_lists = [
        extract_grid_patches(
            image=img,
            grid_size=args.grid_size,
            resize_size=args.resize_size,
        )
        for img in images
    ]
    print(f"Extracted patches for all images. patches/image = {args.grid_size * args.grid_size}")

    # 4) 提取 patch 特征
    patch_features = extract_patch_features(
        patch_lists=patch_lists,
        preprocess=preprocess,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Extracted patch features: {patch_features.shape}")  # (N, M, D)

    # 5) 计算局部相似度矩阵
    if args.sim_mode == "avg":
        sim_matrix = compute_pairwise_local_similarity_avg(patch_features)
    elif args.sim_mode == "maxavg":
        sim_matrix = compute_pairwise_local_similarity_maxavg(patch_features)
    else:
        raise ValueError(f"Unsupported sim_mode: {args.sim_mode}")

    print(f"Local similarity matrix shape: {sim_matrix.shape}")

    # 6) 转换成 affinity matrix
    affinity = local_similarity_to_affinity(
        sim_matrix=sim_matrix,
        knn_k=args.knn_k,
    )
    print(f"Affinity matrix shape: {affinity.shape}")

    # 7) 谱聚类
    pred_labels = run_spectral_clustering(
        affinity=affinity,
        num_clusters=len(args.classes),
        random_state=args.seed,
    )

    # 8) 评估
    metrics = evaluate_clustering(labels, pred_labels)
    print("\n=== Clustering Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()