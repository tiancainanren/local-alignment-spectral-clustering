from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from backbone import (
    extract_global_and_patch_features,
    load_clip_model,
)
from datasets.stl10_subset import STL10Subset, STLSubsetConfig
from evaluate import evaluate_clustering
from hybrid_ot_similarity import (
    compute_pairwise_hybrid_ot_distance_matrix,
    hybrid_ot_distance_to_affinity,
    select_topk_patches_by_global_similarity,
)
from patch_extract import extract_grid_patches
from spectral import run_spectral_clustering


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid OT spectral clustering on STL-10")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 2, 8, 9],
        help="Selected STL-10 classes, e.g. 0 2 8 9",
    )
    parser.add_argument("--samples_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--image_batch_size", type=int, default=64)
    parser.add_argument("--patch_batch_size", type=int, default=256)

    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--patch_topk", type=int, default=5)

    parser.add_argument("--sinkhorn_reg", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--knn_k", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) 数据集
    ds_config = STLSubsetConfig(
        root=args.data_root,
        split=args.split,
        download=True,
        selected_classes=args.classes,
        samples_per_class=args.samples_per_class,
        random_seed=args.seed,
    )
    dataset = STL10Subset(ds_config)

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

    # 3) 切 patch
    patch_lists = [
        extract_grid_patches(
            image=img,
            grid_size=args.grid_size,
            resize_size=args.resize_size,
        )
        for img in images
    ]
    patch_num = args.grid_size * args.grid_size
    print(f"Extracted patches for all images. patches/image = {patch_num}")

    # 4) 提全局 + patch 特征
    global_features, labels, patch_features = extract_global_and_patch_features(
        images=images,
        labels=labels,
        patch_lists=patch_lists,
        preprocess=preprocess,
        model=model,
        device=args.device,
        image_batch_size=args.image_batch_size,
        patch_batch_size=args.patch_batch_size,
    )
    print(f"Extracted global features: {global_features.shape}")
    print(f"Extracted patch features:  {patch_features.shape}")

    # 5) patch selection
    selected_patch_features = select_topk_patches_by_global_similarity(
        patch_features=patch_features,
        global_features=global_features,
        topk=args.patch_topk,
    )
    print(f"Selected patch features:   {selected_patch_features.shape}")

    # 6) hybrid OT distance
    dist_matrix = compute_pairwise_hybrid_ot_distance_matrix(
        patch_features=selected_patch_features,
        global_features=global_features,
        reg=args.sinkhorn_reg,
        theta=args.theta,
        verbose=True,
    )
    print(f"Hybrid OT distance matrix shape: {dist_matrix.shape}")

    # 7) affinity
    affinity = hybrid_ot_distance_to_affinity(
        dist_matrix=dist_matrix,
        gamma=args.gamma,
        knn_k=args.knn_k,
    )
    print(f"Affinity matrix shape: {affinity.shape}")

    # 8) 谱聚类
    pred_labels = run_spectral_clustering(
        affinity=affinity,
        num_clusters=len(args.classes),
        random_state=args.seed,
    )

    # 9) 评估
    metrics = evaluate_clustering(labels, pred_labels)
    print("\n=== Clustering Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()