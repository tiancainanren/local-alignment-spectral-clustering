from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from backbone import extract_global_features, load_clip_model
from datasets.cifar10_subset import CIFAR10Subset, CIFARSubsetConfig
from evaluate import evaluate_clustering
from spectral import build_affinity_matrix, run_spectral_clustering


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Global baseline on CIFAR-10 subset")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train", action="store_true", help="Use CIFAR-10 train split")
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
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--affinity_mode",
        type=str,
        default="cosine_shift",
        choices=["cosine_shift", "rbf_from_cosine_distance"],
    )
    parser.add_argument("--gamma", type=float, default=5.0)
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
    print(f"Mapped labels: 0 ~ {len(args.classes) - 1}")

    # 2) 模型
    model, preprocess = load_clip_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    print(f"Loaded CLIP model: {args.model_name} ({args.pretrained}) on {args.device}")

    # 3) 全局特征提取
    features, labels = extract_global_features(
        images=images,
        labels=labels,
        preprocess=preprocess,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Extracted global features: {features.shape}")

    # 4) 构造 affinity matrix
    affinity = build_affinity_matrix(
        features=features,
        mode=args.affinity_mode,
        gamma=args.gamma,
        knn_k=args.knn_k,
    )
    print(f"Affinity matrix shape: {affinity.shape}")

    # 5) 谱聚类
    pred_labels = run_spectral_clustering(
        affinity=affinity,
        num_clusters=len(args.classes),
        random_state=args.seed,
    )

    # 6) 评估
    metrics = evaluate_clustering(labels, pred_labels)
    print("\n=== Clustering Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()