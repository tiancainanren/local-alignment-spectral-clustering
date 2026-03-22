from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import open_clip
except ImportError as exc:
    raise ImportError(
        "Please install open_clip_torch first: pip install open_clip_torch"
    ) from exc


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, preprocess):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.preprocess(self.images[idx])
        label = self.labels[idx]
        return image, label


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def extract_global_features(
    images,
    labels,
    preprocess,
    model,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = ImageListDataset(images, labels, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        feats = model.encode_image(batch_images)
        feats = F.normalize(feats, dim=-1)

        all_features.append(feats.cpu())
        all_labels.append(batch_labels)

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return features, labels


@torch.no_grad()
def extract_patch_features(
    patch_lists,
    preprocess,
    model,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 2,
) -> np.ndarray:
    """
    提取所有图像的 patch 特征。

    Args:
        patch_lists:
            List[List[PIL.Image]]
            长度为 N，每个元素是一张图的 patch 列表，长度为 M
        preprocess:
            CLIP preprocess
        model:
            CLIP model

    Returns:
        patch_features: np.ndarray, shape (N, M, D)
    """
    flattened_patches = []
    patch_count_per_image = None

    for patches in patch_lists:
        if patch_count_per_image is None:
            patch_count_per_image = len(patches)
        else:
            if len(patches) != patch_count_per_image:
                raise ValueError("All images must have the same number of patches.")
        flattened_patches.extend(patches)

    dummy_labels = [0] * len(flattened_patches)
    dataset = ImageListDataset(flattened_patches, dummy_labels, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    all_features: List[torch.Tensor] = []

    for batch_images, _ in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        feats = model.encode_image(batch_images)
        feats = F.normalize(feats, dim=-1)
        all_features.append(feats.cpu())

    flat_features = torch.cat(all_features, dim=0).numpy()  # (N*M, D)

    num_images = len(patch_lists)
    patch_features = flat_features.reshape(num_images, patch_count_per_image, -1)
    return patch_features

from typing import Sequence


@torch.no_grad()
def extract_global_and_patch_features(
    images: Sequence,
    labels,
    patch_lists,
    preprocess,
    model,
    device: str = "cuda",
    image_batch_size: int = 64,
    patch_batch_size: int = 256,
    num_workers: int = 2,
):
    """
    同时提取全局特征和 patch 特征。

    Args:
        images: List[PIL.Image]
        labels: List[int] or np.ndarray
        patch_lists: List[List[PIL.Image]]
        preprocess: CLIP preprocess
        model: CLIP model

    Returns:
        global_features: np.ndarray, (N, D)
        labels:          np.ndarray, (N,)
        patch_features:  np.ndarray, (N, M, D)
    """
    global_features, labels = extract_global_features(
        images=images,
        labels=labels,
        preprocess=preprocess,
        model=model,
        device=device,
        batch_size=image_batch_size,
        num_workers=num_workers,
    )

    patch_features = extract_patch_features(
        patch_lists=patch_lists,
        preprocess=preprocess,
        model=model,
        device=device,
        batch_size=patch_batch_size,
        num_workers=num_workers,
    )

    return global_features, labels, patch_features
