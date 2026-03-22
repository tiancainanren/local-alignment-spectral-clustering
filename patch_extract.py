from __future__ import annotations

from typing import List
from PIL import Image


def extract_grid_patches(
    image: Image.Image,
    grid_size: int = 4,
    resize_size: int = 224,
) -> List[Image.Image]:
    """
    先将图像 resize 到固定大小，再切成 grid_size x grid_size 个规则 patch。

    Args:
        image: PIL.Image
        grid_size: 网格大小，例如 4 表示 4x4 = 16 个 patch
        resize_size: 先将整图 resize 到 resize_size x resize_size

    Returns:
        patches: List[PIL.Image]
    """
    image = image.convert("RGB")
    image = image.resize((resize_size, resize_size))

    patch_w = resize_size // grid_size
    patch_h = resize_size // grid_size

    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * patch_w
            upper = row * patch_h
            right = left + patch_w
            lower = upper + patch_h
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

    return patches