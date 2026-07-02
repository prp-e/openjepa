"""
Offline-safe datasets (Section 8). NO dataset is ever downloaded automatically.
"""
from __future__ import annotations

import os
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # Pillow ships as a torchvision dependency; not a separate install

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp")


def _list_images(root: str) -> List[str]:
    """Recursively collect image paths -- works for a flat folder OR class subfolders
    (labels are never read, since I-JEPA pretraining is self-supervised)."""
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(IMG_EXTENSIONS):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


class LocalImageFolderDataset(Dataset):
    """
    Loads images from a local directory only. Transform pipeline is intentionally minimal
    per Section 8: Resize -> RandomCrop/CenterCrop -> ToTensor -> Normalize.
    NO color-jitter, NO multi-crop -- I-JEPA's learning signal comes from masking, not
    augmentation invariance.
    """

    def __init__(self, root: str, img_size: int = 224, train: bool = True):
        self.root = root
        self.paths = _list_images(root)
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images with extensions {IMG_EXTENSIONS} found under '{root}'. "
                f"Use SyntheticDataset for an offline smoke test instead."
            )
        crop = transforms.RandomCrop(img_size) if train else transforms.CenterCrop(img_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),  # resize short side to img_size, keep aspect ratio
                crop,
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)  # (3, img_size, img_size)


class SyntheticDataset(Dataset):
    """
    Pure-random-tensor dataset for offline pipeline validation (shapes, mask sampling,
    forward/backward pass, checkpoint save/load) when no local image directory is available.

    IMPORTANT: loss will NOT meaningfully decrease when training on this data -- there is no
    real structure in random noise for the model to learn. This class exists ONLY to verify
    that the code runs end-to-end without errors, never to validate representation quality.
    """

    def __init__(self, num_samples: int = 256, img_size: int = 224):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(3, self.img_size, self.img_size)
