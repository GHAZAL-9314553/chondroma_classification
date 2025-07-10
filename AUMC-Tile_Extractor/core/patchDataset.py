# core/datasets/patch_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image


class PatchDataset(Dataset):
    def __init__(
        self,
        patch_dir: str,
        transform: Optional[Callable] = None,
        format: str = 'png'
    ):
        """
        Dataset for loading image patches.

        Args:
            patch_dir (str): Directory containing image patches.
            transform (Callable, optional): Optional transform to be applied.
            format (str): File format to load. Supported: ['png', 'npy']
        """
        self.patch_dir = patch_dir
        self.transform = transform
        self.format = format.lower()

        supported = ['png', 'npy']
        if self.format not in supported:
            raise ValueError(f"Unsupported format '{self.format}'. Supported formats: {supported}")

        self.file_list = sorted([
            f for f in os.listdir(self.patch_dir)
            if f.endswith(f'.{self.format}')
        ])

        if not self.file_list:
            raise FileNotFoundError(f"No .{self.format} files found in {self.patch_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        path = os.path.join(self.patch_dir, fname)

        if self.format == 'png':
            image = Image.open(path).convert("RGB")
            image = self._apply_transform(image)

        elif self.format == 'npy':
            arr = np.load(path)
            image = self._to_tensor(arr)

        return image

    def _apply_transform(self, image: Image.Image) -> torch.Tensor:
        if self.transform:
            return self.transform(image)
        image_np = np.array(image)
        return self._to_tensor(image_np)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        Convert ndarray (HWC or HW) to torch Tensor (CHW).
        """
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).float() / 255.0
