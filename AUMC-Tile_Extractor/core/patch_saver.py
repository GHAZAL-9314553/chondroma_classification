import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Literal


class BaseSaver(ABC):
    @abstractmethod
    def save(self, patch: np.ndarray, path: str):
        """Save a single patch to the given path or buffer."""
        pass

    def close(self):
        """Optionally close resources like HDF5 files."""
        pass


class PngSaver(BaseSaver):
    def save(self, patch: np.ndarray, path: str):
        from PIL import Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image = Image.fromarray(patch)
        image.save(path)


class HDF5Saver(BaseSaver):
    def __init__(self, hdf5_path: str, dataset_name: str = "patches"):
        import h5py
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name
        self._initialized = False
        self.index = 0
        self.h5file = h5py.File(hdf5_path, "w")

    def save(self, patch: np.ndarray, path: Optional[str] = None):
        if not self._initialized:
            self.dataset = self.h5file.create_dataset(
                self.dataset_name,
                shape=(0,) + patch.shape,
                maxshape=(None,) + patch.shape,
                dtype=patch.dtype,
                chunks=True
            )
            self._initialized = True

        self.dataset.resize((self.index + 1,) + patch.shape)
        self.dataset[self.index] = patch
        self.index += 1

    def close(self):
        self.h5file.close()


def get_saver(format: Literal["png", "hdf5"], path: str) -> BaseSaver:
    """
    Utility to return the correct saver based on format.
    Example:
        saver = get_saver("png", "/output/tile_0_0.png")
        saver = get_saver("hdf5", "/output/patches.h5")
    """
    if format == "png":
        return PngSaver()
    elif format == "hdf5":
        return HDF5Saver(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
