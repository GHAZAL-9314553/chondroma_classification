from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

# CuCIM Setup
try:
    import cucim
    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False

# OpenSlide Setup
try:
    import openslide
    from PIL import Image
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

class BaseWSIReader(ABC):
    """Abstract base class for WSI readers."""
    def __init__(self, wsi_path: str):
        self.wsi_path = wsi_path

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_downsample_levels(self) -> int:
        pass

    @abstractmethod
    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        pass


class CuCIMWSIReader(BaseWSIReader):
    """WSI reader using CuCIM backend."""

    def __init__(self, wsi_path: str):
        if not CUCIM_AVAILABLE:
            raise ImportError("CuCIM is not available.")
        super().__init__(wsi_path)
        self._slide = None

    def _load_slide(self):
        if self._slide is None:
            self._slide = cucim.CuImage(self.wsi_path)

    def get_dimensions(self) -> Tuple[int, int]:
        self._load_slide()
        width, height = self._slide.size()[:2]
        return int(width), int(height)

    def get_downsample_levels(self) -> int:
        self._load_slide()
        return len(self._slide.resolutions['level_dimensions'])

    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        self._load_slide()
        tile = self._slide.read_region(
            location=(x, y),
            size=(width, height),
            level=level
        )
        return np.array(tile)[:, :, :3]


class OpenSlideWSIReader(BaseWSIReader):
    """WSI reader using OpenSlide backend."""

    def __init__(self, wsi_path: str):
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is not available.")
        super().__init__(wsi_path)
        self._slide = openslide.OpenSlide(wsi_path)

    def get_dimensions(self) -> Tuple[int, int]:
        return self._slide.level_dimensions[0]  # width, height

    def get_downsample_levels(self) -> int:
        return self._slide.level_count

    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        tile = self._slide.read_region((x, y), level, (width, height)).convert("RGB")
        return np.array(tile)


def get_wsi_reader(backend: str, wsi_path: str) -> BaseWSIReader:
    backend = backend.lower()
    if backend == "cucim":
        return CuCIMWSIReader(wsi_path)
    elif backend == "openslide":
        return OpenSlideWSIReader(wsi_path)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Choose 'cucim' or 'openslide'.")
