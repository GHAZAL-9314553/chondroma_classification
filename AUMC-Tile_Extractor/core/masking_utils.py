
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from histolab.filters.image_filters import BluePenFilter, GreenPenFilter, RedPenFilter
from PIL import Image

def ensure_array(tile):
    if isinstance(tile, Image.Image):
        return np.array(tile)
    return tile

class BaseMasker(ABC):
    @abstractmethod
    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        pass

class OtsuMasker(BaseMasker):
    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        tile = ensure_array(tile)
        gray = rgb2gray(tile)
        thresh = threshold_otsu(gray)
        mask = gray < thresh  # Tissue assumed to be darker than background

        # OPTIONAL: skip tile if tissue < 5% of area
        if np.sum(mask) < 0.05 * mask.size:
            return np.zeros_like(mask, dtype=bool)

        return mask.astype(bool)

class PenFilterMasker(BaseMasker):
    def __init__(self):
        self.filters = [RedPenFilter(), GreenPenFilter(), BluePenFilter()]

    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        tile_img = Image.fromarray(tile) if not isinstance(tile, Image.Image) else tile
        for f in self.filters:
            tile_img = f(tile_img)
        tile_np = np.array(tile_img)
        return np.ones(tile_np.shape[:2], dtype=bool)  # No mask logic, just cleanup

class AnnotationMasker(BaseMasker):
    def __init__(self, annotation_mask: np.ndarray):
        self.annotation_mask = annotation_mask.astype(bool)

    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        tile = ensure_array(tile)
        return self.annotation_mask

class UnifiedMasker(BaseMasker):
    def __init__(self, use_otsu=False, use_pen_filter=False, use_annotation=False, annotation_mask=None):
        self.use_otsu = use_otsu
        self.use_pen_filter = use_pen_filter
        self.use_annotation = use_annotation
        self.annotation_mask = annotation_mask

        self.maskers = []
        if use_pen_filter:
            self.maskers.append(PenFilterMasker())
        if use_otsu:
            self.maskers.append(OtsuMasker())
        if use_annotation and annotation_mask is not None:
            self.maskers.append(AnnotationMasker(annotation_mask))

    def get_mask(self, tile: np.ndarray) -> np.ndarray:
        tile_np = ensure_array(tile)
        if not self.maskers:
            return np.ones(tile_np.shape[:2], dtype=bool)

        mask = np.ones(tile_np.shape[:2], dtype=bool)
        for masker in self.maskers:
            mask &= masker.get_mask(tile_np)

        return mask
