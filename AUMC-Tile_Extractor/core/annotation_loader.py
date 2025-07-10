import os
import json
import numpy as np
from typing import Tuple, Optional
from PIL import Image


def load_annotation_mask(path: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load annotation mask from a PNG or JSON file.

    Parameters:
        path (str): Path to the annotation file (.png or .json).
        shape (tuple, optional): Required if using JSON mask to define output shape (H, W).

    Returns:
        np.ndarray: Binary mask (bool) where annotated regions are True.
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".png":
        mask = np.array(Image.open(path).convert("L"))
        return mask > 0

    elif ext == ".json":
        if shape is None:
            raise ValueError("Shape (H, W) must be provided for JSON annotations.")
        return mask_from_json(path, shape)

    else:
        raise ValueError(f"Unsupported annotation format: '{ext}'. Supported: .png, .json")


def mask_from_json(json_path: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a binary mask from a JSON polygon annotation.

    Parameters:
        json_path (str): Path to the JSON file.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask (bool).
    """
    try:
        from shapely.geometry import Polygon
        from rasterio.features import rasterize
    except ImportError as e:
        raise ImportError("shapely and rasterio are required for JSON annotations.") from e

    with open(json_path, 'r') as f:
        data = json.load(f)

    polygons = [
        Polygon(obj["points"])
        for obj in data.get("shapes", [])
        if obj.get("type", "polygon") == "polygon"
    ]

    if not polygons:
        return np.zeros(shape, dtype=bool)

    mask = rasterize(
        [(poly, 1) for poly in polygons],
        out_shape=shape,
        fill=0,
        default_value=1,
        dtype='uint8'
    )
    return mask.astype(bool)
