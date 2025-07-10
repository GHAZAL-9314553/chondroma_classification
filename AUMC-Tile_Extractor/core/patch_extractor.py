import os
import numpy as np
from typing import Optional
from PIL import Image
import multiprocessing as mp

from core.masking_utils import UnifiedMasker
from core.tile_reader import get_wsi_reader


def process_tile_worker(args):
    wsi_path, x, y, level, patch_size, save_dir, wsi_name, masker_config, mask_dir = args

    reader = get_wsi_reader("cucim", wsi_path)
    tile = reader.read_region(x, y, level, patch_size, patch_size)
    if isinstance(tile, Image.Image):
        tile = np.array(tile)

    if tile.shape[0] != patch_size or tile.shape[1] != patch_size:
        return

    masker = UnifiedMasker(**masker_config) if masker_config else None
    if masker:
        mask = masker.get_mask(tile)
        if not np.any(mask):
            return
    else:
        mask = None

    if tile.std() < 30:
        return

    if tile.dtype != np.uint8:
        tile = np.clip(tile, 0, 255).astype(np.uint8)
    if tile.ndim == 2:
        tile = np.stack([tile] * 3, axis=-1)
    elif tile.shape[2] == 4:
        tile = tile[:, :, :3]

    os.makedirs(save_dir, exist_ok=True)
    patch_filename = f"{wsi_name}_x{x}_y{y}.png"
    Image.fromarray(tile).save(os.path.join(save_dir, patch_filename))

    if mask_dir and mask is not None:
        mask_save_dir = os.path.join(mask_dir, wsi_name)
        os.makedirs(mask_save_dir, exist_ok=True)
        mask_img = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_img).save(os.path.join(mask_save_dir, patch_filename))


class EfficientPatchExtractor:
    def __init__(
        self,
        wsi_path: str,
        masker_config: Optional[dict] = None,
        patch_size: int = 256,
        stride: Optional[int] = None,
        level: int = 0,
        processes: int = 1
    ):
        self.wsi_path = wsi_path
        self.masker_config = masker_config
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.level = level
        self.processes = processes

    def extract(self, save_dir: str, mask_dir: Optional[str] = None) -> None:
        wsi_name = os.path.splitext(os.path.basename(self.wsi_path))[0]

        # ? ??? save_dir ???? wsi_name ????? ???? ?????? ????? ???
        if os.path.basename(os.path.normpath(save_dir)) == wsi_name:
            save_path = save_dir
        else:
            save_path = os.path.join(save_dir, wsi_name)

        if os.path.isdir(save_path):
            png_files = [f for f in os.listdir(save_path) if f.endswith(".png")]
            if len(png_files) > 0:
                print(f"? Skipping {wsi_name}: {len(png_files)} tiles already exist.")
                return

        reader = get_wsi_reader("cucim", self.wsi_path)
        width, height = reader.get_dimensions()
        factor = 2 ** self.level

        coords = [
            (self.wsi_path, x, y, self.level, self.patch_size, save_path, wsi_name, self.masker_config, mask_dir)
            for y in range(0, height, self.stride * factor)
            for x in range(0, width, self.stride * factor)
        ]

        if self.processes > 1:
            with mp.get_context("spawn").Pool(processes=self.processes) as pool:
                pool.map(process_tile_worker, coords)
        else:
            for coord in coords:
                process_tile_worker(coord)
