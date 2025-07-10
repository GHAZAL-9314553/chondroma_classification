import os
from typing import List
from core.patch_extractor import BasePatchExtractor

class BatchExtractor:
    def __init__(
        self,
        extractors: List[BasePatchExtractor],
        save_dirs: List[str],
        mask_dirs: List[str] = None,
    ):
        assert len(extractors) == len(save_dirs), "Length mismatch."
        if mask_dirs:
            assert len(mask_dirs) == len(extractors), "Mismatch with mask_dirs"

        self.extractors = extractors
        self.save_dirs = save_dirs
        self.mask_dirs = mask_dirs

    def run(self):
        for idx, extractor in enumerate(self.extractors):
            save_dir = self.save_dirs[idx]
            mask_dir = self.mask_dirs[idx] if self.mask_dirs else None
            print(f"[INFO] Extracting patches to: {save_dir}")
            extractor.extract(save_dir, mask_dir)
