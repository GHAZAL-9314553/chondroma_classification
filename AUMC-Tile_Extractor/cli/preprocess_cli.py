
import argparse
import yaml
import os
import glob

from core.masking_utils import UnifiedMasker
from core.annotation_loader import load_annotation_mask
from core.patch_extractor import EfficientPatchExtractor

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="CuCIM-Safe Tile Extractor CLI")
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--override_save_dir', type=str, help="Optional override save directory root")
    parser.add_argument('--processes', type=int, default=8, help="Number of parallel processes for extraction")
    return parser.parse_args()

def expand_wsi_paths(paths, extensions=[".svs", ".tif", ".tiff"]):
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            for ext in extensions:
                expanded.extend(glob.glob(os.path.join(path, f"*{ext}")))
        else:
            expanded.append(path)
    return expanded

def main():
    args = parse_args()
    config = load_config(args.config)

    raw_wsi_paths = config['data']['wsi_paths']
    save_root = config['data']['save_dirs']
    annotation_paths = config['data'].get('annotation_paths', [])

    patch_size = config['extraction']['patch_size']
    stride = config['extraction'].get('stride', patch_size)
    level = config['extraction'].get('level', 0)

    use_otsu = config['masking'].get('use_otsu', True)
    use_pen_filter = config['masking'].get('use_pen_filter', True)
    use_annotation = config['masking'].get('use_annotation', False)

    wsi_paths = expand_wsi_paths(raw_wsi_paths)

    for wsi_path in wsi_paths:
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        base_save_dir = args.override_save_dir or save_root
        save_dir = os.path.join(base_save_dir, wsi_name)

        annotation_mask = None
        if use_annotation:
            matching_annotation = next((a for a in annotation_paths if wsi_name in a), None)
            if matching_annotation:
                import numpy as np
                from core.tile_reader import get_wsi_reader
                shape = get_wsi_reader("cucim", wsi_path).get_dimensions()[::-1]
                annotation_mask = load_annotation_mask(matching_annotation, shape)

        masker_config = {
            "use_otsu": use_otsu,
            "use_pen_filter": use_pen_filter,
            "use_annotation": use_annotation,
            "annotation_mask": annotation_mask
        }

        extractor = EfficientPatchExtractor(
            wsi_path=wsi_path,
            masker_config=masker_config,
            patch_size=patch_size,
            stride=stride,
            level=level,
            processes=args.processes
        )
        extractor.extract(save_dir)

if __name__ == "__main__":
    main()
