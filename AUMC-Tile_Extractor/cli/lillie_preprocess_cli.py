import argparse
import yaml
import os
import glob

from core.masking_utils import UnifiedMasker
from core.tile_reader import get_wsi_reader
from core.patch_extractor import EfficientPatchExtractor
from core.batch_extractor import BatchExtractor
from core.annotation_loader import load_annotation_mask

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Tile Extractor CLI")
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--override_save_dir', type=str, help="Optional override save directory root")
    return parser.parse_args()

def expand_wsi_paths(paths, extensions=[".svs", ".tif", ".tiff"]):
    """Expand directories to full WSI paths"""
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

    # Load config entries
    raw_wsi_paths = config['data']['wsi_paths']
    save_root = config['data']['save_dirs']
    annotation_paths = config['data'].get('annotation_paths', [])

    reader_backend = config['reader']['backend']
    patch_size = config['extraction']['patch_size']
    stride = config['extraction'].get('stride', patch_size)
    level = config['extraction'].get('level', 0)

    use_otsu = config['masking'].get('use_otsu', True)
    use_pen_filter = config['masking'].get('use_pen_filter', True)
    use_annotation = config['masking'].get('use_annotation', False)

    # Expand WSI file paths
    wsi_paths = expand_wsi_paths(raw_wsi_paths)

    extractors = []

    for wsi_path in wsi_paths:
        reader = get_wsi_reader(reader_backend, wsi_path)

        # Setup save_dir per WSI (flattened structure)
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        base_save_dir = args.override_save_dir or save_root
        save_dir = os.path.join(base_save_dir, wsi_name)

        # Setup annotation if used
        annotation_mask = None
        if use_annotation:
            matching_annotation = next((a for a in annotation_paths if wsi_name in a), None)
            if matching_annotation:
                shape = (reader.get_dimensions()[1], reader.get_dimensions()[0])
                annotation_mask = load_annotation_mask(matching_annotation, shape)

        masker = UnifiedMasker(
            use_otsu=use_otsu,
            use_pen_filter=use_pen_filter,
            use_annotation=use_annotation,
            annotation_mask=annotation_mask
        )

        extractor = EfficientPatchExtractor(
            reader=reader,
            masker=masker,
            patch_size=patch_size,
            stride=stride,
            level=level
        )

        extractors.append((extractor, save_dir))

    batch = BatchExtractor(
        extractors=[e[0] for e in extractors],
        save_dirs=[e[1] for e in extractors]
    )
    batch.run()

if __name__ == "__main__":
    main()