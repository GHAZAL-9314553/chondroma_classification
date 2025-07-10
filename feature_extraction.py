import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import timm

class PatchDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.paths = patch_paths
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx]

def extract_features_for_slide(slide_dir, model, device, output_path):
    patch_paths = sorted([
        os.path.join(slide_dir, f)
        for f in os.listdir(slide_dir)
        if f.endswith(('.png', '.jpg'))
    ])

    dataset = PatchDataset(patch_paths, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ]))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model.eval()
    model.to(device)
    all_feats = []

    with torch.no_grad():
        for batch, _ in tqdm(loader, desc=f"{os.path.basename(slide_dir)}"):
            batch = batch.to(device)
            feats = model.forward_features(batch)  # timm ViT models use forward_features()
            feats = model.forward_head(feats, pre_logits=True)
            all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    torch.save(all_feats, output_path)
    print(f"✅ Saved features to {output_path}")

def main(data_root, output_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_small_patch16_224", pretrained=True)

    os.makedirs(output_root, exist_ok=True)
    slide_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d))]

    for slide_dir in slide_dirs:
        slide_name = os.path.basename(slide_dir)
        output_path = os.path.join(output_root, f"{slide_name}.pt")
        extract_features_for_slide(slide_dir, model, device, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()

    main(args.data_root, args.output_root)
