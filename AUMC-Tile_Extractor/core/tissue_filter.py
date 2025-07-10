import torch
import numpy as np
from torchvision import transforms


class TissueClassifier:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Make sure model input size matches this
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def is_tissue(self, tile: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Classifies a tile as tissue or non-tissue.

        Args:
            tile (np.ndarray): RGB image array (H, W, 3)
            threshold (float): Probability threshold

        Returns:
            bool: True if tissue, False otherwise
        """
        if tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError("Expected a 3-channel RGB tile.")

        input_tensor = self.transform(tile).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(input_tensor)).item()

        return prob >= threshold
