# core/normalizer.py
import numpy as np
import cv2
import json
from typing import Optional


class MacenkoNormalizer:
    def __init__(self, alpha: int = 1, beta: float = 0.15, light_intensity: int = 255):
        self.alpha = alpha
        self.beta = beta
        self.light_intensity = light_intensity
        self.stain_matrix: Optional[np.ndarray] = None
        self.max_sat: Optional[np.ndarray] = None

    def fit(self, target_image: np.ndarray):
        target_image = self._standardize_brightness(target_image)
        od = self._rgb_to_od(target_image)
        od = od[~np.any(od < self.beta, axis=1)]

        _, eigvecs = np.linalg.eigh(np.cov(od.T))
        eigvecs = eigvecs[:, [1, 2]]  # Keep top 2 eigenvectors

        # Make sure direction is consistent
        eigvecs[:, 0] *= np.sign(eigvecs[0, 0])
        eigvecs[:, 1] *= np.sign(eigvecs[0, 1])

        projected = np.dot(od, eigvecs)
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        min_phi = np.percentile(angles, self.alpha)
        max_phi = np.percentile(angles, 100 - self.alpha)

        v1 = np.dot(eigvecs, [np.cos(min_phi), np.sin(min_phi)])
        v2 = np.dot(eigvecs, [np.cos(max_phi), np.sin(max_phi)])

        self.stain_matrix = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)
        concentrations = np.linalg.lstsq(self.stain_matrix, od.T, rcond=None)[0]
        self.max_sat = np.percentile(concentrations, 99, axis=1, keepdims=True)

    def transform(self, image: np.ndarray) -> np.ndarray:
        assert self.stain_matrix is not None and self.max_sat is not None, "Normalizer not fitted."
        image = self._standardize_brightness(image)
        od = self._rgb_to_od(image)
        concentrations = np.linalg.lstsq(self.stain_matrix, od.T, rcond=None)[0]
        concentrations /= self.max_sat
        od_reconstructed = np.dot(self.stain_matrix, concentrations)
        recon_rgb = self._od_to_rgb(od_reconstructed.T)
        return recon_rgb.reshape(image.shape)

    def save_vector(self, path: str):
        data = {
            "stain_vectors": self.stain_matrix.tolist(),
            "max_sat": self.max_sat.tolist()
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_vector(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.stain_matrix = np.array(data["stain_vectors"])
        self.max_sat = np.array(data["max_sat"])

    @staticmethod
    def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32)
        rgb[rgb == 0] = 1  # avoid log(0)
        return -np.log(rgb / 255.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        return np.clip(255 * np.exp(-od), 0, 255).astype(np.uint8)

    @staticmethod
    def _standardize_brightness(img: np.ndarray) -> np.ndarray:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab[:, :, 0] = 128
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


def macenko_normalization(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    normalizer = MacenkoNormalizer()
    normalizer.fit(target)
    return normalizer.transform(source)
