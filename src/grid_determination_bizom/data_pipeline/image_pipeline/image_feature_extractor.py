import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

IMAGE_DIR = Path(__file__).parents[4] / "data" / "delhi_grid_images"


class ImageFeatureExtractor:
    def __init__(self):
        self.image_dir = IMAGE_DIR

    def __call__(self) -> pd.DataFrame:
        features_list = []
        for filename in tqdm(
            os.listdir(self.image_dir), desc="Extracting Image Features"
        ):
            try:
                grid_id = int(filename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue

            filepath = self.image_dir / filename
            features = self.extract_image_features(str(filepath))

            if features:
                features["grid_id"] = grid_id
                features_list.append(features)

        return pd.DataFrame(features_list)

    def extract_image_features(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            return None

        self.total_pixels = self.img.shape[0] * self.img.shape[1]

        return {
            "green_cover_ratio": self.extract_green_cover_ratio(),
            "edge_density": self.calculate_edge_density(),
            "structural_complexity": self.calculate_structural_complexity(),
        }

    def extract_green_cover_ratio(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(green_mask)
        green_cover_ratio = green_pixels / self.total_pixels

        return green_cover_ratio

    def calculate_edge_density(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

        edge_pixels = cv2.countNonZero(edges)
        edge_density = edge_pixels / self.total_pixels

        return edge_density

    def calculate_structural_complexity(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=10000, qualityLevel=0.01, minDistance=5
        )

        corner_count = len(corners) if corners is not None else 0
        structural_complexity = (corner_count / self.total_pixels) * 10000

        return structural_complexity
