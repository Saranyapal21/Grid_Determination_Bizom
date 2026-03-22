from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from grid_determination_bizom.utils.math_utils import safe_divide
from grid_determination_bizom.constant import COLS_TO_KEEP


class DataFrameFeatureTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=True)
        self.scaler = StandardScaler()

        hub_threshold = self.df["total_building_density"].quantile(0.95)
        hubs_df = self.df[self.df["total_building_density"] >= hub_threshold][
            ["latitude", "longitude"]
        ].dropna()
        hubs_rad = np.radians(hubs_df[["latitude", "longitude"]].values)

        self.all_coords_rad = np.radians(self.df[["latitude", "longitude"]].values)
        self.tree = BallTree(hubs_rad, metric="haversine")

    def __call__(self, k: int = 1, radius: float = 6371.0) -> pd.DataFrame:
        self.get_primary_transformed_features()
        self.get_distance_to_nearest_hub(k=k, radius=radius)

        return self.scale_and_finalize_df(), self.df

    def get_primary_transformed_features(self) -> None:
        self.df["usable_area_percent"] = 100.0 - (
            self.df["empty"] + self.df["water_percent"]
        )
        self.df["kirana_potential_index"] = np.log2(
            1 + self.df["yellow_building_count"]
        ) * safe_divide(
            self.df["yellow_building_density"], self.df["total_building_density"]
        )
        self.df["local_road_proportion"] = safe_divide(
            self.df["local_road_density"], self.df["total_road_density"]
        )

    def get_distance_to_nearest_hub(self, k: int = 1, radius: float = 6371.0) -> None:
        distances_rad, indices = self.tree.query(self.all_coords_rad, k=k)
        self.df["dist_to_nearest_hub"] = distances_rad * radius

    def scale_and_finalize_df(
        self, cols_to_keep: List[str] = COLS_TO_KEEP
    ) -> pd.DataFrame:
        df_reduced = self.df[cols_to_keep]
        X_scaled = StandardScaler().fit_transform(df_reduced)

        return X_scaled


def convert_to_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    sorted_df = df.sort_values(by="grid_id", ascending=True).reset_index(drop=True)
    return sorted_df
