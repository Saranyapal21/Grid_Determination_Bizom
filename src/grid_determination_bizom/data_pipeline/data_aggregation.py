import pandas as pd

from grid_determination_bizom.data_pipeline.csv_pipeline.data_loader_and_transformer import (
    DataFrameFeatureTransformer,
)
from grid_determination_bizom.data_pipeline.image_pipeline.image_feature_extractor import (
    ImageFeatureExtractor,
)


class DataAggregator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.csv_extractor = DataFrameFeatureTransformer(self.df)
        self.img_extractor = ImageFeatureExtractor()

    def __call__(self) -> pd.DataFrame:
        self.csv_features_df = self.csv_extractor()
        self.img_features_df = self.img_extractor()

        self.combined_df = (
            pd.merge(
                self.csv_features_df, self.img_features_df, on="grid_id", how="inner"
            )
            .sort_values(by="grid_id", ascending=True)
            .reset_index(drop=True)
        )

        self.calculate_combined_features()
        return self.combined_df

    def calculate_combined_features(self):
        self.combined_df["market_activity_index"] = (
            self.combined_df["structural_complexity"]
            * self.combined_df["yellow_building_proportion"]
        )
        self.combined_df["street_exposure_index"] = (
            self.combined_df["structural_complexity"]
            * self.combined_df["local_road_proportion"]
        )


def convert_to_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    sorted_df = df.sort_values(by="grid_id", ascending=True).reset_index(drop=True)
    return sorted_df
