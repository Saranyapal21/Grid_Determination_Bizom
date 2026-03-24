import os
import time
from pathlib import Path

import pandas as pd

from grid_determination_bizom.constant import SAVE_DIR
from grid_determination_bizom.data_clustering.clustering import ClusteringModel
from grid_determination_bizom.data_pipeline.data_aggregation import (
    DataAggregator,
    convert_to_dataframe,
)


def save_df_as_csv(df: pd.DataFrame, output_dir: Path, csv_name: str) -> None:
    full_path = os.path.join(output_dir, csv_name)
    print(f"Saving csv file at: {full_path}")
    df.to_csv(full_path, index=False)


def run_clustering_pipeline(
    file_path: str,
    estimator,
    save_as_csv: bool = False,
    output_dir: str = SAVE_DIR,
    **kwargs,
) -> pd.DataFrame:

    df = convert_to_dataframe(file_path)
    transformer = DataAggregator(df=df)
    X_scaled, new_df = transformer()

    model = ClusteringModel(estimator=estimator(**kwargs))
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    new_df["labels"] = labels
    if probs is not None:
        for i in range(probs.shape[1]):
            new_df[f"prob_cluster_{i}"] = probs[:, i]
    else:
        print(f"Note: No probability columns added for {estimator.__name__}.")

    print(new_df["labels"].value_counts())

    if save_as_csv:
        timestamp = int(time.time() * 1000)
        csv_name = "_clustered_" + str(timestamp) + ".csv"
        save_df_as_csv(new_df, output_dir, csv_name)
