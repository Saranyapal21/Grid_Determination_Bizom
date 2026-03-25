import os
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from grid_determination_bizom.constant import SAVE_DIR, COLS_TO_KEEP_1, COLS_TO_KEEP_2
from grid_determination_bizom.data_clustering.clustering import ClusteringModel
from grid_determination_bizom.data_pipeline.data_aggregation import (
    DataAggregator,
    convert_to_dataframe,
)


def save_df_as_csv(df: pd.DataFrame, output_dir: str | Path, csv_name: str) -> None:
    full_path = os.path.join(output_dir, csv_name)
    print(f"Saving csv file at: {full_path}")
    df.to_csv(full_path, index=False)


def apply_clustering(
    df: pd.DataFrame,
    estimator,
    cols_to_keep: List[str],
    label_names: Tuple[str, str],
    stage_name: str,
    **kwargs,
) -> pd.DataFrame:

    df_reduced = df[cols_to_keep]
    X_scaled = StandardScaler().fit_transform(df_reduced)

    model = ClusteringModel(estimator=estimator(**kwargs))
    labels = model.predict(X_scaled)
    new_df = df.copy()
    new_df["labels"] = labels

    median_0 = new_df[new_df["labels"] == 0]["yellow_building_count"].median()
    median_1 = new_df[new_df["labels"] == 1]["yellow_building_count"].median()

    low_cluster_label = 0 if median_0 < median_1 else 1
    lower_name, higher_name = label_names

    new_df["predicted potential"] = new_df["labels"].apply(
        lambda x: lower_name if x == low_cluster_label else higher_name
    )
    print(f"{stage_name} Results:")
    print(new_df["predicted potential"].value_counts())

    return new_df


def run_clustering_pipeline(
    file_path: str,
    estimator,
    cols_to_keep_1: List[str] = COLS_TO_KEEP_1,
    cols_to_keep_2: List[str] = COLS_TO_KEEP_2,
    save_as_csv: bool = False,
    output_dir: str = SAVE_DIR,
    **kwargs,
) -> pd.DataFrame:

    raw_df = convert_to_dataframe(file_path)
    transformer = DataAggregator(df=raw_df)
    processed_master_df = transformer()

    stage_1_df = apply_clustering(
        df=processed_master_df,
        estimator=estimator,
        cols_to_keep=cols_to_keep_1,
        label_names=("low potential", "stage_2_candidate"),
        stage_name="Stage_1",
        **kwargs,
    )

    low_potential_df = stage_1_df[
        stage_1_df["predicted potential"] == "low potential"
    ].copy()
    candidate_df = stage_1_df[
        stage_1_df["predicted potential"] == "stage_2_candidate"
    ].copy()
    next_input_df = processed_master_df.loc[candidate_df.index].copy()

    stage_2_df = apply_clustering(
        df=next_input_df,
        estimator=estimator,
        cols_to_keep=cols_to_keep_2,
        label_names=("medium potential", "high potential"),
        stage_name="Stage_2",
        **kwargs,
    )

    final_combined_df = pd.concat(
        [low_potential_df, stage_2_df], axis=0, ignore_index=True
    )
    final_combined_df.drop(columns=["labels"], inplace=True)

    print("Final Output")
    print(final_combined_df["predicted potential"].value_counts())

    if save_as_csv:
        timestamp = int(time.time() * 1000)
        csv_name = "_clustered_final_" + str(timestamp) + ".csv"
        save_df_as_csv(final_combined_df, output_dir, csv_name)

    return final_combined_df
