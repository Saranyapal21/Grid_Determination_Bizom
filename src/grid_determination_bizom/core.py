import os
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    probs = model.predict_proba(X_scaled)

    new_df = df.copy()
    new_df["labels"] = labels

    median_0 = new_df[new_df["labels"] == 0]["yellow_building_count"].median()
    median_1 = new_df[new_df["labels"] == 1]["yellow_building_count"].median()

    low_cluster_label = 0 if median_0 < median_1 else 1
    high_cluster_label = 1 if low_cluster_label == 0 else 0

    lower_name, higher_name = label_names

    new_df["grid_segment"] = new_df["labels"].apply(
        lambda x: lower_name if x == low_cluster_label else higher_name
    )

    new_df[f"{stage_name}_prob_higher"] = probs[:, high_cluster_label]
    print(f"{stage_name} Results:")
    print(new_df["grid_segment"].value_counts())

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
        label_names=("bad", "stage_2_candidate"),
        stage_name="Stage_1",
        **kwargs,
    )

    low_potential_df = stage_1_df[stage_1_df["grid_segment"] == "bad"].copy()
    candidate_df = stage_1_df[stage_1_df["grid_segment"] == "stage_2_candidate"].copy()
    next_input_df = candidate_df.copy()

    stage_2_df = apply_clustering(
        df=next_input_df,
        estimator=estimator,
        cols_to_keep=cols_to_keep_2,
        label_names=("neutral", "good"),
        stage_name="Stage_2",
        **kwargs,
    )

    final_combined_df = pd.concat(
        [low_potential_df, stage_2_df], axis=0, ignore_index=True
    )
    final_combined_df.drop(columns=["labels"], inplace=True)

    features_for_score = final_combined_df[
        ["kirana_potential_index", "market_activity_index", "total_building_density"]
    ]

    scaler = MinMaxScaler()
    scaled_feats = scaler.fit_transform(features_for_score)
    raw_continuous = (
        scaled_feats[:, 0] * 0.50
        + scaled_feats[:, 1] * 0.30
        + scaled_feats[:, 2] * 0.20
    )
    c_min = raw_continuous.min()
    c_max = raw_continuous.max()
    if c_max == c_min:
        final_combined_df["grid_score"] = 50.0
    else:
        final_combined_df["grid_score"] = (
            1.0 + ((raw_continuous - c_min) / (c_max - c_min)) * 99.0
        )

    final_combined_df["grid_score"] = final_combined_df["grid_score"].round(2)
    final_combined_df.drop(
        columns=["Stage_1_prob_higher", "Stage_2_prob_higher"], inplace=True
    )

    print("Final Output Segments:")
    print(final_combined_df["grid_segment"].value_counts())

    if save_as_csv:
        timestamp = int(time.time() * 1000)
        csv_name = "_clustered_final_" + str(timestamp) + ".csv"
        save_df_as_csv(final_combined_df, output_dir, csv_name)
        # Stable-named copy so the viewer app never needs the timestamp.
        save_df_as_csv(final_combined_df, output_dir, "clustered_final.csv")
        save_df_as_csv(
            final_combined_df[["grid_id", "grid_score", "grid_segment"]],
            output_dir,
            "enriched_output.csv",
        )

    return final_combined_df
