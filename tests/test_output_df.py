from pathlib import Path
from sklearn.mixture import GaussianMixture

from grid_determination_bizom.core import run_clustering_pipeline

FILE_PATH = Path(__file__).parents[1] / "data" / "delhi_grids.csv"


def test_run_clustering_pipeline():
    final_df = run_clustering_pipeline(
        FILE_PATH,
        save_as_csv=False,
        estimator=GaussianMixture,
        n_components=2,
        covariance_type="full",
        random_state=42,
    )

    all_columns = final_df.columns

    expected_columns = {
        "grid_id",
        "latitude",
        "longitude",
        "building_count",
        "yellow_building_count",
        "building_density",
        "yellow_building_density",
        "total_building_density",
        "main_road_density",
        "local_road_density",
        "total_road_density",
        "waterbody",
        "water_percent",
        "empty",
        "usable_area_percent",
        "kirana_potential_index",
        "yellow_building_proportion",
        "local_road_proportion",
        "avg_yellow_building_size",
        "water_percent_log",
        "kirana_potential_index_log",
        "dist_to_nearest_hub",
        "green_cover_ratio",
        "edge_density",
        "structural_complexity",
        "market_activity_index",
        "street_exposure_index",
        "grid_segment",
        "grid_score",
    }

    assert expected_columns.issubset(set(all_columns)), (
        f"Missing columns: {expected_columns - set(all_columns)}"
    )
