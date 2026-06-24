from pathlib import Path

#   This will be used for segregating low potential from (medium + high) potential [Stage 1 clustering]
COLS_TO_KEEP_1 = [
    "total_building_density",
    "kirana_potential_index_log",
    "dist_to_nearest_hub",
    "green_cover_ratio",
]


#   This will be used for segregating medium potential from high potential [Stage 2 clustering]
COLS_TO_KEEP_2 = [
    "avg_yellow_building_size",
    "local_road_proportion",
    "edge_density",
    "market_activity_index",
]


ALL_COLS = [
    #   CSV FEATURES
    "grid_id",
    "latitude",
    "longitude",
    "building_count",
    "yellow_building_count",
    "building_density",
    "yellow_building_density",
    "total_building_density",
    "yellow_building_proportion",
    "avg_yellow_building_size",
    "main_road_density",
    "local_road_density",
    "total_road_density",
    "water_percent",
    "water_percent_log",
    "waterbody",
    "empty",
    "usable_area_percent",
    "kirana_potential_index",
    "kirana_potential_index_log",
    "local_road_proportion",
    "dist_to_nearest_hub",
    #   IMAGE FEATURES
    "green_cover_ratio",
    "edge_density",
    "structural_complexity",
    #   COMBINED FEATURES
    "market_activity_index",
    "street_exposure_index",
    #   OUTPUT COLUMNS
    "grid_score",
    "grid_segment",
]


SAVE_DIR = Path(__file__).parents[2] / "outputs"
VIZ_DIR = SAVE_DIR / "clustered_final.csv"
