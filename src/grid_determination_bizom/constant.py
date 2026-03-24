from pathlib import Path

#   This is being used...
COLS_TO_KEEP = [
    #   CSV FEATURES
    # "building_density",
    # "yellow_building_density",
    # "yellow_building_proportion",
    "avg_yellow_building_size",
    "total_building_density",
    # "main_road_density",
    # "local_road_density",
    # "total_road_density",
    # "water_percent",
    "water_percent_log",
    # "usable_area_percent",
    # "kirana_potential_index",
    "kirana_potential_index_log",
    # "yellow_building_count",
    "local_road_proportion",
    "dist_to_nearest_hub",
    
    #   IMAGE FEATURES
    "green_cover_ratio",
    # "edge_density",
    "structural_complexity",

    #   COMBINED FEATURES
    # "market_activity_index",
    # "street_exposure_index",
]


SAVE_DIR = Path(__file__).parents[2] / "outputs"
VIZ_DIR = Path(__file__).parents[2] / "outputs" / "_clustered_1774389117329.csv"

print(SAVE_DIR)
print(VIZ_DIR)