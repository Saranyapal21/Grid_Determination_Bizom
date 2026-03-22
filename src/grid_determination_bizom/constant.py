#   NOTE:
#   'yellow_building_count' seems a very important column
#   It alone makes the classes completely seperable

#   When we drop this column, classes start overlapping
#   We need to verify both of the cases (with and without the 'yellow_building_count')
from pathlib import Path

COLS_TO_DROP = [
    "grid_id",
    "latitude",
    "longitude",
    "waterbody",
    "building_count",
    "water_percent",
    "empty",
]


COLS_TO_KEEP = [
    "building_density",
    "yellow_building_density",
    "total_building_density",
    "main_road_density",
    "local_road_density",
    "total_road_density",
    "usable_area_percent",
    "kirana_potential_index",
    "yellow_building_count",
    "local_road_proportion",
    "dist_to_nearest_hub",
]


SAVE_DIR = Path(__file__).parents[2] / "outputs"
