from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from grid_determination_bizom.core import run_clustering_pipeline


FILE_PATH = Path(__file__).parents[2] / "data" / "delhi_grids.csv"

if __name__ == "__main__":
    print("Running Gaussian Mixture model clustering....")
    run_clustering_pipeline(
        FILE_PATH,
        save_as_csv=False,
        estimator=GaussianMixture,
        n_components=3,
        covariance_type="full",
        random_state=20,
    )

    # print("Running K-Means clustering....")
    # run_clustering_pipeline(
    #     FILE_PATH,
    #     save_as_csv=False,
    #     estimator=KMeans,
    #     n_clusters=3,
    #     random_state=20
    # )
