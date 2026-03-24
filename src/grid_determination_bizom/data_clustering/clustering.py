from typing import List
import numpy as np


class ClusteringModel:
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator

    def predict(self, X_scaled: np.ndarray) -> List[str]:
        labels = self.estimator.fit_predict(X_scaled)
        return labels

    def predict_proba(self, X_scaled: np.ndarray) -> List[str]:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X_scaled)
        else:
            print(
                f"Warning: {self.estimator.__class__.__name__} does not support probabilities."
            )
            return None
