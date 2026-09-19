"""K-means model selection using the same Chi Index as the standalone metric."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .metrics import _evaluate, _labels, _positive_integer
from .utils.tools import save_dataframe


class ChiIndex:
    """Fit K-means for an inclusive range of k and retain the best model.

    Construction performs the search. ``df`` must contain a ``Class`` column
    and finite numeric features. The caller's frame is never modified. Ties
    select the smallest k. ``random_state=0`` makes the default reproducible.
    ``save_results=False`` disables all automatic file output.

    ``list_chi`` contains (k, row chi, row max, column chi, column max, score).
    ``results_`` holds the same information as a DataFrame. ``kmeans_model``,
    ``labels_`` and ``cluster_centers_`` belong to ``optimum_k``.
    """

    def __init__(
        self,
        df,
        k_ini=2,
        k_end=5,
        results_path=".",
        save_results=True,
        *,
        random_state=0,
        n_init=100,
        max_iter=500,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not df.columns.is_unique or "Class" not in df.columns:
            raise ValueError("df must have unique columns including 'Class'.")
        if "clusters" in df.columns:
            raise ValueError("'clusters' is reserved for output; remove it from the input.")
        _labels(df["Class"], "Class")
        k_ini = _positive_integer(k_ini, "k_ini")
        k_end = _positive_integer(k_end, "k_end")
        if not k_ini <= k_end <= len(df):
            raise ValueError("Require k_ini <= k_end <= number of samples.")
        features = df.drop(columns="Class")
        if features.shape[1] == 0 or any(
            not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
            for dtype in features.dtypes
        ):
            raise ValueError("Features must be non-empty, real numeric columns.")
        X = features.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(X).all():
            raise ValueError("Features must contain only finite values.")
        if k_end > np.unique(X, axis=0).shape[0]:
            raise ValueError("k_end must not exceed the number of distinct feature rows.")
        self.results_path = results_path
        self.save_results = save_results
        self.random_state = random_state
        self.n_init = _positive_integer(n_init, "n_init")
        self.max_iter = _positive_integer(max_iter, "max_iter")
        self.list_chi = []
        self.optimum_chi = -np.inf
        self.optimum_k = None
        best_model = None
        for k in range(k_ini, k_end + 1):
            self.kmeans(k, X)
            working = df.copy(deep=True)
            working["clusters"] = self.kmeans_model.labels_
            self.chi_index(working, k)
            score = self.list_chi[-1][-1]
            if score > self.optimum_chi:
                self.optimum_chi, self.optimum_k = score, k
                best_model = self.kmeans_model
            if save_results:
                save_dataframe(working, results_path, f"{k}_kmeans_winner.csv")
        self.kmeans_model = best_model
        self.labels_ = best_model.labels_.copy()
        self.cluster_centers_ = best_model.cluster_centers_.copy()
        self.results_ = pd.DataFrame(
            self.list_chi,
            columns=[
                "k",
                "chi1",
                "chi1_max",
                "chi2",
                "chi2_max",
                "chi_index",
            ],
        )
        if save_results:
            save_dataframe(self.results_, results_path, "chi_index_result.csv")

    def kmeans(self, k, X):
        """Fit a K-means model and store it in ``kmeans_model``."""
        self.kmeans_model = KMeans(
            n_clusters=k,
            n_init=self.n_init,
            max_iter=self.max_iter,
            init="random",
            random_state=self.random_state,
        ).fit(X)

    def chi_index(self, df, k):
        """Append the score for an already labelled DataFrame to ``list_chi``."""
        k = _positive_integer(k, "k")
        rows, cols, values = _evaluate(df["clusters"], df["Class"])
        self.list_chi.append((k, *values))
        if self.save_results:
            save_dataframe(rows, self.results_path, f"{k}_df_cluster_winner.csv")
            save_dataframe(cols, self.results_path, f"{k}_df_features_winner.csv")

    def save_centroids(self):
        """Explicitly save the selected model's centroids as a PNG; return its path.

        Requires the optional ``plot`` extra, even if automatic exports are off.
        """
        from .utils.plot_utils import Visualizer

        vis = Visualizer("Centroids", "features", "values", self.results_path, "data")
        try:
            return vis.centroids(self.kmeans_model, save_image=True)
        finally:
            vis.close()
