"""Evaluate K-means partitions with the standalone metric."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from chi_index import chi_index_score


def main():
    source = Path(__file__).resolve().parent / "data" / "iris.data"
    df = pd.read_csv(source, header=None)
    X, classes = df.iloc[:, :-1], df.iloc[:, -1]
    for k in range(2, 11):
        labels = KMeans(
            n_clusters=k,
            n_init=100,
            max_iter=500,
            init="random",
            random_state=0,
        ).fit_predict(X)
        print(k, chi_index_score(labels, classes))


if __name__ == "__main__":
    main()
