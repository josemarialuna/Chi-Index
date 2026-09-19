"""Model selection, side effects, validation and plotting regressions."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from chi_index import ChiIndex, Visualizer, chi_index_score, whatTimeIsIt


@pytest.fixture
def frame():
    X, y = make_blobs(n_samples=60, centers=3, cluster_std=3, random_state=17)
    result = pd.DataFrame(X, columns=["x", "y"])
    result["Class"] = y.astype(str)
    return result


def test_selects_middle_winner_and_preserves_input(frame, tmp_path):
    original = frame.copy(deep=True)
    destination = tmp_path / "absent"
    model = ChiIndex(
        frame, k_ini=2, k_end=4, save_results=False, results_path=destination, n_init=5
    )
    assert model.optimum_k == 3
    assert model.optimum_chi == pytest.approx(1.6287373737373738)
    assert model.kmeans_model.n_clusters == 3
    assert model.cluster_centers_.shape == (3, 2)
    assert chi_index_score(model.labels_, frame.Class) == model.optimum_chi
    pd.testing.assert_frame_equal(frame, original)
    assert not destination.exists()
    second = ChiIndex(frame, 2, 4, save_results=False, n_init=5)
    np.testing.assert_array_equal(model.labels_, second.labels_)
    pd.testing.assert_frame_equal(model.results_, second.results_)


def test_exported_scores_match_standalone_metric(frame, tmp_path):
    model = ChiIndex(frame, 2, 4, results_path=tmp_path, n_init=5)
    for k, *values in model.list_chi:
        saved = pd.read_csv(tmp_path / f"{k}_kmeans_winner.csv", sep="\t", index_col=0)
        assert chi_index_score(saved.clusters, saved.Class) == pytest.approx(values[-1])
        rows = pd.read_csv(tmp_path / f"{k}_df_cluster_winner.csv", sep="\t", index_col=0)
        np.testing.assert_allclose(rows.sum(axis=1), 100)
    assert (tmp_path / "chi_index_result.csv").exists()
    assert len(list(tmp_path.iterdir())) == 10


def test_ties_choose_smallest_k(frame):
    frame["Class"] = "same"
    model = ChiIndex(frame, 1, 3, save_results=False, n_init=2)
    assert model.optimum_k == 1
    assert model.optimum_chi == 0


@pytest.mark.parametrize(
    "params",
    [
        {"k_ini": 0},
        {"k_ini": 4, "k_end": 2},
        {"k_end": 61},
        {"k_end": True},
        {"n_init": 0},
        {"max_iter": 0},
    ],
)
def test_invalid_parameters(frame, params, tmp_path):
    with pytest.raises(ValueError):
        ChiIndex(frame, results_path=tmp_path / "absent", **params)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing_class",
        "missing_features",
        "text",
        "nan",
        "infinity",
        "complex",
        "missing_label",
        "duplicate_columns",
        "reserved",
        "too_few_distinct",
        "empty",
    ],
)
def test_invalid_frames(frame, kind, tmp_path):
    if kind == "missing_class":
        frame = frame.drop(columns="Class")
    elif kind == "missing_features":
        frame = frame[["Class"]]
    elif kind == "text":
        frame["x"] = "text"
    elif kind in {"nan", "infinity", "complex"}:
        frame["x"] = {"nan": np.nan, "infinity": np.inf, "complex": 1 + 2j}[kind]
    elif kind == "missing_label":
        frame.loc[0, "Class"] = None
    elif kind == "duplicate_columns":
        frame.columns = ["x", "x", "Class"]
    elif kind == "reserved":
        frame["clusters"] = 0
    elif kind == "too_few_distinct":
        frame[["x", "y"]] = 0
    elif kind == "empty":
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        ChiIndex(frame, results_path=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_non_frame_input():
    with pytest.raises(TypeError):
        ChiIndex([[1, 2]])


def test_nullable_numeric_features(frame):
    frame["x"] = frame["x"].astype("Float64")
    assert ChiIndex(frame, 2, 3, save_results=False, n_init=5).optimum_k == 3


def test_centroids_save_best_model_without_leaking_figures(frame, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = ChiIndex(frame, 2, 4, save_results=False, results_path=tmp_path, n_init=5)
    before = plt.get_fignums()
    path = model.save_centroids()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == before
    vis = Visualizer("test", "features", "values", tmp_path, "test")
    with pytest.raises(RuntimeError):
        vis.savefig(tmp_path / "before.png")
    vis.start()
    assert vis.centroids(model.kmeans_model) is None
    assert len(vis.ax.lines) == 3
    for line, center in zip(vis.ax.lines, model.cluster_centers_, strict=True):
        np.testing.assert_array_equal(line.get_ydata(), center)
    vis.close()


def test_time_format():
    datetime.strptime(whatTimeIsIt(), "%d/%m/%Y")
    datetime.strptime(whatTimeIsIt("%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M")
