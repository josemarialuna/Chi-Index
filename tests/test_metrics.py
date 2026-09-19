"""Numerical regression tests and mathematical invariants."""

import builtins
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from chi_index import chi_index_score
from chi_index.metrics import _evaluate


def labels_from_counts(counts):
    clusters, classes = [], []
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            clusters.extend([i] * count)
            classes.extend([j] * count)
    return clusters, classes


def reference_score(counts):
    # Independent scalar Pearson implementation of paper Eqs. (1), (5)-(13).
    r, c = len(counts), len(counts[0])
    row_totals = [sum(row) for row in counts]
    col_totals = [sum(row[j] for row in counts) for j in range(c)]
    scores = []
    for by_row in [True, False]:
        table = [
            [100 * counts[i][j] / (row_totals[i] if by_row else col_totals[j]) for j in range(c)]
            for i in range(r)
        ]
        rows = [sum(row) for row in table]
        cols = [sum(row[j] for row in table) for j in range(c)]
        total = sum(rows)
        statistic = 0.0
        for i in range(r):
            for j in range(c):
                expected = rows[i] * cols[j] / total
                statistic += (table[i][j] - expected) ** 2 / expected
        scores.append(statistic / (100 * (r if by_row else c) * (min(r, c) - 1)))
    return 2 * min(scores)


@pytest.mark.parametrize(
    "counts",
    [
        [[2, 8, 6], [6, 2, 0]],
        [[0, 2, 6], [6, 2, 0], [2, 6, 0]],
        [[0, 2, 5], [4, 1, 0], [2, 1, 0], [2, 6, 1]],
        [[3, 1], [2, 7]],
        [[6, 0, 0], [0, 0, 3], [0, 9, 0]],
    ],
)
def test_paper_counts_against_independent_pearson(counts):
    assert chi_index_score(*labels_from_counts(counts)) == pytest.approx(reference_score(counts))


@pytest.mark.parametrize("size", [2, 3, 10])
def test_perfect_partition_has_score_two(size):
    labels = np.repeat(np.arange(size), np.arange(1, size + 1))
    assert chi_index_score(labels, labels) == pytest.approx(2.0)


def test_independent_partition_has_score_zero():
    assert chi_index_score(*labels_from_counts([[2, 2, 2], [1, 1, 1], [3, 3, 3]])) == 0


@pytest.mark.parametrize("a,b", [([0, 0], [1, 2]), ([0, 1], [2, 2]), ([0], [1])])
def test_degenerate_partitions(a, b):
    assert chi_index_score(a, b) == 0


def test_random_tables_symmetry_permutations_and_replication():
    rng = np.random.default_rng(73)
    for _ in range(40):
        counts = rng.integers(1, 15, size=tuple(rng.integers(2, 8, size=2)))
        a, b = map(np.asarray, labels_from_counts(counts))
        score = chi_index_score(a, b)
        assert 0 <= score <= 2
        assert score == pytest.approx(reference_score(counts.tolist()))
        assert score == pytest.approx(chi_index_score(b, a))
        order = rng.permutation(len(a))
        assert score == pytest.approx(chi_index_score(a[order] + 11, -b[order]))
        assert score == pytest.approx(chi_index_score(np.tile(a, 3), np.tile(b, 3)))


def test_series_are_paired_by_position_and_unused_categories_ignored():
    a = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", "unused"]), index=[4, 5])
    b = pd.Series(["x", "y"], index=[9, 8])
    assert chi_index_score(a, b) == 2


def test_mixed_labels_are_not_coerced_to_strings():
    assert chi_index_score([1, "1", 1, "1"], ["a", "b", "a", "b"]) == 2


@pytest.mark.parametrize(
    "a,b",
    [
        ([], []),
        ([0], [0, 1]),
        ([[0], [1]], [0, 1]),
        ([None, 1], [0, 1]),
        ([0, np.nan], [0, 1]),
        ([0, np.inf], [0, 1]),
        ([0, pd.NA], [0, 1]),
        ([{}, 1], [0, 1]),
        ([0, 1], [0, None]),
    ],
)
def test_invalid_labels(a, b):
    with pytest.raises(ValueError):
        chi_index_score(a, b)


@pytest.mark.parametrize("k", [0, -1, True, 2.5, "2"])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        chi_index_score([0, 1], [0, 1], k=k)


def test_export_and_verbose(tmp_path, capsys):
    destination = tmp_path / "nested"
    assert (
        chi_index_score([0, 1], [0, 1], verbose=True, save_results=True, results_path=destination)
        == 2
    )
    assert "chi_row_max=200" in capsys.readouterr().out
    rows = pd.read_csv(destination / "2_df_cluster_result.csv", sep="\t", index_col=0)
    assert rows.to_numpy().sum() == 2
    assert (destination / "2_df_features_result.csv").exists()


def test_export_errors_propagate(tmp_path):
    target = tmp_path / "file"
    target.write_text("occupied")
    with pytest.raises(OSError):
        chi_index_score([0, 1], [0, 1], save_results=True, results_path=target)


def test_rectangular_normalization():
    _, _, result = _evaluate(*labels_from_counts([[2, 8, 6], [6, 2, 0]]))
    assert result[1] == 200
    assert result[3] == 300


def test_k_is_export_metadata_not_normalization(tmp_path):
    assert chi_index_score([0, 1], [0, 1], k=7, save_results=True, results_path=tmp_path) == 2
    assert (tmp_path / "7_df_cluster_result.csv").exists()


def test_perfect_score_does_not_imply_identical_partitions():
    assert chi_index_score(*labels_from_counts([[2, 0, 0], [0, 2, 2]])) == 2


def test_paper_worked_example_scores():
    tables = [
        [[2, 8, 6], [6, 2, 0]],
        [[0, 2, 6], [6, 2, 0], [2, 6, 0]],
        [[0, 2, 5], [4, 1, 0], [2, 1, 0], [2, 6, 1]],
    ]
    scores = [chi_index_score(*labels_from_counts(table)) for table in tables]
    assert scores == pytest.approx([0.9047619047619048, 0.925, 0.7602339181286547])
    assert np.argmax(scores) + 2 == 3


def test_import_and_metric_do_not_require_matplotlib(tmp_path):
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('matplotlib'):
        raise ImportError('Matplotlib deliberately unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from chi_index import chi_index_score
assert chi_index_score([0, 1], [0, 1]) == 2
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_dependency_message(monkeypatch):
    from chi_index import Visualizer

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ImportError, match="chi-index"):
        Visualizer("x", "x", "y", ".", "x").start()
