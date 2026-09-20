"""External clustering validation using the Chi Index (Luna-Romera et al., 2019)."""

from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .utils.tools import save_dataframe


def _labels(values, name):
    """Validate labels without aligning pandas indices or coercing mixed types."""
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    for value in values:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable scalar labels.") from exc
        missing = pd.isna(value)
        if not np.isscalar(missing) or missing:
            raise ValueError(f"{name} must contain non-missing scalar labels.")
        if isinstance(value, (float, complex, np.inexact)) and not np.isfinite(value):
            raise ValueError(f"{name} must not contain infinite labels.")
    return values


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _contingency(cluster_labels, class_labels):
    clusters = _labels(cluster_labels, "cluster_labels")
    classes = _labels(class_labels, "class_labels")
    if len(clusters) != len(classes):
        raise ValueError("cluster_labels and class_labels must have equal lengths.")
    row_codes, row_labels = pd.factorize(clusters, sort=False)
    col_codes, col_labels = pd.factorize(classes, sort=False)
    counts = np.zeros((len(row_labels), len(col_labels)), dtype=np.int64)
    np.add.at(counts, (row_codes, col_codes), 1)
    return pd.DataFrame(
        counts,
        index=pd.Index(row_labels, dtype=object, name="cluster"),
        columns=pd.Index(col_labels, dtype=object, name="class"),
    )


def _evaluate(cluster_labels, class_labels):
    """Return percentage tables and (row chi, row max, col chi, col max, score)."""
    counts = _contingency(cluster_labels, class_labels)
    rows = counts.div(counts.sum(axis=1), axis=0) * 100.0
    cols = counts.div(counts.sum(axis=0), axis=1) * 100.0
    r, c = counts.shape
    row_max = 100.0 * r * (min(r, c) - 1)
    col_max = 100.0 * c * (min(r, c) - 1)
    # The paper's denominators vanish for a single cluster or class.
    # Define these uninformative partitions to have score zero.
    if min(r, c) == 1:
        return rows, cols, (0.0, row_max, 0.0, col_max, 0.0)
    row_chi = float(chi2_contingency(rows, correction=False)[0])
    col_chi = float(chi2_contingency(cols, correction=False)[0])
    score = float(np.clip(2.0 * min(row_chi / row_max, col_chi / col_max), 0.0, 2.0))
    return rows, cols, (row_chi, row_max, col_chi, col_max, score)


def chi_index_score(
    cluster_labels,
    class_labels,
    k=None,
    verbose=False,
    save_results=False,
    *,
    results_path="results",
):
    """Return the Chi Index in [0, 2]; larger values indicate better agreement.

    Labels are paired by position, regardless of pandas indices. Missing labels,
    empty inputs and unequal lengths are rejected. Single-cluster or single-class
    partitions return zero by convention. ``k`` is an optional positive integer
    used only in export filenames; normalization uses observed label counts.
    Exports are tab-separated relative-frequency tables (fractions, not percent).
    File-system errors propagate to the caller. No files are written by default.
    """
    if k is not None:
        k = _positive_integer(k, "k")
    rows, cols, values = _evaluate(cluster_labels, class_labels)
    if k is None:
        k = len(rows)
    if save_results:
        save_dataframe(rows / 100.0, results_path, f"{k}_df_cluster_result.csv")
        save_dataframe(cols / 100.0, results_path, f"{k}_df_features_result.csv")
    if verbose:
        print(
            f"r={rows.shape[0]}, c={rows.shape[1]}, chi_row={values[0]}, "
            f"chi_col={values[2]}, chi_row_max={values[1]}, chi_col_max={values[3]}"
        )
    return values[-1]
