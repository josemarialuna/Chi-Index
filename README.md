# Chi Index

[![CI](https://github.com/josemarialuna/Chi-Index/actions/workflows/ci.yml/badge.svg)](https://github.com/josemarialuna/Chi-Index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chi-index.svg)](https://pypi.org/project/chi-index/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

Chi Index evaluates a clustering partition against known class labels using
two normalized Pearson chi-squared statistics. Scores range from **0 to 2**;
larger is better. Class labels are used for evaluation, never as clustering
features. The standalone metric works with any hard clustering algorithm.

This repository implements the index from
[Luna-Romera et al., Information Sciences 487 (2019), 1–17](https://doi.org/10.1016/j.ins.2019.02.046).
See [mathematical details and paper discrepancies](docs/mathematics.md).
The paper's experiments used Spark; this package runs locally with NumPy,
pandas, SciPy and scikit-learn.

## Installation

The current development version is **3.0.0** and requires Python **3.10+**.
Version 3.0.0 is not published automatically by this repository.

```bash
git clone https://github.com/josemarialuna/Chi-Index.git
cd Chi-Index
python -m pip install .
# Optional plotting:
python -m pip install ".[plot]"
```

To install the latest published release (which may differ from this checkout):

```bash
python -m pip install chi-index
```

## Score an existing partition

```python
from chi_index import chi_index_score

score = chi_index_score(
    cluster_labels=[0, 0, 1, 1],
    class_labels=["a", "a", "b", "b"],
)
print(score)  # 2.0
```

Both sequences must be non-empty, one-dimensional, equally sized and contain
non-missing, finite, hashable scalar labels. Pairing is **positional**: pandas
Series indices are ignored. Unused categorical levels are ignored.

A single observed cluster or class returns **0**, including two constant
partitions. This is an explicit convention for zero denominators in the paper.
A score of 2 does not uniquely identify equal partitions: some rectangular
contingency tables also reach this value.

The optional third positional argument `k` remains supported. It only names
export files; normalization always uses the observed cluster and class counts.
By default the function writes no files. Set `save_results=True` and optionally
`results_path="results"` to export relative-frequency tables as fractions.
`verbose=True` prints the component statistics. File-system errors propagate.

## Select a K-means model

```python
from sklearn.datasets import load_iris
from chi_index import ChiIndex

iris = load_iris(as_frame=True)
df = iris.data.copy()
df["Class"] = iris.target

chi = ChiIndex(df, k_ini=2, k_end=5, save_results=False, random_state=0)
print(chi.results_)
print(chi.optimum_k, chi.optimum_chi)
labels = chi.labels_
centers = chi.cluster_centers_
```

Construction immediately fits each K-means model in the **inclusive** range.
The input DataFrame must have unique column names, a `Class` column, and at
least one finite real numeric feature. `clusters` is reserved for output.
Preprocess categorical features, missing values and feature scales yourself.
The input DataFrame is not modified.

The range must satisfy `1 <= k_ini <= k_end <= n_samples`; `k_end` must not
exceed the number of distinct feature rows. Defaults are `k_ini=2`, `k_end=5`,
`n_init=100`, `max_iter=500`, and `random_state=0`. A fixed seed makes repeated
runs reproducible within the same software environment; results can change
across dependency versions. Pass `random_state=None` for nondeterministic runs.

- `optimum_k`: k with maximum Chi Index; exact ties choose the smallest k.
- `optimum_chi`: selected score.
- `kmeans_model`, `labels_`, `cluster_centers_`: selected model and its outputs.
- `list_chi`: tuples of `(k, chi1, chi1_max, chi2, chi2_max, chi_index)`.
- `results_`: the same summary as a DataFrame.

The class retains the historical default `save_results=True`. Set it to
`False` to avoid **all automatic writes**. When enabled, `results_path`
(default `"."`) receives three tables per k and `chi_index_result.csv`.
Despite their historical `.csv` extension, all exports are **tab-separated**
and include an index column. Class exports contain percentages; standalone
metric exports contain fractions for compatibility.

To explicitly save the winning model's centroid plot (requires `plot`):

```python
chi.results_path = "result"
image_path = chi.save_centroids()
```

This explicit request writes `centroids_data.png` even if automatic output
was disabled. No model is serialized. Existing output filenames are overwritten.

The compatibility methods `kmeans(k, X)` and `chi_index(df, k)` remain available:
the former replaces `kmeans_model`; the latter appends a score to `list_chi`.
They do not rerun model selection or refresh the summary attributes. Construct
a new `ChiIndex` for a new search.

## Examples and development

The Iris data is included; no external dataset download is needed.
After installation, these commands work from the repository root:

```bash
python test/example_1.py
python test/example_2.py
python -m pip install -e ".[dev,plot]"
python -m pytest --cov --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m build
python -m twine check --strict dist/*
```

Tests cover the paper's contingency tables with an independent Pearson
calculation, invariance properties, degenerate and invalid inputs, model
selection, file exports and optional plotting. CI runs tests on Linux and
Windows with Python 3.10–3.14, and checks a built wheel without Matplotlib.
See [contributing](CONTRIBUTING.md) and [migration notes](CHANGELOG.md).

## Citation

José María Luna-Romera, María Martínez-Ballesteros, Jorge García-Gutiérrez,
José C. Riquelme. *External clustering validity index based on chi-squared
statistical test*. Information Sciences 487 (2019), 1–17.
[DOI: 10.1016/j.ins.2019.02.046](https://doi.org/10.1016/j.ins.2019.02.046).

## License

[MIT](LICENSE.txt). Please follow the [code of conduct](CODE_OF_CONDUCT.md).
