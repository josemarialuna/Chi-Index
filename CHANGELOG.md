# Changelog

## 3.0.0 — 2026-09-20

This major version makes input and side-effect contracts explicit
and raises the minimum supported Python version to 3.10.

### Correctness

- Share one metric implementation between the function and K-means class.
- Use the class count for column normalization, including rectangular tables.
- Use Pearson without Yates correction, as defined by the paper.
- Select the actual maximum score and retain that model for centroid export.
- Treat single-cluster/single-class inputs as score zero; reject invalid labels.
- Pair pandas labels by position and ignore unused categorical levels.
- Honor `save_results=False` throughout the class.
- Fix the time utility and release figures after centroid export.

### Migration from 2.1.1

- Requires Python 3.10 or newer.
- `ChiIndex` no longer modifies the input DataFrame. Read `labels_` instead.
- Input `clusters` columns are rejected to prevent accidental feature leakage.
- Invalid ranges, missing/nonfinite inputs and excess k over distinct rows
  raise clear exceptions.
- `random_state=0` is now the default; use `None` for the former random behavior.
- `kmeans_model` now contains the best model, rather than the last fitted model.
- Install `chi-index[plot]` for plotting. Core imports do not need Matplotlib.
- Standalone `k` is optional. Positive supplied values still name export files.
- Export errors now propagate instead of being printed and ignored.
- Numerical results may change due to the normalization and Pearson fixes.
- Generated results and Python bytecode are removed from version control.

### Maintenance

- Add modern package metadata, tests, coverage gate, linting and CI.
- Add paper verification notes, executable examples and contributor guidance.
- Preserve MIT licensing, authorship and the original scientific citation.
