# Contributing

Bug reports and pull requests are welcome. Please follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development environment

Use Python 3.10 or newer and create an isolated virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev,plot]"
```

## Required checks

```bash
python -m pytest --cov --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m build
python -m twine check --strict dist/*
```

Tests treat warnings as errors and require at least 90% coverage including
branches. CI also tests installation of the wheel without the plotting extra.
Do not commit virtual environments, generated CSV/PNG results, bytecode,
credentials or the publisher's PDF.

Changes to numerical behavior must reference the paper's equations or explain
an intentional deviation in `docs/mathematics.md`. Add regression tests using
independently calculated expectations, not only a second call to the function
under test. Keep tests deterministic and free of network access.

Document API changes in README.md and CHANGELOG.md. Preserve compatibility when
possible and call out breaking changes. Open a pull request describing the
problem, resulting behavior and validation. A maintainer reviews and merges it.
Package releases are a separate maintainer action. See [releasing](docs/releasing.md)
for the workflow that publishes approved GitHub releases to PyPI.
