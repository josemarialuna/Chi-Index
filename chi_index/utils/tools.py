"""Small file and time utilities."""

from datetime import datetime
from pathlib import Path


def whatTimeIsIt(time_format="%d/%m/%Y"):
    """Return local time formatted using ``datetime.strftime`` directives."""
    return datetime.now().strftime(time_format)


def save_dataframe(df, results_path, filename):
    """Save a tab-separated DataFrame, creating the destination directory."""
    directory = Path(results_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, sep="\t", encoding="utf-8")
    return path
