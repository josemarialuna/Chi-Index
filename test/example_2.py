"""Select a K-means model; plotting is optional and requested with --plot."""

import argparse
from pathlib import Path

import pandas as pd

from chi_index import ChiIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Export centroid PNG (plot extra)")
    parser.add_argument("--output", type=Path, default=Path("result"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "data" / "iris.data"
    df = pd.read_csv(source, header=None).rename(columns={4: "Class"})
    chi = ChiIndex(df, results_path=args.output, random_state=0)
    print(chi.results_.to_string(index=False))
    print(f"Best k: {chi.optimum_k}; Chi Index: {chi.optimum_chi:.6f}")
    if args.plot:
        print(chi.save_centroids())


if __name__ == "__main__":
    main()
