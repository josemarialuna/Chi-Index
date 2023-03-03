import pandas as pd
from chi_index.model import ChiIndex


def main():
    df = pd.read_csv('./test/data/iris.data', delimiter=",", header=None)
    print(df.columns)
    print(df.head())
    df.rename(columns={4: 'Class'}, inplace=True)

    chi = ChiIndex(df, results_path='result')
    print(chi.list_chi)
    print(chi.optimum_chi)
    print(chi.optimum_k)
    chi.save_centroids()


if __name__ == "__main__":
    main()
