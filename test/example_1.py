import pandas as pd
from chi_index import metrics
from sklearn import cluster
import numpy as np

def main():
    df = pd.read_csv('./test/data/iris.data', delimiter=",", header=None)
    print(df.columns)
    print(df.head())
    df.rename(columns={4: 'Class'}, inplace=True)

    X = np.array(df.drop(['Class'], axis=1))

    for clusters_num in range(2,11):        
        # Clustering stage
        kmeans_model = cluster.KMeans(n_clusters=clusters_num, n_init=100, max_iter=500, init='random').fit(X)
        labels = kmeans_model.predict(X)
        df.loc[:, 'cluster'] = labels   # saves the clustering labels into 'cluster' new column

        # chi_index_score receives the clustering result array and the class array
        valor = metrics.chi_index_score(df['cluster'], df['Class'], k=clusters_num)
        print(clusters_num , '\t', valor)


if __name__ == "__main__":
    main()