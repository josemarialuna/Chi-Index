"""Implements the main logic of Clustering Validity Chi Index."""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn import cluster
from chi_index.utils.tools import save_dataframe
from chi_index.utils.plot_utils import Visualizer


class ChiIndex:
    """Main class of the library.

    Parameters
    ----------
    df : pandas.DataFrame
        Where the dataset is going to be loaded.
    k_ini: int
        Initial number of clusters. Default= 2.
    k_end: int
        Last number of clusters. Default= 5.
    results_path: str
        Path where the results are going to be saved. Default= '.'
    log_file: Optional[str]
    save_results: bool
        Would you like to save the results or not?

    Attributes
    ----------
    self.results_path: str
        Path where the results are going to be saved.
    self.list_chi: list
        List of the chi index values.

    self.optimum_chi: int
        The optimum chi value .
    self.optimum_k: int
        The optimum number of clusters.

    Methods
    -------
    kmeans(self, k: int, X)
        Applies the k-means algorithm.
    chi_index(self, df: pd.DataFrame, k: int)
        Calculate the chi index to the clustering solution with the given number of clusters (k).
    save_dataframe(self, df:pd.Dataframe, filename:str):
        Save the dataframe (df) into the self.results_path with the 'filename'.
    save_centroids(self):
        Save the centroids into a file

    """

    def __init__(
            self,
            df: pd.DataFrame,
            k_ini: int = 2,
            k_end: int = 5,
            results_path: str = '.',
            #log_file: Optional[str] = None,
            save_results: bool = True
    ) -> None:
        self.results_path = results_path
        self.list_chi = []
        self.optimum_chi = None
        self.optimum_k = None

        k_width = range(k_ini, k_end + 1)

        X = np.array(df.drop(['Class'], axis=1))

        if save_results:
            os.makedirs(self.results_path, exist_ok=True)

        for k in k_width:
            self.kmeans(k, X)

            # # Create the default pairplot
            # sns_plot = sns.pairplot(df, hue='Class')
            # sns_plot.savefig(f'{results_dir}{k}_pairclass.png')

            labels = self.kmeans_model.predict(X)
            df.loc[:, 'clusters'] = labels
            if save_results:
                save_dataframe(df, self.results_path, f'{k}_kmeans_winner.csv')

            # # Create the default pairplot
            # sns_plot2 = sns.pairplot(df, hue='clusters')
            # sns_plot2.savefig((f'{results_dir}{k}_paircluster.png')
            #
            # sns_plot3 = sns.pairplot(df, hue='Class')
            # sns_plot3.savefig((f'{results_dir}{k}_class_cluster.png')

            self.chi_index(df, k)

        df_chi = pd.DataFrame(self.list_chi, columns=[
            'k', 'chi1', 'chi1_max', 'chi2', 'chi2_max', 'chi_index'])
        self.optimum_chi = df_chi['chi_index'].max()
        self.optimum_k = int(df_chi['k'].where(
            df_chi['chi_index'] == self.optimum_chi).values[0])

        save_dataframe(df_chi, self.results_path,'chi_index_result.csv')

    def kmeans(self, k: int, X: np.array) -> None:
        """Launch the sklearn k-means algorithm

        Parameters
        ----------
        k: int
            The number of clusters.
        X: np.array
            The data to be clustered. These data must not have the class value.

        Returns
        -------
        None
            It saves the model into self.kmeans_model
        """
        self.kmeans_model = cluster.KMeans(
            n_clusters=k, n_init=100, max_iter=500, init='random').fit(X)

    def chi_index(self, df: pd.DataFrame, k: int) -> None:
        """Calculate the chi index value for the given k

        Parameters
        ----------
        df: pd.DataFrame
            The clustering solution to calculate the chi index value.
        k: int
            The number of clusters.        

        Returns
        -------
        None
            It saves the chi index value into self.list_chi
        """
        df_cluster = pd.crosstab(df.clusters, df.Class, normalize='index')
        df_features = pd.crosstab(df.Class, df.clusters, normalize='index').T

        df_cluster *= 100
        df_features *= 100

        chi1 = chi2_contingency(df_cluster)[0]
        chi2 = chi2_contingency(df_features)[0]

        r = df_cluster.shape[0]
        c = df_features.shape[0]

        if r <= c:
            chi1_max = 100 * r * (r - 1)
            chi2_max = 100 * c * (r - 1)
        else:
            chi1_max = 100 * r * (c - 1)
            chi2_max = 100 * c * (c - 1)

        chi_index_value = (chi1 / chi1_max) + (chi2 / chi2_max) - \
            abs((chi1 / chi1_max) - (chi2 / chi2_max))

        self.list_chi.append(
            (k, chi1, chi1_max, chi2, chi2_max, chi_index_value))

        save_dataframe(df_cluster, self.results_path, f'{k}_df_cluster_winner.csv')
        save_dataframe(df_features, self.results_path, f'{k}_df_features_winner.csv')

    def save_centroids(self):
        """Save the centroids into a file"""
        vis = Visualizer('Centroids', 'features', 'values',
                         path=self.results_path, filename='data')
        vis.centroids(self.kmeans_model, save_image=True)
