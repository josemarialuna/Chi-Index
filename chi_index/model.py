"""Implements the main logic of Clustering Validity Chi Index."""
from typing import Optional

import numpy as np
import logging
import os

import matplotlib.pyplot as plt
import pandas
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn import cluster

from chi_index.utils.plot_utils import Visualizer
from chi_index.utils.tools import Tools


class ChiIndex:
    def __init__(
            self,
            df: pandas.DataFrame,
            k_ini: int = 2,
            k_end: int = 5,
            results_path: str = '.',
            log_file: Optional[str] = None,
            save_results = True
    ) -> None:
        self.results_path = results_path
        self.list_chi = []

        k_width = range(k_ini, k_end + 1)

        X = np.array(df.drop(['Class'], axis=1))

        if save_results: os.makedirs(self.results_path, exist_ok=True)

        for k in k_width:
            self.kmeans(k, X)

            # # Create the default pairplot
            # sns_plot = sns.pairplot(df, hue='Class')
            # sns_plot.savefig(f'{results_dir}{k}_pairclass.png')

            labels = self.kmeans_model.predict(X)
            df.loc[:, 'clusters'] = labels
            if save_results: self.save_dataframe(df=df, filename=f'{k}_kmeans_winner.csv')

            # # Create the default pairplot
            # sns_plot2 = sns.pairplot(df, hue='clusters')
            # sns_plot2.savefig((f'{results_dir}{k}_paircluster.png')
            #
            # sns_plot3 = sns.pairplot(df, hue='Class')
            # sns_plot3.savefig((f'{results_dir}{k}_class_cluster.png')


            self.chi_index(df, k)

        self.df_chi = pd.DataFrame(self.list_chi, columns=['k', 'chi1', 'chi1_max', 'chi2', 'chi2_max', 'chi_index'])
        self.optimum_chi = self.df_chi['chi_index'].max()
        self.optimum_k = int(self.df_chi['k'].where(self.df_chi['chi_index']==self.optimum_chi).values[0])
        self.save_dataframe(df=self.df_chi, filename='chi_index_result.csv')

    def kmeans(self, k: int, X):
        self.kmeans_model = cluster.KMeans(n_clusters=k, n_init=100, max_iter=500, init='random').fit(X)

    def chi_index(self, df: pandas.DataFrame, k: int):
        df_cluster = pd.crosstab(df.clusters, df.Class, normalize='index')
        df_features = pd.crosstab(df.Class, df.clusters, normalize='index').T

        df_cluster *= 100
        df_features *= 100

        chi1, p, dof, ex = chi2_contingency(df_cluster)
        chi2, p, dof, ex = chi2_contingency(df_features)

        r = df_cluster.shape[0]
        c = df_features.shape[0]

        if r <= c:
            chi1_max = 100 * r * (r - 1)
            chi2_max = 100 * c * (r - 1)
        else:
            chi1_max = 100 * r * (c - 1)
            chi2_max = 100 * c * (c - 1)

        chi_index_value = (chi1 / chi1_max) + (chi2 / chi2_max) - abs((chi1 / chi1_max) - (chi2 / chi2_max))

        self.list_chi.append((k, chi1, chi1_max, chi2, chi2_max, chi_index_value))

        self.save_dataframe(df_cluster, f'{k}_df_cluster_winner.csv')
        self.save_dataframe(df_features, f'{k}_df_features_winner.csv')

    def save_dataframe(self, df, filename):
        df.to_csv(f'{self.results_path}/{filename}', sep='\t')

    def save_centroids(self):
        vis = Visualizer('Centroids', 'features', 'values', path=self.results_path, filename='data')
        vis.centroids(self.kmeans_model, save_image=True)

    def get_optimum_chi(self):
        return self.optimum_chi
