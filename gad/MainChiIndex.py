import matplotlib.pyplot as plt
from sklearn import cluster
from tslearn.generators import random_walks
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import datetime
from scipy.stats import chi2_contingency, chisquare


def whatTimeIsIt():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")


## PARAMETERS##
path = "C:/Users/josemaria.luna/PycharmProjects/LondonEnergy/data/"
filename = "dataset_365_full_acorn.csv"

k_ini = 2
k_end = 10
k_width = range(k_ini, k_end + 1)
classIndex = -1
class_name = 'Acorn_grouped'
idIndex = 0

### PRINTING PARAMETERS ###
print("\n*******CALCULATING CHI INDEX********\n")
print("Parameters:")
print("\tFile name: " + str(filename))
print("\nInitializing process...\n")
###########################

the_time = whatTimeIsIt()
script_dir = os.path.dirname(__file__)
titulo = filename
results_dir = os.path.join(script_dir, the_time + "_" + titulo + '_by_' + class_name + "/")

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

df = pd.read_csv(path + filename, delimiter=",")
print(df.head())
df.rename(columns={class_name: 'Class'}, inplace=True)

## Remove class and index columns
cols = ['id', 'Class', 'Acorn']

list_chi = []

X = np.array(df.drop(cols, axis=1))

for k in k_width:

    kmeans = cluster.KMeans(n_clusters=k, n_init=100, max_iter=500, init='random', random_state=101287).fit(X)
    centroids = kmeans.cluster_centers_

    # print("centroids")
    #     # print(centroids)

    plt.figure(f'centroids_{k}')
    plt.title("Centroids")  # Establece el título del gráfico
    plt.xlabel("Features")  # Establece el título del eje x
    plt.ylabel("Values")  # Establece el título del eje y

    plt.grid(True)
    for values in range(len(centroids)):
        plt.plot(range(len(centroids[0])), centroids[values], label="Cluster " + str(values))
    plt.legend(loc='upper left')
    plt.savefig(f'{results_dir}{k}_centroids.png')

    # # Create the default pairplot
    # sns_plot = sns.pairplot(df, hue='Class')
    # sns_plot.savefig(f'{results_dir}{k}_pairclass.png')

    labels = kmeans.predict(X)
    df.loc[:, 'clusters'] = labels
    df.to_csv(f'{results_dir}{k}_kmeans_winner.csv', sep='\t')

    # # Create the default pairplot
    # sns_plot2 = sns.pairplot(df, hue='clusters')
    # sns_plot2.savefig((f'{results_dir}{k}_paircluster.png')
    #
    # sns_plot3 = sns.pairplot(df, hue='Class')
    # sns_plot3.savefig((f'{results_dir}{k}_class_cluster.png')

    df[['id', 'Class', 'clusters']].to_csv(f'{results_dir}{k}_kmeans_winner.csv', sep='\t')

    df_cluster = pd.crosstab(df.clusters, df.Class, normalize='index')
    df_features = pd.crosstab(df.Class, df.clusters, normalize='index').T

    df_cluster *= 100
    df_features *= 100

    chi1, p, dof, ex = chi2_contingency(df_cluster)
    chi2, p, dof, ex = chi2_contingency(df_features)

    r = df_cluster.count()
    c = df_features.count()

    if r <= c:
        chi1_max = 100 * r * (r - 1)
        chi2_max = 100 * c * (r - 1)
    else:
        chi1_max = 100 * r * (c - 1)
        chi2_max = 100 * c * (c - 1)

    chi_index_value = chi1 / chi1_max + chi2 / chi2_max - abs(chi1 / chi1_max + chi2 / chi2_max)

    list_chi.append((k, chi1, chi1_max, chi2, chi2_max, chi_index_value))

    df_cluster.to_csv(f'{results_dir}{k}_df_cluster_winner.csv', sep='\t')
    df_features.to_csv(f'{results_dir}{k}_df_features_winner.csv', sep='\t')

df_chi = pd.DataFrame(list_chi)
df_chi.to_csv(f'{results_dir}chi_index_result.csv', sep='\t')
