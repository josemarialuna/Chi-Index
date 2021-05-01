import matplotlib.pyplot as plt
from sklearn import cluster
from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import datetime


def whatTimeIsIt():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")


## PARAMETERS##
filePath = "C:/Users/josemaria.luna/Downloads/"  # untreated-9-47k #pos-37k.tif
fileName = "-42.csv"
k = 8
classIndex = -1
idIndex = 0

### PRINTING PARAMETERS ###
print("\n*******CALCULATING INDICES********\n")
print("Parameters:")
print("\tImage name: " + str(fileName))
print("\tNumber of clusters: " + str(k))
print("\nInitializing process...\n")
###########################


script_dir = os.path.dirname(__file__)
titulo = fileName + "_" + str(k)
results_dir = os.path.join(script_dir, whatTimeIsIt() + "_" + titulo + "_Res/")

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

df = pd.read_csv(filePath + fileName, delimiter=",")

## Remove class and index columns
cols = ['country', 'brand', 'therapeutic_area', 'num_generics', 'presentation', 'month_name', 'A', 'B', 'C', 'D']

X = np.array(df.drop(cols, axis=1))


kmeans = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=5,
                           max_iter_barycenter=5,
                          random_state=0).fit(X)

centroids = kmeans.cluster_centers_
np.savetxt(results_dir + whatTimeIsIt() + "_centroids.csv", centroids, delimiter=',')

# print("centroids")
#     # print(centroids)

plt.figure("centroids")
plt.title("Centroids")  # Establece el título del gráfico
plt.xlabel("Features")  # Establece el título del eje x
plt.ylabel("Values")  # Establece el título del eje y

plt.grid(True)
for values in range(len(centroids)):
    plt.plot(range(len(centroids[0])), centroids[values], label="Cluster " + str(values))
plt.legend(loc='upper left')
plt.savefig(results_dir + whatTimeIsIt() + "_centroids.png")

# # Create the default pairplot
# sns_plot = sns.pairplot(df, hue='Class')
# sns_plot.savefig(results_dir + whatTimeIsIt() + "_pairclass.png")

labels = kmeans.predict(X)
df.loc[:, 'clusters'] = labels
df.to_csv(results_dir + whatTimeIsIt() + '_kmeans_winner.csv', sep='\t')

# # Create the default pairplot
# sns_plot2 = sns.pairplot(df, hue='clusters')
# sns_plot2.savefig(results_dir + whatTimeIsIt() + "_paircluster.png")
#
# sns_plot3 = sns.pairplot(df, hue='Class')
# sns_plot3.savefig(results_dir + whatTimeIsIt() + "_class_cluster.png")

# df[['id', 'Class', 'clusters']].to_csv(results_dir + whatTimeIsIt() + '_kmeans_winner.csv', sep='\t')
#
# df_cluster = pd.crosstab(df.clusters, df.Class, normalize='index')
# df_features = pd.crosstab(df.Class, df.clusters, normalize='index').T
#
# df_cluster *= 100
# df_features *= 100
#
# df_cluster.to_csv(results_dir + whatTimeIsIt() + '_df_cluster_winner.csv', sep='\t')
# df_features.to_csv(results_dir + whatTimeIsIt() + '_df_features_winner.csv', sep='\t')

