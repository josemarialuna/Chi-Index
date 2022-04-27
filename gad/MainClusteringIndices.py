import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
# from sklearn.metrics import calinski_harabaz_score
import numpy as np
import pandas as pd
import time
import os
import datetime


def whatTimeIsIt():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")


## PARAMETERS##
filePath = "C:/Users/josemaria.luna/PycharmProjects/LondonEnergy/data/"  # untreated-9-47k #pos-37k.tif
fileName = "dataset_365_full.csv"
k_ini = 2
k_end = 10
k_width = range(k_ini, k_end + 1)
classIndex = -1
idIndex = 0

### PRINTING PARAMETERS ###
print("\n*******CALCULATING INDICES********\n")
print("Parameters:")
print("\tImage name: " + str(fileName))
print("\tNumber of clusters: " + str(k_width))
print("\nInitializing process...\n")
###########################


script_dir = os.path.dirname(__file__)
titulo = fileName + "_" + str(k_width)
results_dir = os.path.join(script_dir, whatTimeIsIt() + "_" + titulo + "_indices/")

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

df = pd.read_csv(filePath + fileName, delimiter=",")

## Remove class and index columns
# cols = [classIndex, idIndex]
cols = ['id']
df.drop(cols, axis=1, inplace=True)

X = df.values

list_silhouette = []
list_davies = []
list_calinski = []
list_wssse = []

for k in k_width:
    print("Clustering with k=" + str(k) + "...")
    kmeans_cluster = cluster.KMeans(n_clusters=k, n_init=100, max_iter=500, init='random', random_state=101287,
                                    precompute_distances=False, n_jobs=4)
    cluster_labels = kmeans_cluster.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    list_silhouette.append(silhouette_avg)

    davies_avg = davies_bouldin_score(X, cluster_labels)
    list_davies.append(davies_avg)

    # calinski_avg = calinski_harabaz_score(X, cluster_labels)
    # list_calinski.append(calinski_avg)

    wssse = kmeans_cluster.inertia_
    list_wssse.append(wssse)

    print("Done!")

plt.figure("Silhouette")
plt.title(titulo + "_Silhouette")
plt.xlabel("Number of clusters")
plt.ylabel("Value")
plt.grid(True)
plt.xticks(np.arange(k_ini, k_end + 1, 1.0))
plt.plot(k_width, list_silhouette)
plt.savefig(results_dir + "Silhouette.png")

plt.figure("Davies-Bouldin")
plt.title(titulo + "_Davies-Bouldin")
plt.xlabel("Number of clusters")
plt.ylabel("Value")
plt.grid(True)
plt.xticks(np.arange(k_ini, k_end + 1, 1.0))
plt.plot(k_width, list_davies)
plt.savefig(results_dir + "Davies.png")

# plt.figure("Calinski-Harabaz")
# plt.title(titulo + "_Calinski-Harabaz")
# plt.xlabel("Number of clusters")
# plt.ylabel("Value")
# plt.grid(True)
# plt.xticks(np.arange(k_ini, k_end + 1, 1.0))
# plt.plot(k_width, list_calinski)
# plt.savefig(results_dir + "Calinski.png")

plt.figure("WSSSE")
plt.title(titulo + "_WSSSE")
plt.xlabel("Number of clusters")
plt.ylabel("Value")
plt.grid(True)
plt.xticks(np.arange(k_ini, k_end + 1, 1.0))
plt.plot(k_width, list_wssse)
plt.savefig(results_dir + "WSSSE.png")

# centroids = kmeans_cluster.cluster_centers_
# plt.figure("centroids")
# plt.title("Centroids")  # Establece el título del gráfico
# plt.xlabel("Features")  # Establece el título del eje x
# plt.ylabel("Values")  # Establece el título del eje y
#
# plt.grid(True)
# for values in range(len(centroids)):
#     plt.plot(range(len(centroids[0])), centroids[values], label="Cluster " + str(values))
# plt.legend(loc='upper left')
# plt.savefig(results_dir + "_centroids.png")
