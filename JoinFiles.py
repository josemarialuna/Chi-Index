import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import datetime

# df_original = pd.read_csv("C:/Users/Josem/PycharmProjects/ClusteringIndices/20190604125818_brca_autoencoded.csv_4_Res/20190604125822_kmeans_winner.csv", delimiter="\t")
df_original = pd.read_csv(
    "C:/Users/Josem/PycharmProjects/ClusteringIndices/20190604120737_brca_autoencoded.csv_7_Res/20190604120741_kmeans_winner.csv",
    delimiter="\t")

df_follow = pd.read_csv("C:/datasets/brca/followUP.csv", delimiter="\t")

print(df_original.head())
print(df_follow.head())

df_res = df_original.set_index('id').join(df_follow.set_index('bcr_patient_barcode'))

print(df_res.head())

df_res.to_csv("df_res_7.csv", sep='\t')
