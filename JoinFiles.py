import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import datetime

path = "C:/Users/josemaria.luna/PycharmProjects/LondonEnergy/data/"
filename_1 = "dataset_365_full.csv"
filename_2 = "informations_households.csv"

df_original = pd.read_csv(path + filename_1, delimiter=",")
df_follow = pd.read_csv(path + filename_2, delimiter=",")

print(df_original.head())
print(df_follow.head())

df_res = df_original.set_index('id').join(df_follow.set_index('LCLid'))
df_res.drop(['file', 'stdorToU'], axis=1, inplace=True)

print(df_res.head())

df_res.to_csv("dataset_365_full_acorn.csv", sep=',')
