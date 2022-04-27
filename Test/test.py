import pandas as pd

from ChiIndex.model import ChiIndex

df = pd.read_csv('data/iris.data', delimiter=",", header=None)
print(df.columns)
print(df.head())
df.rename(columns={4: 'Class'}, inplace=True)

chi = ChiIndex(df, results_path='res')
print(chi.list_chi)
print(chi.optimum_chi)

