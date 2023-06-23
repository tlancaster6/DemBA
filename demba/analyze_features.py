import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

features = pd.read_csv(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\testclip\testclip_features.csv", index_col=0)
reducer = umap.UMAP()
scaled_features = StandardScaler().fit_transform(features)
embedding = reducer.fit_transform(scaled_features)
