import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
plt.ion()

comp_n = 5

df = pd.read_excel("/home/jev/sfb/SFBTQ/Stress_results.xlsx")
cols = list(df.columns)

sel_cols = []
for col in cols:
    if re.match("\d*_.*_t", col):
        sel_cols.append(col)
df_sel = df[sel_cols]

mat = np.array(df_sel)
pca = PCA()
trans = pca.fit_transform(mat)
mag = abs(pca.components_).max()
plt.imshow(pca.components_[:comp_n,].T, cmap="seismic", vmin=-mag, vmax=mag)

plt.yticks(np.arange(len(sel_cols)), sel_cols, fontsize=28)
plt.xticks(np.arange(comp_n),
           ["{:0.2f}".format(x) for x in pca.explained_variance_ratio_[:comp_n]],
           fontsize=20)
plt.ylabel("Variables", fontsize=28, rotation="vertical")
plt.xlabel("Component variance explained", fontsize=28)
