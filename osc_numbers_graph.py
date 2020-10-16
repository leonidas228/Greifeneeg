import mne

import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)

import numpy as np
import pandas as pd
from os.path import isdir
import seaborn as sns
import statsmodels.formula.api as smf

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

df = pd.read_pickle("{}grand_df.pickle".format(proc_dir))
# sub_inds = df["Subj"].values.astype(int) >= 31
# df = df[sub_inds]
df = df.query("Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'")

this_df = df.query("OscType=='SO'")
sns.catplot(hue="PrePost", y="Number", x="Cond", data=this_df, kind="box")
plt.title("Number of slow oscillations")

this_df = df.query("OscType=='deltO'")
sns.catplot(hue="PrePost", y="Number", x="Cond", data=this_df, kind="box")
plt.title("Number of delta oscillations")

md = smf.mixedlm("Number ~ Index*C(Cond, Treatment('sham'))*OscType", df,
                 groups=df["Subj"])
res_all = md.fit(reml=False)
print(res_all.summary())
