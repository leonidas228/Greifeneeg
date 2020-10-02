import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
from os.path import isdir
import seaborn as sns

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

df = pd.read_pickle("{}grand_ba_df.pickle")
# sub_inds = df["Subj"].values.astype(int) >= 31
# df = df[sub_inds]

this_df = df.query("Cond=='fix2m' or Cond=='eig2m' and OscType=='SO'")
sns.catplot(x="PrePost", y="Number", hue="Cond", data=this_df, kind="violin",
            split=True)
plt.title("Number of slow oscillations")

this_df = df.query("Cond=='eig2m' or Cond=='fix2m' and OscType=='deltO'")
sns.catplot(x="PrePost", y="Number", hue="Cond", data=this_df, kind="violin",
            split=True)
plt.title("Number of delta oscillations")
