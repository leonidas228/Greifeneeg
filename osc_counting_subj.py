import mne
import numpy as np
import pandas as pd
from mne.time_frequency import read_tfrs
from os.path import isdir
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

baseline = "zscore"
plot_conds = ["sham30s","fix30s","eig30s","sham2m","fix2m","eig2m","sham5m","fix5m","eig5m"]

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr = tfr["OscType=='SO' and PrePost=='Post'"]
subjs = list(tfr.metadata["Subj"].unique())
dfs = {"Subj":[], "df":[], "var":[]}
for subj in subjs:
    this_tfr = tfr["Subj=='{}'".format(subj)]
    these_conds = list(this_tfr.metadata["Cond"].unique())
    counts = this_tfr.metadata.value_counts(subset=["Cond"])
    count_df = {"Cond":[], "Count":[]}
    for pc in plot_conds:
        count_df["Cond"].append(pc)
        if pc in these_conds:
            count_df["Count"].append(counts[pc].values[0])
        else:
            count_df["Count"].append(0)
    count_df = pd.DataFrame.from_dict(count_df)
    dfs["Subj"].append(subj)
    dfs["df"].append(count_df)
    vals = count_df["Count"].values.copy()
    nz_mean = vals[vals!=0].mean() # only compute mean on non-zero values
    vals[vals>nz_mean] = nz_mean # don't penalise variance above the mean
    var = np.sqrt(np.sum((vals - np.ones_like(vals)*nz_mean)**2))
    dfs["var"].append(var)

dfs = pd.DataFrame.from_dict(dfs)
dfs = dfs.sort_values("var")

fig, axes = plt.subplots(6,6)
axes = [ax for axe in axes for ax in axe]
subjs = list(dfs["Subj"].values)
for idx, subj in enumerate(subjs):
    df = dfs.iloc[idx]["df"]
    sns.barplot(data=df, x="Cond", y="Count", order=plot_conds, ax=axes[idx])
    axes[idx].set_title(subj)
plt.tight_layout()
