import mne
from mne.time_frequency import read_tfrs
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import argparse
import pandas as pd
from os.path import isdir
import re
import numpy as np
import matplotlib.pyplot as plt
from mne.stats.cluster_level import _find_clusters
import pickle
plt.ion()
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

def graph_comps(dur_pairs, aics, veclen, freq_n, time_n):
    for dp in dur_pairs:
        aic_pairs = np.zeros((2, veclen))
        for pair_idx in range(2):
            aic_pairs[pair_idx,] = aics[dp[pair_idx]]
        model_map = np.zeros(veclen)
        winner_vecs = np.argsort(aic_pairs, axis=0)[0,]
        model_map[winner_vecs>0] = 1
        for idx in list(np.where(model_map)[0]):
            model_map[idx] = 1 - np.exp((aic_pairs[1,idx] - aic_pairs[0,idx]) / 2)
        model_map = np.reshape(model_map, (freq_n, time_n),
                               order="F")
        plt.figure()
        sns.heatmap(model_map)
        plt.gca().invert_yaxis()
        plt.title(dp[1])

def mass_uv_lmm(data, endog, exog, groups):
    exog_n = exog.shape[1]
    dat = data.reshape(len(data), data.shape[1]*data.shape[2])
    fits = []
    for pnt_idx in range(dat.shape[1]):
        print("{} of {}".format(pnt_idx, dat.shape[1]))
        endog = dat[:, pnt_idx]
        this_mod = MixedLM(endog, exog, groups)
        try:
            fits.append(this_mod.fit(reml=False))
        except:
            print("\nCoudn't fit model.\n")
            fits.append(None)
    return fits

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
chan = "central"
baseline = "zscore"
osc = "SO"
use_group = "nogroup"

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr = tfr["OscType=='{}'".format(osc)]
tfr = tfr["PrePost=='Post'"]

data = np.swapaxes(tfr.data[:,0],1,2)
veclen = data.shape[-2]*data.shape[-1]
if baseline == "mean":
    data *= 1e+12
df = tfr.metadata.copy()
df["Brain"] = np.zeros(len(df),dtype=np.float64)

try:
    with open("{}{}_{}_{}_{}_aics.pickle".format(proc_dir, chan, baseline, osc,
              use_group), "rb") as f:
        aics = pickle.load(f)
except:
    models = {"Null":"Brain ~ 1",
              "Stim":"Brain ~ Stim",
              "StimType":"Brain ~ StimType",
              "Duration_Stim":"Brain ~ Stim*Dur",
              "Duration_StimType":"Brain ~ StimType*Dur",
              "Sync_Stim":"Brain ~ Stim*Dur*Sync",
              "Sync_StimType":"Brain ~ StimType*Dur*Sync",
              "Sync_Stim_NoDur":"Brain ~ Stim*Sync",
              "Sync_StimType_NoDur":"Brain ~ StimType*Sync"}

    aics = {k:np.zeros(data.shape[-2]*data.shape[-1]) for k in models.keys()}

    for mod_name, mod_form in models.items():
        groups = df["Subj"] if use_group == "group" else pd.Series(np.zeros(len(df),dtype=int))
        md = smf.mixedlm(mod_form, df, groups=groups)
        endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
        print(exog_names)
        #main result
        fits = mass_uv_lmm(data, endog, exog, groups)
        for fit_idx, fit in enumerate(fits):
            aics[mod_name][fit_idx] = fit.aic
        del fits

    with open("{}{}_{}_{}_{}_aics.pickle".format(proc_dir, chan, baseline, osc,
              use_group), "wb") as f:
        pickle.dump(aics, f)

aics_arr = np.empty((len(aics), veclen))
for v_idx, v in enumerate(aics.values()):
    aics_arr[v_idx,] = v
winner_vecs = np.argsort(aics_arr, axis=0)
fig, axes = plt.subplots(1,2, figsize=(38.4,21.6))
#axes = [ax for axe in axes for ax in axe]
for ax_idx, ax in enumerate(axes):
    winner_map = np.reshape(winner_vecs[ax_idx,], (data.shape[2], data.shape[1]),
                            order="F")
    sns.heatmap(winner_map, ax=ax, cmap="Set1")
    ax.invert_yaxis()

# assess effect of duration
dur_pairs = [("Sync_StimType_NoDur", "Sync_StimType"),
             ("StimType", "Duration_StimType")]
graph_comps(dur_pairs, aics, veclen, data.shape[-1], data.shape[-2])

# assess effect of sync
sync_pairs = [("StimType", "Sync_StimType_NoDur"),
             ("Duration_StimType", "Sync_StimType")]
graph_comps(sync_pairs, aics, veclen, data.shape[-1], data.shape[-2])

# assess effect of stimtype
sync_pairs = [("Stim", "StimType"),
             ("Duration_Stim", "Duration_StimType")]
graph_comps(sync_pairs, aics, veclen, data.shape[-1], data.shape[-2])
