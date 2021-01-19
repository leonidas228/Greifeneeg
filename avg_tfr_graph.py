import mne
from mne.time_frequency import read_tfrs
import argparse
import pandas as pd
from os.path import isdir
import re
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
import seaborn as sns
import matplotlib
font = {'weight' : 'bold',
        'size'   : 26}
matplotlib.rc('font', **font)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"
img_dir = root_dir+"images/"

n_jobs = 8
chan = "central"
baseline = "zscore"
osc = "SO"
durs = ["30s","2m","5m"]
syncs = ["async", "sync"]
syncs = ["async"]
conds = ["sham","fix","eig"]
balance_conds = False
vmin, vmax = -4, 4
sync_titles = {"async":"non-synchronised included", "sync":"synchronised only"}
toi = .26
foi = 15

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr = tfr["OscType=='{}' and PrePost=='Post'".format(osc)]
epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")
epo = epo["OscType=='{}' and PrePost=='Post'".format(osc)]
epo.resample(tfr.info["sfreq"], n_jobs="cuda")
epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
# calculate global ERP min and max for scaling later on
evo = epo.average()
ev_min, ev_max = evo.data.min(), evo.data.max()
mask = None

if toi != None and foi != None:
    toi_idx = find_nearest(tfr.times, toi)
    foi_idx = find_nearest(tfr.freqs, foi)
    mask = np.zeros(tfr.data.shape[2:])
    mask[...,foi_idx,toi_idx] = 1

for sync in syncs:
    avg_fig, avg_axes = plt.subplots(3, 3, figsize=(38.4, 21.6))
    std_fig, std_axes = plt.subplots(3, 3, figsize=(38.4, 21.6))
    hist_data = {"Subj":[], "Cond":[], "TFR":[], "Dur":[]}
    mods = []
    for dur_idx,dur in enumerate(durs):

        with open("{}main_fits_{}_cond_{}_{}_{}.pickle".format(proc_dir,baseline,osc,dur,sync), "rb") as f:
            fits = pickle.load(f)

        subjs = np.unique(tfr.metadata["Subj"].values)
        col = "Cond"
        bad_subjs = []
        # if sync then remove all subjects recorded under asynchronous conditions (<31)
        if sync == "sync":
            for subj in list(subjs):
                if int(subj) < 31:
                    bad_subjs.append(subj)
        if balance_conds:
        # check for missing conditions in each subject
            for subj in subjs:
                this_df = tfr.metadata.query("Subj=='{}'".format(subj))
                these_conds = list(np.unique(this_df[col].values))
                checks = [c in these_conds for c in list(np.unique(tfr.metadata[col].values))]
                if not all(checks):
                    bad_subjs.append(subj)
        # remove all subjects with missing conditions or not meeting synchronicity criterion
        bad_subjs = list(set(bad_subjs))
        for bs in bad_subjs:
            print("Removing subject {}".format(bs))
            tfr = tfr["Subj!='{}'".format(bs)]
            epo = epo["Subj!='{}'".format(bs)]

        subjs = np.unique(tfr.metadata["Subj"].values)
        for cond_idx, cond in enumerate(conds):

            # get osc ERP and normalise
            evo = epo["Cond=='{}{}'".format(cond,dur)].average()
            evo_data = evo.data
            evo_data = (evo_data - ev_min) / (ev_max - ev_min)
            evo_data = evo_data*3 + 13

            this_tfr = tfr["Cond=='{}{}'".format(cond,dur)]
            this_avg = this_tfr.average()
            this_avg.plot(picks="central", axes=avg_axes[dur_idx][cond_idx],
                          colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis",
                          mask=mask, mask_style="contour")
            avg_axes[dur_idx][cond_idx].plot(tfr.times, evo_data[0,],
                                             color="gray", alpha=0.8,
                                             linewidth=10)
            this_std = this_avg.copy()
            this_std.data = this_tfr.data.std(axis=0)
            this_std.plot(picks="central", axes=std_axes[dur_idx][cond_idx],
                          colorbar=False, vmin=0, vmax=15, cmap="hot",
                          mask=mask, mask_style="contour")
            std_axes[dur_idx][cond_idx].plot(tfr.times, evo_data[0,],
                                             color="gray", alpha=0.8,
                                             linewidth=10)
            if dur_idx == 0:
                avg_axes[dur_idx][cond_idx].set_title("{}".format(cond))
                std_axes[dur_idx][cond_idx].set_title("{}".format(cond))
            if cond_idx == len(conds)-1:
                avg_rax = avg_axes[dur_idx][cond_idx].twinx()
                avg_rax.set_ylabel("{}".format(dur))
                avg_rax.set_yticks([])
                std_rax = std_axes[dur_idx][cond_idx].twinx()
                std_rax.set_ylabel("{}".format(dur))
                std_rax.set_yticks([])

            if toi != None and foi != None:
                for trial_idx in range(len(this_tfr.data)):
                    hist_data["Subj"].append(this_tfr.metadata["Subj"].iloc[trial_idx])
                    hist_data["Cond"].append(cond)
                    hist_data["Dur"].append(dur)
                    hist_data["TFR"].append(this_tfr.data[trial_idx,0,foi_idx,toi_idx])

                mods.append(fits["fits"][foi_idx*toi_idx])

    if toi != None and foi != None:
        hist_data = pd.DataFrame.from_dict(hist_data)
        for cond in conds:
            this_hist_data = hist_data.query("Cond=='{}'".format(cond))
            plt.figure(figsize=(38.4,21.6))
            sns.histplot(data=this_hist_data, bins="auto", x="TFR", hue="Dur")
            plt.title("TFR point value histogram, {} {} {}".format(osc, cond, sync_titles[sync]))
            plt.savefig("../images/{}_{}_{}_hist.tif".format(osc, cond, sync))

            plt.figure(figsize=(38.4,21.6))
            sns.countplot("Subj", hue="Dur", order=list(subjs),
                          hue_order=["30s","2m","5m"], data=this_hist_data)
            plt.title("Counts of {} {} {}".format(osc, cond, sync_titles[sync]))
            plt.savefig("../images/{}_{}_{}_subjcount.tif".format(osc, cond, sync))

            plt.figure(figsize=(38.4,21.6))
            sns.countplot(x="Dur",data=this_hist_data)
            plt.title("Counts of {} {} {}".format(osc, cond, sync_titles[sync]))
            plt.savefig("../images/{}_{}_{}_count.tif".format(osc, cond, sync))

    avg_fig.suptitle("Raw average {}, {}".format(osc, sync_titles[sync]))
    avg_fig.savefig("../images/{}_{}_rawavg.tif".format(osc, sync))
    std_fig.suptitle("Standard deviation {}, {}".format(osc, sync_titles[sync]))
    std_fig.savefig("../images/{}_{}_rawstd.tif".format(osc, sync))

    df = tfr.metadata.copy()
    condlist = list(np.unique(df["Cond"].values))
    trial_ns = {}
    for cond in condlist:
        trial_ns[cond] = len(df.query("Cond=='{}'".format(cond)))
