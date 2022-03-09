import mne
from mne.time_frequency import read_tfrs
from os.path import isdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 42}
matplotlib.rc('font', **font)

def time_as_index(time, times):
    idx = np.argmin(np.abs((times - time)))
    return idx

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

chan = "central"
baseline = "zscore"
badsubjs = ["002","003","028"]
twin = (.275, .4)
fwin = (13, 16)

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]

twin_inds = [time_as_index(twi, tfr.times) for twi in twin]
fwin_inds = [time_as_index(fwi, tfr.freqs) for fwi in fwin]

tfr = tfr["OscType=='SO' and PrePost=='Post'"]
# remove all subjects with missing conditions or not meeting synchronicity criterion
for bs in badsubjs:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]

subjs = list(tfr.metadata["Subj"].unique())
df_dict = {"Subj":[], "EigFreq":[], "sham":[], "eig":[], "fix":[]}
for subj in subjs:
    df_dict["Subj"].append(subj)
    this_tfr = tfr["Subj=='{}'".format(subj)]
    eig = this_tfr[0].metadata["EigFreq"].values[0]
    df_dict["EigFreq"].append(eig)
    for stimtype in ["sham", "eig", "fix"]:
        cond_tfr = this_tfr["StimType=='{}'".format(stimtype)]
        data = np.squeeze(cond_tfr.data)
        data = np.mean(data[...,twin_inds[0]:twin_inds[1]], axis=-1)
        data = np.mean(data[...,fwin_inds[0]:fwin_inds[1]], axis=-1)
        df_dict[stimtype].append(data.mean())

df = pd.DataFrame.from_dict(df_dict)
df["EigFreqDelta"] = df["EigFreq"] - .75
df["FixDelta"] = df["fix"] - df["sham"]
df["EigDelta"] = df["eig"] - df["sham"]
df["EigFix"] = (df["eig"] - df["sham"]) - (df["fix"] - df["sham"])

ylabels = ["Eigen - Sham", "Fix (0.75) - Sham", "Eigen - Fix (0.75)"]
keys = ["EigDelta", "FixDelta", "EigFix"]
colours = ["red", "green", "blue"]
fig, axes = plt.subplots(1, 3, figsize=(38.4, 21.6))
for idx, (ax, key, ylab, col) in enumerate(zip(axes, keys, ylabels, colours)):
    x = df["EigFreqDelta"].values
    y = df[key].values
    ax.scatter(x, y, c=col, s=300, alpha=0.8)
    ax.set_ylim(-5, 7.5)
    p = np.poly1d(np.polyfit(x, y, 1))
    xp = [p(x.min()), p(x.max())]
    ax.plot([x.min(), x.max()], xp, linewidth=8, alpha=0.8, color=col)
    if idx == 0:
        ax.set_ylabel("Spindle Coupling\n{}".format(ylab))
    else:
        ax.set_ylabel(ylab)
        ax.set_yticks([])
    if idx == 1:
        ax.set_xlabel("Eigenfrequency - 0.75")

plt.suptitle("Spindle Coupling by Eigenfreqency")
