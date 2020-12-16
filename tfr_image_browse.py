import mne
import numpy as np
from os.path import isdir
from mne.time_frequency import read_tfrs
import matplotlib.pyplot as plt
plt.ion()

baseline = "logratio"
chan = "central"
osc = "SO"
conds = ["eig30s", "eig2m", "eig5m"]
freqs = [15, 19]
sync = "sync"

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

infile = "{}grand_{}_{}-tfr.h5".format(proc_dir, chan, baseline)

tfr = read_tfrs(infile)[0]
subjs = np.unique(tfr.metadata["Subj"].values)
bad_subjs = []
if sync == "sync":
    for subj in list(subjs):
        if int(subj) < 31:
            bad_subjs.append(subj)
bad_subjs = list(set(bad_subjs))
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]
freq_inds = (tfr.freqs >= freqs[0]) & (tfr.freqs <= freqs[1])
for cond in conds:
    this_tfr = tfr["Cond=='{}' and PrePost=='Post' and OscType=='{}'".format(cond, osc)]
    data = this_tfr.data[:, 0, freq_inds,].mean(axis=1, keepdims=True)

    info = mne.create_info(["tfr"], this_tfr.info["sfreq"], ch_types="misc")
    epo = mne.EpochsArray(data, info)
    epo.plot_image(picks="tfr")
    plt.title(cond)
