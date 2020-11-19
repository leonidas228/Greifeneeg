import mne
from mne.time_frequency import read_tfrs
from mne.stats import f_mway_rm, f_threshold_mway_rm
from os.path import isdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

def stat_fun(*args):
    # get f-values only.
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
chan = "central"
osc = "SO"
conds = ["eig30s", "fix30s"]
factor_levels = [2]
effects = 'A'
tfce = dict(start=0, step=0.2)

epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
tfr = read_tfrs("{}grand_central-tfr.h5".format(proc_dir))[0]
tfr = tfr["OscType=='{}'".format(osc)]
epo = epo["OscType=='{}'".format(osc)]

df = epo.metadata.copy()
subjs = list(np.unique(df["Subj"].values))
results = []
for cond in conds:
    this_tfr = tfr["Cond=='{}' or Cond=='sham'".format(cond)]
    data = [[], []]
    for subj in subjs:
        subj_tfr = this_tfr["Subj=='{}'".format(subj)]
        sham_tfr = subj_tfr["Cond=='sham'"].data[:,0,] * 1e+10
        cond_tfr = subj_tfr["Cond=='{}'".format(cond)].data[:,0,] * 1e+10
        if not len(sham_tfr) or not len(cond_tfr):
            continue
        data[0].append(sham_tfr.mean(axis=0))
        data[1].append(cond_tfr.mean(axis=0))
    data[0] = np.swapaxes(np.array(data[0]), 1, 2)
    data[1] = np.swapaxes(np.array(data[1]), 1, 2)

    f_thresh = f_threshold_mway_rm(len(subjs), factor_levels, effects, 0.05)
    stats = mne.stats.spatio_temporal_cluster_test(data, stat_fun=stat_fun,
                                                   threshold=tfce, tail=1,
                                                   n_jobs=8,
                                                   n_permutations=1000,
                                                   out_type="mask")
    results.append(stats)

#(t_obs, clusters, cluster_p_values, h0) = stats
