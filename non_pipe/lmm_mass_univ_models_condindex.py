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
osc = "deltO"
durs = ["30s","2m","5m"]
syncs = ["sync","async"]
bootstrap = True

for sync in syncs:
    for dur in durs:
        tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
        df = tfr.metadata
        inds = (df["PrePost"]=="Pre").values & (df["Index"]==0).values
        df.loc[inds, "Index"] = "Pre"
        df.loc[inds, "PrePost"] = "Post"
        tfr = tfr["OscType=='{}' and PrePost=='Post'".format(osc)]
        subjs = np.unique(tfr.metadata["Subj"].values)
        col = "Cond"
        t_string = "sham{}".format(dur)
        tfr = tfr["Cond=='eig{}' or Cond=='fix{}' or Cond=='sham{}'".format(dur,dur,dur)]
        bad_subjs = []
        # if sync then remove all subjects recorded under asynchronous conditions (<31)
        if sync == "sync":
            for subj in list(subjs):
                if int(subj) < 31:
                    bad_subjs.append(subj)
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

        data = np.swapaxes(tfr.data[:,0],1,2)
        if baseline == "mean":
            data *= 1e+12
        df = tfr.metadata.copy()
        df["Brain"] = np.zeros(len(df),dtype=np.float64)

        formula = "Brain ~ C({}, Treatment('{}'))*C(Index, Treatment('Pre'))".format(col,t_string)
        outfile = "{}main_fits_{}_condidx_{}_{}_{}.pickle".format(proc_dir, baseline, osc, dur, sync)
        md = smf.mixedlm(formula, df, groups=df["Subj"])
        endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
        print(exog_names)
        #main result
        fits = mass_uv_lmm(data, endog, exog, groups)
        fits = {"exog_names":exog_names, "fits":fits}
        with open(outfile, "wb") as f:
            pickle.dump(fits, f)
