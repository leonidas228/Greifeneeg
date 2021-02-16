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
            fits.append(this_mod.fit(reml=True))
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
baseline = "logmean"
osc = "SO"
durs = ["30s","2m","5m"]
sync_facts = ["syncfact", "nosyncfact"]
#sync_facts = ["syncfact"]
use_groups = ["group", "nogroup"]
use_groups = ["nogroup"]
use_badsubjs = {"all_subj":[]}


for bs_name, bad_subjs in use_badsubjs.items():
    for use_group in use_groups:
        for sync_fact in sync_facts:
            tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
            tfr = tfr["OscType=='{}'".format(osc)]
            #tfr = tfr["PrePost=='Post'"]
            subjs = np.unique(tfr.metadata["Subj"].values)
            # remove all subjects with missing conditions or not meeting synchronicity criterion
            for bs in bad_subjs:
                print("Removing subject {}".format(bs))
                tfr = tfr["Subj!='{}'".format(bs)]
            data = np.swapaxes(tfr.data[:,0],1,2)
            if baseline == "mean":
                data *= 1e+12
            df = tfr.metadata.copy()
            df["Brain"] = np.zeros(len(df),dtype=np.float64)

            if sync_fact == "syncfact":
                formula = "Brain ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))*C(PrePost, Treatment('Pre'))*C(Sync, Treatment('sync'))"
                formula = "Brain ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))*C(Sync, Treatment('sync'))"
            else:
                formula = "Brain ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))*C(PrePost, Treatment('Pre'))"
                formula = "Brain ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))"
            outfile = "{}main_fits_{}_grand_{}_{}_{}_{}.pickle".format(proc_dir, baseline, osc, bs_name, use_group, sync_fact)
            groups = df["Subj"] if use_group == "group" else pd.Series(np.zeros(len(df),dtype=int))
            md = smf.mixedlm(formula, df, groups=groups)
            endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
            print(exog_names)
            #main result
            fits = mass_uv_lmm(data, endog, exog, groups)
            fits = {"exog_names":exog_names, "fits":fits}
            with open(outfile, "wb") as f:
                pickle.dump(fits, f)
            del fits
