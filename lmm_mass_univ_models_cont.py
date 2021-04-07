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

def mass_uv_lmm(data, endog, exog, groups, exog_re):
    exog_n = exog.shape[1]
    dat = data.reshape(len(data), data.shape[1]*data.shape[2])
    fits = []
    for pnt_idx in range(dat.shape[1]):
        print("{} of {}".format(pnt_idx, dat.shape[1]))
        endog = dat[:, pnt_idx]
        this_mod = MixedLM(endog, exog, groups, exog_re)
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

stim_type = "eig"
cont_vars = ["Null", "FehlerEig", "Fehler75", "Fehler875"]

n_jobs = 8
chan = "central"
baseline = "zscore"
osc = "SO"
use_groups = ["nogroup"]
use_badsubjs = {"no2,3,28":["002","003","028"]}

for cont_var in cont_vars:
    for bs_name, bad_subjs in use_badsubjs.items():
        for use_group in use_groups:
            tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
            tfr = tfr["OscType=='{}' and PrePost=='Post'".format(osc)]
            if stim_type:
                tfr = tfr["StimType=='{}'".format(stim_type)]
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


            outfile = "{}main_fits_{}_{}_{}_{}_cont_{}.pickle".format(proc_dir, baseline, osc, bs_name, use_group, cont_var)

            groups = df["Subj"] if use_group == "group" else pd.Series(np.zeros(len(df),dtype=int))

            formula = "Brain ~ {}".format(cont_var)
            if cont_var == "FehlerEig":
                df["FehlerEig"] = (df["EigFreq"] - df["AvgSOFreq"]).abs()
            elif cont_var == "Fehler75":
                df["Fehler75"] = (df["EigFreq"] - 0.75).abs()
            elif cont_var == "Fehler875":
                df["Fehler875"] = (df["EigFreq"] - 0.875).abs()
            elif cont_var == "Null":
                formula = "Brain ~ 1"

            md = smf.mixedlm(formula, df, groups=groups)
            endog, exog, groups, exog_names, exog_re = md.endog, md.exog, md.groups, md.exog_names, md.exog_re
            print(exog_names)
            #main result
            fits = mass_uv_lmm(data, endog, exog, groups, exog_re)
            fits = {"exog_names":exog_names, "fits":fits}
            with open(outfile, "wb") as f:
                pickle.dump(fits, f)
            del fits
