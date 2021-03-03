import mne
from mne.time_frequency import read_tfrs
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import argparse
import pandas as pd
from os.path import isdir
import re
import numpy as np
from mne.stats.cluster_level import _find_clusters
import pickle
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

parser = argparse.ArgumentParser()
parser.add_argument('--perm', type=int, default=64)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--baseline', type=str, default="zscore")
parser.add_argument('--osc', type=str, default="SO")
parser.add_argument('--syncfact', default="nosyncfact")
parser.add_argument('--badsubjs', default="all_subj")
parser.add_argument('--group', action='store_true')
opt = parser.parse_args()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
chan = "central"
baseline = opt.baseline
osc = opt.osc
sync_fact = opt.syncfact
use_group = opt.group
bad_subjs = opt.badsubjs
bad_subjs_dict = {"all_subj":[], "no2,3":["002", "003"]}
perm_n = opt.perm

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr = tfr["OscType=='{}'".format(osc)]
tfr_shape = tfr.data.shape[-2:]
tfr = tfr["PrePost=='Post'"]
subjs = np.unique(tfr.metadata["Subj"].values)
# remove all subjects with missing conditions or not meeting synchronicity criterion
for bs in bad_subjs_dict[bad_subjs]:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]
subjs = np.unique(tfr.metadata["Subj"].values)
data = np.swapaxes(tfr.data[:,0],1,2)
if baseline == "mean":
    data *= 1e+12
df = tfr.metadata.copy()
df["Brain"] = np.zeros(len(df),dtype=np.float64)

if sync_fact == "syncfact":
    formula = "Brain ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))*C(Sync, Treatment('sync'))"
else:
    formula = "Brain ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))"

groups = df["Subj"] if use_group else pd.Series(np.zeros(len(df),dtype=int))
md = smf.mixedlm(formula, df, groups=groups)
endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
print(exog_names)
t_vals = np.zeros((perm_n, len(exog_names), np.product(tfr_shape)))
data_inds = np.arange(len(data))
for perm_idx in range(perm_n):
    if use_group:
        # if we have group as a factor, we shuffle data only within subjects
        for subj in subjs:
            subj_inds = data_inds[df["Subj"]==subj]
            np.random.shuffle(subj_inds)
            data_inds[df["Subj"]==subj] = subj_inds
    else:
        np.random.shuffle(data_inds)
    data = data[data_inds,]
    fits = mass_uv_lmm(data, endog, exog, groups)
    for mf_idx, mf in enumerate(fits):
        for exog_idx, en in enumerate(exog_names):
            t_vals[perm_idx, exog_idx, mf_idx] = mf.tvalues[exog_names.index(en)]
    del fits

t_vals = np.reshape(t_vals, (perm_n, len(exog_names), *tfr_shape), order="F")
outfile = "{}perm_{}_grand_{}_{}_{}_{}_{}.pickle".format(proc_dir, baseline,
                                                         osc, bad_subjs,
                                                         use_group, sync_fact,
                                                         opt.iter)
out_dict = {"exog_names":exog_names, "t_vals":t_vals}
with open(outfile, "wb") as f:
    pickle.dump(out_dict, f)
