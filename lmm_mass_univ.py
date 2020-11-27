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
    t_vals = np.zeros((exog_n, dat.shape[1]))
    for pnt_idx in range(dat.shape[1]):
        print("{} of {}".format(pnt_idx, dat.shape[1]))
        endog = dat[:, pnt_idx]
        this_mod = MixedLM(endog, exog, groups)
        fit = this_mod.fit(reml=False)
        t_vals[:, pnt_idx] = fit.tvalues[:exog_n]
    t_vals = t_vals.reshape((exog_n, *data.shape[1:]))
    return t_vals

parser = argparse.ArgumentParser()
parser.add_argument('--perm', type=int, default=64)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--model', type=str, default="cond")
parser.add_argument('--baseline', type=str, default="mean")
parser.add_argument('--osc', type=str, default="SO")
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
osc = opt.osc
factor_levels = [2]
effects = 'A'
tfce_params = dict(start=0, step=0.2)
perm_n = opt.perm
bootstrap = True

#epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, opt.baseline))[0]
tfr = tfr["OscType=='{}' and PrePost=='Post'".format(osc)]
#epo = epo["OscType=='{}'".format(osc)]

if opt.model == "simple":
    col = "Stim"
elif opt.model == "cond":
    col = "Cond"
else:
    raise ValueError("Model not recognised.")

tfr = tfr["Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'"]
subjs = np.unique(tfr.metadata["Subj"].values)
# check for missing conditions in each subject
bad_subjs = []
for subj in subjs:
    this_df = tfr.metadata.query("Subj=='{}'".format(subj))
    these_conds = list(np.unique(this_df[col].values))
    checks = [c in these_conds for c in list(np.unique(tfr.metadata[col].values))]
    if not all(checks):
        bad_subjs.append(subj)
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]

data = np.swapaxes(tfr.data[:,0],1,2)
if opt.baseline == "mean":
    data *= 1e+12
df = tfr.metadata.copy()
df["Brain"] = np.zeros(len(df),dtype=np.float64)
conds = list(np.unique(df[col].values))
conds.remove("sham")

md = smf.mixedlm("Brain ~ C({}, Treatment('sham'))".format(col), df, groups=df["Subj"])
endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
# main result
if opt.iter == 0: # only do main result if this is the first node
    t_vals = mass_uv_lmm(data, endog, exog, groups)
    main_result = {}
    main_result["raw_t"] = {k:t_vals[idx,] for idx,k in enumerate(exog_names)}
    main_result["tfce_pos"] = {k:None for k in exog_names}
    main_result["tfce_neg"] = {k:None for k in exog_names}
    for idx, k in enumerate(exog_names):
        # positive values
        masked_tvals = t_vals[idx,].copy()
        masked_tvals[masked_tvals<0] = 0
        tfce = np.reshape(_find_clusters(masked_tvals, tfce_params)[1],
                          t_vals[idx,].shape)
        main_result["tfce_pos"][k] = tfce
        # negative values
        masked_tvals = t_vals[idx,].copy()
        masked_tvals[masked_tvals>0] = 0
        tfce = np.reshape(_find_clusters(abs(masked_tvals), tfce_params)[1],
                          t_vals[idx,].shape)
        main_result["tfce_neg"][k] = tfce
    with open("{}{}/{}/main_result_{}.pickle".format(proc_dir, opt.baseline, osc, opt.model), "wb") as f:
        pickle.dump(main_result, f)

# permute
perm_data = data.copy()
subjs = list(np.unique(groups))
perm_results = []
for perm_idx in range(perm_n):
    print("\n\nPermutation {} of {}\n\n".format(perm_idx+1, perm_n))
    perm_result = {}
    perm_result["tfce_pos"] = {k:None for k in exog_names}
    perm_result["tfce_neg"] = {k:None for k in exog_names}
    for subj in subjs:
        if bootstrap:
            sham_inds = np.where((df["Subj"]==subj) & (df[col]=='sham').values)[0]
            for cond in conds:
                cond_inds = np.where((df["Subj"]==subj) & (df[col]==cond).values)[0]
                if len(cond_inds) < 2:
                    continue
                perm_inds = np.random.choice(cond_inds, size=len(cond_inds)//2+len(cond_inds)%2)
                perm_inds = np.concatenate((perm_inds, np.random.choice(sham_inds,
                                           size=len(cond_inds)//2)))
                perm_data[cond_inds,] = data[perm_inds,]
            non_sham_inds = np.where((df["Subj"]==subj) & (df[col]!='sham').values)[0]
            perm_inds = np.random.choice(sham_inds, size=len(sham_inds)//2+len(sham_inds)%2)
            perm_inds = np.concatenate((perm_inds, np.random.choice(non_sham_inds,
                                       size=len(sham_inds)//2)))
            perm_data[sham_inds,] = data[perm_inds,]

        else:
            # permute
            subj_inds = np.where(groups==subj)[0]
            temp_slice = data[subj_inds,]
            np.random.shuffle(temp_slice)
            data[subj_inds,] = temp_slice

    t_vals = mass_uv_lmm(perm_data, endog, exog, groups)
    for idx, k in enumerate(exog_names):
        # positive values
        masked_tvals = t_vals[idx,].copy()
        masked_tvals[masked_tvals<0] = 0
        tfce = np.reshape(_find_clusters(masked_tvals, tfce_params)[1],
                          t_vals[idx,].shape)
        perm_result["tfce_pos"][k] = tfce
        # negative values
        masked_tvals = t_vals[idx,].copy()
        masked_tvals[masked_tvals>0] = 0
        tfce = np.reshape(_find_clusters(abs(masked_tvals), tfce_params)[1],
                          t_vals[idx,].shape)
        perm_result["tfce_neg"][k] = tfce

    perm_results.append(perm_result)

with open("{}{}/{}/perm_result_{}_{}_{}.pickle".format(proc_dir, opt.baseline, osc, perm_n, opt.iter, opt.model), "wb") as f:
    pickle.dump(perm_results, f)
