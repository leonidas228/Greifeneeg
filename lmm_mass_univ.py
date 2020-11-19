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
parser.add_argument('--perm', type=int, default=500)
parser.add_argument('--iter', type=int, default=0)
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
osc = "SO"
conds = ["eig30s", "fix30s"]
factor_levels = [2]
effects = 'A'
tfce = dict(start=0, step=0.2)
perm_n = opt.perm

#epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
tfr = read_tfrs("{}grand_central-tfr.h5".format(proc_dir))[0]
tfr = tfr["OscType=='{}'".format(osc)]
#epo = epo["OscType=='{}'".format(osc)]

tfr = tfr["Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'"]
data = np.swapaxes(tfr.data[:,0],1,2)*1e+10
df = tfr.metadata
df["Brain"] = np.zeros(len(df),dtype=np.float64)
md = smf.mixedlm("Brain ~ C(Cond, Treatment('sham'))", df,
                 groups=df["Subj"])
endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
# main result
if opt.iter == 0: # only do main result if this is the first node
    t_vals = mass_uv_lmm(data, endog, exog, groups)
    main_result = {"raw_t":{k:t_vals[idx,] for idx,k in enumerate(exog_names)},
                   "tfce_t":{k:np.reshape(_find_clusters(t_vals[idx,], tfce)[1],
                             t_vals[idx,].shape) for idx,k in enumerate(exog_names)}}
    with open("{}main_result.pickle".format(proc_dir), "wb") as f:
        pickle.dump(main_result, f)

# permute
subjs = list(np.unique(groups))
perm_maxs = []
for perm_idx in range(perm_n):
    print("\n\nPermutation {} of {}\n\n".format(i, perm_n))
    for subj in subjs:
        subj_inds = np.where(groups==subj)[0]
        temp_slice = data[subj_inds,]
        np.random.shuffle(temp_slice)
        data[subj_inds,] = temp_slice
    t_vals = mass_uv_lmm(data, endog, exog, groups)
    maxs = []
    for t_idx in range(len(t_vals)):
        tfce_vals = np.reshape(_find_clusters(t_vals[t_idx,], tfce)[1],
                               t_vals[t_idx,].shape)
        maxs.append(tfce_vals.max())
    perm_maxs.append(maxs)
perm_maxs = np.array(perm_maxs)
perm_result = {k:perm_maxs[:,idx] for idx, k in enumerate(exog_names)}
with open("{}perm_result_{}_{}.pickle".format(proc_dir, perm_n, opt.iter), "wb") as f:
    pickle.dump(perm_result, f)
