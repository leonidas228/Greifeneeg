import mne
from mne.time_frequency import read_tfrs
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
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
    aics = []
    for pnt_idx in range(dat.shape[1]):
        print("{} of {}".format(pnt_idx, dat.shape[1]))
        endog = dat[:, pnt_idx]
        this_mod = MixedLM(endog, exog, groups)
        fit = this_mod.fit(reml=False)
        aics.append(fit.aic)
    aics = np.array(aics).reshape(*data.shape[1:])
    return np.array(aics)

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
tfce_params = dict(start=0, step=0.2)

#epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
tfr = read_tfrs("{}grand_central-tfr.h5".format(proc_dir))[0]
tfr = tfr["OscType=='{}' and PrePost=='Post'".format(osc)]
#epo = epo["OscType=='{}'".format(osc)]

tfr = tfr["Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'"]
data = np.swapaxes(tfr.data[:,0],1,2)

df = tfr.metadata
df["Brain"] = np.zeros(len(df),dtype=np.float64)

try:
    aics = np.load("{}aics.np".format(proc_dir))
except:
    aics = np.zeros((3, *data.shape[1:]))
    md = smf.mixedlm("Brain ~ 1", df, groups=df["Subj"])
    endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
    aics[0] = mass_uv_lmm(data, endog, exog, groups)
    md = smf.mixedlm("Brain ~ C(Stim, Treatment('sham'))", df, groups=df["Subj"])
    endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
    aics[1] = mass_uv_lmm(data, endog, exog, groups)
    md = smf.mixedlm("Brain ~ C(Cond, Treatment('sham'))", df, groups=df["Subj"])
    endog, exog, groups, exog_names = md.endog, md.exog, md.groups, md.exog_names
    aics[2] = mass_uv_lmm(data, endog, exog, groups)
    np.save("{}aics.np".format(proc_dir),aics)

mins = np.argmin(aics,axis=0).astype(np.float)
plt.figure()
plt.imshow(mins.T)

fig, axes = plt.subplots(2,1)
plt.suptitle("Null superiority")
inds = mins==0
simp_over_null = mins.copy()
simp_over_null[~inds] = np.nan
simp_over_null[inds] = 1-np.exp((aics[0,inds]-aics[1,inds])/2)
axes[0].imshow(simp_over_null.T, vmin=0, vmax=1)
cond_over_null = mins.copy()
cond_over_null[~inds] = np.nan
cond_over_null[inds] = 1-np.exp((aics[0,inds]-aics[2,inds])/2)
axes[1].imshow(cond_over_null.T, vmin=0, vmax=1)
plt.tight_layout()

fig, axes = plt.subplots(2,1)
plt.suptitle("Simple superiority")
inds = mins==1
null_over_simp = mins.copy()
null_over_simp[~inds] = np.nan
null_over_simp[inds] = 1 - np.exp((aics[1,inds]-aics[0,inds])/2)
axes[0].imshow(null_over_simp.T, vmin=0, vmax=1)
cond_over_simp = mins.copy()
cond_over_simp[~inds] = np.nan
cond_over_simp[inds] = 1 - np.exp((aics[1,inds]-aics[2,inds])/2)
axes[1].imshow(cond_over_simp.T, vmin=0, vmax=1)
plt.tight_layout()

fig, axes = plt.subplots(2,1)
plt.suptitle("Condition superiority")
inds = mins==2
null_over_cond = mins.copy()
null_over_cond[~inds] = np.nan
null_over_cond[inds] = 1 - np.exp((aics[2,inds]-aics[0,inds])/2)
axes[0].imshow(null_over_cond.T, vmin=0, vmax=1)
simp_over_cond = mins.copy()
simp_over_cond[~inds] = np.nan
simp_over_cond[inds] = 1 - np.exp((aics[2,inds]-aics[1,inds])/2)
axes[1].imshow(simp_over_cond.T, vmin=0, vmax=1)
plt.tight_layout()
