import mne
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

'''
Plot a continuous variable by condition
'''

def cond2vec(exog_names, params, keys_cond):
    out_vec = np.zeros(len(exog_names))
    for param in params:
        out_vec[exog_names.index(keys_cond[param])] = 1
    return out_vec

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

cond_keys = {"EigFreq":"EigFreq",
             "EigFreq:C(StimType, Treatment('sham'))[T.eig]":"Eigenfrequency",
             "EigFreq:C(StimType, Treatment('sham'))[T.fix]":"Fixed frequency",
             "EigFreq:C(Dur, Treatment('30s'))[T.2m]":"2m",
             "EigFreq:C(Dur, Treatment('30s'))[T.5m]":"5m",
             "EigFreq:C(Dur, Treatment('30s'))[T.2m]:C(StimType, Treatment('sham'))[T.eig]":"Eigenfrequency 2m",
             "EigFreq:C(Dur, Treatment('30s'))[T.2m]:C(StimType, Treatment('sham'))[T.fix]":"Fixed frequency 2m",
             "EigFreq:C(Dur, Treatment('30s'))[T.5m]:C(StimType, Treatment('sham'))[T.eig]":"Eigenfrequency 5m",
             "EigFreq:C(Dur, Treatment('30s'))[T.5m]:C(StimType, Treatment('sham'))[T.fix]":"Fixed frequency 5m"
            }

durs = ["30s", "2m", "5m"]
conds = ["sham","fix","eig"]
osc = "SO"
baseline = "zscore"
use_group = "nogroup"
badsubjs = "all_subj"
cont_var = "EigFreq"
if baseline == "zscore":
    vmin, vmax = -6, 6
elif baseline == "logmean":
    vmin, vmax = -.35, .35

keys_cond = {v:k for k,v in cond_keys.items()}

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()
epo = mne.read_epochs(proc_dir+"grand_central_finfo-epo.fif")

e = epo["OscType=='{}'".format(osc)]
e.resample(tfr.info["sfreq"], n_jobs="cuda")
e.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
if osc == "deltO":
    tfr_avg.crop(tmin=-0.75, tmax=0.75)
    epo.crop(tmin=-0.75, tmax=0.75)
# calculate global ERP min and max for scaling later on
evo = e.average()
ev_min, ev_max = evo.data.min(), evo.data.max()
# get osc ERP and normalise
evo_data = evo.data
evo_data = (evo_data - ev_min) / (ev_max - ev_min)
evo_data = evo_data*4 + 12


stat_conds = list(cond_keys.keys())
tfr_c = tfr_avg.copy()
dat_shape = tfr_c.data.shape[1:]
infile = "{}main_fits_{}_{}_{}_{}_cont_{}.pickle".format(proc_dir, baseline,
                                                         osc, badsubjs,
                                                         use_group, cont_var)
with open(infile, "rb") as f:
    fits = pickle.load(f)
exog_names = fits["exog_names"]
modfit = fits["fits"]

fig, axes = plt.subplots(3, 3, figsize=(38.4,21.6))
axes = [ax for axe in axes for ax in axe]
for en_idx,en in enumerate(list(cond_keys.keys())):
    data = np.zeros((3, len(modfit)))
    for mf_idx, mf in enumerate(modfit):
        data[0, mf_idx] = mf.params[exog_names.index(en)]
        data[1, mf_idx] = mf.tvalues[exog_names.index(en)]
        data[2, mf_idx] = mf.pvalues[exog_names.index(en)]

    pvals = data[2,].reshape(*dat_shape, order="F")
    pvals[np.isnan(pvals)] = 1
    mask = pvals<0.05
    if "Intercept" in en:
        mask = None
    dat = data[0,].reshape(*dat_shape, order="F")
    dat[np.isnan(dat)] = 0
    tfr_c.data[0,] = dat
    tfr_c.plot(picks="central", axes=axes[en_idx], colorbar=False,
               vmin=vmin, vmax=vmax, cmap="viridis", mask=mask,
               mask_style="contour")
    axes[en_idx].plot(tfr.times, evo_data[0,],
                      color="gray", alpha=0.8,
                      linewidth=10)
    axes[en_idx].set_title(cond_keys[en])

    fig.suptitle("{} Power by Eigenfrequency".format(osc))
    fig.tight_layout()
    fig.savefig("../images/lmmtfr_contcat_{}_{}_{}_{}.tif".format(cont_var, osc, badsubjs, use_group))
