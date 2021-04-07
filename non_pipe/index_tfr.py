import mne
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

syncs = ["async"]
durs = ["30s"]
conds = ["sham","fix","eig"]
oscs = ["SO"]
baseline = "zscore"
cond_keys = {"Intercept":"Pre",
             "C(Index, Treatment('Pre'))[T.0]":0,
             "C(Index, Treatment('Pre'))[T.1]":1,
             "C(Index, Treatment('Pre'))[T.2]":2,
             "C(Index, Treatment('Pre'))[T.3]":3,
             "C(Index, Treatment('Pre'))[T.4]":4}

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

for osc in oscs:
    for sync in syncs:
        for dur in durs:
            for cond in conds:
                fig, axes = plt.subplots(5,2)
                tfr_c = tfr_avg.copy()
                dat_shape = tfr_c.data.shape[1:]
                with open("{}main_fits_index_{}_{}_{}{}_{}.pickle".format(proc_dir,baseline,osc,cond,dur,sync), "rb") as f:
                    fits = pickle.load(f)
                exog_names = fits["exog_names"]
                modfit = fits["fits"]
                for en_idx,en in enumerate(exog_names[1:]):
                    data = np.zeros((3, len(modfit)))
                    for mf_idx, mf in enumerate(modfit):
                        data[0, mf_idx] = mf.params[exog_names.index(en)]
                        data[1, mf_idx] = mf.tvalues[exog_names.index(en)]
                        data[2, mf_idx] = mf.pvalues[exog_names.index(en)]
                    pvals = data[2,].reshape(*dat_shape, order="F")
                    pvals[np.isnan(pvals)] = 1
                    mask = pvals<0.05
                    for idx,val in enumerate(["coef","tval"]):
                        dat = data[idx,].reshape(*dat_shape, order="F")
                        dat[np.isnan(dat)] = 0
                        tfr_c.data[0,] = dat
                        if val == "tval":
                            vmin, vmax = -4, 4
                        else:
                            vmin, vmax = -4, 4
                        tfr_c.plot(picks="central", axes=axes[en_idx][idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")
                        tfr_c.save("{}index_tfr_{}_{}_{}{}_{}-tfr.h5".format(proc_dir,baseline,sync,cond,dur,cond_keys[en]), overwrite=True)
                plt.suptitle("{} {} {} {}".format(osc, sync, dur, cond))
