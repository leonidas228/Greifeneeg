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
dur = "30s"
conds = ["sham","fix","eig"]
oscs = ["SO"]
baseline = "zscore"
cond_keys = {'Intercept':"sham",
             "C(Cond, Treatment('sham{}'))[T.eig{}]".format(dur,dur):"eig{}".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]".format(dur,dur):"fix{}".format(dur),
             "C(Index, Treatment('Pre'))[T.0]":"sham{}_0".format(dur),
             "C(Index, Treatment('Pre'))[T.1]":"sham{}_1".format(dur),
             "C(Index, Treatment('Pre'))[T.2]":"sham{}_2".format(dur),
             "C(Index, Treatment('Pre'))[T.3]":"sham{}_3".format(dur),
             "C(Index, Treatment('Pre'))[T.4]":"sham{}_4".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.0]".format(dur,dur):"eig{}_0".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.0]".format(dur,dur):"fix{}_0".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.1]".format(dur,dur):"eig{}_1".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.1]".format(dur,dur):"fix{}_1".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.2]".format(dur,dur):"eig{}_2".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.2]".format(dur,dur):"fix{}_2".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.3]".format(dur,dur):"eig{}_3".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.3]".format(dur,dur):"fix{}_3".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.4]".format(dur,dur):"eig{}_4".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.4]".format(dur,dur):"fix{}_4".format(dur)}
key_conds = {v:k for k,v in cond_keys.items()}

inds = ["0","1","2","3","4"]
conds = ["sham","fix","eig"]

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

for osc in oscs:
    for sync in syncs:
        fig, axes = plt.subplots(5,6)
        tfr_c = tfr_avg.copy()
        dat_shape = tfr_c.data.shape[1:]
        with open("{}main_fits_{}_condidx_{}_{}_{}.pickle".format(proc_dir,baseline,osc,dur,sync), "rb") as f:
            fits = pickle.load(f)
        modfit = fits["fits"]
        exog_names = fits["exog_names"]
        for ind_idx, ind in enumerate(inds):
            for cond_idx, cond in enumerate(conds):
                data = np.zeros((3, len(modfit)))
                for mf_idx, mf in enumerate(modfit):
                    data[0, mf_idx] = mf.params[exog_names.index(key_conds["{}{}_{}".format(cond,dur,ind)])]
                    data[1, mf_idx] = mf.tvalues[exog_names.index(key_conds["{}{}_{}".format(cond,dur,ind)])]
                    data[2, mf_idx] = mf.pvalues[exog_names.index(key_conds["{}{}_{}".format(cond,dur,ind)])]
                pvals = data[2,].reshape(*dat_shape, order="F")
                pvals[np.isnan(pvals)] = 1
                mask = pvals<0.05
                for val_idx,val in enumerate(["coef","tval"]):
                    dat = data[val_idx,].reshape(*dat_shape, order="F")
                    dat[np.isnan(dat)] = 0
                    tfr_c.data[0,] = dat
                    if val == "tval":
                        vmin, vmax = -4, 4
                    else:
                        vmin, vmax = -4, 4
                    tfr_c.plot(picks="central", axes=axes[ind_idx][cond_idx*2+val_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")

        fig, axes = plt.subplots(2,2)
        for cond_idx,cond in enumerate(["fix","eig"]):
            data = np.zeros((3, len(modfit)))
            for mf_idx, mf in enumerate(modfit):
                data[0, mf_idx] = mf.params[exog_names.index(key_conds["{}{}".format(cond,dur)])]
                data[1, mf_idx] = mf.tvalues[exog_names.index(key_conds["{}{}".format(cond,dur)])]
                data[2, mf_idx] = mf.pvalues[exog_names.index(key_conds["{}{}".format(cond,dur)])]
            pvals = data[2,].reshape(*dat_shape, order="F")
            pvals[np.isnan(pvals)] = 1
            mask = pvals < 0.05
            for val_idx,val in enumerate(["coef","tval"]):
                dat = data[val_idx,].reshape(*dat_shape, order="F")
                dat[np.isnan(dat)] = 0
                tfr_c.data[0,] = dat
                if val == "tval":
                    vmin, vmax = -4, 4
                else:
                    vmin, vmax = -4, 4
                tfr_c.plot(picks="central", axes=axes[cond_idx][val_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")
