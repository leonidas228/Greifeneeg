import mne
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

syncs = ["async", "sync"]
#syncs = ["sync"]
durs = ["30s", "2m", "5m"]
#durs = ["30s"]
baseline = "logratio"

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

for sync in syncs:
    for dur in durs:
        cond_keys = {"Intercept":"sham",
                     "C(Cond, Treatment('sham{}'))[T.eig{}]".format(dur,dur):"eig{}".format(dur),
                     "C(Cond, Treatment('sham{}'))[T.fix{}]".format(dur,dur):"fix{}".format(dur),
                     "C(Cond, Treatment('sham'))[T.eig{}]".format(dur):"eig{}".format(dur),
                     "C(Cond, Treatment('sham'))[T.fix{}]".format(dur):"fix{}".format(dur),
                     "C(Stim, Treatment('sham'))[T.stim]":"stim"}
        tfr_c = tfr_avg.copy()
        dat_shape = tfr_c.data.shape[1:]
        with open("{}main_fits_{}_cond_SO_{}_{}.pickle".format(proc_dir,baseline,dur,sync), "rb") as f:
            fits = pickle.load(f)
        exog_names = fits["exog_names"]
        modfit = fits["fits"]
        for en in exog_names:
            exog_array = np.zeros(len(exog_names))
            exog_array[0] = 1
            ex_idx = exog_names.index(en)
            exog_array[ex_idx] = 1
            dat = np.zeros(len(modfit))
            for mf_idx, mf in enumerate(modfit):
                dat[mf_idx] = mf.predict(exog_array)[0]
            dat = dat.reshape(*dat_shape, order="F")
            tfr_c.data[0,] = dat
            tfr_c.save("{}predicted_cond_SO_{}_{}_{}_{}-tfr.h5".format(proc_dir,baseline,dur,sync,cond_keys[en]), overwrite=True)
