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

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

cond_keys = {'Intercept':"Intercept (sham30s)",
             "C(Stimtype, Treatment('sham'))[T.eig]":"Eigenfrequency",
             "C(Stimtype, Treatment('sham'))[T.fix]":"Fixed frequency",
             "C(Dur, Treatment('30s'))[T.2m]":"2m",
             "C(Dur, Treatment('30s'))[T.5m]":"5m",
             "C(Sync, Treatment('sync'))[T.async]":"non-synchronised",
             "C(Stimtype, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.2m]":"Eigenfrequency 2m",
             "C(Stimtype, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.2m]":"Fixed frequency 2m",
             "C(Stimtype, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.5m]":"Eigenfrequency 5m",
             "C(Stimtype, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.5m]":"Fixed frequency 5m",
             "C(Stimtype, Treatment('sham'))[T.eig]:C(Sync, Treatment('sync'))[T.async]":"Eigenfrequency non-synchronised",
             "C(Stimtype, Treatment('sham'))[T.fix]:C(Sync, Treatment('sync'))[T.async]":"Fixed frequency non-synchronised",
             "C(Dur, Treatment('30s'))[T.2m]:C(Sync, Treatment('sync'))[T.async]":"2m non-synchronised",
             "C(Dur, Treatment('30s'))[T.5m]:C(Sync, Treatment('sync'))[T.async]":"5m non-synchronised",
             "C(Stimtype, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.2m]:C(Sync, Treatment('sync'))[T.async]":"Eigenfrequency 2m non-synchronised",
             "C(Stimtype, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.2m]:C(Sync, Treatment('sync'))[T.async]":"Fixed frequency 2m non-synchronised",
             "C(Stimtype, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.5m]:C(Sync, Treatment('sync'))[T.async]":"Eigenfrequency 5m non-synchronised",
             "C(Stimtype, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.5m]:C(Sync, Treatment('sync'))[T.async]":"Fixed frequency 5m non-synchronised"}

vmin, vmax = -2.5, 2.5
durs = ["30s", "2m", "5m"]
conds = ["sham","fix","eig"]
oscs = ["SO", "deltO"]
oscs = ["SO"]
baseline = "zscore"
col_dict = {"sham":"Intercept (sham)", "fix":"Fixed frequency",
            "eig":"Eigenfrequency", "fix_sync":"Fixed frequency\nSynchronisation",
            "eig_sync":"Eigenfrequency\nSynchronisation"}
sync_facts = ["syncfact", "nosyncfact"]
sync_facts = ["syncfact"]
use_groups = ["group", "nogroup"]
use_groups = ["group"]
balance_conds = False
bootstrap = True
use_badsubjs = ["all_subj"]
#use_badsubjs = ["bad10"]
#use_badsubjs = ["async"]

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")
epo.resample(tfr.info["sfreq"], n_jobs="cuda")
epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
# calculate global ERP min and max for scaling later on
evo = epo.average()
ev_min, ev_max = evo.data.min(), evo.data.max()
# get osc ERP and normalise
evo_data = evo.data
evo_data = (evo_data - ev_min) / (ev_max - ev_min)
evo_data = evo_data*3 + 13

for osc in oscs:
    for bs_name in use_badsubjs:
        for use_group in use_groups:
            for sync_fact in sync_facts:
                if sync_fact == "nosyncfact":
                    cond_keys["Intercept"] = "Intercept (sham30s)"
                    syncfact_order = []
                else:
                    cond_keys["Intercept"] = "Intercept (sham30s synchronised)"
                    syncfact_order = [c for c in cond_keys.values() if "non-synchronised" in c]
                nosyncfact_order = [c for c in cond_keys.values() if "non-synchronised" not in c]
                orders = [nosyncfact_order, syncfact_order]
                keys_cond = {v:k for k,v in cond_keys.items()}
                tfr_c = tfr_avg.copy()
                dat_shape = tfr_c.data.shape[1:]
                with open("{}main_fits_{}_grand_{}_{}_{}_{}.pickle".format(proc_dir, baseline, osc, bs_name, use_group, sync_fact), "rb") as f:
                    fits = pickle.load(f)
                exog_names = fits["exog_names"]
                modfit = fits["fits"]
                for order in orders:
                    if len(order) == 0:
                        continue
                    fig, axes = plt.subplots(3, 3, figsize=(38.4,21.6))
                    axes = [ax for axe in axes for ax in axe]
                    for en_idx,en in enumerate(order):
                        data = np.zeros((3, len(modfit)))
                        for mf_idx, mf in enumerate(modfit):
                            data[0, mf_idx] = mf.params[exog_names.index(keys_cond[en])]
                            data[1, mf_idx] = mf.tvalues[exog_names.index(keys_cond[en])]
                            data[2, mf_idx] = mf.pvalues[exog_names.index(keys_cond[en])]
                        pvals = data[2,].reshape(*dat_shape, order="F")
                        pvals[np.isnan(pvals)] = 1
                        mask = pvals<0.05
                        if "Intercept" in en:
                            mask = None
                        dat = data[0,].reshape(*dat_shape, order="F")
                        dat[np.isnan(dat)] = 0
                        tfr_c.data[0,] = dat
                        tfr_c.plot(picks="central", axes=axes[en_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")
                        axes[en_idx].plot(tfr.times, evo_data[0,],
                                          color="gray", alpha=0.8,
                                          linewidth=10)
                        axes[en_idx].set_title(en)

                plt.suptitle("{}_{}_{}_{}".format(osc, bs_name, use_group, sync_fact))
                plt.tight_layout()
                plt.savefig("../images/lmmtfr_grand_{}_{}_{}_{}.tif".format(osc, bs_name, use_group, sync_fact))
