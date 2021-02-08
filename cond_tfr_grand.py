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

cond_keys = {"Intercept":"Intercept (sham30s)",
             "C(StimType, Treatment('sham'))[T.eig]":"Eigenfrequency",
             "C(StimType, Treatment('sham'))[T.fix]":"Fixed frequency",
             "C(Dur, Treatment('30s'))[T.2m]":"2m",
             "C(Dur, Treatment('30s'))[T.5m]":"5m",
             "C(Sync, Treatment('sync'))[T.async]":"non-synchronised",
             "C(StimType, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.2m]":"Eigenfrequency 2m",
             "C(StimType, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.2m]":"Fixed frequency 2m",
             "C(StimType, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.5m]":"Eigenfrequency 5m",
             "C(StimType, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.5m]":"Fixed frequency 5m",
             "C(StimType, Treatment('sham'))[T.eig]:C(Sync, Treatment('sync'))[T.async]":"Eigenfrequency non-synchronised",
             "C(StimType, Treatment('sham'))[T.fix]:C(Sync, Treatment('sync'))[T.async]":"Fixed frequency non-synchronised",
             "C(Dur, Treatment('30s'))[T.2m]:C(Sync, Treatment('sync'))[T.async]":"2m non-synchronised",
             "C(Dur, Treatment('30s'))[T.5m]:C(Sync, Treatment('sync'))[T.async]":"5m non-synchronised",
             "C(StimType, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.2m]:C(Sync, Treatment('sync'))[T.async]":"Eigenfrequency 2m non-synchronised",
             "C(StimType, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.2m]:C(Sync, Treatment('sync'))[T.async]":"Fixed frequency 2m non-synchronised",
             "C(StimType, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.5m]:C(Sync, Treatment('sync'))[T.async]":"Eigenfrequency 5m non-synchronised",
             "C(StimType, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.5m]:C(Sync, Treatment('sync'))[T.async]":"Fixed frequency 5m non-synchronised"}

vmin, vmax = -2.5, 2.5
durs = ["30s", "2m", "5m"]
conds = ["sham","fix","eig"]
oscs = ["SO", "deltO"]
oscs = ["SO"]
baseline = "zscore"
sync_facts = ["syncfact", "nosyncfact"]
sync_facts = ["syncfact"]
use_groups = ["group", "nogroup"]
use_groups = ["group"]
balance_conds = False
bootstrap = True
use_badsubjs = ["all_subj"]
#use_badsubjs = ["bad10"]
#use_badsubjs = ["async"]

cond_exogs_syncfact = {  "Sham 30s synchronised":["Intercept (sham30s synchronised)"],
                "Eigenfrequency 30s synchronised":["Intercept (sham30s synchronised)", "Eigenfrequency"],
                "Fixed frequency 30s synchronised":["Intercept (sham30s synchronised)", "Fixed frequency"],
                "Sham 2m synchronised":["Intercept (sham30s synchronised)", "2m"],
                "Eigenfrequency 2m synchronised":["Intercept (sham30s synchronised)", "2m", "Eigenfrequency", "Eigenfrequency 2m"],
                "Fixed frequency 2m synchronised":["Intercept (sham30s synchronised)", "2m", "Fixed frequency", "Fixed frequency 2m"],
                "Sham 5m synchronised":["Intercept (sham30s synchronised)", "5m"],
                "Eigenfrequency 5m synchronised":["Intercept (sham30s synchronised)", "5m", "Eigenfrequency", "Eigenfrequency 5m"],
                "Fixed frequency 5m synchronised":["Intercept (sham30s synchronised)", "Fixed frequency", "Fixed frequency 5m"],
                "Sham 30s non-synchronised":["Intercept (sham30s synchronised)", "non-synchronised"],
                "Eigenfrequency 30s non-synchronised":["Intercept (sham30s synchronised)", "Eigenfrequency", "non-synchronised", "Eigenfrequency non-synchronised"],
                "Fixed frequency 30s non-synchronised":["Intercept (sham30s synchronised)", "Fixed frequency", "non-synchronised", "Fixed frequency non-synchronised"],
                "Sham 2m non-synchronised":["Intercept (sham30s synchronised)", "2m", "non-synchronised"],
                "Eigenfrequency 2m non-synchronised":["Intercept (sham30s synchronised)", "2m", "Eigenfrequency", "Eigenfrequency 2m", "non-synchronised", "Eigenfrequency non-synchronised", "Eigenfrequency 2m non-synchronised"],
                "Fixed frequency 2m non-synchronised":["Intercept (sham30s synchronised)", "2m", "Fixed frequency", "Fixed frequency 2m", "non-synchronised", "Fixed frequency non-synchronised", "Fixed frequency 2m non-synchronised"],
                "Sham 5m non-synchronised":["Intercept (sham30s synchronised)", "5m", "non-synchronised"],
                "Eigenfrequency 5m non-synchronised":["Intercept (sham30s synchronised)", "5m", "Eigenfrequency", "Eigenfrequency 5m", "non-synchronised", "Eigenfrequency non-synchronised", "Eigenfrequency 5m non-synchronised"],
                "Fixed frequency 5m non-synchronised":["Intercept (sham30s synchronised)", "Fixed frequency", "Fixed frequency 5m", "non-synchronised", "Fixed frequency non-synchronised", "Fixed frequency 5m non-synchronised"]
            }

cond_exogs_nosyncfact = {  "Sham 30s":["Intercept (sham30s)"],
                "Eigenfrequency 30s":["Intercept (sham30s)", "Eigenfrequency"],
                "Fixed frequency 30s":["Intercept (sham30s)", "Fixed frequency"],
                "Sham 2m":["Intercept (sham30s)", "2m"],
                "Eigenfrequency 2m":["Intercept (sham30s)", "2m", "Eigenfrequency", "Eigenfrequency 2m"],
                "Fixed frequency 2m":["Intercept (sham30s)", "2m", "Fixed frequency", "Fixed frequency 2m"],
                "Sham 5m":["Intercept (sham30s)", "5m"],
                "Eigenfrequency 5m":["Intercept (sham30s)", "5m", "Eigenfrequency", "Eigenfrequency 5m"],
                "Fixed frequency 5m":["Intercept (sham30s)", "Fixed frequency", "Fixed frequency 5m"]
            }

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()
epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")

for osc in oscs:
    e = epo["OscType=='{}'".format(osc)]
    e.resample(tfr.info["sfreq"], n_jobs="cuda")
    e.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
    # calculate global ERP min and max for scaling later on
    evo = e.average()
    ev_min, ev_max = evo.data.min(), evo.data.max()
    # get osc ERP and normalise
    evo_data = evo.data
    evo_data = (evo_data - ev_min) / (ev_max - ev_min)
    evo_data = evo_data*4 + 12
    for bs_name in use_badsubjs:
        for use_group in use_groups:
            for sync_fact in sync_facts:
                if sync_fact == "nosyncfact":
                    cond_exogs = cond_exogs_nosyncfact
                    cond_keys["Intercept"] = "Intercept (sham30s)"
                    syncfact_order = []
                else:
                    cond_exogs = cond_exogs_syncfact
                    cond_keys["Intercept"] = "Intercept (sham30s synchronised)"
                    syncfact_order = [c for c in cond_keys.values() if "non-synchronised" in c]
                nosyncfact_order = [c for c in cond_keys.values() if "non-synchronised" not in c]
                orders = [nosyncfact_order, syncfact_order]
                keys_cond = {v:k for k,v in cond_keys.items()}
                stat_conds = list(cond_keys.keys())
                tfr_c = tfr_avg.copy()
                dat_shape = tfr_c.data.shape[1:]
                with open("{}main_fits_{}_grand_{}_{}_{}_{}.pickle".format(proc_dir, baseline, osc, bs_name, use_group, sync_fact), "rb") as f:
                    fits = pickle.load(f)
                exog_names = fits["exog_names"]
                modfit = fits["fits"]
                for order_idx, order in enumerate(orders):
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

                    fig.suptitle("{}_{}_{}_{}".format(osc, bs_name, use_group, sync_fact))
                    if sync_fact == "syncfact":
                        fig.suptitle("LME model parameters of SO spindle power, synchronicity tested")
                    else:
                        fig.suptitle("LME model parameters of SO spindle power, synchronicity not tested")
                    fig.tight_layout()
                    fig.savefig("../images/lmmtfr_grand_{}_{}_{}_{}_{}.tif".format(osc, bs_name, use_group, sync_fact, order_idx))

                fig0, axes0 = plt.subplots(3, 3, figsize=(38.4,21.6))
                if sync_fact == "syncfact":
                    fig1, axes1 = plt.subplots(3, 3, figsize=(38.4,21.6))
                else:
                    axes1 = []
                axes = [ax for axes in [axes0, axes1] for axe in axes for ax in axe]
                for cond_idx, (cond_k, cond_v) in enumerate(cond_exogs.items()):
                    data = np.zeros(len(modfit))
                    cond_vec = cond2vec(exog_names, cond_v, keys_cond)
                    for mf_idx, mf in enumerate(modfit):
                        data[mf_idx] = mf.predict(cond_vec)
                    data = data.reshape(*dat_shape, order="F")
                    data[np.isnan(data)] = 0
                    tfr_c.data[0,] = data
                    tfr_c.plot(picks="central", axes=axes[cond_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis")
                    axes[cond_idx].plot(tfr.times, evo_data[0,],
                                        color="gray", alpha=0.8,
                                        linewidth=10)
                    axes[cond_idx].set_title(cond_k)

                if sync_fact == "syncfact":
                    fig0.suptitle("LME model predictions of SO spindle power, synchronicity tested")
                    fig1.suptitle("LME model predictions of SO spindle power, synchronicity tested")
                    fig1.tight_layout()
                    fig1.savefig("../images/lmmtfr_grand_predict_{}_{}_{}_{}_1.tif".format(osc, bs_name, use_group, sync_fact))
                else:
                    fig0.suptitle("LME model predictions of SO spindle power, synchronicity not tested")
                fig0.tight_layout()
                fig0.savefig("../images/lmmtfr_grand_predict_{}_{}_{}_{}_0.tif".format(osc, bs_name, use_group, sync_fact))
