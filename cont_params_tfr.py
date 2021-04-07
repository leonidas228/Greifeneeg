import mne
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle
from mne.stats import fdr_correction
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
import seaborn as sns

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

group_slope = False
cont_var = "Fehler875"
vmin, vmax = -5, 5
#vmin, vmax = -0.3, 0.3
durs = ["30s", "2m", "5m"]
conds = ["sham","fix","eig"]
oscs = ["SO", "deltO"]
oscs = ["SO"]
baseline = "zscore"
sync_facts = ["syncfact", "nosyncfact"]
sync_facts = ["nosyncfact"]
use_groups = ["group", "nogroup"]
use_groups = ["nogroup"]
balance_conds = False
bootstrap = True
use_badsubjs = ["no2,3,28"]
#use_badsubjs = ["bad10"]
#use_badsubjs = ["async"]

fdr_cor = True
toi = .308
foi = 13
# toi = .365
# foi = 16
# toi = None
# foi = None

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

if toi != None and foi != None:
    toi_idx = find_nearest(tfr.times, toi)
    foi_idx = find_nearest(tfr.freqs, foi)
    roi_mask = np.zeros(tfr.data.shape[2:])
    roi_mask[...,foi_idx,toi_idx] = 1
    dat_idx = toi_idx*len(tfr.freqs) + foi_idx

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
            tfr_c = tfr_avg.copy()
            dat_shape = tfr_c.data.shape[1:]
            outfile = "{}main_fits_{}_{}_{}_{}_cont_{}.pickle".format(proc_dir, baseline, osc, bs_name, use_group, cont_var)
            if group_slope:
                outfile = "{}main_fits_{}_{}_{}_{}_cont_{}_indslope.pickle".format(proc_dir, baseline, osc, bs_name, use_group, cont_var)
            with open(outfile, "rb") as f:
                fits = pickle.load(f)
            exog_names = fits["exog_names"]
            modfit = fits["fits"]
            fig, axes = plt.subplots(1, len(exog_names), figsize=(38.4,21.6))
            #axes = [ax for axe in axes for ax in axe]
            for en_idx,en in enumerate(exog_names):
                data = np.zeros((3, len(modfit)))
                for mf_idx, mf in enumerate(modfit):
                    data[0, mf_idx] = mf.params[exog_names.index(en)]
                    data[1, mf_idx] = mf.tvalues[exog_names.index(en)]
                    data[2, mf_idx] = mf.pvalues[exog_names.index(en)]
                    if mf_idx == dat_idx:
                        moi = mf

                pvals = data[2,].reshape(*dat_shape, order="F")
                pvals[np.isnan(pvals)] = 1

                if fdr_cor:
                    _, pvals = fdr_correction(pvals)

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
            plt.tight_layout()
            plt.suptitle("Estimated parameters")
            plt.savefig("../images/lmmtfr_{}_{}_{}_cont_{}.tif".format(osc, bs_name, use_group, cont_var))

            if cont_var == "OscFreq":
                fig, axes = plt.subplots(1,2, figsize=(38.4,21.6))
                tfr_c.plot(picks="central", colorbar=False, vmin=vmin,
                           vmax=vmax, cmap="viridis", mask=roi_mask,
                           mask_style="contour", axes=axes[0])
                axes[0].set_title(en+" Time-Frequency of interest")
                axes[0].plot(tfr.times, evo_data[0,],
                            color="gray", alpha=0.8,
                            linewidth=10)

                sns.scatterplot(x=moi.model.exog[:,1], y=moi.model.endog, ax=axes[1])
                z = np.polyfit(moi.model.exog[:,1], moi.model.endog, 1)
                p = np.poly1d(z)
                axes[1].plot(moi.model.exog[:,1], p(moi.model.exog[:,1]))
                axes[1].set_ylabel("Log power")
                axes[1].set_xlabel("SO frequency")
                axes[1].set_title("Log power*SO frequency at TFOI")
                plt.suptitle("")
                plt.savefig("../images/lmmtfr_{}_{}_{}_cont_tfoi_{}.tif".format(osc, bs_name, use_group, cont_var))

                fig, axes = plt.subplots(5, 6, figsize=(38.4,21.6))
                axes = [ax for axe in axes for ax in axe]
                subjs = list(np.unique(moi.model.groups))
                subjs.sort()
                for off_ax in axes[len(subjs):]:
                    off_ax.axis("off")
                for ax, subj in zip(axes, subjs):
                    inds = moi.model.groups==subj
                    sns.scatterplot(x=moi.model.exog[inds,1], y=moi.model.endog[inds], ax=ax)
                    ax.set_ylim((-2.5,2.5))
                    z = np.polyfit(moi.model.exog[inds,1], moi.model.endog[inds], 1)
                    p = np.poly1d(z)
                    ax.plot(moi.model.exog[inds,1], p(moi.model.exog[inds,1]))
                    ax.set_title(subj)
                plt.tight_layout()
                plt.savefig("../images/lmmtfr_{}_{}_{}_cont_tfoisubj_{}.tif".format(osc, bs_name, use_group, cont_var))

            elif cont_var == "StimFreq":
                fig, axes = plt.subplots(1,3, figsize=(38.4,21.6))
                tfr_c.plot(picks="central", colorbar=False, vmin=vmin,
                           vmax=vmax, cmap="viridis", mask=roi_mask,
                           mask_style="contour", axes=axes[0])
                axes[0].set_title(en+" Time-Frequency of interest")
                axes[0].plot(tfr.times, evo_data[0,],
                            color="gray", alpha=0.8,
                            linewidth=10)

                sns.scatterplot(x=moi.model.exog[:,1], y=moi.model.endog, ax=axes[1])
                z = np.polyfit(moi.model.exog[:,1], moi.model.endog, 1)
                p = np.poly1d(z)
                axes[1].plot(moi.model.exog[:,1], p(moi.model.exog[:,1]))
                axes[1].set_ylabel("Log power")
                axes[1].set_xlabel("SO frequency")
                axes[1].set_title("Log power*Stim frequency at TFOI")

                subjs = list(np.unique(moi.model.groups))
                subjs.sort()
                subj_avgs = []
                subj_freqs = []
                for subj in subjs:
                    inds = moi.model.groups==subj
                    subj_avgs.append(moi.model.endog[inds].mean())
                    subj_freqs.append(moi.model.exog[inds,1].mean())
                sns.scatterplot(x=subj_freqs, y=subj_avgs, ax=axes[2])
                z = np.polyfit(subj_freqs, subj_avgs, 1)
                p = np.poly1d(z)
                axes[2].plot(subj_freqs, p(subj_freqs))
                axes[2].set_ylabel("Log power")
                axes[2].set_xlabel("SO frequency")
                axes[2].set_title("Log power*Stim frequency at TFOI, Subject average")

                plt.suptitle("")
                plt.savefig("../images/lmmtfr_{}_{}_{}_cont_tfoisubj_{}.tif".format(osc, bs_name, use_group, cont_var))

                plt.tight_layout()
