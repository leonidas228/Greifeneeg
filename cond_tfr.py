import mne
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 26}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

syncs = ["async", "sync"]
durs = ["30s", "2m", "5m"]
conds = ["sham","fix","eig"]
oscs = ["SO", "deltO"]
#oscs = ["SO"]
baseline = "zscore"
to_plot = ["coef"]
col_titles = ["Intercept (sham)", "Fixed frequency", "Eigenfrequency"]

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")
epo.resample(tfr.info["sfreq"], n_jobs="cuda")
epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
# calculate global ERP min and max for scaling later on
evo = epo.average()
ev_min, ev_max = evo.data.min(), evo.data.max()

for osc in oscs:
    for sync in syncs:
        e = epo.copy()
        if sync == "sync":
            subjs = epo.metadata["Subj"]
            bad_subjs = []
            if sync == "sync":
                for subj in list(subjs):
                    if int(subj) < 31:
                        bad_subjs.append(subj)
            bad_subjs = list(set(bad_subjs))
            for bs in bad_subjs:
                print("Removing subject {}".format(bs))
                e = e["Subj!='{}'".format(bs)]

        fig, axes = plt.subplots(len(durs),len(conds)*len(to_plot), figsize=(38.4,21.6))
        for dur_idx,dur in enumerate(durs):
            cond_keys = {"Intercept":"sham{}".format(dur),
                         "C(Cond, Treatment('sham{}'))[T.eig{}]".format(dur,dur):"eig{}".format(dur),
                         "C(Cond, Treatment('sham{}'))[T.fix{}]".format(dur,dur):"fix{}".format(dur)}

            tfr_c = tfr_avg.copy()
            dat_shape = tfr_c.data.shape[1:]
            with open("{}main_fits_{}_cond_{}_{}_{}.pickle".format(proc_dir,baseline,osc,dur,sync), "rb") as f:
                fits = pickle.load(f)
            exog_names = fits["exog_names"]
            modfit = fits["fits"]
            for en_idx,en in enumerate(exog_names[0:]):

                # get osc ERP and normalise
                evo = e["OscType=='{}' and Cond=='{}'".format(osc, cond_keys[en])].average()
                evo_data = evo.data
                evo_data = (evo_data - ev_min) / (ev_max - ev_min)
                evo_data = evo_data*3 + 13

                data = np.zeros((3, len(modfit)))
                for mf_idx, mf in enumerate(modfit):
                    data[0, mf_idx] = mf.params[exog_names.index(en)]
                    data[1, mf_idx] = mf.tvalues[exog_names.index(en)]
                    data[2, mf_idx] = mf.pvalues[exog_names.index(en)]
                pvals = data[2,].reshape(*dat_shape, order="F")
                pvals[np.isnan(pvals)] = 1
                mask = pvals<0.05
                if en == "Intercept":
                    mask = None
                for idx,val in enumerate(to_plot):
                    dat = data[idx,].reshape(*dat_shape, order="F")
                    dat[np.isnan(dat)] = 0
                    tfr_c.data[0,] = dat
                    if val == "tval":
                        if baseline == "zscore":
                            vmin, vmax = -4, 4
                        elif baseline == "zlogratio":
                            vmin, vmax = None, None
                    else:
                        vmin, vmax = -1.5, 1.5
                    tfr_c.plot(picks="central", axes=axes[dur_idx][en_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")
                    axes[dur_idx][en_idx].plot(tfr.times, evo_data[0,],
                                               color="gray", alpha=0.8,
                                               linewidth=10)
                    if dur_idx == 0:
                        axes[dur_idx][en_idx].set_title(col_titles[en_idx])
                    if en_idx*len(to_plot)+idx == len(exog_names)*len(to_plot)-1:
                        rax = axes[dur_idx][en_idx].twinx()
                        plt.ylabel("{}".format(dur))
                        plt.yticks(ticks=[], labels=[])

        if sync == "sync":
            plt.suptitle("{}, synchronised only".format(osc))
        else:
            plt.suptitle("{}, non-synchronised included".format(osc))
        plt.savefig("../images/lmmtfr_{}_{}.tif".format(osc, sync))
